"""
Minimal roll-your-own auth: bcrypt password hashes + signed session cookie
(via Starlette's SessionMiddleware, added in main.py). No third-party auth
service — fine for a friends-only test, not a substitute for a hardened
provider if this ever handles real researcher IP at scale.

Uses the `bcrypt` library directly rather than passlib — passlib's backend
self-check has a known incompatibility with bcrypt>=4.0 that raises a
spurious "password cannot be longer than 72 bytes" error on every hash call.

Single-use tokens (password reset, email verification) are minted and checked
here too. Only their SHA-256 is ever stored, so a database dump does not hand
over the ability to take over an account.
"""
import hashlib
import hmac
import secrets

import bcrypt
from fastapi import Request
from fastapi.responses import RedirectResponse

from . import db

# bcrypt silently truncates beyond this, so reject rather than let two
# different passwords hash to the same value.
MAX_PASSWORD_BYTES = 72

# 32 bytes of entropy, URL-safe. The link in the email *is* the credential, so
# this has to be unguessable rather than merely unique.
TOKEN_BYTES = 32


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def new_token() -> tuple:
    """Mint a token. Returns `(plaintext, sha256_hex)`.

    The plaintext goes in the email and is never stored; only the hash is. They
    are returned together because losing the plaintext before the mail goes out
    would be unrecoverable.
    """
    plaintext = secrets.token_urlsafe(TOKEN_BYTES)
    return plaintext, hash_token(plaintext)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(candidate: str, stored_hash: str) -> bool:
    """Constant-time comparison of a presented token against the stored hash.

    The row is fetched by hash — an indexed lookup — but the comparison itself is
    still done through `compare_digest` rather than `==`, so a difference in the
    first byte cannot be measured off response time.
    """
    return hmac.compare_digest(hash_token(candidate or ""), stored_hash or "")


def start_session(request: Request, user):
    """Put a user in the session, along with the epoch that keeps them here.

    The single place session keys are written. login, signup and password-change
    all go through it, so a new session key cannot be added to one path and
    forgotten in another — which is the failure that would show up as "some
    sessions never get revoked", the least debuggable version of this bug.

    `epoch` is copied from the row rather than defaulted, because it is the
    counter get_current_user() compares against.
    """
    request.session.clear()
    request.session["user_id"] = user["id"]
    request.session["session_epoch"] = user["session_epoch"]


def get_current_user(request: Request):
    """The signed-in user, or None.

    Also the single place the RLS tenant is installed (invariant: a request can
    never hold a user without also holding a tenant). Every route starts with
    `login_or_redirect` or ends in `render()`, and both come through here — so
    putting it anywhere else would be a place that could be forgotten.
    """
    user_id = request.session.get("user_id")
    if not user_id:
        db.clear_tenant()
        return None
    # Install the tenant from the session id BEFORE reading the row. The `users`
    # policy admits only your own row, so reading first would always return
    # nothing. The cookie is signed with SESSION_SECRET_KEY, so this id is a
    # trustworthy claim rather than user input — and even a forged one would only
    # ever surface the row it claims to be, which is the thing being claimed.
    db.set_tenant(user_id=user_id)
    user = db.get_user_by_id(user_id)
    if not user:
        # Stale session (account deleted, or a different database than the cookie
        # was signed against). Clear it rather than leaving it to fail later.
        db.clear_tenant()
        request.session.clear()
        return None
    # Session revocation (M0.7). A missing epoch counts as revoked, not as
    # epoch 1: cookies minted before this existed are exactly the ones the change
    # exists to kill, so assuming 1 would let them all survive the deploy. The
    # cost of that choice is a one-time logout on deploy, which is the correct
    # default for a security change.
    #
    # Clear the tenant on this path exactly as on the stale-row path above. A
    # revoked cookie that left the previous user's tenant installed would be a
    # cross-tenant read waiting for the next query that forgot to check.
    if request.session.get("session_epoch") != user["session_epoch"]:
        db.clear_tenant()
        request.session.clear()
        return None
    db.set_tenant(user_id=user["id"], lookup_email=user["email"])
    return user


def as_lookup_email(email: str):
    """Scope the pre-authentication paths — login, signup, forgot-password.

    They have to find a user row before anyone is signed in. The `users` RLS
    policy admits exactly the one row whose address matches this setting, so this
    is no wider than the query the app already issues: it is the tenant
    equivalent of looking somebody up by email.
    """
    db.set_tenant(user_id=None, lookup_email=email.strip().lower())


def require_login(request: Request):
    """Returns the user row, or a RedirectResponse to /login if not signed in.
    Callers should check `isinstance(result, RedirectResponse)`."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return user
