"""
Minimal roll-your-own auth: bcrypt password hashes + signed session cookie
(via Starlette's SessionMiddleware, added in main.py). No third-party auth
service — fine for a friends-only test, not a substitute for a hardened
provider if this ever handles real researcher IP at scale.

Uses the `bcrypt` library directly rather than passlib — passlib's backend
self-check has a known incompatibility with bcrypt>=4.0 that raises a
spurious "password cannot be longer than 72 bytes" error on every hash call.
"""
import bcrypt
from fastapi import Request
from fastapi.responses import RedirectResponse

from . import db

# bcrypt silently truncates beyond this, so reject rather than let two
# different passwords hash to the same value.
MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return db.get_user_by_id(user_id)


def require_login(request: Request):
    """Returns the user row, or a RedirectResponse to /login if not signed in.
    Callers should check `isinstance(result, RedirectResponse)`."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return user
