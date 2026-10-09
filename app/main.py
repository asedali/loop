import json
import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import (
    auth, config, db, dedupe, llm, mailer, mentors, quota, ratelimit, sources,
    upload,
)
from .constants import (
    ACTION_STEP_STATUSES,
    MENTOR_KEYS,
    MENTOR_SUBJECT_KINDS,
    is_critical,
)

# Every knob below is read from app/config.py, which reads the environment (and
# .env) once. See .env.example for the full list and each default.
DEBUG = config.DEBUG

SECRET_KEY = config.SECRET_KEY
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "dev-only-insecure-secret"
    else:
        raise RuntimeError(
            "SESSION_SECRET_KEY is not set. Generate one with "
            "`python -c \"import secrets; print(secrets.token_hex(32))\"` and set it "
            "in the environment (Fly: `fly secrets set SESSION_SECRET_KEY=...`)."
        )
if SECRET_KEY.startswith("dev-only") and not DEBUG:
    raise RuntimeError("SESSION_SECRET_KEY is still the development default. Set a real one.")

# SameSite=lax already blocks cross-site form POSTs, which covers CSRF for this
# app's endpoints; https_only just makes sure the cookie never rides plaintext.
SESSION_HTTPS_ONLY = config.SESSION_HTTPS_ONLY



@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    # Name the NOBYPASSRLS role so get_conn() switches to it and the RLS policies
    # actually apply. Without this the connection stays as whatever
    # DATABASE_URL says — which on Supabase is service_role, and service_role
    # bypasses every policy regardless of FORCE.
    db.set_app_role(config.db_app_role())
    if db.rls_active():
        print(f"[boot] row-level security ACTIVE (role {config.db_app_role()!r})")
    else:
        print(
            f"[boot] row-level security INERT: the current role bypasses RLS.\n"
            f"[boot] Tenant isolation rests on the user_id filters in app/db.py alone.\n"
            f"[boot] To fix: create a NOBYPASSRLS role, GRANT it to the connecting "
            f"role, and set DB_APP_ROLE to its name."
        )
    yield


app = FastAPI(title="LaunchLoop MVP", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=SESSION_HTTPS_ONLY,
    same_site="lax",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# The mentor catalogue and its "this is not a quote" notice are available to every
# template as globals rather than being passed through each render call. They are
# static, they are needed by a shared macro on three different pages, and adding a
# context key to three render sites for a constant is exactly the drift this repo
# keeps fixing elsewhere. db.py's equivalent note is invariant 10.
templates.env.globals["mentors"] = mentors.all_mentors()
templates.env.globals["mentors_notice"] = mentors.NOT_A_QUOTE

# The function itself, not its result, so the flag is read per render rather
# than frozen at import — and so it reaches the 429 and 500 handlers, which
# build their context by hand and never go through render().
templates.env.globals["email_verification_enabled"] = config.email_verification_enabled


def render(request, template_name, **context):
    context["user"] = auth.get_current_user(request)
    # `request` must be the first positional argument on current Starlette
    # versions — passing it inside the context dict (the old calling
    # convention) causes an internal argument mix-up on newer Starlette
    # releases ("unhashable type: dict" from the template cache lookup).
    return templates.TemplateResponse(request, template_name, context)


def back_to_venture(venture_id: int, message: str = None) -> RedirectResponse:
    """Redirect back to a venture, optionally carrying a message.

    Routes that can't re-render their page (they only ever redirect) used to
    swallow failures outright, so a spent quota or a provider outage looked
    like the button simply not working."""
    url = f"/venture/{venture_id}"
    if message:
        url += f"?err={quote(message[:300])}"
    return RedirectResponse(url=url, status_code=303)


def login_or_redirect(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return None, RedirectResponse(url="/login", status_code=303)
    return user, None


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def throttle_llm(request: Request, user_id: int, back_url: str, saved: str = None) -> Response:
    """Per-request rate limit for the routes that call the model.

    Returns None when the request may proceed, or the 429 response when it may
    not. The check is a single `ratelimit.check` call — it spends a token only
    when the request is allowed, and the number it returns on refusal is the
    same number the page quotes, so the two cannot disagree.

    Rendered in place rather than redirected, with a real reason and a way back,
    because the failure mode this replaces was a swallowed refusal that looked
    like a dead button.
    """
    retry_after = ratelimit.check(user_id, _client_ip(request))
    if retry_after is None:
        return None

    per_user = config.llm_rate_limit_per_min()
    per_ip = config.llm_rate_limit_ip_per_min()
    # Name the tighter of the two, so the sentence matches the bucket that fired.
    scope_note = " from your network" if per_ip and (not per_user or per_ip <= per_user) else ""
    response = templates.TemplateResponse(
        request,
        "rate_limited.html",
        {
            "user": auth.get_current_user(request),
            "limit": min([v for v in (per_user, per_ip) if v] or [0]),
            "window": config.llm_rate_limit_window_seconds(),
            "retry_after": retry_after,
            "back_url": back_url,
            "saved": saved,
            "scope_note": scope_note,
        },
        status_code=429,
    )
    response.headers["Retry-After"] = str(retry_after)
    return response


@app.exception_handler(500)
async def server_error(request: Request, exc: Exception):
    """Last-resort handler. LLM failures are already caught per-route and shown
    inline, so anything reaching here is a genuine bug or an environment
    problem (locked DB, missing template). Render something actionable instead
    of Starlette's bare 'Internal Server Error' body.

    The traceback still goes to the server log — only the response changes.
    """
    traceback.print_exc()
    detail = str(exc) if DEBUG else ""
    return templates.TemplateResponse(
        request,
        "error.html",
        {"user": auth.get_current_user(request), "detail": detail,
         "status_code": 500},
        status_code=500,
    )


@app.get("/healthz")
def healthz():
    """Liveness + DB reachability, for the deploy health check. Deliberately does
    not touch the LLM provider: a provider outage shouldn't fail the deploy
    health check and get the machine cycled.

    `rls` is reported rather than assumed. A connection as a superuser or as
    Supabase's service_role bypasses every RLS policy regardless of FORCE, so a
    deploy can have the policies installed and still have no enforcement — and
    that is exactly the sort of thing that should be visible rather than assumed.
    """
    if not db.ping():
        return {"status": "degraded", "rls": False}
    return {"status": "ok", "rls": db.rls_active()}


# ==================== Auth ====================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    if auth.get_current_user(request):
        return RedirectResponse(url="/dashboard", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request):
    return render(request, "signup.html", error=None, email="")


@app.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    # Pre-authentication: no session yet, so the lookup runs under the
    # address-scoped policy instead of a tenant.
    auth.as_lookup_email(email)
    if db.get_user_by_email(email):
        return render(request, "signup.html", error="An account with that email already exists.", email=email)
    if len(password) < config.min_password_length():
        return render(
            request, "signup.html",
            error=f"Password must be at least {config.min_password_length()} characters.",
            email=email,
        )
    user_id = db.create_user(email, auth.hash_password(password))
    # Best-effort and never blocking: a mail outage must not stop someone signing
    # up, and the unverified banner still lets them work while it is sorted.
    db.set_tenant(user_id=user_id, lookup_email=email)
    if config.email_verification_enabled():
        _issue_email_verification(user_id, email)
    # start_session, not a bare assignment: the session has to carry the epoch
    # that get_current_user() compares, and this is the second of the two places
    # that write one. See auth.start_session.
    auth.start_session(request, db.get_user_by_id(user_id))
    return RedirectResponse(url="/dashboard", status_code=303)


# ---------------- password reset + email verification ----------------

# The forgot-password response is deliberately identical for every outcome —
# registered, unregistered, or mail provider down. Anything that varied would
# turn that form into an account enumeration oracle, so even the redirect target
# does not differ.
_RESET_EXPIRED = "That link is invalid or has expired. Request a new one."


def _token_expiry():
    return db.now() + timedelta(minutes=config.password_reset_token_minutes())


def _issue_email_verification(user_id: int, email: str):
    plaintext, token_hash = auth.new_token()
    db.create_reset_token(user_id, "email_verify", token_hash, _token_expiry())
    return mailer.send_email_verification(email, plaintext)


def _issue_password_reset(user_id: int, email: str):
    plaintext, token_hash = auth.new_token()
    db.create_reset_token(user_id, "password_reset", token_hash, _token_expiry())
    return mailer.send_password_reset(email, plaintext)


@app.get("/forgot-password", response_class=HTMLResponse)
def forgot_password_form(request: Request):
    return render(request, "forgot_password.html", error=None,
                  sent=request.query_params.get("sent") == "1",
                  token_minutes=config.password_reset_token_minutes(), email="")


@app.post("/forgot-password")
def forgot_password(request: Request, email: str = Form(...)):
    email = email.strip().lower()
    auth.as_lookup_email(email)
    user = db.get_user_by_email(email)
    if user:
        # From here on the work touches that user's own rows, so the tenant is
        # installed — the pre-auth lookup above only covered reading `users`.
        db.set_tenant(user_id=user["id"], lookup_email=email)
        _issue_password_reset(user["id"], user["email"])
    # Same redirect whether the account exists, does not exist, or the mail
    # provider is down. A mail failure is not reported here: telling the user
    # "we couldn't send it" would also tell them the account exists.
    return RedirectResponse(url="/forgot-password?sent=1", status_code=303)


def _resolve_token(token: str, purpose: str):
    """Find the row a presented token points at, and install that row's tenant.

    The RLS policy admits this one row by token hash, because the token is the
    credential the user holds *before* anyone knows whose account it belongs to —
    the same shape as the address-scoped login lookup. Once resolved, every write
    runs under the real tenant.
    """
    if not token:
        db.clear_tenant()
        return None
    token_hash = auth.hash_token(token)
    db.set_tenant(lookup_token=token_hash)
    row = db.redeemable_token(token_hash, purpose)
    if row:
        db.set_tenant(user_id=row["user_id"], lookup_token=token_hash)
    return row


@app.get("/reset-password", response_class=HTMLResponse)
def reset_password_form(request: Request):
    done = request.query_params.get("done") == "1"
    token = request.query_params.get("token") or ""
    # A used token is now the normal state of this page after a successful reset,
    # so `done` is checked before the token is validated.
    row = _resolve_token(token, "password_reset")
    return render(request, "reset_password.html",
                  error=None if (done or row) else _RESET_EXPIRED,
                  token=token, done=done,
                  token_minutes=config.password_reset_token_minutes())


@app.post("/reset-password")
async def reset_password(request: Request):
    form = await request.form()
    token = str(form.get("token", "")).strip()
    password = str(form.get("password", ""))
    confirm = str(form.get("confirm", ""))

    row = _resolve_token(token, "password_reset")

    def back(error):
        return render(request, "reset_password.html", error=error, token=token,
                      done=False,
                      token_minutes=config.password_reset_token_minutes())

    if not row:
        # Checked before the password, so an invalid link and a weak password
        # cannot be told apart from the error message.
        return back(_RESET_EXPIRED)
    if password != confirm:
        return back("Those two passwords do not match.")
    if len(password) < config.min_password_length():
        return back(f"Password must be at least {config.min_password_length()} characters.")
    if len(password.encode("utf-8")) > auth.MAX_PASSWORD_BYTES:
        return back(f"Password must be at most {auth.MAX_PASSWORD_BYTES} bytes.")

    db.set_password(row["user_id"], auth.hash_password(password))
    # Burn this token and every other live reset token for the account, so a link
    # captured before the reset cannot be used after it.
    db.consume_user_tokens(row["user_id"], "password_reset")
    return RedirectResponse(url="/reset-password?done=1", status_code=303)


@app.get("/verify-email", response_class=HTMLResponse)
def verify_email(request: Request):
    """Confirm an address from the link in the email.

    A GET because it is a link the user clicks. A third party cannot trigger it
    without the token, and a confirm button would add a step to every account.
    The deliberate POST — resending — is its own route below.
    """
    row = _resolve_token(request.query_params.get("token") or "", "email_verify")
    if row:
        db.mark_email_verified(row["user_id"])
        db.consume_reset_token(row["id"])
        return RedirectResponse(url="/dashboard?note=Email+confirmed.+Thank+you.",
                                status_code=303)
    return RedirectResponse(url=f"/login?err={quote(_RESET_EXPIRED)}", status_code=303)


@app.post("/resend-verification")
def resend_verification(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    # With the flag off no link was ever sent, so there is nothing to resend.
    # The route stays mounted so a stale bookmark from an earlier build lands
    # somewhere honest instead of a bare 404.
    if not config.email_verification_enabled():
        note = "Email confirmation is not enabled."
    elif user["email_verified_at"] is None:
        _issue_email_verification(user["id"], user["email"])
        note = "Verification email sent. Check your inbox."
    else:
        note = "That address is already confirmed."
    return RedirectResponse(url=f"/dashboard?note={quote(note)}", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return render(request, "login.html", error=request.query_params.get("err") or None,
                  note=request.query_params.get("note") or None, email="")


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    ip = _client_ip(request)
    auth.as_lookup_email(email)

    window = config.login_window_minutes()
    since = datetime.now(timezone.utc) - timedelta(minutes=window)
    locked = (
        "Too many failed attempts. Try again in "
        f"{window} minutes."
    )
    # Two counters, because they catch different attacks. Per-email stops password
    # guessing against one account; per-IP stops one address spraying a single
    # password across thousands of accounts, where every individual account has
    # exactly one failed attempt and looks innocent.
    if db.recent_failed_logins(email, since) >= config.max_failed_logins():
        return render(request, "login.html", error=locked, email=email)
    if db.recent_failed_logins_from_ip(ip, since) >= config.max_failed_logins() * 5:
        return render(request, "login.html", error=locked, email=email)

    user = db.get_user_by_email(email)
    ok = bool(user) and auth.verify_password(password, user["password_hash"])
    db.record_login_attempt(email, ip, ok)
    if not ok:
        return render(request, "login.html", error="Invalid email or password.", email=email)

    # Clear the failure history on success. Without this, six typos in a row lock
    # the account out for the rest of the window and the *next* legitimate login
    # is refused — which reads as the app being broken, not as a security feature.
    db.clear_failed_logins(email)
    auth.start_session(request, user)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


@app.get("/logout")
def logout_get(request: Request):
    """Kept so existing bookmarks keep working, but nothing in the UI links
    here — a GET logout can be triggered cross-site by an <img> tag."""
    return logout(request)


# ==================== Dashboard ====================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    ideas = db.list_ideas(user["id"])
    candidates = [i for i in ideas if i["status"] == "candidate"]
    return render(
        request, "dashboard.html",
        ideas=ideas,
        candidates=candidates,
        # Over the candidate set this page renders, so every "Also extracted as"
        # link points at a card that is actually on screen.
        duplicates=dedupe.find_near_duplicates(candidates),
        ideas_by_id={i["id"]: i for i in candidates},
        ventures=db.list_ventures(user["id"]),
        bmc_elements=db.get_bmc_map_for_user(user["id"]),
        quota_used=db.count_llm_calls_this_month(user["id"]),
        quota_limit=quota.monthly_limit(),
        note=request.query_params.get("note") or None,
    )


# ==================== Phase 1: Idea Discovery ====================

@app.get("/phase1/new", response_class=HTMLResponse)
def phase1_new(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    return _phase1(request)


def _upload_limit_label() -> str:
    """Human-readable size for the upload form — nobody reads '8388608'."""
    mib = config.max_upload_bytes() / (1024 * 1024)
    return f"{mib:.0f} MB" if mib >= 1 else f"{config.max_upload_bytes() // 1024} KB"


def _phase1(request, **overrides):
    """Render `/phase1/new` with everything that page needs.

    Four separate ways into this page (paste, upload, import, and the error paths
    of each) all render the same template, and a context variable forgotten in one
    of them is a Jinja undefined that only shows up for that one failure mode. So
    the baseline is built here once and callers override just what differs.
    """
    context = {
        "error": None,
        "raw_text": "",
        "from_upload": None,
        "from_import": None,
        "max_upload_bytes": _upload_limit_label(),
        "max_pdf_pages": config.max_pdf_pages(),
        "import_enabled": config.source_import_enabled(),
        "import_sources": sources.supported(),
    }
    context.update(overrides)
    return render(request, "phase1_new.html", **context)


@app.get("/phase1/ideas", response_class=HTMLResponse)
def phase1_ideas(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    candidates = [i for i in db.list_ideas(user["id"]) if i["status"] == "candidate"]
    shown = candidates[:config.idea_card_limit()]
    # Near-duplicates are computed over the cards this page actually shows, not
    # over every candidate. Otherwise a card could link to a near-duplicate that
    # IDEA_CARD_LIMIT pushed below the fold, and the link would go nowhere.
    return render(
        request, "phase1_ideas.html",
        ideas=shown,
        duplicates=dedupe.find_near_duplicates(shown),
        ideas_by_id={i["id"]: i for i in shown},
            idea_challenges={
            idea["id"]: _challenge_cards(user["id"], idea_id=idea["id"])
            for idea in shown
        },
        dismissed=[i for i in db.list_ideas(user["id"], "rejected")],
        note=request.query_params.get("note") or None,
    )


@app.post("/phase1/dismiss/{idea_id}")
def phase1_dismiss(request: Request, idea_id: int):
    """Retire an idea card the researcher does not want to see again.

    The card is marked `rejected`, not deleted. Nothing in this app throws work
    away, and an idea the model extracted from someone's unpublished research is
    the one thing they cannot get back by pasting the text in again.

    Only a `candidate` can be dismissed: a card already selected into a venture
    is what that venture was built from, and rejecting it would leave the venture
    pointing at an idea the account calls rejected. `kill_venture` returns a
    venture's idea to `candidate`, and it can be dismissed from there.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    idea = db.get_idea(idea_id, user["id"])
    # The venture lookup is the tenant check: an id the caller does not own is
    # simply not found, so a guessed id cannot dismiss somebody else's card.
    if not idea or idea["status"] != "candidate":
        return RedirectResponse(url="/phase1/ideas", status_code=303)
    db.set_idea_status(idea_id, "rejected")
    return RedirectResponse(
        url=f"/phase1/ideas?note={quote('Card dismissed. It is still on this page under Dismissed ideas, if you change your mind.')}",
        status_code=303,
    )


@app.post("/phase1/restore/{idea_id}")
def phase1_restore(request: Request, idea_id: int):
    """Undo a dismissal. Same tenancy check, same status guard."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    idea = db.get_idea(idea_id, user["id"])
    if not idea or idea["status"] != "rejected":
        return RedirectResponse(url="/phase1/ideas", status_code=303)
    db.set_idea_status(idea_id, "candidate")
    return RedirectResponse(
        url=f"/phase1/ideas?note={quote('Card restored to your candidates.')}",
        status_code=303,
    )


@app.post("/phase1/extract")
def phase1_extract(request: Request, raw_text: str = Form(...)):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    if len(raw_text.strip()) < 40:
        return _phase1(
            request, raw_text=raw_text,
            error="Paste a bit more material — a sentence or two isn't enough signal "
                  "to extract ideas from.")
    throttled = throttle_llm(request, user["id"], "/phase1/new")
    if throttled is not None:
        return throttled
    try:
        quota.check(user["id"])
        idea_cards = llm.extract_ideas(raw_text, user_id=user["id"])
    except llm.LLMError as e:
        return _phase1(request, raw_text=raw_text, error=str(e))

    for card in idea_cards:
        db.create_idea(
            user_id=user["id"],
            title=card["title"],
            commercial_framing=card["commercial_framing"],
            strength_signal=card["strength_signal"],
            raw_claims=card["raw_claims"],
        )
    # Redirect rather than render, so refreshing the results page doesn't
    # re-run the extraction and duplicate every card.
    return RedirectResponse(url="/phase1/ideas", status_code=303)


@app.post("/phase1/upload")
async def phase1_upload(request: Request, file: UploadFile = File(...)):
    """Turn an uploaded PDF/DOCX/TXT into editable material — and discard the file.

    The bytes never reach the disk: they are read into memory, parsed, and dropped.
    The extracted text comes back in the same preview box pasted text uses, so
    submitting it is an ordinary `POST /phase1/extract` — no new AI-call path, so
    no quota change and nothing new for the rate limiter to cover.

    The preview is the point, not a nicety: a thesis chapter arrives with a
    reference list, an acknowledgements page and possibly a co-author's section,
    and the user should get to remove those before an AI call spends their quota
    on them.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    if not (file.filename or "").strip():
        return _phase1(request, error="Pick a file first.")

    limit = config.max_upload_bytes()
    data = bytearray()
    try:
        # Read in chunks and stop the moment it is over the cap, rather than
        # reading a 2 GB body into memory in order to reject it afterwards.
        # Starlette's UploadFile is not async-iterable, so this is read(size).
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > limit:
                return _phase1(
                    request,
                    error=f"That file is larger than {_upload_limit_label()}. "
                          "Upload the chapter or section you care about.")
    finally:
        await file.close()

    try:
        text = upload.extract(bytes(data), filename=file.filename or "")
    except upload.UploadError as e:
        return _phase1(request, error=str(e))

    if len(text.strip()) < 40:
        return _phase1(
            request, raw_text=text,
            error="Only a little text came out of that — paste more, or paste "
                  "the section directly.")
    return _phase1(request, raw_text=text, from_upload=file.filename or "file")


@app.post("/phase1/import")
def phase1_import(request: Request, identifier: str = Form("")):
    """Turn an ORCID iD, DOI, or arXiv ID into editable material.

    Same shape as the upload route, for the same reasons: the fetched metadata
    comes back in the preview box and submitting it is an ordinary
    `POST /phase1/extract`. No AI call, so no quota, and the untrusted-data wrapper
    in `llm.extract_ideas` covers this material exactly as it covers a paste —
    fetched text is third-party text and gets no exemption for having travelled
    through our own server.

    Rate-limited separately from the model routes and under its own prefix, because
    it has the same shape of risk — a user-triggered outbound request holding a
    worker thread — at a far smaller cost. Checked *before* the fetch, so a refused
    import never touches a third party.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    identifier = (identifier or "").strip()

    if not config.source_import_enabled():
        return _phase1(request, error="Identifier import is turned off on this deployment.")

    retry_after = ratelimit.check(
        user["id"], _client_ip(request),
        per_user=config.import_rate_limit_per_min(),
        per_ip=config.import_rate_limit_ip_per_min(),
        prefix="import:")
    if retry_after is not None:
        return _phase1(request, error=f"Too many imports in a row. Try again in "
                                      f"about {retry_after} seconds.")

    try:
        record = sources.import_material(identifier)
    except sources.SourceError as e:
        return _phase1(request, error=str(e))

    if len(record.body.strip()) < 40:
        return _phase1(request,
                       error=f"{record.source} gave back too little to work with. "
                             "Add a line of your own in the box, or paste the material.")
    return _phase1(request, raw_text=record.body, from_import=record)


@app.post("/phase1/select/{idea_id}")
def phase1_select(request: Request, idea_id: int):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    idea = db.get_idea(idea_id, user["id"])
    if not idea:
        return RedirectResponse(url="/dashboard", status_code=303)
    db.set_idea_status(idea_id, "selected")
    venture_id = db.create_venture(user["id"], idea_id)
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


# ==================== Account & data ====================

# GET, not POST: a download changes nothing, so there is nothing for a cross-site
# request to cause, and a GET is what a browser can fetch without JavaScript.
@app.get("/account/export")
def account_export(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    payload = db.export_user_data(user["id"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Response(
        content=json.dumps(payload, indent=2, default=str),
        media_type="application/json",
        headers={
            # `no-store` because this file is the user's research and must not sit
            # in a shared machine's cache or history.
            "Cache-Control": "no-store",
            "Content-Disposition": f'attachment; filename="launchloop-{stamp}.json"',
        },
    )


@app.post("/account/password")
async def account_password(request: Request):
    """Change the password, and end every session that predates it.

    This is the security control the stateless cookie could not previously
    express: a stolen session cookie stops working the moment the owner rotates
    the credential, which is the whole reason anyone rotates one.

    Asks for the current password for the same reason `/account/delete` does —
    the people who most need to rotate are the ones whose session may be
    compromised, and without this gate an attacker who found the cookie could
    change the password and lock the owner out for good.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    form = await request.form()
    current = str(form.get("current_password", ""))
    password = str(form.get("password", ""))
    confirm = str(form.get("confirm", ""))

    def back(error):
        return render(request, "account.html", error=error, confirm="",
                      export_rows=_export_row_count(user["id"]))

    if not auth.verify_password(current, user["password_hash"]):
        return back("That is not your current password.")
    if password != confirm:
        return back("Those two passwords do not match.")
    if len(password) < config.min_password_length():
        return back(f"Password must be at least {config.min_password_length()} characters.")
    if len(password.encode("utf-8")) > auth.MAX_PASSWORD_BYTES:
        return back(f"Password must be at most {auth.MAX_PASSWORD_BYTES} bytes.")
    # Reusing the current password is a failed rotation that reads as success, so
    # it is refused here rather than accepted and then reported as a change.
    if auth.verify_password(password, user["password_hash"]):
        return back("That is already your password. Pick a different one.")

    epoch = db.set_password(user["id"], auth.hash_password(password))
    # Re-issue THIS session with the new epoch. set_password() has already
    # invalidated every other one, and revoking the caller's own session would
    # sign them out of the tab they just used to secure the account.
    refreshed = dict(user, session_epoch=epoch)
    auth.start_session(request, refreshed)
    return RedirectResponse(
        url=f"/account?note={quote('Password changed. Every other session has been signed out.')}",
        status_code=303,
    )

@app.post("/account/delete")
async def account_delete(request: Request):
    """Irreversible. Two gates: the password, and typing DELETE.

    A checkbox would be one stray click; a password means an unlocked laptop left
    on a desk is not enough to destroy someone's research.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    form = await request.form()
    password = str(form.get("password", ""))
    confirm = str(form.get("confirm", "")).strip()

    def back(error):
        return render(request, "account.html", error=error, confirm="",
                      export_rows=_export_row_count(user["id"]))

    if confirm != "DELETE":
        return back("Type DELETE exactly, in capitals, to confirm.")
    # Checked before the deletion, obviously, and the order matters: a wrong
    # password must never look like a confirmation problem and vice versa.
    if not auth.verify_password(password, user["password_hash"]):
        return back("That password is not right.")

    removed = db.delete_user_account(user["id"])
    request.session.clear()
    print(f"[account] deleted user {user['id']} ({user['email']}): {removed}")
    # The counts go to the log, not the page: "1 rows of research removed" is a
    # worse confirmation than saying what actually happened.
    return RedirectResponse(
        url="/login?note=" + quote(
            "Account deleted. Everything in it has been removed."),
        status_code=303)


def _export_row_count(user_id: int) -> dict:
    """Row counts for the account page, so the export/download buttons state what
    they are about to move rather than making the user trust a word."""
    data = db.export_user_data(user_id)
    return {k: len(v) for k, v in data.items() if isinstance(v, list)}


@app.get("/account", response_class=HTMLResponse)
def account_view(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    return render(request, "account.html", error=None, confirm="",
                  note=request.query_params.get("note") or None,
                  export_rows=_export_row_count(user["id"]))


# ==================== Phase 2 state machine ====================

def venture_phase_state(venture, current_run) -> dict:
    """Single source of truth for what a venture will let the user do next.

    A "run" is one pass at one canvas block. Only one run is in flight per
    venture at a time, exactly as before — what changed is that a run is now
    scoped to a single block, so the per-segment cap is what stops progress
    rather than a venture-wide cycle budget.
    """
    awaiting_results = current_run is not None and not current_run["results_json"]
    awaiting_analysis = (
        current_run is not None
        and bool(current_run["results_json"])
        and not current_run["analysis_json"]
    )
    # A run with no block attached is a pre-segment one: the scoring path refuses
    # it by design, so it can never be completed. Treating it as in-flight would
    # block every future run with no way for the user to clear it, which is how a
    # venture migrated from the old flat loop got permanently stuck.
    scoreable = current_run is None or bool(current_run["segment"])
    in_flight = scoreable and (awaiting_results or awaiting_analysis)
    terminal = venture["status"] in ("killed", "pivoted")
    at_backstop = venture["cycle_count"] >= venture["max_cycles"]

    kind = (
        "phase3" if (venture["phase"] == 3 or venture["status"] == "validated")
        else "killed" if venture["status"] == "killed"
        else "pivoted" if venture["status"] == "pivoted"
        else "stopped" if terminal or at_backstop
        else "looping"
    )

    return {
        "kind": kind,
        "awaiting_results": awaiting_results,
        "awaiting_analysis": awaiting_analysis,
        "in_flight": in_flight,
        "at_backstop": at_backstop,
        "terminal": terminal,
        "can_start_new_cycle": not terminal and not in_flight and not at_backstop,
        "start_blocked_reason": _start_blocked_reason(venture, in_flight, at_backstop),
    }


def _start_blocked_reason(venture, in_flight: bool, at_backstop: bool):
    if venture["status"] == "killed":
        return "This venture was killed — the idea is back in your candidate list."
    if venture["status"] == "pivoted":
        return "This venture was pivoted. Open the new venture it created to keep going."
    if in_flight:
        return "Finish the run in progress first — log its results, or retry the analysis."
    if at_backstop:
        return (
            f"This venture has used its backstop of {venture['max_cycles']} runs. "
            "Start a new venture if you want to keep going."
        )
    return None


def segment_run_state(segment, venture) -> dict:
    """What this one block will let the user do next.

    A block that has hit its own cap is parked rather than pausing the venture,
    which is the whole point of moving the cap onto the segment.
    """
    resolved = segment["outcome"] in db.SEGMENT_RESOLVED
    at_cap = segment["cycle_count"] >= segment["max_cycles"]
    return {
        "resolved": resolved,
        "at_cap": at_cap,
        "parked_by_cap": at_cap and not resolved,
        "can_run": (
            segment["outcome"] != "passed"
            and venture["status"] not in ("killed", "pivoted")
            and venture["cycle_count"] < venture["max_cycles"]
        ),
    }


def spawn_pivot_venture(venture, note: str) -> int:
    """A 'pivot' verdict has to actually change something. Creates a sibling
    venture on a fresh nine-block canvas carrying the revised hypothesis, and
    retires this one as 'pivoted' so the state machine has exactly one live
    branch per idea.

    Blocks this venture already PASSED are carried across with their evidence.
    A pivot changes the hypothesis about one block, not the eight you proved
    around it — re-testing those would throw away real work and real AI calls.
    """
    venture_id = venture["id"]
    idea_id = db.create_idea(
        user_id=venture["user_id"],
        title=f"{venture['idea_title']} (pivot)",
        commercial_framing=note,
        strength_signal="early",
        raw_claims=f"Pivoted from venture #{venture_id} after {venture['cycle_count']} runs.",
    )
    new_venture_id = db.create_venture(
        venture["user_id"], idea_id,
        parent_venture_id=venture_id, pivot_note=note,
    )
    carried = db.carry_passed_segments(venture_id, new_venture_id)
    db.update_venture(venture_id, status="pivoted", pivot_note=note)
    if carried:
        print(f"[pivot] carried {carried} validated block(s) into venture {new_venture_id}")
    return new_venture_id


def apply_verdict(venture, run, analysis) -> int | None:
    """Persist one run's verdict and move the venture on. Returns the id of a
    newly spawned pivot venture, if any.

    The verdict/evidence pairing is already enforced by llm.analyze_segment_run,
    so db.set_segment_outcome will not hit a CHECK constraint here.

    A critical block failing does NOT end the venture here. It marks the block
    failed and leaves the decision to the user, because a model should not get to
    close someone's venture on its own. `needs_decision` in the rendered context
    drives the kill-or-pivot card; the routes that resolve it are kill_venture()
    and pivot_venture().
    """
    verdict = analysis["verdict"]
    db.save_run_verdict(run["id"], analysis, verdict)
    segment_key = run["segment"]
    venture_id = venture["id"]

    if verdict == "pivot":
        return spawn_pivot_venture(venture, analysis.get("pivot_suggestion")
                                   or "The model recommended a pivot on this block.")

    if verdict == "pass":
        # On a pass, revised_hypothesis is the block's confirmed answer — the value
        # proposition, the customer segment, whatever the block is about. It lands in
        # `hypothesis` and the canvas cell renders it as the block's answer, which is
        # why the prompt asks for the answer itself rather than a finding summary.
        # outcome_note keeps the evidence that settled it, shown beneath.
        db.set_segment_outcome(venture_id, segment_key, "passed",
                               note=analysis.get("evidence_note"),
                               hypothesis=analysis.get("revised_hypothesis"))
    elif verdict == "fail":
        if segment_key in db.SEGMENT_KEYS and is_critical(segment_key):
            # Left for the user to resolve via kill or pivot.
            db.set_segment_outcome(venture_id, segment_key, "failed",
                                   note=analysis.get("verdict_reasoning"))
        else:
            # A secondary block failing just parks that one block with a
            # workaround; the venture keeps going.
            db.set_segment_outcome(venture_id, segment_key, "parked",
                                   note=analysis.get("workaround")
                                   or analysis.get("verdict_reasoning"))
    else:  # "iterate"
        db.update_segment(venture_id, segment_key, status="mixed",
                          notes=analysis.get("evidence_note"))
        db.set_segment_hypothesis(venture_id, segment_key,
                                  analysis.get("revised_hypothesis"))

    # Every block resolved (passed or parked) unlocks Phase 3. A failed critical
    # block is deliberately not resolved, so this cannot fire while a decision
    # is outstanding.
    if db.all_resolved(venture_id):
        db.update_venture(venture_id, phase=3, status="validated")
    return None


# ==================== Venture (Phase 2 + 3 router) ====================

def _venture_context(request, venture, user, focus_key=None, **extra):
    """Every context key the phase-2 template needs, assembled once so the view,
    the error re-render and the retry path cannot drift apart.

    Focus is resolved here rather than in the template: whichever block the run
    panel shows gets a consistent block of context (its runs, its state, whether
    it needs a decision). It falls back to the recommended block so that loading a
    venture directly always answers "what do I do next" at the top of the page,
    instead of showing no panel at all.
    """
    venture_id = venture["id"]
    segments = db.get_segments(venture_id)
    current_run = db.get_current_cycle(venture_id)
    state = venture_phase_state(venture, current_run)
    active_segment = None
    if current_run and current_run["segment"]:
        active_segment = db.get_segment(venture_id, current_run["segment"])
    recommended = db.next_recommended_segment(venture_id)
    resolved, total = db.resolved_count(venture_id)

    # A failed critical block is unresolved until the user picks kill or pivot.
    failed_critical = next(
        (s for s in segments
         if s["outcome"] == "failed" and is_critical(s["element_name"])), None)
    needs_decision = failed_critical is not None and venture["status"] not in (
        "killed", "pivoted", "validated")

    # Focus order, most urgent first:
    #   1. the block the URL names
    #   2. the one with a run in flight
    #   3. one awaiting a kill-or-pivot decision — it blocks Phase 3, and putting
    #      it anywhere but the top would mean the user has to notice it on the
    #      board themselves
    #   4. otherwise the block we would attack next
    focus_segment = None
    if focus_key:
        focus_segment = db.get_segment(venture_id, focus_key)
    if focus_segment is None:
        focus_segment = active_segment or (failed_critical if needs_decision else None) \
                        or recommended

    ctx = {
        "venture": venture,
        "kind": state["kind"],
        "segments": segments,
        "segment_by_key": {s["element_name"]: s for s in segments},
        "cycles": db.list_cycles(venture_id),
        "current_cycle": current_run,
        "active_segment": active_segment,
        "recommended_segment": recommended,
        "resolved": resolved,
        "total_segments": total,
        # Keyed `focus` because that is the name the template uses throughout.
        "focus": focus_segment,
        "focus_runs": db.list_segment_runs(venture_id, focus_segment["element_name"])
                      if focus_segment else [],
        "focus_state": segment_run_state(focus_segment, venture) if focus_segment else None,
        "failed_critical": failed_critical,
        "needs_decision": needs_decision,
        "current_todos": current_run["todos_json"] if current_run else None,
        "current_results": current_run["results_json"] if current_run else None,
        "current_analysis": current_run["analysis_json"] if current_run else None,
        "awaiting_results": state["awaiting_results"],
        "awaiting_analysis": state["awaiting_analysis"],
        "can_start_new_cycle": state["can_start_new_cycle"],
        "start_blocked_reason": state["start_blocked_reason"],
        "pivot_ventures": db.get_child_ventures(venture_id),
        "cap_extension": config.segment_cap(),
        "quota_used": db.count_llm_calls_this_month(user["id"]),
        "quota_limit": quota.monthly_limit(),
        # subject_kind is not optional here: a block challenge and a plan
        # challenge share a venture_id, so filtering by venture alone put the
        # launch plan's questions on top of every block panel.
        "segment_challenges": (
            _challenge_cards(user["id"], venture_id=venture_id,
                             subject_kind="segment")
            if focus_segment else []
        ),
        "error": request.query_params.get("err") or None,
        # A success message needs its own key: reusing `err` would render "email
        # confirmed" in the same red box as a quota failure.
        "note": request.query_params.get("note") or None,
    }
    ctx.update(extra)
    return ctx


@app.get("/venture/{venture_id}", response_class=HTMLResponse)
def venture_view(request: Request, venture_id: int):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)

    ctx = _venture_context(request, venture, user)
    if ctx["current_cycle"] and ctx["current_cycle"]["analysis_json"]:
        ctx["awaiting_analysis"] = False

    if ctx["kind"] == "phase3":
        strategy = db.get_launch_strategy(venture_id)
        # Every context key comes off ctx rather than being re-listed here. This
        # branch used to hand-build its own dict, which is how `error` and
        # `note` went missing: back_to_venture() attaches a message as ?err=,
        # the template had no {{ error }} block either, so a provider outage or
        # an exhausted quota while generating the strategy redirected back to a
        # page that looked unchanged and said nothing at all — the swallowed
        # refusal back_to_venture's docstring says it exists to prevent.
        ctx.update(
            gaps=db.open_gaps(venture_id),
            strategy={
                "funding_matches": strategy["funding_matches_json"],
                "gtm_channels": strategy["gtm_channels_json"],
                "action_plan": strategy["action_plan_json"],
            } if strategy else None,
        )
        if strategy:
            plan = ctx["strategy"]["action_plan"] or []
            ctx["steps"] = db.get_action_steps(venture_id)
            ctx["plan_items"] = db.join_action_plan(ctx["steps"], plan)
            ctx["progress"] = db.action_plan_progress(ctx["steps"], plan)
        else:
            ctx["steps"], ctx["plan_items"], ctx["progress"] = {}, [], None
            ctx["plan_challenges"] = _challenge_cards(
                user["id"], venture_id=venture_id, subject_kind="plan")
        return render(request, "venture_phase3.html", **ctx)
    return render(request, "venture_phase2.html", **ctx)


@app.get("/venture/{venture_id}/segment/{segment_key}", response_class=HTMLResponse)
def segment_view(request: Request, venture_id: int, segment_key: str):
    """Focused screen for one block: its hypothesis, its runs, and the run in
    progress if it is the block being worked on."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)
    segment = db.get_segment(venture_id, segment_key)
    if not segment:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    # focus_key lets _venture_context resolve the panel and its run history in
    # one place, exactly as the default-focus path does.
    ctx = _venture_context(request, venture, user, focus_key=segment_key)
    return render(request, "venture_phase2.html", **ctx)


@app.post("/venture/{venture_id}/segment/{segment_key}/run")
def start_run(request: Request, venture_id: int, segment_key: str):
    """Generate the task list for one run against one block."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    segment = db.get_segment(venture_id, segment_key)
    if not venture or not segment:
        # The venture check is the tenant check: an unowned id never reaches
        # the model, so a guessed id cannot spend a user's quota.
        return RedirectResponse(url="/dashboard", status_code=303)

    state = venture_phase_state(venture, db.get_current_cycle(venture_id))
    if not state.get("can_start_new_cycle"):
        return _phase2_retry(request, venture,
                             state.get("start_blocked_reason") or "Can't start a run right now.")
    seg_state = segment_run_state(segment, venture)
    if not seg_state["can_run"]:
        return _phase2_retry(request, venture, "That block is already passed.")
    if seg_state["parked_by_cap"]:
        return _phase2_retry(request, venture,
                             f"You've used all {segment['max_cycles']} runs on this block. "
                             "Extend it first if you want another.")

    others = [dict(r) for r in db.get_segments(venture_id)
              if r["element_name"] != segment_key]
    previous = db.list_segment_runs(venture_id, segment_key)

    throttled = throttle_llm(request, user["id"],
                             f"/venture/{venture_id}/segment/{segment_key}")
    if throttled is not None:
        return throttled
    try:
        quota.check(user["id"])
        designed = llm.generate_segment_tasks(
            venture["idea_title"], venture["commercial_framing"], dict(segment),
            segment["cycle_count"] + 1, others,
            hypothesis=segment["hypothesis"],
            previous_runs=[r["analysis_json"] for r in previous if r["analysis_json"]],
            user_id=user["id"],
        )
    except llm.LLMError as e:
        return _phase2_retry(request, venture, str(e), segment_key=segment_key)

    # The model's hypothesis is stored before the run so the run panel leads
    # with something even if the user never wrote one down.
    if designed.get("hypothesis"):
        db.set_segment_hypothesis(venture_id, segment_key, designed["hypothesis"])

    run_number = venture["cycle_count"] + 1
    db.bump_segment_run(venture_id, segment_key)
    db.create_cycle(venture_id, run_number, designed["tasks"], segment=segment_key)
    db.update_venture(venture_id, cycle_count=run_number)
    return RedirectResponse(url=f"/venture/{venture_id}/segment/{segment_key}", status_code=303)


def _phase2_retry(request: Request, venture, message: str, segment_key: str = None):
    """Re-render Phase 2 with an error while preserving the full view state.

    Keeps the block the user was working on in focus, so an error never dumps
    them back onto the default panel and loses their place."""
    user = auth.get_current_user(request)
    ctx = _venture_context(request, venture, user, focus_key=segment_key, error=message)
    return render(request, "venture_phase2.html", **ctx)


async def _read_results(form, tasks: list) -> list:
    """Collect the per-task outcome + sample size the user filled in."""
    return [
        {
            "outcome": str(form.get(f"outcome_{i}", "")).strip()[:4000],
            "sample_size": str(form.get(f"sample_size_{i}", "")).strip()[:100],
            "title": tasks[i].get("title") if i < len(tasks) else "",
        }
        for i in range(len(tasks))
    ]


def _score_run(request, venture, run, user, results=None):
    """Shared by log-results and retry-analysis: score a run and apply it."""
    segment_key = run["segment"]
    if not segment_key:
        # A pre-segment run with no attributable block. Nothing sensible to do
        # with a verdict, so send the user back to the board.
        return None, back_to_venture(venture["id"],
                                     "That run predates per-block tracking.")
    segment = db.get_segment(venture["id"], segment_key)
    previous = [r["analysis_json"] for r in
                db.list_segment_runs(venture["id"], segment_key)
                if r["analysis_json"] and r["id"] != run["id"]]

    # Re-read the run: the caller may have logged results after fetching it, and
    # the stale mapping would hand the model a None and crash on zip().
    logged = results if results is not None else db.get_cycle(run["id"], venture["id"])
    if isinstance(logged, list):
        logged = {"todos_json": run["todos_json"], "results_json": logged}
    if not logged or not logged["results_json"]:
        return None, back_to_venture(venture["id"], "Log the results before scoring.")

    # Throttle *after* the results are known to be on disk. This route is reached
    # from log_run (which persists them first, invariant 5) and from retry-analysis
    # (where they were persisted earlier), so refusing here never costs the user
    # their evidence — the retry path stays available.
    throttled = throttle_llm(
        request, user["id"], f"/venture/{venture['id']}/segment/{segment_key}",
        saved="Your logged results are saved — the verdict is the only thing "
              "that has not run yet, and retrying it will not cost you anything "
              "but time.",
    )
    if throttled is not None:
        return None, throttled

    try:
        quota.check(user["id"])
        analysis = llm.analyze_segment_run(
            venture["idea_title"], dict(segment), segment["hypothesis"],
            logged["todos_json"], logged["results_json"], previous, user_id=user["id"],
        )
    except llm.LLMError as e:
        # Results are already saved; the page offers a retry button.
        return None, back_to_venture(venture["id"], str(e))

    return apply_verdict(venture, run, analysis), None


@app.post("/venture/{venture_id}/run/{run_id}/log")
async def log_run(request: Request, venture_id: int, run_id: int):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    run = db.get_cycle(run_id, venture_id)
    if not venture or not run:
        return RedirectResponse(url="/dashboard", status_code=303)

    form = await request.form()
    results = await _read_results(form, run["todos_json"])
    if not any(r["outcome"] for r in results):
        return _phase2_retry(request, venture, "Log at least one outcome first.")

    db.log_cycle_results(run_id, results)
    new_venture_id, error_response = _score_run(request, venture, run, user, results=results)
    if error_response is not None:
        return error_response
    if new_venture_id:
        return RedirectResponse(url=f"/venture/{new_venture_id}", status_code=303)
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


@app.post("/venture/{venture_id}/run/{run_id}/retry-analysis")
def retry_analysis(request: Request, venture_id: int, run_id: int):
    """If the analysis call failed after results were logged, retry it here
    instead of forcing the user to re-log everything."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    run = db.get_cycle(run_id, venture_id)
    if not venture or not run or not run["results_json"]:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    new_venture_id, error_response = _score_run(request, venture, run, user)
    if error_response is not None:
        return error_response
    if new_venture_id:
        return RedirectResponse(url=f"/venture/{new_venture_id}", status_code=303)
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


@app.post("/venture/{venture_id}/segment/{segment_key}/extend")
def extend_segment(request: Request, venture_id: int, segment_key: str):
    """Raise one block's run cap. Scoped to the block so exhausting one segment
    never parks the whole venture the way the old global cap did."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    segment = db.get_segment(venture_id, segment_key)
    if not venture or not segment:
        return RedirectResponse(url="/dashboard", status_code=303)
    db.extend_segment(venture_id, segment_key)
    return RedirectResponse(url=f"/venture/{venture_id}/segment/{segment_key}", status_code=303)


@app.post("/venture/{venture_id}/segment/{segment_key}/unpark")
def unpark_segment(request: Request, venture_id: int, segment_key: str):
    """Re-open a parked block so the user can take one more run at it."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    segment = db.get_segment(venture_id, segment_key)
    if not venture or not segment:
        return RedirectResponse(url="/dashboard", status_code=303)
    if segment["outcome"] != "parked":
        return RedirectResponse(url=f"/venture/{venture_id}/segment/{segment_key}", status_code=303)
    db.set_segment_outcome(venture_id, segment_key, "pending",
                           note=segment["outcome_note"])
    return RedirectResponse(url=f"/venture/{venture_id}/segment/{segment_key}", status_code=303)


@app.post("/venture/{venture_id}/segment/{segment_key}/park")
async def park_segment(request: Request, venture_id: int, segment_key: str):
    """Close out a block the user doesn't want to spend more runs on.

    This is the exit that was missing: a block that hit its cap could only be
    extended, so the venture could neither progress nor reach Phase 3. Parking
    counts as resolved and carries the user's note into the launch strategy, so
    the gap is documented rather than silently dropped.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    segment = db.get_segment(venture_id, segment_key)
    if not venture or not segment:
        return RedirectResponse(url="/dashboard", status_code=303)
    if segment["outcome"] == "passed":
        return RedirectResponse(url=f"/venture/{venture_id}/segment/{segment_key}", status_code=303)

    form = await request.form()
    note = str(form.get("note", "")).strip()[:1000]
    if not note:
        return back_to_venture(
            venture_id,
            "Say what you learned before parking this block — a park with no "
            "reason is a hole in the model, and the launch plan needs to know.",
        )
    db.set_segment_outcome(venture_id, segment_key, "parked", note=note)
    if db.all_resolved(venture_id):
        db.update_venture(venture_id, phase=3, status="validated")
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


# ---------- resolving a failed critical block ----------
# The model marks the block failed; it does not get to end the venture.

@app.post("/venture/{venture_id}/kill")
def kill_venture(request: Request, venture_id: int):
    """The user chooses to end a venture whose critical block came back negative."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)
    if venture["status"] in ("killed", "pivoted", "validated"):
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)
    db.update_venture(venture_id, status="killed")
    db.set_idea_status(venture["idea_id"], "candidate")
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


@app.post("/venture/{venture_id}/pivot")
async def pivot_venture(request: Request, venture_id: int):
    """The user chooses to pivot instead. Blocks already passed carry across."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)
    if venture["status"] in ("killed", "pivoted", "validated"):
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    form = await request.form()
    note = str(form.get("note", "")).strip()[:1000]
    if not note:
        # Fall back to what the model suggested on the run that failed.
        failed = [s for s in db.get_segments(venture_id) if s["outcome"] == "failed"]
        note = (failed[0]["outcome_note"] if failed else "") or "Pivoted on the researcher's call."
    new_venture_id = spawn_pivot_venture(venture, note)
    return RedirectResponse(url=f"/venture/{new_venture_id}", status_code=303)


# ==================== Phase 3: Launch Strategy ====================

@app.post("/venture/{venture_id}/phase3/generate")
def generate_strategy(request: Request, venture_id: int):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)
    if venture["status"] != "validated" and venture["phase"] != 3:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    segments = [dict(row) for row in db.get_segments(venture_id)]
    throttled = throttle_llm(request, user["id"], f"/venture/{venture_id}")
    if throttled is not None:
        return throttled
    try:
        quota.check(user["id"])
        strategy = llm.generate_launch_strategy(
            venture["idea_title"], venture["commercial_framing"], segments,
            user_id=user["id"],
        )
    except llm.LLMError as e:
        return back_to_venture(venture_id, str(e))

    db.save_launch_strategy(
        venture_id,
        strategy["funding_matches"],
        strategy["gtm_channels"],
        strategy["action_plan"],
    )
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

# How much a single action-plan outcome note may be. The run-outcome fields cap
# at 4000; a plan step is one sentence of "what happened", so this is tighter.
MAX_STEP_NOTE_CHARS = 2000

# Confirmation per status, on the success path only. Kept apart from the refusal
# copy above because a success message reusing the red error box reads as a
# failure (invariant 10: `error` and `note` are different channels on purpose).
STEP_SAVED_NOTES = {
    "done": "Step marked done. The outcome you recorded is saved.",
    "blocked": "Step marked blocked. The reason you recorded is saved.",
    "pending": "Step reopened.",
}

@app.post("/venture/{venture_id}/phase3/step")
async def phase3_step(request: Request, venture_id: int):
    """Record how far one launch-plan step got.

    The form posts the step's INDEX and the step_key it was *rendered* with, and
    the server re-derives the key from the stored plan and requires the two to
    agree. Both halves are load-bearing:

    * Deriving the key server-side is what stops a crafted POST writing to an
      arbitrary row. A form that simply posted a step_key would let anyone with
      a session write to any row in the table by supplying a hash, and name a
      step the researcher was never shown.
    * Comparing against the *rendered* key is what makes a stale form safe. The
      user confirmed a specific instruction at render time; if the strategy was
      regenerated since, index 0 names a different instruction, and writing
      there would record a completion against work nobody did. Without this
      check the identity the user agreed to and the identity written would be
      resolved at two different moments, and could differ.
    """
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)
    if venture["status"] != "validated" and venture["phase"] != 3:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    strategy = db.get_launch_strategy(venture_id)
    if not strategy:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)
    plan = [i for i in (strategy["action_plan_json"] or [])
            if isinstance(i, dict) and isinstance(i.get("step"), str)
            and i["step"].strip()]

    form = await request.form()
    try:
        index = int(str(form.get("index", "")).strip())
    except (TypeError, ValueError):
        index = -1
    # Bounds-checked against the *stored* plan, not the posted length: a negative
    # index would otherwise wrap around to the last step.
    if not 0 <= index < len(plan):
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)
    key = db.step_key(plan[index]["step"])
    if str(form.get("step_key", "")).strip() != key:
        # A strategy was regenerated between rendering this form and submitting
        # it, so the step the user confirmed is no longer the step at this
        # position. Refusing beats guessing which one they meant.
        return back_to_venture(
            venture_id,
            "That step is no longer the one this page showed — the strategy was "
            "regenerated. Reload to see the current plan and record it there.",
        )

    status = str(form.get("status", "")).strip()
    if status not in ACTION_STEP_STATUSES:
        return back_to_venture(venture_id, "That is not a state a plan step can be in.")
    note = str(form.get("note", "")).strip()[:MAX_STEP_NOTE_CHARS]

    # `done` and `blocked` are assertions about the real world, so both need a
    # reason. Same rule, and for the same reason, as parking a block: a step
    # marked done with nothing recorded is the unevidenced claim this app exists
    # to replace. Reverting to pending asserts nothing, so it needs no note.
    if status in ("done", "blocked") and not note:
        word = "done" if status == "done" else "blocked"
        return back_to_venture(
            venture_id,
            f"Say what happened before marking this step {word} — a step with "
            f"no outcome is just a claim, and the claim is the thing this app "
            f"is meant to replace.",
        )

    if not db.set_action_step(venture_id, key, status, note=note or None):
        # The row is gone: the plan was regenerated into a different list
        # between rendering this form and submitting it.
        return back_to_venture(
            venture_id,
            "That step is no longer part of the current plan — the strategy was "
            "regenerated. Reload to see the new one.",
        )
    return RedirectResponse(
        url=f"/venture/{venture_id}?note={quote(STEP_SAVED_NOTES[status])}",
        status_code=303,
    )

# ==================== Mentor challenges (M2.1) ====================

def _challenge_cards(user_id: int, **kwargs) -> list:
    """This subject's past challenges, each carrying its playbook for rendering.

    The label and attribution are joined in here rather than stored on the row,
    because they are properties of the playbook, not of the challenge — and
    copying them into the table would mean a catalogue edit left old rows quoting
    a label that no longer exists.
    """
    cards = []
    for row in db.list_mentor_challenges(user_id, **kwargs):
        mentor = mentors.get(row["mentor_key"]) or {}
        cards.append({
            **row,
            "mentor": mentor,
            "mentor_label": mentor.get("label", row["mentor_key"]),
        })
    return cards

def _challenge_state_for_idea(idea: dict) -> list:
    """What is actually on record for one idea card, as lines.

    Deliberately thin: an idea card has a title, a framing and a strength signal,
    and inventing more would mean inventing facts. The `raw_claims` field is the
    model's own text about the researcher's work, so it is included but capped.
    """
    lines = [
        f"The idea: {idea['title']}",
        f"Its commercial framing: {idea.get('commercial_framing') or '(none)'}",
        f"Strength signal the model gave it: {idea.get('strength_signal') or '(none)'}",
    ]
    if idea.get("raw_claims"):
        lines.append(f"Built from this technical claim: {idea['raw_claims'][:600]}")
    lines.append(f"Idea status in the account: {idea['status']}")
    return lines


def _challenge_state_for_segment(venture: dict, segment: dict, runs: list) -> list:
    """The block's own evidence, plus where the rest of the canvas stands."""
    lines = [
        f"Block: {segment.get('label') or segment.get('element_name')}",
        f"Current outcome: {segment.get('outcome')}",
        f"Evidence read: {segment.get('status')}",
        f"Hypothesis on record: {segment.get('hypothesis') or '(none written yet)'}",
    ]
    if segment.get("outcome_note"):
        lines.append(f"What the researcher recorded: {segment['outcome_note'][:600]}")
    if segment.get("notes"):
        lines.append(f"Notes: {segment['notes'][:600]}")

    if runs:
        lines.append(f"Runs logged on this block: {len(runs)}")
        for run in runs[-3:]:
            results = run.get("results_json") or []
            if isinstance(results, list):
                for result in results[:4]:
                    if not isinstance(result, dict):
                        continue
                    title = result.get("title") or "(untitled task)"
                    outcome = (result.get("outcome") or "").strip()[:300]
                    size = result.get("sample_size") or "no sample size stated"
                    lines.append(f"  - {title}: logged \"{outcome}\" ({size})")
            analysis = run.get("analysis_json")
            if isinstance(analysis, dict) and analysis.get("verdict"):
                lines.append(f"  - verdict given: {analysis['verdict']}")
    else:
        lines.append("No runs logged on this block yet.")

    others = [
        f"{other.get('label') or other.get('element_name')}: {other.get('outcome')}"
        for other in db.get_segments(venture["id"])
        if other.get("element_name") != segment.get("element_name")
    ]
    if others:
        lines.append("Rest of the canvas: " + "; ".join(others))
    return lines


def _challenge_state_for_plan(venture: dict, segments: list, steps: dict,
                              plan: list) -> list:
    """The canvas, the gaps, and what has actually been ticked off."""
    lines = [
        f"Idea: {venture['idea_title']}",
        f"Commercial framing: {venture.get('commercial_framing') or '(none)'}",
    ]
    for segment in segments:
        note = (segment.get("outcome_note") or "").strip()
        line = (f"{segment.get('label') or segment.get('element_name')}: "
                f"{segment.get('outcome')} (evidence: {segment.get('status')})")
        if segment.get("outcome") == "parked" and note:
            line += f" — PARKED, workaround: {note[:300]}"
        elif note:
            line += f" — recorded: {note[:300]}"
        lines.append(line)

    progress = db.action_plan_progress(steps, plan or [])
    if progress["total"]:
        lines.append(f"Action plan: {progress['done']} of {progress['total']} steps done"
                     + (f", {progress['blocked']} blocked" if progress["blocked"] else ""))
        for item in db.join_action_plan(steps, plan or []):
            state = item["status"]
            lines.append(f"  - [{state}] {item['step'][:200]}"
                         + (f" — researcher recorded: {item['note'][:200]}"
                            if item["note"] else ""))
    else:
        lines.append("No action plan generated yet.")
    return lines


def _challenge_other_segments(venture: dict) -> list:
    """Every block on the canvas, as plain dicts.

    A helper rather than inlined into each state builder so none of them has to
    know how another one fetches the canvas.
    """
    return [dict(s) for s in db.get_segments(venture["id"])]


async def _challenge(request: Request, user: dict, *, mentor_key: str,
                     subject_kind: str, subject: str, state: list,
                     idea_id: int = None, venture_id: int = None,
                     segment: str = None, back_url: str = None) -> Response:
    """Shared body of the three challenge routes.

    The mentor is refused before the call if the playbook is unknown, is throttled,
    then charged to the monthly quota, then called, then persisted — and
    redirected. Nothing in here can change a block's outcome, a run's verdict or a
    venture's status; that is the whole point of the feature (invariant 4) and
    TestMentorChallenge asserts it rather than trusting this docstring.

    `state` is already-composed lines of what the researcher has recorded. This
    function decides how they are presented; the caller decides which facts are
    relevant.
    """
    back_url = back_url or (f"/venture/{venture_id}" if venture_id else "/phase1/ideas")
    if mentor_key not in MENTOR_KEYS or subject_kind not in MENTOR_SUBJECT_KINDS:
        # Refused before the limiter is touched or a prompt exists. An unknown key
        # would otherwise put a real founder's name next to generated text with no
        # principle behind it — the failure app/mentors.py exists to prevent.
        return RedirectResponse(url=f"{back_url}?err={quote(mentors.MENTOR_BAD_KEY)}",
                                status_code=303)

    throttled = throttle_llm(request, user["id"], back_url)
    if throttled is not None:
        return throttled
    try:
        quota.check(user["id"])
        result = llm.challenge_with_mentor(
            mentor_key, subject, state, user_id=user["id"])
    except llm.LLMError as e:
        # Nothing is written. back_to_venture so a Phase 2/3 subject lands back on
        # the page they were reading; back_to_venture also works for Phase 1
        # because the ideas page renders `error` from ?err= like the others.
        return RedirectResponse(
            url=f"{back_url}?err={quote(str(e)[:300])}", status_code=303)

    db.save_mentor_challenge(
        user["id"], mentor_key, subject_kind, result["questions"],
        idea_id=idea_id, venture_id=venture_id, segment=segment,
        dropped=result.get("dropped", 0),
    )
    return RedirectResponse(
        url=f"{back_url}?note={quote(mentors.CHALLENGE_READY)}", status_code=303)


@app.post("/mentor/idea/{idea_id}")
async def mentor_idea(request: Request, idea_id: int):
    """Challenge one candidate idea card, before it becomes a venture."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    # The lookup IS the tenant check: an idea belonging to somebody else is simply
    # not found, so a guessed id cannot spend this user's quota.
    idea = db.get_idea(idea_id, user["id"])
    if not idea:
        return RedirectResponse(url="/dashboard", status_code=303)
    form = await request.form()
    return await _challenge(
        request, user,
        mentor_key=str(form.get("mentor_key", "")).strip(),
        subject_kind="idea", subject=f'the idea card "{idea["title"]}"',
        state=_challenge_state_for_idea(dict(idea)),
        idea_id=idea_id, back_url=f"/phase1/ideas#idea-{idea_id}",
    )


@app.post("/venture/{venture_id}/mentor/segment/{segment_key}")
async def mentor_segment(request: Request, venture_id: int, segment_key: str):
    """Challenge one canvas block against the evidence actually logged on it."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    segment = db.get_segment(venture_id, segment_key) if venture else None
    if not venture or not segment:
        return RedirectResponse(url="/dashboard", status_code=303)
    form = await request.form()
    return await _challenge(
        request, user,
        mentor_key=str(form.get("mentor_key", "")).strip(),
        subject_kind="segment",
        subject=f'the "{segment.get("label") or segment_key}" block of this canvas',
        state=_challenge_state_for_segment(
            dict(venture), dict(segment),
            db.list_segment_runs(venture_id, segment_key)),
        venture_id=venture_id, segment=segment_key,
        back_url=f"/venture/{venture_id}/segment/{segment_key}",
    )


@app.post("/venture/{venture_id}/mentor/plan")
async def mentor_plan(request: Request, venture_id: int):
    """Challenge the launch plan against the resolved canvas and the recorded steps."""
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)

    segments = [dict(s) for s in db.get_segments(venture_id)]
    strategy = db.get_launch_strategy(venture_id)
    plan = (strategy["action_plan_json"] if strategy else []) or []
    steps = db.get_action_steps(venture_id) if strategy else {}

    form = await request.form()
    return await _challenge(
        request, user,
        mentor_key=str(form.get("mentor_key", "")).strip(),
        subject_kind="plan",
        subject="this launch plan and the canvas it was built from",
        state=_challenge_state_for_plan(dict(venture), segments, steps, plan),
        venture_id=venture_id, back_url=f"/venture/{venture_id}",
    )
