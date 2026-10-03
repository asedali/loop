import os
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import auth, config, db, llm, quota
from .constants import is_critical

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
        from urllib.parse import quote
        url += f"?err={quote(message[:300])}"
    return RedirectResponse(url=url, status_code=303)


def login_or_redirect(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return None, RedirectResponse(url="/login", status_code=303)
    return user, None


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
    """Liveness + DB reachability, for the Fly health check. Deliberately does
    not touch the LLM provider: a provider outage shouldn't fail the deploy
    health check and get the machine cycled."""
    if not db.ping():
        return {"status": "degraded"}
    return {"status": "ok"}


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
    if db.get_user_by_email(email):
        return render(request, "signup.html", error="An account with that email already exists.", email=email)
    if len(password) < config.min_password_length():
        return render(
            request, "signup.html",
            error=f"Password must be at least {config.min_password_length()} characters.",
            email=email,
        )
    user_id = db.create_user(email, auth.hash_password(password))
    request.session["user_id"] = user_id
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return render(request, "login.html", error=None, email="")


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()

    window = config.login_window_minutes()
    since = datetime.now(timezone.utc) - timedelta(minutes=window)
    if db.recent_failed_logins(email, since) >= config.max_failed_logins():
        return render(
            request, "login.html",
            error=f"Too many failed attempts. Try again in {window} minutes.",
            email=email,
        )

    user = db.get_user_by_email(email)
    ok = bool(user) and auth.verify_password(password, user["password_hash"])
    db.record_login_attempt(email, request.client.host if request.client else None, ok)
    if not ok:
        return render(request, "login.html", error="Invalid email or password.", email=email)

    request.session["user_id"] = user["id"]
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
    return render(
        request, "dashboard.html",
        ideas=db.list_ideas(user["id"]),
        ventures=db.list_ventures(user["id"]),
        bmc_elements=db.get_bmc_map_for_user(user["id"]),
        quota_used=db.count_llm_calls_this_month(user["id"]),
        quota_limit=quota.monthly_limit(),
    )


# ==================== Phase 1: Idea Discovery ====================

@app.get("/phase1/new", response_class=HTMLResponse)
def phase1_new(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    return render(request, "phase1_new.html", error=None, raw_text="")


@app.get("/phase1/ideas", response_class=HTMLResponse)
def phase1_ideas(request: Request):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    candidates = [i for i in db.list_ideas(user["id"]) if i["status"] == "candidate"]
    return render(request, "phase1_ideas.html", ideas=candidates[:config.idea_card_limit()])


@app.post("/phase1/extract")
def phase1_extract(request: Request, raw_text: str = Form(...)):
    user, redirect = login_or_redirect(request)
    if redirect:
        return redirect
    if len(raw_text.strip()) < 40:
        return render(
            request, "phase1_new.html", raw_text=raw_text,
            error="Paste a bit more material — a sentence or two isn't enough signal to extract ideas from.",
        )
    try:
        quota.check(user["id"])
        idea_cards = llm.extract_ideas(raw_text, user_id=user["id"])
    except llm.LLMError as e:
        return render(request, "phase1_new.html", raw_text=raw_text, error=str(e))

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
        "error": request.query_params.get("err") or None,
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
        return render(
            request, "venture_phase3.html",
            venture=venture, segments=ctx["segments"],
            resolved=ctx["resolved"], total_segments=ctx["total_segments"],
            gaps=db.open_gaps(venture_id),
            strategy={
                "funding_matches": strategy["funding_matches_json"],
                "gtm_channels": strategy["gtm_channels_json"],
                "action_plan": strategy["action_plan_json"],
            } if strategy else None,
            quota_used=ctx["quota_used"], quota_limit=ctx["quota_limit"],
        )
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
