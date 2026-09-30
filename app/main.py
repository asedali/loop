import json
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from . import db, auth, llm

app = FastAPI(title="LaunchLoop MVP")

SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "dev-only-insecure-secret-change-me")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
def on_startup():
    db.init_db()


def render(request, template_name, **context):
    context["user"] = auth.get_current_user(request)
    # `request` must be the first positional argument on current Starlette
    # versions — passing it inside the context dict (the old calling
    # convention) causes an internal argument mix-up on newer Starlette
    # releases ("unhashable type: dict" from the template cache lookup).
    return templates.TemplateResponse(request, template_name, context)


# ==================== Auth ====================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = auth.get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard", status_code=303)
    return RedirectResponse(url="/login", status_code=303)


@app.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request):
    return render(request, "signup.html", error=None)


@app.post("/signup")
def signup(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    if db.get_user_by_email(email):
        return render(request, "signup.html", error="An account with that email already exists.")
    if len(password) < 6:
        return render(request, "signup.html", error="Password must be at least 6 characters.")
    user_id = db.create_user(email, auth.hash_password(password))
    request.session["user_id"] = user_id
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return render(request, "login.html", error=None)


@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()
    user = db.get_user_by_email(email)
    if not user or not auth.verify_password(password, user["password_hash"]):
        return render(request, "login.html", error="Invalid email or password.")
    request.session["user_id"] = user["id"]
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


# ==================== Dashboard ====================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    ideas = db.list_ideas(user["id"])
    ventures = db.list_ventures(user["id"])
    return render(request, "dashboard.html", ideas=ideas, ventures=ventures, cycle_cap=db.CYCLE_CAP)


# ==================== Phase 1: Idea Discovery ====================

@app.get("/phase1/new", response_class=HTMLResponse)
def phase1_new(request: Request):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return render(request, "phase1_new.html", error=None)


@app.post("/phase1/extract")
def phase1_extract(request: Request, raw_text: str = Form(...)):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    if len(raw_text.strip()) < 40:
        return render(request, "phase1_new.html", error="Paste a bit more material — a sentence or two isn't enough signal to extract ideas from.")
    try:
        idea_cards = llm.extract_ideas(raw_text)
    except llm.LLMError as e:
        return render(request, "phase1_new.html", error=str(e))

    saved = []
    for card in idea_cards:
        idea_id = db.create_idea(
            user_id=user["id"],
            title=card.get("title", "Untitled idea"),
            commercial_framing=card.get("commercial_framing", ""),
            strength_signal=card.get("strength_signal", "early"),
            raw_claims=card.get("raw_claims", ""),
        )
        saved.append(db.get_idea(idea_id, user["id"]))
    return render(request, "phase1_ideas.html", ideas=saved)


@app.post("/phase1/select/{idea_id}")
def phase1_select(request: Request, idea_id: int):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    idea = db.get_idea(idea_id, user["id"])
    if not idea:
        return RedirectResponse(url="/dashboard", status_code=303)
    db.set_idea_status(idea_id, "selected")
    venture_id = db.create_venture(user["id"], idea_id)
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


# ==================== Venture (Phase 2 + 3 router) ====================

@app.get("/venture/{venture_id}", response_class=HTMLResponse)
def venture_view(request: Request, venture_id: int):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)

    bmc_elements = db.get_bmc_elements(venture_id)
    cycles = db.list_cycles(venture_id)
    current_cycle = db.get_current_cycle(venture_id)

    if venture["phase"] == 3 or venture["status"] == "validated":
        strategy = db.get_launch_strategy(venture_id)
        strategy_data = None
        if strategy:
            strategy_data = {
                "funding_matches": json.loads(strategy["funding_matches_json"]),
                "gtm_channels": json.loads(strategy["gtm_channels_json"]),
                "action_plan": json.loads(strategy["action_plan_json"]),
            }
        return render(
            request, "venture_phase3.html",
            venture=venture, bmc_elements=bmc_elements, strategy=strategy_data,
        )

    current_todos = json.loads(current_cycle["todos_json"]) if current_cycle else None
    current_results = json.loads(current_cycle["results_json"]) if current_cycle and current_cycle["results_json"] else None
    current_analysis = json.loads(current_cycle["analysis_json"]) if current_cycle and current_cycle["analysis_json"] else None
    awaiting_results = current_cycle is not None and current_results is None
    awaiting_analysis = current_cycle is not None and current_results is not None and current_analysis is None
    cycle_done = current_cycle is not None and current_analysis is not None
    can_start_new_cycle = (
        (current_cycle is None or cycle_done)
        and venture["status"] == "active"
        and venture["cycle_count"] < db.CYCLE_CAP
    )
    cap_reached = venture["cycle_count"] >= db.CYCLE_CAP and venture["status"] == "active" and (current_cycle is None or cycle_done)

    return render(
        request, "venture_phase2.html",
        venture=venture, bmc_elements=bmc_elements, cycles=cycles,
        current_cycle=current_cycle, current_todos=current_todos,
        current_results=current_results, current_analysis=current_analysis,
        awaiting_results=awaiting_results, awaiting_analysis=awaiting_analysis,
        can_start_new_cycle=can_start_new_cycle, cap_reached=cap_reached,
        cycle_cap=db.CYCLE_CAP, error=None,
    )


@app.post("/venture/{venture_id}/cycle/new")
def new_cycle(request: Request, venture_id: int):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)

    bmc_elements = [dict(row) for row in db.get_bmc_elements(venture_id)]
    next_cycle_number = venture["cycle_count"] + 1
    try:
        todos = llm.generate_todos(
            venture["idea_title"], venture["commercial_framing"], bmc_elements, next_cycle_number
        )
    except llm.LLMError as e:
        bmc_elements_full = db.get_bmc_elements(venture_id)
        return render(
            request, "venture_phase2.html", venture=venture, bmc_elements=bmc_elements_full,
            cycles=db.list_cycles(venture_id), current_cycle=None, current_todos=None,
            current_results=None, current_analysis=None, awaiting_results=False,
            awaiting_analysis=False, can_start_new_cycle=True, cap_reached=False,
            cycle_cap=db.CYCLE_CAP, error=str(e),
        )
    db.create_cycle(venture_id, next_cycle_number, todos)
    db.update_venture(venture_id, cycle_count=next_cycle_number)
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


@app.post("/venture/{venture_id}/cycle/{cycle_id}/log")
async def log_cycle(request: Request, venture_id: int, cycle_id: int):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    venture = db.get_venture(venture_id, user["id"])
    cycle = db.get_cycle(cycle_id, venture_id)
    if not venture or not cycle:
        return RedirectResponse(url="/dashboard", status_code=303)

    form = await request.form()
    todos = json.loads(cycle["todos_json"])
    results = []
    for i in range(len(todos)):
        results.append({
            "outcome": form.get(f"outcome_{i}", "").strip(),
            "sample_size": form.get(f"sample_size_{i}", "").strip(),
        })
    db.log_cycle_results(cycle_id, results)

    bmc_elements = [dict(row) for row in db.get_bmc_elements(venture_id)]
    try:
        analysis = llm.analyze_cycle(venture["idea_title"], todos, results, bmc_elements)
    except llm.LLMError:
        # results are saved; user can retry analysis from the page
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    for el in analysis.get("elements", []):
        db.update_bmc_element(
            venture_id, el["element_name"], el["status"], el.get("notes", "")
        )
    decision = analysis.get("recommendation", "persevere")
    db.save_cycle_analysis(cycle_id, analysis, decision)

    if db.all_confirmed(venture_id):
        db.update_venture(venture_id, phase=3, status="validated")
    elif decision == "kill":
        db.update_venture(venture_id, status="killed")
        db.set_idea_status(venture["idea_id"], "candidate")
    elif venture["cycle_count"] >= db.CYCLE_CAP:
        db.update_venture(venture_id, status="paused")

    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


@app.post("/venture/{venture_id}/cycle/{cycle_id}/retry-analysis")
def retry_analysis(request: Request, venture_id: int, cycle_id: int):
    """If the analysis call failed after results were logged, retry it here
    instead of forcing the user to re-log everything."""
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    venture = db.get_venture(venture_id, user["id"])
    cycle = db.get_cycle(cycle_id, venture_id)
    if not venture or not cycle or not cycle["results_json"]:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    todos = json.loads(cycle["todos_json"])
    results = json.loads(cycle["results_json"])
    bmc_elements = [dict(row) for row in db.get_bmc_elements(venture_id)]
    try:
        analysis = llm.analyze_cycle(venture["idea_title"], todos, results, bmc_elements)
    except llm.LLMError:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)

    for el in analysis.get("elements", []):
        db.update_bmc_element(venture_id, el["element_name"], el["status"], el.get("notes", ""))
    decision = analysis.get("recommendation", "persevere")
    db.save_cycle_analysis(cycle_id, analysis, decision)

    if db.all_confirmed(venture_id):
        db.update_venture(venture_id, phase=3, status="validated")
    elif decision == "kill":
        db.update_venture(venture_id, status="killed")
        db.set_idea_status(venture["idea_id"], "candidate")
    elif venture["cycle_count"] >= db.CYCLE_CAP:
        db.update_venture(venture_id, status="paused")

    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


@app.post("/venture/{venture_id}/resume")
def resume_venture(request: Request, venture_id: int):
    """Manual override to keep looping past the cycle cap."""
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    venture = db.get_venture(venture_id, user["id"])
    if venture and venture["status"] == "paused":
        db.update_venture(venture_id, status="active")
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)


# ==================== Phase 3: Launch Strategy ====================

@app.post("/venture/{venture_id}/phase3/generate")
def generate_strategy(request: Request, venture_id: int):
    user = auth.require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    venture = db.get_venture(venture_id, user["id"])
    if not venture:
        return RedirectResponse(url="/dashboard", status_code=303)
    bmc_elements = [dict(row) for row in db.get_bmc_elements(venture_id)]
    try:
        strategy = llm.generate_launch_strategy(
            venture["idea_title"], venture["commercial_framing"], bmc_elements
        )
    except llm.LLMError:
        return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)
    db.save_launch_strategy(
        venture_id,
        strategy.get("funding_matches", []),
        strategy.get("gtm_channels", []),
        strategy.get("action_plan", []),
    )
    return RedirectResponse(url=f"/venture/{venture_id}", status_code=303)
