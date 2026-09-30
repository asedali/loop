# LaunchLoop MVP

A working end-to-end prototype of the three-phase LaunchLoop workflow:

**Idea Discovery** (paste research material → AI-extracted Idea Cards) →
**Validation Loop** (repeating TODO → real-world action → AI analysis cycles against
a five-element Business Model Canvas, cycle-capped at 6) →
**Launch Strategy** (funding-type matches, GTM channels, sequenced action plan).

Each person who signs up gets their own private ideas and ventures (SQLite + a
plain email/password session — no third-party auth service).

Built with FastAPI + Jinja2 + SQLite + Gemini 2.0 Flash. Deploys as a single
long-running process (not serverless) so the SQLite file persists on disk.

## What this MVP deliberately leaves out

- No LinkedIn/X/Google Scholar ingestion — paste-text only for Phase 1
- No live Grants.gov/VC directory — funding matches are Gemini's reasoned category-level
  suggestions, clearly labeled as such in the UI
- No Zero Data Retention guarantee on the LLM calls — this is a prototype for a
  friends-only test, not a place to put real unpublished research/IP you'd want kept
  confidential at a legal level
- One shared Gemini free-tier quota across everyone who signs up — fine for a small
  group of friends, will hit rate limits with real traffic

## 1. Run it locally

```bash
cd launchloop
python -m venv venv && source venv/bin/activate   # or your preferred env tool
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste your free Gemini key from https://aistudio.google.com/apikey,
# and set SESSION_SECRET_KEY to a random string (see the comment in .env.example)

uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, sign up, and try the flow: Phase 1 → select an idea →
Phase 2 → run a cycle → log results → repeat until validated (or killed, or capped) →
Phase 3 → generate the launch strategy.

## 2. Deploy to Fly.io (so you can share a real URL with friends)

Fly.io works here specifically because it runs a persistent process with a
mountable volume — the SQLite file needs that; it would NOT work on a serverless
host like Vercel, which discards local disk between requests.

```bash
# one-time setup
curl -L https://fly.io/install.sh | sh     # installs the flyctl CLI
fly auth login

cd launchloop
fly launch --no-deploy   # walks you through picking an app name + region;
                          # it'll detect the Dockerfile automatically.
                          # Say NO to "would you like a Postgres database" —
                          # we don't need one.

# create the persistent volume the SQLite file lives on (1GB is plenty for an MVP)
fly volumes create launchloop_data --size 1 --region <the region fly.toml picked>

# set your secrets (never commit these — they're stored encrypted by Fly, not in fly.toml)
fly secrets set GEMINI_API_KEY=your-real-key-here
fly secrets set SESSION_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")

# deploy
fly deploy
```

Fly will print your app's URL (something like `https://launchloop-mvp.fly.dev`) —
that's what you share with friends. Each of them signs up with their own email/password
on that URL and gets their own isolated data.

### Updating after changes
Just run `fly deploy` again from the project directory — it rebuilds and redeploys
against the same persistent volume, so existing accounts/data survive.

### Backing up the SQLite file
Since everything lives in one file on one volume, back it up occasionally:
```bash
fly ssh sftp get /data/launchloop.db ./backup-launchloop.db
```

## Architecture notes

- `app/db.py` — all SQLite schema + queries. Deliberately plain `sqlite3`, no ORM,
  so it's easy to read end to end and port to Postgres later if you outgrow this.
- `app/llm.py` — the Gemini wrapper. Every call demands JSON and retries up to 3
  times if the model wraps its answer in prose or markdown fences (free-tier models
  do this more often than paid ones) — this is what keeps every AI touchpoint a
  structured card/form rather than open chat, per the product's core design rule.
- `app/auth.py` — bcrypt password hashing + a signed session cookie
  (Starlette's `SessionMiddleware`). No password reset flow, no email verification —
  fine for a friends-only test, not production-grade auth.
- `app/main.py` — all routes. The Phase 2 state machine (persevere/pivot/kill/cap)
  lives here rather than in a separate module, since it's short enough to read in one pass.

## Known rough edges (MVP, not production)
- No "forgot password" flow — if a friend forgets their password, you'd need to
  reset it manually via `fly ssh console` and a Python one-liner against the DB.
- Pivot suggestions are shown but not turned into a new structured hypothesis
  automatically — the researcher just sees the suggestion as text for now.
- Action plan items in Phase 3 are read-only (no "mark done" checkboxes yet).
