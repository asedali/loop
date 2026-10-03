# LaunchLoop MVP

A working end-to-end prototype of the three-phase LaunchLoop workflow:

**Idea Discovery** (paste research material → AI-extracted Idea Cards) →
**Validation Loop** (repeating: state a hypothesis about one Business Model Canvas
block → get a detailed research plan for it → go do it in the real world → log
outcomes → get a pass/iterate/fail/pivot verdict, block by block) →
**Launch Strategy** (funding-type matches, GTM channels, sequenced action plan).

Each person who signs up gets their own private ideas and ventures (Postgres +
a plain email/password session — no third-party auth service).

Built with FastAPI + Jinja2 + Postgres (Supabase). The LLM backend is any
OpenAI-SDK-shaped provider, chosen by environment variable.

## What this MVP deliberately leaves out

- No LinkedIn/X/Google Scholar ingestion — paste-text only for Phase 1
- No live Grants.gov/VC directory. Funding matches are the model's reasoned
  category-level suggestions and are labelled as unverified in the UI; the model
  is explicitly instructed never to invent deadlines, amounts, or links
- No "forgot password" flow, no email verification — not production-grade auth
- No row-level security. Every query filters by `user_id` in `app/db.py`, and a
  migration revokes the tables from Supabase's `anon`/`authenticated` roles so
  the public anon key cannot read them. Adding RLS is the next step if this
  grows past people you trust
- One shared provider account. There is a per-user monthly call cap
  (`LLM_MONTHLY_LIMIT_PER_USER`) but no billing or payout of any kind

## 1. Run it locally

```bash
cd launchloop
python -m venv venv && source venv/bin/activate   # or your preferred env tool
pip install -r requirements-dev.txt              # includes pytest

cp .env.example .env
# edit .env: set LLM_API_KEY, LLM_MODEL, and SESSION_SECRET_KEY to a random
# string, then paste your Supabase connection string into DATABASE_URL
#   python -c "import secrets; print(secrets.token_hex(32))"

# Use the SESSION POOLER (port 5432) string, not the Transaction pooler on
# 6543 — see .env.example. The schema is created on first boot, but you can
# also run it by hand:
alembic upgrade head

uvicorn app.main:app --reload
```

Open http://127.0.0.1.8000, sign up, and try the flow: Phase 1 → select an idea →
Phase 2 → run a cycle → log results → repeat until validated (or pivoted, killed,
or capped) → Phase 3 → generate the launch strategy.

Run the tests with `pytest -q` (87 tests). They need a Postgres but no network:
each test gets its own schema, built by running the real Alembic migrations, and
dropped afterwards. Set `TEST_DATABASE_URL`, or leave it unset and the fixtures
fall back to `DATABASE_URL` and then to `localhost/launchloop_test`:

```bash
createdb launchloop_test
pytest -q
```

Never point `TEST_DATABASE_URL` at production — the suite drops schemas.

## Waiting on the model, and when it breaks

An AI call takes 10-60s, and the provider retries transient failures up to
three times, so a slow one can run past two minutes. Three things keep that
from looking like a hung page:

- **A "Thinking…" overlay** on every form that calls the model, with a
  stage-by-stage description of what's happening, an elapsed timer, and a
  *Stop waiting* escape hatch. The page behind it stays on screen — the forms
  are submitted with `fetch` and the response is swapped in, rather than
  navigating to a blank page and back.
- **Plain-English errors.** `_friendly_error()` in `app/llm.py` maps provider
  exceptions to what a user can act on: a dropped connection, a timeout, a
  rate limit, a rejected key, and a bad model all get their own message
  instead of `Connection error`. Every one says the work is saved.
- **A friendly 500 page** for genuinely unexpected faults, so a bug never
  shows a stack trace.

If the provider is unreachable the app does **not** crash: `call_json` retries,
records the failure in `llm_calls`, raises `LLMError`, and each route renders
it inline. A 401 fails immediately rather than burning the retry budget.
`TestProviderFailure` covers all of this.

Everything above is progressive enhancement. With JavaScript off the forms
still POST natively and the whole flow works — you just get a blank page
during the wait instead of an overlay.

## Theming

Dark and light, following the OS by default, with a toggle in the nav that
persists to `localStorage`. A short inline script in `<head>` resolves the
theme before first paint so there's no flash of the wrong one.

The light palette isn't an inversion: status colours are darkened
(`--mixed` goes from `#fbbf24` to `#a16207`) because several of the dark
theme's values fall below 3:1 contrast on white. Surfaces and the default
button are driven by tokens (`--surface-deep`, `--btn-top`, …) rather than
hard-coded hexes, so the two themes can't drift.

## 2. Choose an LLM provider

`app/llm.py` speaks to any OpenAI-SDK-shaped endpoint. Set `LLM_BASE_URL`,
`LLM_API_KEY`, and `LLM_MODEL`; if those are absent it falls back to
`OPENROUTER_API_KEY` / `OPENROUTER_MODEL`.

> **Set all three `LLM_*` variables together, or none of them.** The fallback
> only triggers when `LLM_BASE_URL` *and* `LLM_API_KEY` are both unset. If you
> set only `LLM_BASE_URL`, the client points at Zen but still sends the
> OpenRouter key, and every call fails with a 401. `.env` ships with the Zen
> lines commented out for exactly this reason.

**OpenCode Zen** (https://opencode.ai/auth — billing details required):

```
LLM_BASE_URL=https://opencode.ai/zen/v1
LLM_API_KEY=<your Zen key>
LLM_MODEL=glm-5.3
```

Model IDs on Zen are flat, with no vendor prefix — `glm-5.3`, not
`z-ai/glm-5.3`. Only some models live on the OpenAI-compatible
`/chat/completions` path (GLM, DeepSeek, Kimi, Qwen, MiniMax, the free tiers);
the GPT/Claude/Gemini models are served from `/responses`, `/messages`, and
`/models/{id}` respectively and would need a second adapter.

**A privacy note that matters here.** Users paste unpublished research into
this app. Not all Zen models are equal: `big-pickle`, `mimo-*` and `ling-*` are
documented as *possibly training on your prompts*; `space-bunny-free` and
`longcat-2.5-preview-free` are zero-retention; OpenAI and Anthropic endpoints
retain requests for 30 days. The default here is `glm-5.3` for that reason. If
you widen the audience beyond people you know, keep it on a zero-retention
model.

## 3. Deploy to Fly.io (so you can share a real URL)

The app is stateless now that the database is Postgres, so it runs anywhere.
`render.yaml` and `fly.toml` both build the same Dockerfile.

### Render (recommended — free tier works)

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, pick the repo. Render reads `render.yaml`.
3. It prompts for the three secrets. Paste:

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the Supabase **session pooler** (port 5432) URI — see below |
   | `LLM_API_KEY` | your provider key |
   | `LLM_BASE_URL` | `https://opencode.ai/zen/v1` |

   `SESSION_SECRET_KEY` is generated for you.
4. First deploy applies the schema automatically.

Use the **session pooler (5432)**, never the transaction pooler (6543): pgbouncer
in transaction mode breaks the server-side prepared statements SQLAlchemy sends,
which shows up as intermittent runtime errors rather than a clean startup failure.

```bash
postgresql://postgres.PROJECT:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=verify-full&sslrootcert=certs/supabase-ca.crt
```

Three details that are easy to get wrong:

- The **username must include the project ref** — `postgres.PROJECT`, not plain
  `postgres`. The pooler parses it to work out which project you mean, and fails
  with `FATAL: no tenant identifier provided` otherwise.
- `sslmode=verify-full` is required, not `require`. The pooler identifies the
  tenant by TLS SNI, which libpq omits under `require`.
- `sslrootcert` points at `certs/supabase-ca.crt`, because Supabase's cert is
  signed by Supabase's own CA rather than a public one. It is committed to the
  repo and copied into the image; re-download it from
  `https://supabase-downloads.s3-ap-southeast-1.amazonaws.com/prod/ssl/prod-ca-2021.crt`
  if it ever expires.

The free plan sleeps after ~15 idle minutes, so the first request after a pause
takes a few extra seconds while the container boots and re-runs migrations. That
is harmless: migrations are idempotent and run under a Postgres advisory lock, so
concurrent boots cannot collide.

### Fly.io (alternative)

```bash
curl -L https://fly.io/install.sh | sh     # installs the flyctl CLI
fly auth login

fly launch --no-deploy   # pick an app name; it detects the Dockerfile
                         # Say NO to "would you like a Postgres database" —
                         # Supabase is the database. The region should match
                         # your Supabase project's to keep the latency down.

fly secrets set SESSION_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
fly secrets set LLM_API_KEY=your-real-key-here
fly secrets set DATABASE_URL='postgresql://postgres.PROJECT:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=verify-full&sslrootcert=certs/supabase-ca.crt'

fly deploy
```

### Updating after changes

Push to GitHub (Render auto-deploys) or run `fly deploy`. Either way
`alembic upgrade head` runs before the new code serves traffic.

### Backing up

Supabase handles this: point-in-time recovery is on by default, and you can take
a logical dump with `pg_dump`:

```bash
pg_dump "$DATABASE_URL" -Fc -f launchloop-$(date +%F).dump
```

## Architecture notes

- `app/db.py` — every query, as SQLAlchemy Core statements over psycopg3. No ORM
  models: the tables in `app/schema.py` are used to build queries, and functions
  return mappings so callers keep using `row["col"]`. Reads a row-mapped result
  and writes with `RETURNING`, which replaces SQLite's `lastrowid`.
- `app/schema.py` / `app/constants.py` — table definitions and the enum-like
  values, kept free of any engine import so Alembic can read them without
  opening a connection. The CHECK constraints are generated from the constants,
  so they cannot drift from what the app validates in Python.
- `migrations/` — Alembic, versioned. `init_db()` runs `upgrade head` on boot
  under a Postgres advisory lock, so concurrent boots are safe.
- Tenant isolation is `user_id` in the WHERE clause, not RLS. A migration revokes
  all tables from Supabase's `anon` and `authenticated` roles — without it the
  project's public anon key can read `users.password_hash`.
- `app/llm.py` — the provider wrapper. Every response is validated against an
  allowlist (canvas element names, statuses, decisions, task types) before it
  leaves this file; unrecognised values are dropped rather than persisted,
  because one hallucinated enum value used to make Phase 2's completion check
  unreachable. Retries only on transient failures — a 401 fails immediately.
- `app/main.py` — all routes. `venture_phase_state()` is the single source of
  truth for what a venture will let you do next, and the POST routes enforce it;
  the cycle cap is not a template-only convention any more.
- `app/quota.py` — per-user monthly call cap, counted from the `llm_calls` table
  that doubles as the request/cost log.
- `app/static/app.js` — the thinking overlay and the theme toggle. The overlay
  needs no backend support: those routes already redirect with 303, so
  `fetch` follows the redirect and the rendered HTML is swapped into `<main>`.
  A server that answers the POST in place (a quota or provider error) keeps the
  current URL rather than pushing the POST endpoint into the address bar.
- `app/templates/macros.html` — the Business Model Canvas, shared by Phase 2 and
  Phase 3 so the two can't drift. It renders the classic five-column layout with
  a financial row beneath. The loop tracks five of the seven shown blocks; Key
  Partners and Cost Structure render dimmed and explicitly labelled "not tracked
  by the loop" rather than being quietly dropped, so the canvas reads as the
  real thing without overstating coverage.

## The validation loop

The canvas is the full nine-block Osterwalder model, and the loop works **one
block at a time**. Each block carries its own hypothesis, its own run history,
and its own run cap.

| # | Block | Severity |
|---|-------|----------|
| 1 | Customer Segments | critical |
| 2 | Problem & Key Activities | critical |
| 3 | Value Propositions | critical |
| 4 | Revenue Streams | critical |
| 5 | Channels | important |
| 6 | Cost Structure | important |
| 7 | Key Partners | important |
| 8 | Key Resources | important |
| 9 | Customer Relationships | important |

The order is a recommendation, not a constraint — every block is clickable, so
you can test pricing first if you already know your channel is dead.

### What a run looks like

1. The model states a **falsifiable hypothesis** for that block.
2. It writes 2–4 tasks, each with a **method** drawn from a per-block list
   (`problem_interview`, `preorder_test`, `vendor_quote`, `dependency_map`, …),
   concrete steps, a target sample size, an effort estimate, and — importantly —
   a **success criterion** you can judge true or false afterwards.
3. You do the work and log what happened.
4. The model returns a **verdict** for that block:

| Verdict | Effect |
|---|---|
| `pass` | block → `passed`, evidence forced to `confirmed` |
| `iterate` | evidence was thin; run again, up to the block's cap |
| `fail` on a **critical** block | the venture is killed and the idea returns to candidates |
| `fail` on an **important** block | the block is **parked** with a recorded workaround; the venture carries on |
| `pivot` | spawns a sibling venture on a fresh nine-block canvas |

### Why severity matters

Losing your Key Partners block should not suggest killing an otherwise validated
business, and a Customer Segments failure should not be parkable. `severity` is
what makes those two cases behave differently.

### Why parked counts as resolved

Phase 3 unlocks when all nine blocks are `passed` **or** `parked`. Finishing with
documented gaps is a legitimate outcome, and those gaps are carried explicitly
into the launch-strategy prompt rather than quietly dropped. A `parked` block can
be reopened at any time.

### Run caps are per block

Each block gets 3 runs (`SEGMENT_CAP`). Exhausting one parks *that block* and
offers "+3 more runs" — it never pauses the whole venture, which is what the old
global cap did. The venture-level `max_cycles` is only a runaway backstop
(9 × 3 = 27).

### Cost

A full nine-block validation is roughly **54 AI calls** (9 blocks × 3 runs ×
2 calls: design the run, then score it). The default per-user monthly cap is
500, so about nine complete validations.

### Storage

A "run" is one row in `cycles`, scoped by the new `segment` column. Rows written
before this change spanned several blocks at once, so the migration attributes
them to a block only where every task named the same one, and translates the old
`persevere`/`kill` decisions to `iterate`/`fail`. Blocks the old loop had already
marked `confirmed` come back as `passed`, so existing work is not thrown away.

## Known rough edges (MVP, not production)

- No password reset. If someone forgets theirs, you have to reset it by hand
  against the DB (Supabase SQL editor, or `psql $DATABASE_URL`).
- `llm_calls` stores prompts' metadata but not the prompts themselves, so you
  can see spend and latency but cannot audit what was actually sent.
- No per-request rate limiting on the LLM-touching routes; the monthly quota is
  the only brake, so a single user can still tie up a worker thread for the
  duration of a slow provider call.
- Action plan items in Phase 3 are read-only (no "mark done" checkboxes yet).
- The per-block cap is a blunt instrument: a block that needs a fourth good-faith
  pass has to be extended explicitly, which is the intended friction but is
  still friction.
- No RLS. Isolation is `user_id` in the WHERE clause, so a bug in a query is a
  data leak. The `anon`/`authenticated` grants are revoked, but `service_role`
  still bypasses everything — keep the connection string in a Fly secret.
- The connection pool is fixed at 5+5. Supabase's pooler has its own limits, and
  a `shared-cpu-1x` machine running many machines will queue rather than scale.
