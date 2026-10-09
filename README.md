# LaunchLoop MVP

A working end-to-end prototype of the three-phase LaunchLoop workflow:

**Idea Discovery** (paste research material → AI-extracted Idea Cards) →
**Validation Loop** (one Business Model Canvas block at a time: state a
falsifiable hypothesis → get a research plan for it → go do it in the real
world → log outcomes → get a pass/iterate/fail/pivot verdict) →
**Launch Strategy** (funding-type matches, GTM channels, and a sequenced action
plan you work through — steps marked done or blocked with the outcome recorded,
progress surviving a regeneration of the plan).

Each person who signs up gets their own private ideas and ventures (Postgres +
a plain email/password session — no third-party auth service).

Built with FastAPI + Jinja2 + Postgres (Supabase). The LLM backend is any
OpenAI-SDK-shaped provider, chosen by environment variable.

**Further reading**

| Document | What it is for |
|---|---|
| `README.md` (this file) | Features, how to run it, how it is deployed, current limitations |
| [`docs/TECHNICAL.md`](docs/TECHNICAL.md) | Architecture, data model, state machines, module contracts — for engineers and coding agents changing the code |
| [`docs/FLOW.md`](docs/FLOW.md) | The product process end to end, and exactly how the user and the system interact at each step, with eight diagrams in [`docs/img/`](docs/img/) |

The diagrams are standalone SVGs drawn with the app's own colour tokens, so they
follow the reader's light/dark theme and need no tooling to regenerate.

## Status

Working prototype, deployed-shaped, not production-hardened. The whole three-phase
flow works, per-block validation is the default, and the suite is green
(**464 tests**, `pytest -q`).

What that means in practice:

- All nine canvas blocks are tracked, clickable, and independently testable.
- The state machine is enforced server-side, not just hidden in templates.
- Kill-or-pivot on a failed critical block is a **user decision** — the model
  marks the block failed and stops there.
- Every AI call is logged with purpose, model, attempts, tokens, latency and
  outcome; the same table drives the per-user monthly quota.
- Tenant isolation is three layers: `user_id` filtering in every query, row-level
  security policies in the database, and revoked Supabase `anon`/`authenticated`
  grants. `/healthz` reports whether the RLS layer is actually enforcing — see
  the deploy note below, because it is inert without one manual step.
- Auth is bcrypt + a signed session cookie, with sliding-window login
  throttling on both account and source address that clears on success.
  Password reset works end to end, over single-use hashed tokens with a
  30-minute lifetime. Email confirmation is built and tested but switched off
  by default (`EMAIL_VERIFICATION_ENABLED=0`) — see below.
- Changing your password — while signed in, or through a reset link — **signs out
  every other session**, enforced by a `session_epoch` counter compared on every
  request. The check costs no extra query, because the user row is already loaded
  to install the RLS tenant.
- Account & data: download everything as versioned JSON, or delete the account
  and everything in it behind two gates (password + typed `DELETE`). The export
  carries no passwords and no reset-token hashes.
- **"Think like a founder"** — pick one of six operator playbooks (Jobs on
  subtraction, Musk on first principles, Graham on unscalable work, Ries on
  falsification, Christensen on demand, plus the app's own evidence standard) and
  it asks questions about what you have actually recorded. It **never answers**:
  the output has no field for advice, first-person and name-naming lines are
  dropped, and nothing in the path can change a block, a verdict or a venture.
  Every card carries a "not a quote, this person is not involved" notice.
- Near-duplicate idea cards are **flagged, never merged** — each says "Also
  extracted as" and links to the other, and you can dismiss a card you don't want
  (dismissed cards stay in a "Dismissed ideas" list with a Restore button; nothing
  is deleted). Detection is character-trigram similarity with the threshold
  calibrated against measured pairs, not a model call, so it costs no quota.
- The Phase 3 action plan is a checklist, not a document: steps are marked
  `done` or `blocked` and **the outcome is mandatory** — a step with no record of
  what happened is the unevidenced claim this app exists to replace. Recording one
  is a plain write, so it costs no quota.

## What this MVP deliberately leaves out

- **Identifier import** — paste an ORCID iD, a DOI, or an arXiv ID and the title,
  authors and abstract come back as editable material. Open public metadata APIs
  only (ORCID, Crossref, arXiv); no account, no cookie, nothing sent about you.
  Nothing about the record is persisted — it is cached in memory for a day and
  nothing is stored
- No LinkedIn/X/Google Scholar ingestion — those need auth and are the fragile
  ones. The three identifier sources here are open and unauthenticated
- File upload covers **PDF, Word and plain text**. The file is read in memory,
  turned into text, and **discarded** — nothing is stored, so there is nothing to
  clean up or leak. A scanned document has no text layer and the app says so
  rather than guessing. OCR is not supported.
- No live Grants.gov/VC directory. Funding matches are the model's reasoned
  category-level suggestions and are labelled as unverified in the UI; the model
  is explicitly instructed never to invent deadlines, amounts, or links
- Email confirmation is **off by default** (`EMAIL_VERIFICATION_ENABLED`). No
  mail provider is configured, so a live flag would show every account a banner
  asking for a link that never arrives
- Email is `console`-backed by default, so a real deployment must set
  `MAIL_BACKEND=smtp` and `APP_BASE_URL` or nobody receives a reset link
- One shared provider account. There is a per-user monthly call cap
  (`LLM_MONTHLY_LIMIT_PER_USER`) but no billing or payout of any kind
- The action-plan checklist is not a task manager — no owners, no due dates, no
  reminders, and nothing is tracked once the plan is done

## 1. Run it locally

```bash
cd launchloop
python -m venv venv && source venv/bin/activate   # or your preferred env tool
pip install -r requirements-dev.txt              # includes pytest

cp .env.example .env
# edit .env: set LLM_API_KEY, LLM_MODEL, and SESSION_SECRET_KEY to a random
# string, then paste your Supabase connection string into DATABASE_URL
#   python -c "import secrets; print(secrets.token_hex(32))"
#   set SOURCE_IMPORT_ENABLED=false to disable all outbound third-party calls

# Use the SESSION POOLER (port 5432) string, not the Transaction pooler on
# 6543 — see .env.example. The schema is created on first boot, but you can
# also run it by hand:
alembic upgrade head

uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, sign up, and try the flow: Phase 1 → select an idea →
Phase 2 → open a canvas block → run a cycle → log results → repeat until resolved
(or pivoted, killed, or capped) → Phase 3 → generate the launch strategy.

### Tests

```bash
createdb launchloop_test
pytest -q            # 464 tests, ~7m
```

They need a Postgres but no network. Each test gets its own schema, built by
running the real Alembic migrations, and dropped afterwards — so the suite
exercises the same DDL production does. Set `TEST_DATABASE_URL`, or leave it
unset and the fixtures fall back to `DATABASE_URL` and then to
`localhost/launchloop_test`. A fresh clone can run the suite with no `.env` at
all: `tests/conftest.py` sets a throwaway session secret before importing the
app.

Never point `TEST_DATABASE_URL` at production — the suite drops schemas.

`tests/app_js.test.js` runs under `node` (shelled out to by `test_frontend.py`),
so the front-end contract tests need Node installed. There is no skip: a missing
`node` binary fails that one test.

### Migrating an old local SQLite file

`launchloop.db` from the pre-Postgres version can be copied in one shot. Primary
keys are preserved, every insert is an upsert, so an interrupted run can simply
be repeated:

```bash
alembic upgrade head      # the script assumes the target schema exists
python scripts/migrate_sqlite_to_supabase.py --dry-run
python scripts/migrate_sqlite_to_supabase.py
```

## Waiting on the model, and when it breaks

An AI call takes 10-60s, and the provider retries transient failures up to
three times, so a slow one can run past two minutes. Four things keep that
from looking like a hung page:

- **A "Thinking…" overlay** on every form that calls the model, with a
  stage-by-stage description of what's happening (per route, not generic), an
  elapsed timer, and a *Stop waiting* escape hatch. The page behind it stays on
  screen — the forms are submitted with `fetch` and the response is swapped in,
  rather than navigating to a blank page and back.
- **Plain-English errors.** `_friendly_error()` in `app/llm.py` maps provider
  exceptions to what a user can act on: a dropped connection, a timeout, a
  rate limit, a rejected key, a denied model, a bad request, and a truncated
  response each get their own message instead of `Connection error`. Every one
  says the work is saved.
- **Nothing is lost on failure.** Logged results are persisted before the
  scoring call, so a failed verdict offers *Retry verdict* rather than making
  the user re-enter what they typed.
- **A friendly 500 page** for genuinely unexpected faults, so a bug never
  shows a stack trace.

If the provider is unreachable the app does **not** crash: `call_json` retries,
records the failure in `llm_calls`, raises `LLMError`, and each route renders
it inline. A 401 fails immediately rather than burning the retry budget. A
truncated response also fails immediately — the same ceiling gives the same
result, so retrying only burns the budget again. `TestProviderFailure` covers
all of this.

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

`.env.example` lists only those three. The alternate endpoints
(`LLM_ZEN_BASE_URL`, `LLM_OPENROUTER_BASE_URL`), the fallback model
(`LLM_DEFAULT_MODEL`), and the log label (`LLM_PROVIDER`) have built-in defaults
in `app/config.py` and are only read on the fallback path, so a normal setup
never sets them — they are documented as prose at the end of the LLM section
for the rare provider switch.

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
retain requests for 30 days. `LLM_DEFAULT_MODEL` — the built-in fallback,
consulted only when `LLM_MODEL` is unset — is `glm-5.3` for that reason. If you
widen the audience beyond people you know, keep it on a zero-retention model.

> **The shipped `.env.example` sets `LLM_MODEL=space-bunny-free`, and so does
> `render.yaml`,** which is a zero-retention free tier — consistent with the
> privacy rule, but a different model from the `glm-5.3` documented above. Pick
> one deliberately per environment; the free tier is rate-limited and will hit
> the empty-response path described in `app/llm.py` under load.

Two knobs worth knowing about, because the defaults bite:

- `LLM_REASONING_EFFORT` (default `low`) bounds a reasoning model's internal
  reasoning. Reasoning tokens draw from the *same* ceiling as output, so left
  unbounded a reasoning model can spend the whole budget thinking and return an
  empty message. Set it to `none` for models that reject the parameter — the
  app also auto-drops the parameter if the provider names it in a 400.
- `LLM_MAX_OUTPUT_TOKENS` (default 4096; `render.yaml` ships 2048) is the
  ceiling that truncation is measured against.

## 3. Deploy

The app is stateless now that the database is Postgres, so it runs anywhere. Two
platforms are configured; both deploy the same code.

| | Render | Vercel |
|---|---|---|
| Build | `Dockerfile` (`render.yaml`) | native Python, from `requirements.txt` |
| Runtime shape | one long-lived container | function that scales to zero |
| Migrations | at boot | run manually before deploying |
| Connection pool | 5 + 5 overflow | 1, no overflow |
| Max upload | 8 MB | 4 MB (platform caps bodies at 4.5 MB) |

Vercel ignores the `Dockerfile`: its container path looks for `Dockerfile.vercel`
at the repo root, which this repo does not have, so the two platforms never
contend for the same file.

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

The health check hits `/healthz`, which does a real Postgres round-trip but
deliberately does **not** touch the LLM provider — a provider outage should not
fail the deploy health check and get the machine cycled.

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
  if it ever expires. `app/db.py` rewrites a relative `sslrootcert` to an
  absolute path anchored at the project root, so starting the app from another
  working directory still works — and it fails loudly if the file is missing.

The free plan sleeps after ~15 idle minutes, so the first request after a pause
takes a few extra seconds while the container boots and re-runs migrations. That
is harmless: migrations are idempotent and run under a Postgres advisory lock, so
concurrent boots cannot collide.

### Vercel

Runs on the native Python runtime, so there is no image and no `Dockerfile` in
the path. `vercel.json` pins the region and duration, and dependencies come from
`requirements.txt` — the same file the Dockerfile installs, so there is exactly
one list of what this app needs.

There is deliberately **no `pyproject.toml`**. Its mere presence makes Vercel
prefer it over `requirements.txt` and then run `uv lock`, which requires a PEP 621
`[project]` table this repo has no reason to maintain — and the build fails with
`No 'project' table found`. The pytest config that used to live there is now in
`pytest.ini`.

```bash
vercel link          # once, to bind this directory to a project
alembic upgrade head # migrations are NOT run at boot here — see below
vercel deploy --prod
```

`vercel dev` runs it locally with the same routing and env wiring.

**Migrations are a manual step on Vercel, deliberately.** A function that scales
to zero re-runs its boot hook on the first request after every idle period, which
would put twelve migrations and an advisory-lock round trip in front of requests
that do not need them — and make Supabase's pooler the ceiling on concurrency. So
`RUN_MIGRATIONS_ON_BOOT=0` there and you run `alembic upgrade head` yourself
first. Render keeps it on: one container, one boot, free thereafter.

A failed migration at boot is **logged, not raised**, on both platforms. Boot is
where a transient database blip is most likely to land, and raising there means
every subsequent request 500s too.

Set these in the Vercel dashboard (Settings → Environment Variables):

| Key | Value | Why |
|---|---|---|
| `DATABASE_URL` | the session pooler URI | see the Render section above for the full form |
| `SESSION_SECRET_KEY` | generate one | the app refuses to boot without a non-default |
| `LLM_API_KEY` | your provider key | |
| `SESSION_HTTPS_ONLY` | `1` | **Vercel does not use the Dockerfile**, which is the only place this is set on Render. It defaults to off, so session cookies would otherwise go out over plaintext. |
| `RUN_MIGRATIONS_ON_BOOT` | `0` | see above |
| `DB_POOL_SIZE` | `1` | every concurrent invocation is its own process |
| `DB_POOL_MAX_OVERFLOW` | `0` | so 20 simultaneous users do not ask the pooler for 200 connections |
| `MAX_UPLOAD_BYTES` | `4194304` | Vercel rejects bodies over 4.5 MB before the app sees them; 4 MB leaves room for the multipart envelope |
| `DB_APP_ROLE` | `launchloop_app` | see RLS below |
| `APP_BASE_URL` | your Vercel URL | every link in every email is built from this |

`vercel.json` sets `regions: ["bom1"]` — Mumbai, `ap-south-1`, matching an
`aws-0-ap-south-1` Supabase pooler. Vercel's default is Washington D.C., which
would put a trans-Pacific round trip in front of every query.

`maxDuration: 300` is the floor that works, not a comfort setting: a single
request can spend `LLM_TIMEOUT` × `LLM_MAX_ATTEMPTS` plus backoff, which is 186
seconds at the shipped defaults. Raising either of those without raising this
means 504s mid-verdict.

### Enabling row-level security (one manual step)

The RLS policies are installed by migration `e7a2b4c9d016`, but **they are inert
until the app connects as a role that cannot bypass them.** Postgres superusers
bypass RLS regardless of `FORCE ROW LEVEL SECURITY`, and so does Supabase's
`service_role` — which is what `DATABASE_URL` points at.

Until you do the step below, tenant isolation rests on the `user_id` filter in
every query. That is real defence, but it is defence by convention: one forgotten
`WHERE` clause is a cross-tenant read, and nothing would notice.

In the **Supabase SQL editor**, signed in as the `postgres` superuser:

```sql
CREATE ROLE launchloop_app NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
GRANT launchloop_app TO service_role;
```

Do **not** change `DATABASE_URL`. The app switches roles at runtime with
`SET LOCAL ROLE` rather than connecting as a different user, so your pooler URL,
TLS settings and certificate keep working unchanged.

Then confirm it took effect:

```bash
curl -s "$APP_URL/healthz"     # {"status":"ok","rls":true}
```

`rls: false` means the grant is missing. The app keeps working either way — the
switch is membership-tested per connection precisely so that a missing grant
degrades to the `user_id` filters instead of 500-ing every request.

Locally this is automatic: the test suite grants itself the role and asserts the
policies are genuinely being enforced, so a silent bypass fails CI.

### Updating after changes

Push to GitHub (Render auto-deploys). `alembic upgrade head` runs before the new
code serves traffic.

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
  and writes with `RETURNING`, which replaces SQLite's `lastrowid`. Every call
  opens and commits its own transaction via the `get_conn()` context manager.
- `app/schema.py` / `app/constants.py` — table definitions and the enum-like
  values, kept free of any engine import so Alembic can read them without
  opening a connection. The CHECK constraints are generated from the constants,
  so they cannot drift from what the app validates in Python.
- `app/config.py` — every environment-tunable value in one place, split into
  boot-time settings (read once at import, because the session layer needs them
  before the first request) and runtime settings (read per call through a
  function). Readers degrade to the default on blank, junk, or out-of-range
  values, so a typo in a deploy config cannot wedge the app.
- `migrations/` — Alembic, versioned: `f1ceb3f3319f` initial schema →
  `a1c9f0e2b7d1` revoke Supabase anon access → `b3d7e91c4a02` per-segment loop.
  `init_db()` runs `upgrade head` on boot under a Postgres advisory lock, so
  concurrent boots are safe.
- Tenant isolation is `user_id` in the WHERE clause, not RLS. A migration revokes
  all tables from Supabase's `anon` and `authenticated` roles — without it the
  project's public anon key can read `users.password_hash`.
- `app/llm.py` — the provider wrapper. Every response is validated against an
  allowlist (canvas element names, statuses, decisions, task types) before it
  leaves this file; unrecognised values are dropped rather than persisted,
  because one hallucinated enum value used to make Phase 2's completion check
  unreachable. Retries only on transient failures — a 401 fails immediately.
- `app/main.py` — all routes. `venture_phase_state()` and `segment_run_state()`
  are the single source of truth for what a venture and a block will let you do
  next, and the POST routes enforce them; the run cap is not a template-only
  convention any more.
- `app/quota.py` — per-user monthly call cap, counted from the `llm_calls` table
  that doubles as the request/cost log. Failed calls are logged too, so they
  count against the cap.
- `app/static/app.js` — the thinking overlay, the page-transition bar, and the
  theme toggle. None of it needs backend support: those routes already redirect
  with 303, so `fetch` follows the redirect and the rendered HTML is swapped into
  `<main>`. A server that answers the POST in place (a quota or provider error)
  keeps the current URL rather than pushing the POST endpoint into the address bar.
- `app/templates/macros.html` — the Business Model Canvas, shared by Phase 2 and
  Phase 3 so the two can't drift. It renders the canonical Osterwalder layout
  (Key Partners / Key Activities / Value Propositions / Customer Relationships /
  Customer Segments across the top, Key Resources and Channels stacked
  underneath their neighbours, Cost Structure and Revenue Streams across the
  financial row), placed by named CSS grid areas rather than document order, and
  **all nine blocks are live and clickable** — there is no "untracked" branch
  left. A passed block shows its extracted answer as the headline — the value
  proposition for Value Propositions, the segment for Customer Segments — with
  the supporting evidence in smaller type beneath.
- `tests/` — 464 tests. `tests/test_launchloop.py` covers the state machine,
  LLM output validation, caps, isolation, auth, quota, the call log, rendering,
  provider failure, identifier import, action-plan tracking, session revocation,
  duplicate detection and mentor challenges; `tests/test_frontend.py` guards the two ways the front
  end breaks silently (JS route→copy mapping and template form actions that no
  longer exist), and shells out to `tests/app_js.test.js` under Node.

Full detail, including the data model and the exact state transitions, is in
[`docs/TECHNICAL.md`](docs/TECHNICAL.md).

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
3. You do the work and log what happened, per task, with a sample size.
4. The model returns a **verdict** for that block:

| Verdict | Effect |
|---|---|
| `pass` | block → `passed`, evidence forced to `confirmed` |
| `iterate` | evidence was thin; block → `mixed`, hypothesis revised, run again |
| `fail` on a **critical** block | block → `failed`, and **you** choose kill or pivot |
| `fail` on an **important** block | the block is **parked** with a recorded workaround; the venture carries on |
| `pivot` | spawns a sibling venture on a fresh nine-block canvas |

### Why severity matters

Losing your Key Partners block should not suggest killing an otherwise validated
business, and a Customer Segments failure should not be parkable. `severity` is
what makes those two cases behave differently.

### Why the model does not get to kill your venture

A `fail` on a critical block marks the block `failed` and stops. `apply_verdict`
deliberately does not close the venture: it renders a kill-or-pivot card with the
model's reasoning attached, and waits. The user resolves it with `kill_venture`
or `pivot_venture`. A model should not get to close someone's venture on its own.

### Why parked counts as resolved

Phase 3 unlocks when all nine blocks are `passed` **or** `parked`. Finishing with
documented gaps is a legitimate outcome, and those gaps are carried explicitly
into the launch-strategy prompt and listed on the Phase 3 page rather than
quietly dropped. A `parked` block can be reopened at any time, and a block at its
cap can be parked by hand with a note.

### Run caps are per block

Each block gets 3 runs (`SEGMENT_CAP`). Exhausting one parks *that block* and
offers two exits — "+3 more runs", or park it with a note — and never pauses the
whole venture, which is what the old global cap did. The venture-level
`max_cycles` is only a runaway backstop (9 × 3 = 27).

### Cost

A full nine-block validation is roughly **54 AI calls** (9 blocks × 3 runs ×
2 calls: design the run, then score it). The default per-user monthly cap is
500, so about nine complete validations.

### Storage

A "run" is one row in `cycles`, scoped by the `segment` column. Rows written
before this change spanned several blocks at once, so the migration attributes
them to a block only where every task named the same one, and translates the old
`persevere`/`kill` decisions to `iterate`/`fail`. Blocks the old loop had already
marked `confirmed` come back as `passed`, so existing work is not thrown away.

## Known rough edges (MVP, not production)

- **Vercel has a 4 MB upload ceiling** against Render's 8 MB, because Vercel
  rejects any request body over 4.5 MB before the app runs. Same code, smaller
  limit, set per platform. A researcher with a 6 MB PDF is served by Render and
  refused by Vercel.
- **Vercel migrations are a manual pre-deploy step** (`alembic upgrade head`).
  That is the trade for not re-running twelve migrations on the first request
  after every idle period, but it means a deploy is not self-contained — forget
  the step and the app runs against the previous schema.
- **Vercel cold starts are paid on the first request after an idle period**,
  because the native runtime imports `psycopg[binary]`, `openai`, `pypdf` and
  `python-docx` at module load. The container path would avoid this at the cost
  of Active CPU billing instead of the free tier.
- **Password reset works**, but email is sent through the `console` backend by
  default, which prints the link to stdout. A real deployment must set
  `MAIL_BACKEND=smtp` and `APP_BASE_URL`, or researchers get no mail at all —
  and on `console` a reset link is only ever printed, never delivered.
- **Email confirmation is disabled by default** (`EMAIL_VERIFICATION_ENABLED=0`).
  The flow is implemented and tested — signup emails a link, `/verify-email`
  redeems it, single-use and idempotent — but with no provider configured the
  banner would nag every account for a message that cannot arrive. Set the flag
  to `1` once `MAIL_BACKEND=smtp` and a real `MAIL_FROM` domain are in place.
  Password reset is not affected by the flag and still runs either way.
- **Changing your password does sign out every other session**, and you can do it
  while signed in (Account & data) as well as through the reset link. It works by
  a `session_epoch` counter compared on every request, not a session table — so
  there is no way to *list* or name your sessions, and no "sign out of one device
  only". Deploying this change signs everyone out once, which is the point.
- `llm_calls` records which prompt template produced a call
  (`prompt_version`) and its SHA-256, so you can attribute and compare results
  across a prompt edit — but not the prompts themselves, by design. The bodies
  embed the pasted research material verbatim and the table deliberately outlives
  account deletion, so storing them would copy unpublished IP into the one table
  that survives the account.
- Login throttling keys on email **and** IP (the IP ceiling is 5× the per-account
  one, because universities share NAT addresses) and clears on success. It stops
  password guessing; it is not a full brute-force defence. The LLM routes have
  their own token bucket — see `app/ratelimit.py`.
- The LLM rate limiter is per **process**, so a multi-instance deploy multiplies
  the effective ceiling by the instance count. Render's free plan is a single
  instance, which matches; horizontal scaling needs a shared store first.
- Action plan items in Phase 3 are read-only (no "mark done" checkboxes yet).
- The per-block cap is a blunt instrument: a block that needs a fourth good-faith
  pass has to be extended explicitly, which is the intended friction but is
  still friction.
- Per-run success criteria are not yet hashed and locked, so "the criteria were
  written down before the results" is not yet provable from the database.
- No RLS. Isolation is `user_id` in the WHERE clause, so a bug in a query is a
  data leak. The `anon`/`authenticated` grants are revoked, but `service_role`
  still bypasses everything — keep the connection string in a platform secret.
- The connection pool is fixed at 5+5. Supabase's pooler has its own limits, and
  a `shared-cpu-1x` machine running many machines will queue rather than scale.
- The test suite drops schemas on the configured database. It refuses to run
  against a URL that still contains `.env.example` placeholders, but that is a
  string check, not a safety interlock.