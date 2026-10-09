# LaunchLoop — Technical Reference

Audience: engineers and coding agents who will change this codebase. It is the
contract-level view: how the pieces fit, what the state machines actually
guarantee, and which invariants a change must not break.

- Product/process view, including the full user journey: [`FLOW.md`](FLOW.md)
- Setup, deployment, and the current feature list: [`../README.md`](../README.md)

---

## 1. The one-paragraph model

LaunchLoop is a three-phase research tool. Phase 1 turns pasted material into
idea cards. Phase 2 is the substance: a **per-block validation loop** over the
nine Business Model Canvas blocks, where the LLM designs a run against one
block, the human does that run in the real world and logs what happened, and the
LLM returns a verdict that advances one block's lifecycle. Phase 3 turns a
resolved canvas into a go-to-market package. Every phase is a server-rendered
Jinja page; every state change is a POST that a server-side state machine
authorises before it happens.

The design bias throughout: **the model proposes, the user disposes.** The LLM
writes hypotheses, tasks and verdicts. It never closes a venture, never spends
quota without a recorded call, and never persists an enum value it invented.

---

## 2. Stack and layout

| Concern | Choice |
|---|---|
| Web framework | FastAPI 0.115 / Starlette 0.38 (sync route handlers, Jinja2 templates) |
| Templates | Jinja2, server-rendered; no client framework |
| Database | PostgreSQL via SQLAlchemy 2.1 **Core** + psycopg3. No ORM models |
| Migrations | Alembic, run on boot under an advisory lock (off on Vercel — see §8) |
| Deploy targets | Render (Docker) and Vercel (native Python), one codebase — see §8 |
| Auth | `bcrypt` directly + Starlette `SessionMiddleware` (itsdangerous signed cookie) |
| LLM | `openai` 1.51 SDK against any OpenAI-SDK-shaped endpoint |
| Outbound HTTP | `httpx` 0.28, one GET per import, redirects disabled |
| Front end | One CSS file, one JS file. Progressive enhancement only |
| Tests | pytest + FastAPI `TestClient`, real Postgres, per-test schema |

```
app/
  main.py        all HTTP routes + the two state machines + verdict application
  db.py          every query; engine/pool; init_db()
  schema.py      SQLAlchemy Core Table objects (engine-free)
  constants.py   domain enums, the nine segments, allowlists (engine-free)
  llm.py         provider wrapper: prompts, retries, output validation
  config.py      every env-tunable value, boot vs runtime
  auth.py        password hashing, current-user lookup, token minting
  mailer.py      transactional email (console | smtp), never raises
  quota.py       per-user monthly call cap
  ratelimit.py   per-request token bucket (user + IP), on the LLM routes and on import
  dedupe.py      near-duplicate idea cards: character trigrams, no model call (M1.4)
  mentors.py     the founder-playbook catalogue and the "not a quote" notice (M2.1)
  upload.py      PDF/DOCX/TXT → text, in memory, then discarded (M1.1)
  sources.py     ORCID / Crossref / arXiv → text. Hardcoded hosts, encoded input (M1.2)
  templates/     base + page templates + macros.html (shared partials)
  static/        style.css, app.js
migrations/
  env.py         reads the URL from the environment, honours a test schema
  versions/      f1ceb3f3319f → a1c9f0e2b7d1 → b3d7e91c4a02 → c4e7a1b9d205
                 → d5f8b2c7a913 → e7a2b4c9d016 → c9e4b2a71d38
                 → d4a7f1c93e20 → f2b8d3a6c5e1
scripts/
  migrate_sqlite_to_supabase.py   one-shot, re-runnable ETL from the old SQLite file
tests/
  conftest.py                    per-test Postgres schema, built with real migrations
  test_launchloop.py             the state machine, output validation, isolation, quota…
  test_frontend.py               template/JS contracts, shells out to app_js.test.js
```

### The engine-free rule

`app/schema.py` and `app/constants.py` **must not** import `app.db`. Alembic's
`env.py` imports the table metadata just to read it; if the import chain reached
`app.db` it would try to build a live engine — and therefore need credentials —
just to print the schema. Keep domain constants in `constants.py`, table
definitions in `schema.py`, and everything that touches a connection in `db.py`.

### Boot order

`app.main` import → `config` loads `.env` → `SECRET_KEY` validated (hard failure
if missing in production) → `SessionMiddleware` installed → `StaticFiles`
mounted. On startup the `lifespan` hook calls `db.init_db()`, which runs
`alembic upgrade head` under `pg_advisory_lock(hashtext('launchloop_alembic'))`.
The URL reaches Alembic via `cfg.attributes["sqlalchemy_url"]`, never
`set_main_option`, because configparser's `%`-interpolation raises on a
percent-encoded `sslrootcert` path before any connection is attempted.

---

## 3. Data model

Eleven tables. All have `users.id → ON DELETE CASCADE` except `llm_calls`
(`SET NULL`, so the audit trail outlives the account) and `login_attempts`
(standalone).

Every table except `login_attempts` also carries **row-level security** — see
§3.1, because the policies only bite for a connection that cannot bypass them,
which is the part that is easy to get wrong.

```
users ─┬─ ideas ──── ventures ─┬─ bmc_elements   (9 rows per venture)
       │                       ├─ cycles          (one row per run)
       │                       └─ launch_strategy (0..1 active row)
       ├─ llm_calls
       └─ password_reset_tokens (CASCADE — a token must not outlive its user)
       login_attempts (standalone)
```

### `users`
`id`, `email` (unique), `password_hash` (bcrypt), `session_epoch`,
`email_verified_at` (nullable), `created_at`.

`session_epoch` is the whole of session revocation (M0.7): login copies it into
the cookie, and `get_current_user()` compares it against the row it has *already
loaded* to install the RLS tenant — so the check costs **no extra query**. It is
not a secret and is not treated as one; forging a cookie needs
`SESSION_SECRET_KEY` regardless. The epoch only decides whether an already-valid
cookie is still current.

`db.set_password()` advances it (`session_epoch + 1`, computed in SQL so two
concurrent resets cannot both write the same value), which makes it impossible to
change a password without revoking sessions — the one function that changes a
password is the one that advances the counter. A cookie with **no** epoch counts
as revoked rather than as epoch 1, so deploying this invalidates every session
issued before it, which is the correct outcome for the change.

`auth.start_session()` is the only writer of session keys, so login, signup and
password-change cannot drift on which keys a session carries.

`email_verified_at` is null until the address is proven. It is deliberately
**not** backfilled for pre-existing users: `created_at` would assert a
verification that never happened. Those accounts get the unverified banner,
which is the truth.

### `password_reset_tokens` — resets *and* verification links
`id`, `user_id` (`CASCADE`), `purpose` (`password_reset` | `email_verify`),
`token_hash` (SHA-256, unique), `expires_at`, `used_at` (nullable), `created_at`.

One table for two flows, because email verification needs exactly the same
primitive as a password reset — unguessable token, stored hashed, expiring,
single-use — and a second near-identical table would be a second set of
token-rotation bugs to maintain. `purpose` is what stops one flow from ever
reading the other's tokens, and it is a CHECK-constrained enum
(`constants.TOKEN_PURPOSES`, invariant 2).

Only the hash is stored, so a database dump cannot be used to take over an
account. `used_at` is the single-use guarantee — the row survives for audit, but
`db.redeemable_token()` stops accepting it.

`db.consume_user_tokens()` burns every unused token of one purpose for a user.
It is called on password reset so that requesting two links and using the second
also kills the first; otherwise resetting your password does not lock anyone out.

### `ideas` — Phase 1 output
`id`, `user_id`, `title`, `commercial_framing`, `strength_signal`
(`strong|moderate|early`), `raw_claims`, `status`
(`candidate|selected|rejected`), `created_at`.

A killed venture's idea goes back to `candidate` so it reappears on the
dashboard as something you could try a different angle on.

### `ventures` — one attempt at one idea
`id`, `user_id`, `idea_id`, `phase` (`2` loop, `3` launch), `cycle_count`,
`max_cycles` (**backstop only** — the real cap is per segment), `status`
(`active|validated|killed|paused|pivoted`), `parent_venture_id`
(`SET NULL` — set on a pivot child), `pivot_note`, `created_at`.

`parent_venture_id` makes the pivot lineage a real tree, so
`get_child_ventures()` can link a pivoted venture to its successor and there is
exactly one live branch per idea.

### `bmc_elements` — the nine canvas blocks
The table predates the per-segment loop and keeps its name; `get_bmc_elements`
is a documented alias of `get_segments`. Rows are treated as **segments**
everywhere in the new code.

| Column | Meaning |
|---|---|
| `venture_id`, `element_name` | unique together — one row per block per venture |
| `status` | **evidence**: `untested|mixed|confirmed|disconfirmed` |
| `outcome` | **decision**: `pending|active|passed|failed|parked` |
| `position` | 0-based index in the recommended testing order |
| `label`, `severity` | display name; `critical|important` |
| `hypothesis` | the assumption currently under test — **and, once a block passes, its confirmed answer** (see §4.3) |
| `outcome_note` | the evidence that settled it; or the workaround if parked |
| `cycle_count`, `max_cycles` | runs used and allowed **for this block** |
| `notes`, `decided_at`, `updated_at` | evidence text and timestamps |

`status` and `outcome` are deliberately **separate columns**. Evidence is what
the world said; the decision is what you did about it. Keeping them apart makes
both queryable — "a passed block whose evidence is thin" is a real question.

Two CHECK constraints encode the pairing, so an inconsistent pair is a database
error rather than a silently wrong board:

```sql
NOT (outcome = 'passed' AND status <> 'confirmed')
NOT (outcome = 'failed' AND status <> 'disconfirmed')
```

### `cycles` — a run
One row per pass at one block. `cycle_number` is the venture-wide running total
(the backstop counter and the display label); `segment` says **which block** this
run tests. `todos_json`, `results_json`, `analysis_json` are JSONB, and
`decision` (`pass|iterate|fail|pivot`) is the denormalised verdict.

`segment` is nullable only for legacy rows written before the per-segment loop;
the migration backfills it where every task in the cycle named the same block.
A row with `segment IS NULL` is deliberately **unscoreable** — the scoring path
refuses it, and `venture_phase_state()` treats it as not-in-flight so it can
never wedge a venture.

Two indexes serve the two access patterns: `(venture_id, cycle_number)` for
"the current run", and `(venture_id, segment, cycle_number)` for "every run on
this block".

### `launch_strategy`, `action_steps`, `llm_calls`, `login_attempts`
`launch_strategy` holds three JSONB arrays (`funding_matches`,
`gtm_channels`, `action_plan`) and is append-only — regenerating inserts a new
row and `get_launch_strategy()` returns the newest, so nothing is overwritten.

`action_steps` is `venture_id`, `step_key`, `step`, `milestone_type`,
`status`, `outcome_note`, `decided_at`, `created_at`, `updated_at`, with
`UNIQUE (venture_id, step_key)`. Two properties are worth knowing before changing
it:

- **`step_key` is the SHA-256 of the step text, not an array position.**
  Regeneration produces a different list, so index 3 of the new plan is not
  index 3 of the old one; keying on position would re-label a completion onto a
  step the user never did. Same wording → same key → status kept. Reworded step
  → new key → correctly pending. The cost: two steps with identical wording
  collapse to one row, so ticking one ticks both — which is right, since they
  are the same instruction.
- **It is deliberately NOT a key inside `action_plan_json`.** The newest strategy
  wins, so progress stored there would be destroyed by "Regenerate strategy", a
  button on the same page as the progress. `save_launch_strategy()` seeds the rows
  in the same transaction as the insert, so a plan can never be visible with no
  steps to record against it.

`db.step_key()` is duplicated inside migration `c9e4b2a71d38` (a migration must
not import a module that builds a live engine); `TestActionStepTracking` pins the
two to the same value so they cannot drift.

`llm_calls` is both the observability log **and** the quota source of truth:
`user_id`, `purpose`, `provider`, `model`, `status`, `attempts`,
`input_tokens`, `output_tokens`, `latency_ms`, `error`, `month` (`YYYY-MM`),
`prompt_version`, `prompt_sha256`, and an index on `(user_id, month)`. Failed
calls are logged too, so they count against the cap — a provider that 500s is
still costing attempts.

`prompt_version` and `prompt_sha256` are **prompt provenance, not prompt
storage** — see §5.6. The two columns exist because editing a prompt would
otherwise silently re-label every historical result, and the obvious fix (keep
the body) is the one thing that must not happen here.

`login_attempts` is `(email, ip, succeeded, created_at)`, indexed on both
`(email, created_at)` and `(ip, created_at)`. Both are read by a sliding-window
count, and they catch different attacks — see §6. `clear_failed_logins()` flips
an account's rows to `succeeded` on a successful login, so a handful of typos
does not lock the account out for the rest of the window.

---

### 3.1 Row-level security

The `user_id` filters are load-bearing but load-bearing *by convention* — one
forgotten `WHERE` clause is a cross-tenant read and nothing notices. The policies
move that guarantee into the database: a query with no `user_id` predicate at all
still cannot cross a tenant boundary.

**RLS does nothing for a role with `BYPASSRLS`.** Superusers bypass it regardless
of `FORCE ROW LEVEL SECURITY`, and so does Supabase's `service_role` — the role
`DATABASE_URL` points at. So the migration creates `launchloop_app`
(`NOSUPERUSER NOBYPASSRLS`), grants it table and sequence privileges, and
`get_conn()` reaches it with `SET LOCAL ROLE`. Two deliberate safety properties,
because this migration runs on boot against production:

- **Role creation is best-effort, never fatal.** A deploy that cannot
  `CREATE ROLE` still migrates cleanly, with policies and no role to apply them.
  Failing the boot would be strictly worse than a documented gap.
- **The switch is membership-tested**, via `pg_has_role`, on every connection. A
  deploy that never granted membership keeps working on the `user_id` filters
  rather than 500-ing.

`db.rls_active()` reports whether the policies are genuinely enforcing, and
`/healthz` publishes it. It reports on **the role transactions will run as**, not
the role the pool connects as — asking the latter answers "bypasses" even on a
fully-enforced deploy, because `current_user` on a raw pooled connection is the
owner. That distinction is the whole reason the health check is trustworthy.

| Table | Policy |
|---|---|
| `ideas`, `ventures`, `llm_calls`, `mentor_challenges` | `user_id = <tenant>` for all commands |
| `bmc_elements`, `cycles`, `launch_strategy`, `action_steps` | `EXISTS (SELECT 1 FROM ventures …)` — they carry `venture_id`, and referring to `ventures` costs no recursion because its policy does not refer back |
| `password_reset_tokens` | the tenant, **or** the one row matching the presented token's hash — see below |
| `users` | SELECT your own row **or** the row matching `app.lookup_email`; INSERT only the address being registered; UPDATE/DELETE only your own |

Two tables need a non-tenant predicate, and in both cases the pre-auth credential
is the only thing the user holds at that moment:

- `users` for **login and signup** — you know the address before you are signed
  in. `app.lookup_email` admits exactly that one row, which is no wider than the
  query the app already issues.
- `password_reset_tokens` for **reset and verification** — the token *is* the
  credential, exactly as the address is for login. `app.lookup_token` admits that
  one row; the caller then installs the resolved row's tenant before writing.

**Unset tenant means DENY.** Every predicate is
`NULLIF(current_setting('app.user_id', true), '')::bigint`. The `NULLIF` is
load-bearing: once a custom setting has been assigned anywhere in a session,
`current_setting(name, true)` returns the **empty string** after it is reset, not
NULL, and `''::bigint` raises `invalid input syntax for type bigint`. That would
turn every reuse of a pooled connection into a 500.

`login_attempts` is deliberately **not** enabled. Both throttling queries are
pre-authentication, and filtering them by a tenant that does not exist yet would
make every count read zero — the lockout would silently never fire, i.e. a
security control failing **open**, which no test of its normal path would catch.
The table holds no research content and no secrets.

#### Carrying the tenant

A `ContextVar` holds the context for the current transaction; `get_conn()`
issues the `SET LOCAL` statements, so the connection returns to the pool clean —
which matters, because it is the pool the next tenant's request will use.

A ContextVar rather than a `user_id` parameter through all of `db.py`: a
parameter can be forgotten at a call site, and a forgotten parameter under RLS
returns *zero rows*, which is a bug that hides. `auth.get_current_user()` installs
it once, and every route and every `render()` already passes through there — so
there is no call site that can forget.

`get_current_user()` installs the tenant from the session id **before** reading
the row, because the `users` policy admits only your own row. The cookie is
signed, so the id is a trustworthy claim; even a forged one would only surface the
row it claims to be.

**Verified, not assumed, that this cannot leak between requests.** Starlette runs
sync handlers in a *reused* AnyIO worker thread, so a plain module global would
hand one tenant's id to the next request. Probed directly before choosing the
mechanism: anyio runs each task inside a fresh `copy_context()`, so a value set
during one request is invisible in the next. `TestIsolation` is the end-to-end
proof — two users in one process, and the second cannot see the first's venture.

## 4. The two state machines

These are the part to read before touching a route. Both live in `app/main.py`
and both are **called by the POST handlers** — the templates only reflect them.

### 4.1 Venture state — `venture_phase_state(venture, current_run)`

Inputs: the venture row, and the highest-`cycle_number` row in `cycles`.

| Signal | Definition |
|---|---|
| `awaiting_results` | a run exists with no `results_json` |
| `awaiting_analysis` | a run has `results_json` but no `analysis_json` |
| `scoreable` | there is no run, **or** the run has a `segment` |
| `in_flight` | `scoreable and (awaiting_results or awaiting_analysis)` |
| `terminal` | status is `killed` or `pivoted` |
| `at_backstop` | `cycle_count >= max_cycles` |

`kind` collapses those into one render branch: `phase3` → `killed` → `pivoted` →
`stopped` (terminal or at backstop) → `looping`.

`can_start_new_cycle = not terminal and not in_flight and not at_backstop`, and
`start_blocked_reason` is the sentence shown when it is false.

The `scoreable` guard is load-bearing. Without it, a legacy unscoreable run
would read as permanently in-flight: the user would be told to finish a run that
the scoring path refuses to finish, with no way to clear it. That is exactly how
a venture migrated from the old flat loop got stuck.

### 4.2 Block state — `segment_run_state(segment, venture)`

| Signal | Definition |
|---|---|
| `resolved` | `outcome` in `SEGMENT_RESOLVED` = `{passed, parked}` |
| `at_cap` | `cycle_count >= max_cycles` |
| `parked_by_cap` | `at_cap and not resolved` |
| `can_run` | `outcome != 'passed'` and the venture is live and under its backstop |

Note what is **not** here: a `parked` block is still `can_run`. `can_run` is
gated on `outcome != 'passed'`, so re-opening a parked block (which sets
`outcome` back to `pending`) is all that is needed to make it runnable again.
And a block at its cap that has *not* been parked is `parked_by_cap`, which the
start-run route rejects in favour of the cap card — extend, or park with a note.

### 4.3 Verdict application — `apply_verdict(venture, run, analysis)`

Persists the verdict, then moves exactly one segment:

| Verdict | Segment write | Venture write |
|---|---|---|
| `pass` | `outcome='passed'`, `status='confirmed'`, note = evidence note, hypothesis ← revised | — |
| `iterate` | `status='mixed'`, notes ← evidence note, hypothesis ← revised | — |
| `fail`, critical | `outcome='failed'`, `status='disconfirmed'`, note = verdict reasoning | **none** — the user decides |
| `fail`, important | `outcome='parked'`, `status='mixed'`, note = workaround or reasoning | — |
| `pivot` | — | spawns a child, sets this one `pivoted` |

Then, for every verdict except `pivot`: if `db.all_resolved(venture_id)`,
`phase=3, status='validated'`.

Three properties worth preserving if you edit this:

1. **A critical `fail` does not end the venture.** It marks the block and stops.
   `apply_verdict` is called from a POST handler that returns a response; the
   decision belongs to the human, via `kill_venture` / `pivot_venture`. The
   failed block stays unresolved, which also means the Phase 3 gate cannot fire
   while the decision is outstanding.
2. **Evidence follows the verdict.** `set_segment_outcome` defaults `status` to
   whatever the outcome implies (`passed→confirmed`, `failed→disconfirmed`,
   `parked→mixed`), because the CHECK constraints would otherwise raise a 500 on
   the page that submitted the verdict. `llm.analyze_segment_run` forces the same
   pairing one layer earlier, so the two can never disagree.
3. **`all_resolved` is defensive.** It returns false unless there are exactly
   nine rows *and* the keys are exactly the nine expected ones. A missing row or
   an off-list key would make the gate silently unreachable — or, worse, wrongly
   true.
4. **On `pass`, `revised_hypothesis` is the block's answer, not a hypothesis.**
   The same JSON key carries two meanings by verdict, which is deliberate: for
   `iterate` it is the sharper claim for the next run, and for `pass` it is what
   the block is now *known to be* — the value proposition, the customer segment,
   whatever the block is about. `analyze_segment_run` v2 asks for that
   explicitly ("name the proposition", "name the segment") and rejects finding
   summaries, because this string is what the canvas cell displays as the block's
   answer. Before v2 the cell showed a description of the evidence instead.

### 4.3a Canvas layout and block text

Two things live in `app/templates/macros.html` + `app/static/style.css`, and
both have tests that pair the constants, the template, and the stylesheet.

**Placement is by named CSS grid area, never by document order.** Auto-flow
across five columns produces a plausible-looking but wrong canvas: it puts
Customer Segments under Key Partners and Channels in the middle. The area map is
declared once in `style.css`:

```
"kp ka vp cr cs"      Key Partners | Key Activities | Value Props | Cust Rel | Cust Seg
"kp kr vp ch cs"      (full height) | Key Resources  | (full height)| Channels | (full height)
"fin fin fin fin fin" Cost Structure | Revenue Streams
```

The template emits `a-<key>` on each cell and the stylesheet maps that to
`grid-area`. `CANVAS_LAYOUT` / `CANVAS_FINANCIAL` in `app/constants.py` carry
the same `area` values, and a test asserts the three files agree — a block whose
rule is missing or points at the wrong area otherwise renders in the right shape
with the wrong contents. Document order is still the canonical *reading* order,
because it is what the narrow-screen single-column stack falls back to.

**A cell shows the block's answer, then its evidence.** For a `passed` block the
headline is `hypothesis` (the confirmed answer) and the smaller line beneath is
`outcome_note` (the evidence that settled it). For `parked`/`failed` the note
takes the headline — a parked block has a workaround, not an answer. Before
resolution the hypothesis is what is being tested. Both are clamped in CSS rather
than truncated in the template, so the full text stays in the DOM for screen
readers and copy-paste.

### 4.4 Pivot lineage

`spawn_pivot_venture(venture, note)`:

1. Create a new `ideas` row — `"<title> (pivot)"`, the note as the commercial
   framing — so the dashboard shows it as a candidate, not as a duplicate.
2. `create_venture(..., parent_venture_id=venture_id, pivot_note=note)`, which
   seeds a **fresh nine-block canvas**, all `pending`.
3. `carry_passed_segments(old, new)` copies `status`, `notes`, `hypothesis`,
   `outcome`, `outcome_note`, `cycle_count` and `decided_at` for every block that
   was `passed`. Blocks that were pending, iterating or failed stay fresh.
4. Mark the old venture `pivoted`.

Step 3 is the whole point of a pivot: a pivot changes the hypothesis about *one*
block, not the eight you already proved around it. Re-testing those would throw
away real work and spend real AI calls. The old venture keeps rendering, with a
link to its child.

The same helper backs both entry points — a model `pivot` verdict, and the
user's own choice on the kill-or-pivot card.

### 4.5 Focus resolution — `_venture_context()`

The Phase 2 page opens **one** panel, and which block that is must be
deterministic. The order, most urgent first:

1. the block named in the URL (`/venture/{id}/segment/{key}`)
2. the block with a run in flight
3. a critical block awaiting a kill-or-pivot decision — it blocks Phase 3, and
   burying it on the board would make the user notice it themselves
4. otherwise the block the loop would attack next (`next_recommended_segment`)

The whole context is assembled in one function so the view, the error re-render
(`_phase2_retry`) and the retry path cannot drift apart. **If you add something
the page needs, add it here.**

---

## 5. The LLM layer

`app/llm.py` is the only module that talks to a provider. Four public
functions, one per call site:

| Function | Purpose constant | Returns |
|---|---|---|
| `extract_ideas(raw_text, user_id)` | `extract_ideas` | `list[card]` |
| `generate_segment_tasks(...)` | `generate_segment_tasks` | `{hypothesis, tasks}` |
| `analyze_segment_run(...)` | `analyze_segment_run` | `{verdict, evidence_status, …}` |
| `generate_launch_strategy(...)` | `generate_launch_strategy` | `{funding_matches, gtm_channels, action_plan}` |

### 5.1 `call_json` — the transport

```
for attempt in 1..LLM_MAX_ATTEMPTS:
    send chat.completions.create(model, messages, max_tokens, temperature=0,
                                  response_format?, reasoning_effort?)
    text = _message_text(choice.message, choice.finish_reason)
    if text is empty and finish_reason == "length":  -> TruncatedResponse, break
    if text is empty:                              -> LLMError (retryable)
    result = _extract_json(text); unwrap if wrap_key
    log ok; return
```

Retry policy, which is the part with real judgement in it:

| Failure | Action |
|---|---|
| `JSONDecodeError` | retry with a corrective suffix appended to the prompt |
| `APITimeoutError`, `APIConnectionError`, `RateLimitError` | sleep `2**attempt`, retry |
| any other exception | same backoff, retry |
| `AuthenticationError`, `PermissionDeniedError` | log and raise **immediately** |
| `BadRequestError` naming `reasoning_effort` | drop that parameter, retry |
| `BadRequestError` naming `response_format` | disable JSON mode, retry |
| any other `BadRequestError` | give up — a malformed request will not fix itself |
| `TruncatedResponse` | give up — the ceiling is deterministic |

`temperature=0` throughout: these are structured-extraction calls and variance
is a cost, not a feature.

### 5.2 JSON extraction

`_extract_json` strips ``` fences, tries a direct parse, then **scans with a
brace matcher that tracks string state and escapes**. A greedy regex would
swallow a `{` from prose in front of the real payload. If nothing parses it
raises, and the retry adds an explicit "respond with ONLY valid JSON" suffix.

`_message_text` is deliberately conservative. Reasoning models behind
OpenAI-compatible gateways can return empty `content` with the answer in
`reasoning` / `reasoning_content` / `reasoning_details`. It will read those —
**but only when `finish_reason` is not `length`.** A truncated response has empty
`content` too, and its reasoning field is the model's internal monologue, which
repeats the prompt's own schema example; scanning that for JSON yields fragments
that *parse* and then become garbage data. Refusing is the only safe option.

`wrap_key` exists because `response_format={"type":"json_object"}` forces a
top-level object: a prompt asking for a bare array gets `{"ideas": [...]}` back.
The wrapper instruction is appended to the prompt, and `_unwrap` tolerates a
model that ignored it (single-key dict, or the payload returned directly).

### 5.3 Output validation — the load-bearing rule

**Every response is validated against an allowlist before it leaves this module.**

- Task `method` must be in `SEGMENT_METHODS[block]`; otherwise it is replaced
  with the first legal method. This is also what keeps the per-block advice
  specific instead of collapsing into "do an interview" for everything.
- `verdict` must be in `RUN_VERDICTS`; if not, `analyze_segment_run` **raises**
  rather than defaulting — the page then offers *Retry verdict*.
- `evidence_status` must be in `SEGMENT_STATUSES`, and is then **forced** to
  match the verdict where the schema requires a pairing.
- `strength_signal` falls back to `early`; `milestone_type` falls back to
  `pilot`.
- `_norm_enum` normalises case and `-`/whitespace to `_` before comparing, so
  `"Problem Interview"` survives.
- Every string is truncated to a column-appropriate length.

The reason this module is strict: a single hallucinated enum value once made
Phase 2's completion check permanently unreachable, because the value the model
invented was neither `passed` nor `parked` and no code path could move it.

### 5.4 Prompt-injection posture

User-supplied text is wrapped in explicit delimiters and labelled:

```
--- BEGIN RESEARCHER MATERIAL (untrusted data) ---
{raw_text[:20000]}
--- END RESEARCHER MATERIAL ---
```

with an instruction above it to treat the contents as data, never as
instructions, and to ignore anything that looks like a command. This is a
mitigation, not a guarantee — the model is still the one parsing the text. It is
also why the README steers deployments away from models documented as training
on prompts.

### 5.5 Errors and cost

`LLMError` is the one exception the routes handle. `QuotaExceeded` and
`TruncatedResponse` subclass it. `_friendly_error()` turns the final exception
into a sentence a researcher can act on — timeout, connection loss, rate limit,
denied model, rejected request, truncation, generic — and every message says the
work is saved, because by then it is.

`_record()` writes the `llm_calls` row and swallows its own exceptions:
telemetry must never take down a user-facing request.

### 5.6 Prompt provenance, and why the bodies are not kept

Every row records two things about the prompt and nothing about its content:

| Column | Value |
|---|---|
| `prompt_version` | `constants.PROMPT_VERSIONS[purpose]` — e.g. `analyze_segment_run.v2` |
| `prompt_sha256` | SHA-256 of the prompt string **as sent**: after the `wrap_key` envelope, before any retry corrective suffix |

The problem this solves: prompts are the highest-churn code in the app, because
every rule in §5.3 exists because a prompt was once wrong. Without a version, an
edit makes it impossible to tell which historical verdicts came from the old
template and which from the new one — and a verdict is the app's most
consequential output.

Why the hash is of the *sent* prompt, and why that is computed before retries:
the `wrap_key` envelope is genuinely part of the request, so it is part of the
fingerprint. The retry suffix is a transport artefact — hashing it would make two
calls of the same template differ for no reason a reader could act on.

Why the body is not stored, which is the load-bearing half. Prompts embed the
researcher's pasted material verbatim inside the untrusted-data wrapper, and
`llm_calls.user_id` is `ON DELETE SET NULL` — these rows deliberately outlive
account deletion. A `prompt_text` column would therefore be a copy of
unpublished IP in the one table that survives the account, which is exactly the
posture §5.4 and §11 are trying to defend. The hash answers "is this the same
prompt I think it is", not "what did it say"; retrieving a prompt by hash would
require storing it.

`PROMPT_VERSIONS` is a **domain** constant, not config, for the same reason the
segment keys are (§8): a deployment must not be able to relabel history. Bumping
a `.vN` is a code change, and therefore reviewable. Bump it on any edit that
changes what the prompt *asks for* — a JSON key, an enum, the output schema — not
on cosmetic wording.

`TestPromptProvenance` holds both halves: one test asserts the string appears in
no column of a logged row, and another asserts the exact column set of
`llm_calls`, so adding a `prompt_text` column later fails the suite rather than
shipping.

---

## 6. HTTP surface

All POST handlers redirect with **303** (POST/redirect/GET, so a refresh never
re-submits) except where an error must render in place.

| Method | Path | Notes |
|---|---|---|
| GET | `/` | → `/dashboard` if signed in, else `/login` |
| GET | `/healthz` | DB round-trip only, deliberately never the LLM |
| GET/POST | `/signup`, `/login`, `/logout` | `/logout` also answers GET for old bookmarks; nothing links to it |
| GET/POST | `/forgot-password` | POST → 303 to `?sent=1`. **Identical response for every outcome**, including a mail failure — anything that varied would be an account-enumeration oracle |
| GET/POST | `/reset-password` | `?token=` on GET, hidden field on POST. Token validated before the password, so an invalid link and a weak password are indistinguishable |
| GET | `/verify-email?token=` | Confirm an address. A GET because it is a link the user clicks; a third party cannot trigger it without the token |
| POST | `/resend-verification` | 303. The *deliberate* action, so it is a POST even though the link above is not |
| GET | `/dashboard` | ventures, candidate ideas, quota, per-venture progress |
| GET | `/phase1/new`, `/phase1/ideas` | |
| POST | `/phase1/extract` | ≥40 chars of material; writes cards, then redirects |
| POST | `/phase1/upload` | PDF/DOCX/TXT read into memory, parsed, **discarded**; returns an editable preview. No AI call, no quota |
| POST | `/phase1/import` | ORCID / DOI / arXiv → editable material. One outbound GET to a hardcoded host; own rate-limit bucket |
| POST | `/phase1/dismiss/{idea_id}` | mark a candidate card rejected. Candidate-only: a card already selected into a venture cannot be dismissed |
| POST | `/phase1/restore/{idea_id}` | undo a dismissal. Rejected-only |
| POST | `/phase1/select/{idea_id}` | marks the idea selected, creates the venture |
| GET | `/venture/{id}` | Phase 2 or Phase 3 depending on `kind` |
| GET | `/venture/{id}/segment/{key}` | Phase 2 focused on one block |
| POST | `/venture/{id}/segment/{key}/run` | **design a run** (LLM call) |
| POST | `/venture/{id}/segment/{key}/extend` | raise this block's cap by `SEGMENT_CAP` |
| POST | `/venture/{id}/segment/{key}/park` | park with a mandatory note |
| POST | `/venture/{id}/segment/{key}/unpark` | re-open a parked block |
| POST | `/venture/{id}/run/{run_id}/log` | save outcomes, then score (LLM call) |
| POST | `/venture/{id}/run/{run_id}/retry-analysis` | re-score a saved run (LLM call) |
| POST | `/venture/{id}/kill` | user ends the venture; idea → `candidate` |
| POST | `/venture/{id}/pivot` | user pivots; spawns the child |
| POST | `/venture/{id}/phase3/generate` | launch strategy (LLM call) |
| POST | `/mentor/idea/{idea_id}` | challenge an unselected idea card (LLM call) |
| POST | `/venture/{id}/mentor/segment/{key}` | challenge one block against its logged evidence (LLM call) |
| POST | `/venture/{id}/mentor/plan` | challenge the launch plan against the canvas (LLM call) |
| POST | `/venture/{id}/phase3/step` | record one action-plan step's status and outcome. No model call, so no quota and no rate-limit interaction |
| GET | `/account` | export + delete, gated on a session |
| GET | `/account/export` | versioned JSON, `no-store`. A GET because it changes nothing and must work without JS |
| POST | `/account/password` | change password; revokes every session that predates it, and re-issues the caller's own. Needs the current password |
| POST | `/account/delete` | 303. Irreversible: needs the password **and** a typed `DELETE` |

### Error rendering

`back_to_venture(id, message)` redirects to `/venture/{id}?err=…` — for routes
that can only redirect, and used to swallow failures so a spent quota looked like
a dead button. `_phase2_retry(request, venture, message, segment_key)` re-renders
Phase 2 in place, preserving focus, for failures that should not lose the user's
place. `_venture_context()` reads `err` back out of the query string.

### Two ways in that are not a paste

Phase 1 has three inputs, and only one of them is typing.

| Route | Produces | Cost |
|---|---|---|
| `/phase1/extract` | idea cards | 1 LLM call, 1 quota |
| `/phase1/upload` | text in the preview box | one file read, then nothing |
| `/phase1/import` | text in the preview box | one outbound GET, then nothing |

### Recording an action-plan step

`POST /venture/{id}/phase3/step` takes `index`, `step_key`, `status` and `note`.
The key handling is the part worth reading twice:

- The **server derives** `step_key` from the stored plan, at the posted index.
  A form that simply posted a key would let a crafted request write to any row
  in the table by supplying a hash, naming a step the user was never shown.
- The posted `step_key` must **equal** that derived value. It is a consistency
  assertion, never an address. This is what makes a stale form safe: the user
  confirmed a specific instruction at render time, so if the strategy was
  regenerated since, index 0 names a different instruction and the write is
  refused rather than recording a completion against work nobody did.
- `status` is re-checked against `ACTION_STEP_STATUSES` here as well as by the
  CHECK constraint, so an unvalidated value is a refusal rather than a 500.
  `done` and `blocked` require a non-empty `note` (capped at
  `MAX_STEP_NOTE_CHARS`); `pending` does not, because undoing asserts nothing.

**Both non-paste Phase 1 routes end in the same place**: they render `phase1_new.html`
with `raw_text` filled in, and the user submits it as an ordinary
`/phase1/extract`. That is the whole reason they are cheap — there is no second
AI-call path, so no new quota accounting, no new prompt, and no new entry in the
rate limiter's list of things to cover. It also means imported and uploaded
material is wrapped as untrusted data by exactly the same line of `llm.py`, which
matters because *imported* material is third-party text that may contain
instructions and has passed through our server, which earns it nothing.

`_phase1(request, **overrides)` builds the baseline context for all four ways into
that page (paste, upload, import, and each one's error path). A context variable
forgotten in one of them would surface as a Jinja undefined in one failure mode
only, which is the kind of bug that hides until a user finds it.

#### `app/sources.py` and the rule that makes it safe

An adapter recognises the input by a **whole-string anchored** regex (`.match`,
never `.search`), takes the single captured group, and **percent-encodes** it into
a hardcoded URL. `follow_redirects=False`. Those four together are why no input can
reach anything but a path segment on one of three fixed hosts:

- anchored, because a searchable pattern accepts a URL carrying a payload in its
  query string — the exact SSRF shape;
- one capture group, because everything else in the input is discarded;
- percent-encoded, because `quote(safe="")` can only emit unreserved characters
  and `%XX`, so no `/`, `:`, `@`, `?`, `#` or space can survive to alter the
  request. A character whitelist would be **weaker and wrong**: DOIs legitimately
  contain most of printable ASCII, so it would reject valid input while still
  depending on the pattern staying tight;
- no redirects, because a 302 from an allowlisted host to a link-local address is
  the whole attack, and this removes it without inspecting a `Location` header.

The reachable host set is asserted in the tests rather than left to inspection.
`MAX_EXTRACTED_CHARS` also caps the material, which is what stops an ORCID iD —
a *person*, a career's worth of works — becoming a 400 KB prompt.

Two parsing notes. arXiv's Atom goes through `ElementTree`, which does not resolve
external entities and raises on undefined ones, and the body is size-capped before
parsing. Crossref's JATS abstract is handled **by regex, not by a parser**, because
the parser is the thing that would have to defend against entity expansion — not
using it removes the question. Entities are decoded *before* tags are stripped, so
`&lt;em&gt;` does not survive as live markup.

Cache: in memory, keyed on `(provider, identifier)`, TTL 24h, bounded at 1 000
entries because the key is attacker-influenced. Deliberately **not** the database
— a missed hit costs one GET, and a table would add a schema, an account-deletion
question and a privacy surface for no user-visible gain.

### Account data control

`db.export_user_data(user_id)` returns a versioned, **flat** document —
`launchloop.user_export.v1`, rows with their ids intact, so M4.3's round-trip
(export, then re-import into a clean database) does not need a nesting scheme
invented now.

It reads through the ordinary tenant-scoped functions, so row-level security
applies: it cannot be used to export somebody else's account.

Three exclusions, and anything added later needs a reason:

| Excluded | Why |
|---|---|
| `users.password_hash` | obviously |
| `password_reset_tokens` | **entirely, not column-wise.** `token_hash` is a live credential for the token's lifetime, and it is the user's own row — so nobody would notice the leak downstream |
| `llm_calls.error` | provider-supplied text: untrusted by invariant 9, and it can quote request fragments back. Same reasoning as not storing prompt bodies (§5.6) |

`llm_calls` *metadata* is included deliberately — it is the user's own spend and
call history, and a "what did you do with my data" request that excluded it would
be incomplete. There are no prompt bodies in it.

`db.delete_user_account(user_id)` deletes the `users` row and lets `ON DELETE
CASCADE` do the rest; `llm_calls.user_id` is `SET NULL`, so spend and latency
metadata outlive the account while the research is gone. Deliberately not a
row-by-row delete — the cascade is what keeps the tables consistent, and doing it
by hand is how orphans get created.

**Cascade under RLS was probed, not assumed.** PostgreSQL applies policies to
referential-integrity actions, and three child policies reach through a
`ventures` subquery whose parent row is mid-delete. It works: RI triggers execute
with the table owner's rights. That matters beyond convenience — the deletion is
authorised by the same policies as everything else, rather than by a
`BYPASSRLS` back door.

Two gates, because the action is irreversible and a checkbox is one stray click:
re-enter the password, and type `DELETE`.

### Rate limiting

`app/ratelimit.py` is a token bucket per `(scope, key)` — one for `user:<id>`, one
for `ip:<addr>`. `throttle_llm(request, user_id, back_url)` in `main.py` returns
`None` to proceed or a rendered **429** to refuse, and sets `Retry-After` to the
same number the page quotes, so the two cannot disagree.

Where it sits on each LLM route is load-bearing:

- **Before** `quota.check` on `/phase1/extract`, `…/segment/{key}/run` and
  `/phase3/generate` — nothing has been saved yet, so refusing is free.
- **After** the results are persisted, inside `_score_run`, for
  `…/run/{run_id}/log` and `…/retry-analysis`. This is invariant 5 under load: the
  logged evidence is the one thing the user cannot recreate, so a 429 must never
  be the reason it is lost. The retry path stays open.

Two properties the tests pin:

1. A refused request writes **no** `llm_calls` row, because the check precedes
   the call — so the throttle cannot bill the quota it exists to protect.
2. Both scopes are refilled *before* either is spent, so a request refused by the
   IP scope does not also drain the per-user bucket. Otherwise one exhausted
   address would throttle everyone behind it twice over.

A third route rides the same bucket with **its own limits and its own prefix**:
`/phase1/import`. It has the same shape of risk as a model call — a user-triggered
outbound request that holds a worker thread for the round trip — at a far smaller
cost, so its limits are looser. The `prefix` matters: import capacity and LLM
capacity must not spend each other, and the test suite pins that exhausting the
import budget leaves the LLM bucket untouched. It is checked **before** the fetch,
so a refused import never touches a third party.

The buckets are per **process**. Render's free plan is a single instance so this
matches the deployed topology, but horizontal scaling multiplies the effective
ceiling and would need a shared store first. The monthly quota remains the real
budget; the pace limit is a throttle in front of it.

### Login throttling

Two sliding windows on `login_attempts`, because they catch different attacks:

| Window | Query | Catches |
|---|---|---|
| per account | `recent_failed_logins(email, since)` | password guessing against one account |
| per address | `recent_failed_logins_from_ip(ip, since)` | one address spraying a single password across thousands of accounts, where every individual account has exactly one failure and looks innocent |

The per-IP ceiling is `MAX_FAILED_LOGINS * 5`, deliberately looser: universities
and small companies put many researchers behind one NAT address, and a per-address
ceiling tight enough to stop a spray would lock out a whole department.

`clear_failed_logins(email)` runs on a successful login. Without it, six typos
lock the account out for the rest of the window and the *next* correct login is
refused — which reads as the app being broken rather than as a security feature.

### Password reset and email verification

Both flows are one primitive: `auth.new_token()` returns `(plaintext, sha256)`,
the plaintext goes in the email and is never stored, and
`db.redeemable_token(hash, purpose)` resolves it back to a row — or to `None` for
every failure mode, deliberately indistinguishable, because the user is told "that
link is invalid or has expired" and never which.

`db.redeemable_token()` takes the **hash**, not the token: minting is `auth.py`'s
business and `db.py` must not import it, since `auth.py` imports `db`.

`app/mailer.py` has two backends and `send()` never raises into a request. That
is not just tidiness — a provider outage must not turn signup into a 500, and a
failure cannot be reported to the user, because the forgot-password response is
identical for a registered and an unregistered address, so "we couldn't send it"
would leak that the account exists.

The `console` backend prints the link. That is what dev and the test suite use,
and it is the whole reason this flow is testable end to end with no provider and
no network: the test reads the link the user would have clicked.

Email confirmation is **off by default** (`EMAIL_VERIFICATION_ENABLED`, default
0). The flow is fully implemented and tested, but the only backend configured
anywhere is `console`, so a live flag would show every account a permanent
banner asking it to click a link that is printed to stdout rather than
delivered. `/verify-email` stays mounted either way, so a link already in
someone's inbox keeps working after the flag is switched off.

Unverified accounts are a **nudge, not a gate** — the banner is rendered from
`user.email_verified_at`, which `render()` already puts in context, so it costs
no per-page plumbing. The flag itself reaches templates as a Jinja global
holding the *function* rather than its result, because the 429 and 500 handlers
build their context by hand and never go through `render()`, and because reading
it per render is what lets tests `monkeypatch` it.

`db.user_is_verified()` is the single place M4.4 (share links) and M5.1 (invites)
will ask, and it returns `True` unconditionally while the flag is off. That is
not a shortcut: with confirmation disabled no account ever gets an
`email_verified_at`, so reading the column directly would report every user as
unverified and lock out the entire app the moment either milestone shipped.

### Mentor challenges: the ask-only guard

`app/mentors.py` holds six playbooks. Each is a real operator's *documented,
published* principle used as an attribution label. **No output ever speaks in a
person's voice, quotes one, or attributes a statement to them** — the name
appears only as the reason a principle is being applied, and every rendered
challenge carries `mentors.NOT_A_QUOTE`.

The reason is not stylistic. These are real people, some living; and this app's
thesis is that unevidenced assertion is the thing to strip out (invariant 4, and
the Phase 3 "a language model can invent plausible-sounding ones" warning). A
mentor that asserts is the defect this codebase exists to avoid — and instructing
a model to be a more *persuasive* founder, inside a loop whose job is resisting
persuasion, is how you build a research tool that argues its user into a bad idea.

Four independent layers enforce "asks only", and the tests assert each:

| Layer | Catches |
|---|---|
| The output shape is `{questions: [{question, principle, why_it_matters}]}` — no field an answer could go in | The common case |
| `_must_be_question()` — a line not ending in `?` is dropped | "I would have killed this feature." |
| `_mentor_name_tokens()` + word-boundary match on the attribution | "Jobs would have said…" |
| `_FIRST_SINGULAR` — first-person singular | "If I were you, would you have tested pricing?" — which ends in a question mark and names nobody, and is still impersonation |

A reply of *only* dropped lines raises `LLMError` rather than rendering an empty
card: an empty card reads as "this playbook had nothing to say", which is
indistinguishable from a feature that works.

Two decisions worth knowing before changing any of it:

- **`we`/`our`/`us` are not banned.** They are ambiguous between "you and I" and a
  company's founders, and banning them strips ordinary phrasing like "which of
  these have we already tested?". The first-person *singular* is the
  impersonation risk; the plural is mostly just English.
- **Name matching is word-bounded, not substring.** "Ries" is a substring of
  "varies", so a plain `in` test silently dropped every question about what varies
  between two blocks — on the one playbook whose name ends in a common word.

Nothing in `app/main.py`'s mentor section calls `apply_verdict`,
`set_segment_outcome` or `update_venture`, and `TestMentorChallenge` asserts a
challenge leaves two failed blocks, their notes, their hypotheses, the venture's
status and the run count untouched.

**Scoping is not optional.** `list_mentor_challenges(subject_kind=…)` is
required on any venture page, because a block challenge and a plan challenge share
a `venture_id`: filtering by venture alone put the launch plan's questions on top
of every block panel.

### Near-duplicate idea cards

`app/dedupe.py` is pure text arithmetic — no model call, no embedding, no new
dependency — because it runs while *rendering* the dashboard. A model asked "are
these the same idea?" would cost a quota unit per pair and would make a page
depend on a model call succeeding.

Titles are compared by **Jaccard similarity over character trigrams**, after
lowercasing and turning punctuation into a separator. Trigrams rather than word
sets because word sets punish exactly the case that matters: "Selling sensors to
hospitals" against "Sell sensors to hospitals" shares few whole words and almost
all of its characters. Punctuation becomes a space rather than being deleted, so
`sensors/to hospitals` and `sensors to hospitals` do not fuse into
`sensorstohospitals` and invent matches that are not there.

**`NEAR_DUPLICATE_THRESHOLD = 0.72` is calibrated, not chosen.** On the pairs
pinned in `tests/test_launchloop.py`, true duplicates score 0.774 and up and
distinct ideas 0.659 and down:

| Pair | Score |
|---|---|
| identical after normalisation | 1.000 |
| case / punctuation / whitespace only | 1.000 |
| same idea, words reordered | 0.893 |
| synonym swapped (`supplies` → `provides`) | 0.876 |
| one word changed in a 120-char title | 0.836 |
| "…to small research groups at universities" vs "…to large pharmaceutical companies" | **0.534** |
| "Detect sepsis from **blood** samples" vs "…from **urine** samples" | **0.659** |

The clean band is **0.68–0.77**, and it is narrow — 0.12 wide. That is the honest
weakness of this approach and it is why the feature **surfaces** duplicates rather
than merging them: a chip the user ignores costs a glance, a wrong merge destroys
an idea nobody can get back. `similarity("", "")` returns 0.0 explicitly, because the
padding means `trigrams("")` is `{'   '}` — a single trigram of the four padding
spaces — so two untitled cards would share it completely and score 1.0, making
every untitled card a duplicate of every other.

Computed over **exactly the cards the page renders**, so an "Also extracted as"
link can never point at a card `IDEA_CARD_LIMIT` pushed below the fold. A
duplicate outside the visible set is therefore not flagged — stated, because the
alternative is a link that goes nowhere.

`IDEA_STATUSES` already contained `"rejected"` and the schema already had its
CHECK constraint; nothing ever wrote it. M1.4 adds the two routes that do, and no
migration.

### Tenant isolation

Every lookup that takes an id also takes `user_id`: `get_venture(id, user_id)`,
`get_idea(id, user_id)`, `get_cycle(id, venture_id)`. In `start_run` the venture
lookup *is* the tenant check — an unowned id never reaches the model, so a
guessed id cannot spend a user's quota. A `@app.exception_handler(500)` renders
`error.html` and logs the traceback, so a bug never shows a stack trace.

---

## 7. Front end

`app/static/app.js` is one IIFE with three jobs, all progressive enhancement.
**Every page works with JS off** — forms POST natively.

### Thinking overlay

Forms marked `data-thinking` are intercepted, an overlay is shown with
per-route stage copy (`extract`, `design`, `analyze`, `retry`, `strategy`), an
elapsed timer, and a *Stop waiting* button. The route→copy mapping is a
regex match on the form `action`; `test_frontend.py` asserts that every form
action in the templates is a real POST route, because the overlay once
advertised the wrong work for a route that no longer existed.

The swap needs no backend support: the POSTs already 303, so `fetch` follows the
redirect and returns fully rendered HTML. `swapIn()` parses it, replaces
`<main>`'s inner HTML and class, updates the title and nav highlight, and
`pushState`s the new URL — **unless** the server answered in place, in which case
the current URL is kept so the POST endpoint never lands in the address bar.

Two details that are easy to break: `swapIn` moves focus to `<main>` after the
swap (a keyboard or screen-reader user would otherwise be dumped at the top of
the document with no announcement), and `popstate` forces a full reload, because
a `pushState` entry has no rendered page behind it.

There is a hard 150s client ceiling. The server's own budget is three attempts
with backoff, and surfacing the real error beats spinning forever behind a dead
socket. *Stop waiting* aborts the fetch and reloads — the server may still have
finished, so reloading gets the real state rather than guessing client-side.

### Page transitions

`initNav()` intercepts plain same-origin left-clicks, shows a slim
`nav-progress` bar (reusing the overlay's sweep gradient, dimming the body after
2s), fetches, and calls the same `swapIn`. It bails on modified clicks,
non-`GET` targets, `download`, `rel=external`, `mailto:`/`tel:`, cross-origin
hrefs, and pure-hash links — all of which must reach the browser. On network
failure it falls back to a real `location.href` navigation so the user is never
stranded on a stale page.

### Theme

An inline script in `<head>` resolves `data-theme` from `localStorage` or the OS
preference **before first paint**. It always writes a concrete value, because the
light palette is keyed on `[data-theme="light"]` and "no attribute" would mean
dark regardless of what the OS asked for. `app.js` then only overrides when the
user has explicitly chosen, and follows the OS live if they never have.

---

## 8. Configuration

`app/config.py` splits settings deliberately:

- **Boot settings** — `LAUNCHLOOP_DEBUG`, `SESSION_SECRET_KEY`,
  `SESSION_HTTPS_ONLY` — read once at import, because the session middleware needs
  them before the first request and a missing secret must fail at boot, not on
  first use. `main.py` refuses to start without a real `SESSION_SECRET_KEY`
  unless `LAUNCHLOOP_DEBUG=1`.
- **Runtime settings** — everything else, read per call through a function, so a
  changed value takes effect without a restart and the tests can exercise it.

Readers (`_int`, `_bool`, `_str`) degrade to the default on blank, junk, or
out-of-range values. A typo in a deploy config must not wedge the app.

Full list with defaults and rationale: `.env.example`. The knobs most likely to
need tuning: `SEGMENT_CAP` (default 3), `LLM_MONTHLY_LIMIT_PER_USER` (500),
`LLM_RATE_LIMIT_PER_MIN` (12), `LLM_MODEL`, `LLM_MAX_OUTPUT_TOKENS` (4096),
`LLM_REASONING_EFFORT` (`low`).

**A reader's range can bite a test.** `_int` degrades an out-of-range value to the
default, silently. `MAX_EXTRACTED_CHARS` has a floor of 1000, so a test asserting
truncation at 60 characters is really asserting truncation at 200 000 and passes
for the wrong reason. The fix is to put the test value *inside* the range and make
the input long enough that the cap is unambiguously the binding constraint. This
has now bitten twice, which is why it is written down.

`SOURCE_IMPORT_ENABLED=false` turns off every outbound third-party call and hides
the import box — for a deployment that must not phone home, or a demo where it
should not.

### Two platforms, one codebase

Render and Vercel differ in **process lifetime**, and that single difference
produces every platform-specific setting in this app. Render runs one
long-lived container for the life of a deploy; a Vercel function scales to zero
and each concurrent invocation is its own process.

| Setting | Render | Vercel | Why |
|---|---|---|---|
| `DB_POOL_SIZE` | 5 | 1 | a pool is connections held for nobody when there is no persistent process |
| `DB_POOL_MAX_OVERFLOW` | 5 | 0 | 20 concurrent users × 10 connections would exhaust Supabase's pooler |
| `RUN_MIGRATIONS_ON_BOOT` | 1 | 0 | a cold function would re-run twelve migrations per idle period |
| `MAX_UPLOAD_BYTES` | 8388608 | 4194304 | Vercel rejects bodies over 4.5 MB before the app runs |

`DB_POOL_SIZE=0` selects SQLAlchemy's `NullPool` — connect on checkout, close on
return. It is *not* the same as `pool_size=0`, which is a valid but useless
one-connection pool that would serialise concurrent requests; `get_engine()`
therefore branches on the class rather than passing the number through.

`RUN_MIGRATIONS_ON_BOOT=0` means a Vercel deploy is not self-contained: run
`alembic upgrade head` first, or the app serves the previous schema. That is the
deliberate trade for not putting migrations in front of every cold request.

**A failed boot migration is logged, not raised.** Boot is where a transient
database blip is most likely to land, and raising there means the schema never
gets its chance to recover — every later request 500s too.

The `Dockerfile` is Render's and Vercel's does not touch it: Vercel's container
path looks for `Dockerfile.vercel` at the repo root, which this repo does not
have, so there is one image definition and it is unambiguous which platform uses
it. Note that `SESSION_HTTPS_ONLY=1` is set as an `ENV` in that Dockerfile, so
**Vercel deployments must set it as an environment variable** — it defaults off,
and nothing else turns it on outside the image.

### Deliberately not configurable

The nine segment keys, `SEGMENT_METHODS`, `RUN_VERDICTS`, `SEGMENT_OUTCOMES`,
`IDEA_STATUSES`, `VENTURE_STATUSES` generate both the Postgres CHECK constraints
and the LLM output allowlists. Making them env-driven would let the running code
and the live schema disagree, turning a verdict the model is allowed to emit into
a 500. Same reasoning for `auth.MAX_PASSWORD_BYTES = 72`: that is bcrypt's limit,
and exposing it would let a deploy silently re-enable silent password
truncation — and for `constants.PROMPT_VERSIONS`, which is telemetry about a code
path: a deploy that could relabel its own history would make §5.6's provenance
claim worthless.

---

## 9. Testing

`pytest -q` → **437 tests**, about 7m. Needs Postgres; no network.

### Isolation: one schema per test

`tmp_db` creates `t_<uuid>`, runs the **real Alembic migrations** into it via
`cfg.attributes["version_table_schema"]`, installs an engine whose every
connection sets `-csearch_path=<schema>`, yields, then drops the schema.

### Testing a migration's *data* handling

`tmp_db` upgrades to head, so a **backfill never runs anywhere in the suite** —
and a backfill is the part of a migration most likely to be wrong and least likely
to be noticed, because it only executes against a database that already has data.

The `migrate_to(revision)` fixture creates its own schema, stops at the named
revision, lets the test write rows there, and then upgrades to head.
`TestActionStepMigration` uses it to check three things `tmp_db` structurally
cannot: that an existing `launch_strategy` gets its steps backfilled, that the
hash the *migration* computes matches the one the *route* computes (if they
differ, the backfilled rows are unreachable and the checklist silently shows
everything as pending), and that downgrade → upgrade leaves no debris.

One trap worth knowing before writing these: Alembic's `search_path` during a
pinned migration is `<test schema>,public`, so an **unqualified** table name can
resolve to a leftover copy of the schema in `public` and pass for entirely the
wrong reason. Qualify every name with the test schema.

Two reasons this design:

1. Transaction rollback is not usable here. `db.get_conn()` opens and commits
   its own transaction per call, so a fixture-level rollback would never see the
   app's writes.
2. Running the real migrations means the tests exercise production DDL. A
   migration that only half-applies fails in CI rather than on deploy.

`conftest.py` sets a throwaway `SESSION_SECRET_KEY` and `LAUNCHLOOP_DEBUG`
*before* importing the app, because `app.main` refuses to import without a
secret — so a fresh clone with no `.env` can still collect and run the suite. It
also refuses a `TEST_DATABASE_URL` that still contains `.env.example`
placeholders (`USER`, `PASSWORD`, `<`, `PROJECT`), which otherwise fails as a
confusing `role "USER" does not exist`.

### What the suite covers

`tests/test_launchloop.py` — `TestValidation` (output allowlists),
`TestSegmentLoop`, `TestVerdicts`, `TestRunCap`, `TestRetryPath`, `TestIsolation`,
`TestAuth`, `TestPasswordReset`, `TestEmailVerification`, `TestLoginThrottling`,
`TestPhase1`, `TestQuota`, `TestRateLimit`, `TestCallLog`, `TestPromptProvenance`,
`TestDataExport`, `TestAccountDeletion`, `TestRowLevelSecurity`,
`TestTenantContext`, `TestDbHelpers`, `TestRendering`, `TestProviderFailure`,
`TestUploadExtraction`, `TestUploadRoute`, `TestSourceRecognition`,
`TestSourceFetching`, `TestSourceCache`, `TestImportRoute`.

`tests/test_frontend.py` — template form actions against the live route table,
CSS classes the templates use, the `<main>` focus contract, `initNav`'s guards,
every template reachable from `main.py`, and `tests/app_js.test.js` under Node.

### Testing without a provider

The LLM is mocked at the `app.llm` function boundary with
`monkeypatch.setattr`. Helpers worth reusing:

- `make_venture(client, monkeypatch)` — sign up, stub `extract_ideas`, extract, select.
- `fake_design(*a, **k)` — a valid `generate_segment_tasks` return.
- `fake_verdict(verdict="pass", **over)` — a valid `analyze_segment_run` return.
- `run_once(client, monkeypatch, venture_id, segment_key, verdict, outcome, size)`
  — drives a whole run and asserts the run is recorded against the segment it was
  started for.

**Mock at those two functions, not at the HTTP client.** The rules under test are
the state machine and the output validation, not the transport; `call_json`
itself is exercised through `BadRequestError` and
`TestProviderFailure`.

### Stub the LLM, or the suite spends real money

`config.py` calls `load_dotenv()` at import, so **a developer's real
`LLM_API_KEY` in `.env` is live inside the test suite**. Any test that reaches
`llm.extract_ideas` (or the other three) without stubbing it makes a real,
billable network call. `make_venture`, `fake_design` and `fake_verdict` are the
stubs that prevent this; a test that needs an LLM route must use one of them.

### The rate limiter is process-global

`app/ratelimit.py`'s buckets are keyed on `(user_id, ip)`, and every test signs up
as user 1 from the same TestClient address, so without intervention the whole
suite would share one bucket. An autouse fixture in `conftest.py` clears it and
raises the ceiling out of the way — inside the readers' accepted range, because
`_int` silently degrades an out-of-range value to the production default, which
would re-impose the throttle the fixture exists to lift. `TestRateLimit` then sets
its own limits, exactly as `TestQuota` sets its own monthly cap.

### Row-level security, and making sure the tests are not lying

Two autouse fixtures exist purely so the RLS tests mean something.

**`_enable_app_role`.** Without it the suite runs as the superuser that owns the
tables, and a superuser bypasses RLS even with `FORCE` — every isolation test
would pass vacuously and prove nothing. The fixture grants the app role and
installs it, and fails with an explanatory `UsageError` if the role does not
exist, rather than skipping quietly.

**`_tenant`.** Wraps each test in `db.as_tenant(1)`. Under RLS an unset tenant
denies everything, so a test that calls `db.py` directly would read zero rows and
fail confusingly. It is deliberately *not* a security control: tests about
cross-tenant behaviour switch tenants explicitly, and assert as the **owner**,
because a cross-tenant assertion run as the attacker would pass for the wrong
reason.

`TestRowLevelSecurity::test_the_suite_is_actually_exercising_rls` is the one to
read first — if the connected role bypasses RLS, every test after it is
decoration, so it asserts that before anything else.

Note the asymmetry when writing assertions: RLS makes a row **invisible**, so an
`UPDATE`/`DELETE` that targets another tenant's row matches **zero rows and does
not raise**. Only a `WITH CHECK` violation — an `INSERT` claiming someone else's
`user_id` — errors. Both are safe; they fail differently.

### Two test helpers that are not about tenants

`counts_by_table()` and `as_owner()` in `tests/test_launchloop.py` both lift the
app-role switch so a query runs with the table owner's rights.

They exist because some assertions are **database administration, not tenant
behaviour**, and RLS makes them look like false failures:

- "did that cascade leave an orphan?" — a tenant cannot see orphans; that is the
  policy working.
- "does that audit row survive account deletion?" — `llm_calls` rows whose
  `user_id` was SET NULL are invisible to *every* tenant by design, so there is
  no tenant-scoped way to check.

Running those as a tenant would make "no orphans left" pass for entirely the wrong
reason, which is the specific failure mode the RLS work was guarding against.

---

## 10. Invariants for anyone changing this

1. **Server-side enforcement.** Any new action on a venture or segment goes
   through `venture_phase_state()` / `segment_run_state()` and re-checks tenancy
   before touching the model. A capability the template merely hides is a
   capability the user does not have.
2. **Never persist an unvalidated enum.** Add to the constants *and* the schema
   CHECK in the same change, and validate in `llm.py`. Constants are not
   configurable for exactly this reason.
3. **`passed` requires `confirmed`; `failed` requires `disconfirmed`.** Use
   `set_segment_outcome()`, which defaults the evidence read, rather than
   `update_segment()` with a hand-picked pair.
4. **The model never disposes.** It marks blocks; the user kills and pivots.
   Keep it that way unless the product decision changes explicitly.
5. **Save before you score.** Logged results are persisted before the LLM call,
   so a failure leaves a retryable run rather than lost work.
6. **Quota before the call, telemetry after, never let telemetry raise.** The
   same holds for the rate limit, with one addition: it must also sit *after*
   `log_cycle_results()` on the log route, or a 429 would cost the user the
   evidence they cannot recreate (invariant 5).
7. **Keep `app/schema.py` and `app/constants.py` engine-free.**
8. **New POST → new 303.** `fetch` depends on the redirect to swap the page; a
   200 that renders in place is only for errors, and must be mirrored in
   `tests/test_frontend.py` if it adds a form action.
9. **New user-visible strings from the provider are untrusted data**, wrapped and
   length-capped on the way out.
10. **Adding a page context key means adding it to `_venture_context()`**, so the
    view, the error re-render and the retry path stay identical.
11. **Never store what a prompt said.** `llm_calls` gets a version and a hash
    (§5.6). A column that could hold the prompt body is a copy of unpublished IP
    in the one table that outlives the account, and `TestPromptProvenance`
    asserts the table's exact column set so the next person cannot add one.
12. **Never let a route trust an identity it could derive.** A posted
    `step_key` is checked against the one derived from the stored plan, not
    written to. The general form: when a form posts an identifier, the server
    derives what that identifier *should* be and compares. Two places this has
    already been got wrong are the stale-form case (the identity the user
    confirmed and the identity written would otherwise be resolved at two
    different moments) and the forged-key case.
13. **Never let a mentor answer, and never let it speak for a person.** The
    output shape has no field for advice, `llm.py` drops any line that is not a
    question / names the attributed person / uses the first-person singular, and a
    reply of only dropped lines is an error rather than an empty card. A new
    playbook goes in `app/mentors.py` with its published source; a new *person* is
    not a catalogue change, it is a product decision.
14. **`action_steps.outcome_note` is text the model has never seen.** Nothing
    sends it anywhere today, which is exactly why it is exempt from the
    untrusted-data wrapper (invariant 9). The moment a slice feeds plan notes
    into a prompt, that exemption ends and the wrapper becomes mandatory.
15. **Never change a password without revoking sessions.** `db.set_password()` is
    the only function that writes `password_hash`, and it advances
    `session_epoch` in the same statement. That placement is the invariant: a new
    route that wants to set a password has to go through it, so there is nothing
    tempting anyone into the `UPDATE users SET password_hash = …` that would skip
    the revocation. A new session key goes in `auth.start_session()`, never in a
    route.

---

## 11. Security posture, honestly

| Control | Status |
|---|---|
| Password storage | bcrypt via `auth.py`, direct (not passlib — passlib's backend self-check breaks on bcrypt ≥ 4.0) |
| Password length | min 8 chars (`MIN_PASSWORD_LENGTH`); >72 bytes rejected outright rather than silently truncated |
| Session | itsdangerous-signed cookie, `SameSite=Lax` (which is also the CSRF defence for these endpoints), `Secure` when `SESSION_HTTPS_ONLY=1`. No `max_age`, so it is a browser-session cookie. **Revocable**: `users.session_epoch` is compared on every request, so changing or resetting the password ends every session that predates it. Not revocable *individually* — there is no session table, so no per-device listing |
| Login throttling | sliding window on account **and** source address (IP ceiling is 5× the account one, so a shared NAT is not locked out); clears on success. Stops password guessing, is not a full brute-force defence |
| Password reset | single-use token, 30-minute expiry, only the SHA-256 stored, compared with `hmac.compare_digest`; a reset burns every other live reset token for that account |
| Email verification | **off by default** (`EMAIL_VERIFICATION_ENABLED`, default 0) — no mail provider is configured, so the banner would ask every account for a message that cannot arrive. When on: unverified accounts use the whole app, the banner is a nudge. `db.user_is_verified()` returns `True` while the flag is off, so M4.4/M5.1 gating on it cannot lock out everyone |
| Email transport | `console` by default — a real deployment must set `MAIL_BACKEND=smtp` or nobody receives a reset link. `send()` never raises into a request |
| Secrets | never in the image; `alembic.ini`'s URL is blank on purpose; platform-injected only |
| Tenant isolation | three layers: `user_id` in every WHERE clause; RLS policies on all 11 tables except `login_attempts`; revoked Supabase `anon`/`authenticated` grants (§3.1) |
| Supabase exposure | `anon`/`authenticated` grants revoked by migration `a1c9f0e2b7d1`. `service_role` bypasses **both** those grants and RLS — keep the URL in a platform secret, and see the one-time `GRANT launchloop_app TO service_role` in the README, without which the RLS layer is inert |
| SQL injection | no string-built queries; `update_venture`/`update_segment` allowlist field names because they build SQL from column names |
| XSS | Jinja autoescape on every page, including `action_steps.outcome_note` and mentor questions — free text rendered into shared templates. Note that `mentors.NOT_A_QUOTE` renders its apostrophe as `&#39;`, which is the autoescape working, not a bug. The only `innerHTML` writes are the overlay's own static markup and `swapIn`'s assignment of a `DOMParser`-parsed server response — and script inserted that way does not execute |
| Impersonation | a mentor can never speak as, quote, or name a real person: enforced by output shape, punctuation, word-bounded name matching and a first-person filter in `llm.py` (invariant 13). Every card states the person is not involved |
| Secrets in logs | no prompt or response bodies stored; `llm_calls` keeps metadata plus a prompt version and hash (§5.6) |
| LLM rate limiting | token bucket per user **and** per IP on all five LLM routes, before the call (§6). Per process, so a multi-instance deploy multiplies the ceiling; the monthly quota is still the budget |
| RLS enforcement | `db.rls_active()` / `GET /healthz` → `{"rls": true}`. **Inert on Supabase until `GRANT launchloop_app TO service_role` is run** — one manual step, documented in the README. The switch is membership-tested, so a missing grant degrades to the `user_id` filters rather than 500-ing |

All three of the items this section originally named as "the first three things
to add if this goes in front of people you do not know" — RLS, password reset,
and per-request rate limiting — now exist. The caveat that matters is the row
above: RLS is installed and inert until a `GRANT` is run, so `/healthz` is the
check, not the migration log.