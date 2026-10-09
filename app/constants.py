"""Enum-like domain constants, and the segment model the validation loop runs on.

Deliberately engine-free: Alembic's env.py imports the table metadata from
app.schema, and both modules import these. If any of them reached app.db, the
migration tooling would try to build a live engine just to read a list.
"""

# ---------------------------------------------------------------------------
# The nine Business Model Canvas blocks, in the order the loop recommends
# attacking them. Front-loading Customer/Problem/Value Prop/Revenue front-loads
# the assumptions that are both most uncertain and most fatal to get wrong;
# the rest are cheaper to reason about once those hold.
#
# `severity` decides what a failed segment means: a critical failure forces a
# pivot-or-kill decision, while an important one is parked with a workaround so
# the venture can keep going.
# ---------------------------------------------------------------------------
SEGMENTS = [
    {"key": "customer", "label": "Customer Segments", "severity": "critical"},
    {"key": "problem", "label": "Problem & Key Activities", "severity": "critical"},
    {"key": "value_prop", "label": "Value Propositions", "severity": "critical"},
    {"key": "revenue", "label": "Revenue Streams", "severity": "critical"},
    {"key": "channel", "label": "Channels", "severity": "important"},
    {"key": "cost_structure", "label": "Cost Structure", "severity": "important"},
    {"key": "key_partners", "label": "Key Partners", "severity": "important"},
    {"key": "key_resources", "label": "Key Resources", "severity": "important"},
    {"key": "customer_relationships", "label": "Customer Relationships", "severity": "important"},
]

SEGMENT_KEYS = [s["key"] for s in SEGMENTS]
SEGMENT_BY_KEY = {s["key"]: s for s in SEGMENTS}
SEGMENT_LABELS = {s["key"]: s["label"] for s in SEGMENTS}
SEGMENT_SEVERITY = {s["key"]: s["severity"] for s in SEGMENTS}

# Where each block sits on the rendered canvas. This is the canonical
# Osterwalder layout, not the recommended testing order — reading order, left to
# right across the two upper rows, then the financial row.
#
# `area` is the CSS grid-area name from app/static/style.css; the template emits
# it as the `a-<area>` class and the stylesheet places the block. Document order
# here must stay in this order, because it is what the narrow-screen single
# column stack falls back to.
CANVAS_LAYOUT = [
    {"key": "key_partners", "label": "Key Partners", "area": "kp"},
    {"key": "problem", "label": "Problem & Key Activities", "area": "ka"},
    {"key": "key_resources", "label": "Key Resources", "area": "kr"},
    {"key": "value_prop", "label": "Value Propositions", "area": "vp", "hero": True},
    {"key": "customer_relationships", "label": "Customer Relationships", "area": "cr"},
    {"key": "channel", "label": "Channels", "area": "ch"},
    {"key": "customer", "label": "Customer Segments", "area": "cs"},
]
CANVAS_FINANCIAL = [
    {"key": "cost_structure", "label": "Cost Structure", "area": "fin"},
    {"key": "revenue", "label": "Revenue Streams", "area": "fin"},
]

# A segment's lifecycle. `passed`/`parked` are both terminal-and-satisfied:
# all nine resolved to one of those is what unlocks Phase 3. A parked segment
# is a known, documented gap rather than a blocker.
SEGMENT_OUTCOMES = ["pending", "active", "passed", "failed", "parked"]
SEGMENT_RESOLVED = ["passed", "parked"]

# What the model concludes about a single run.
RUN_VERDICTS = ["pass", "iterate", "fail", "pivot"]

# Evidence quality per segment, which is what the canvas colours by. Kept
# separate from `outcome` (the decision) so the two stay queryable apart.
SEGMENT_STATUSES = ["untested", "mixed", "confirmed", "disconfirmed"]

# Each run's methods are segment-specific, which is what stops the tasks reading
# as generic "do an interview" advice.
SEGMENT_METHODS = {
    "customer": ["problem_interview", "customer_interview", "segment_survey", "lead_interview"],
    "problem": ["incident_review", "workflow_observation", "data_analysis", "expert_review", "problem_interview"],
    "value_prop": ["prototype_test", "concierge_test", "landing_page_test", "usability_test", "solution_demo"],
    "revenue": ["willingness_to_pay_survey", "price_sensitivity_interview", "preorder_test", "pricing_page_test", "competitor_pricing"],
    "channel": ["channel_trial", "outreach_test", "keyword_test", "ad_test", "community_test"],
    "cost_structure": ["unit_economics_model", "vendor_quote", "bom_analysis", "cost_estimate"],
    "key_partners": ["dependency_map", "partner_interview", "api_probe", "reseller_conversation"],
    "key_resources": ["capability_audit", "tool_trial", "outsourcing_quote", "hiring_scan"],
    "customer_relationships": ["retention_interview", "churn_analysis", "support_ticket_review", "onboarding_test"],
}

# Fallback so a segment with no mapping (e.g. a legacy row) still validates.
DEFAULT_METHODS = ["expert_review", "interview", "survey"]

# Validation runs a segment gets before it's parked and asked to be extended.
SEGMENT_CAP = 3

# Venture-wide backstop. Not the real constraint -- per-segment caps are --
# but it keeps `ventures.max_cycles` meaningful and bounds a runaway venture.
VENTURE_RUN_BACKSTOP = len(SEGMENTS) * SEGMENT_CAP

IDEA_STATUSES = ["candidate", "selected", "rejected"]

VENTURE_STATUSES = ["active", "validated", "killed", "paused", "pivoted"]

# What kind of real-world milestone a launch-plan step is aiming at. A domain
# enum, not config: app/llm.py checks model output against it AND it generates
# the action_steps.milestone_type CHECK, so the running code and the live schema
# cannot disagree (invariant 2).
MILESTONE_TYPES = ["grant", "pilot", "customer"]

# How far along one action-plan step is.
#
# Three states rather than one checkbox, because the realistic outcomes are
# three: done, tried-and-could-not, and not-yet. `blocked` is the one that earns
# its place — it is the outcome a researcher has real information about, and it
# is not terminal, because un-blocking returns the step to `pending` with its
# note intact.
#
# `done` and `blocked` both require an outcome note. A step marked done with
# nothing recorded is the unevidenced claim this whole app is built to replace.
ACTION_STEP_STATUSES = ["pending", "done", "blocked"]

# What a row in `password_reset_tokens` is for. One table serves both flows —
# email verification needs the same primitive (unguessable token, stored hashed,
# expiring, single-use) and a second table would be a second set of rotation
# bugs. This column is what stops one flow from ever reading the other's tokens,
# so it is a CHECK-constrained enum like any other (invariant 2).
TOKEN_PURPOSES = ["password_reset", "email_verify"]

LLM_PURPOSES = [
    "extract_ideas",
    "generate_segment_tasks",
    "analyze_segment_run",
    "generate_launch_strategy",
    "challenge_with_mentor",
]
LLM_STATUSES = ["ok", "error"]

# Which prompt template produced an llm_calls row, so editing a prompt does not
# silently re-label historical results.
#
# This is a domain constant and NOT config. It is telemetry about a code path, so
# a deployment must not be able to relabel history; bumping a version string is a
# code change, and therefore reviewable. The prompt body is deliberately not
# stored — only its SHA-256 (see app/llm.py `_record`).
#
# Bump the `.vN` suffix on any edit that changes what the prompt asks for, in
# particular any edit to a JSON key, an enum, or the output schema. Wording that
# does not change the request does not need a bump.
PROMPT_VERSIONS = {
    "extract_ideas": "extract_ideas.v1",
    "generate_segment_tasks": "generate_segment_tasks.v1",
    "analyze_segment_run": "analyze_segment_run.v2",
    "generate_launch_strategy": "generate_launch_strategy.v1",
    "challenge_with_mentor": "challenge_with_mentor.v1",
}

# Which founder playbook a mentor challenge is applying (M2.1).
#
# The keys only, because app/schema.py needs them to generate the
# mentor_challenges.mentor_key CHECK and this module must stay engine-free
# (invariant 7). The definitions — names, attributions, the principles themselves
# — live in app/mentors.py, which imports this.
#
# MENTOR_SUBJECT_KINDS is the other enum for the same table: what a challenge is
# pointed at. Exactly one of idea_id / venture_id must be set, which the CHECK
# constraint enforces; this list is what the route validates against.
MENTOR_KEYS = [
    "focus",
    "first_principles",
    "demand",
    "falsify",
    "jobs_to_be_done",
    "evidence",
]

MENTOR_SUBJECT_KINDS = ["idea", "segment", "plan"]

# Fields update_venture() will write. It builds SQL from column names, so an
# unchecked key would be injectable.
ALLOWED_VENTURE_FIELDS = {
    "phase", "cycle_count", "max_cycles", "status", "parent_venture_id", "pivot_note",
}

# Same reasoning for the per-segment writers.
ALLOWED_SEGMENT_FIELDS = {
    "status", "notes", "outcome", "outcome_note", "hypothesis",
    "position", "label", "severity", "decided_at", "updated_at",
    "max_cycles",
}


def methods_for(segment_key: str) -> list:
    return SEGMENT_METHODS.get(segment_key) or DEFAULT_METHODS


def is_critical(segment_key: str) -> bool:
    return SEGMENT_SEVERITY.get(segment_key) == "critical"


def label_for(segment_key: str) -> str:
    return SEGMENT_LABELS.get(segment_key) or str(segment_key).replace("_", " ").title()


def _in_clause(values: list) -> str:
    """Render a Python list as a SQL literal list, for the CHECK constraints
    below. Values are all module-level literals, never user input."""
    return ", ".join(f"'{v}'" for v in values)
