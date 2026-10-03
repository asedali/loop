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

# Where each block sits on the rendered canvas (0-based). This is the canonical
# Osterwalder layout, not the recommended testing order.
CANVAS_LAYOUT = [
    {"key": "key_partners", "label": "Key Partners"},
    {"key": "problem", "label": "Problem & Key Activities"},
    {"key": "key_resources", "label": "Key Resources"},
    {"key": "value_prop", "label": "Value Propositions", "hero": True},
    {"key": "customer_relationships", "label": "Customer Relationships"},
    {"key": "customer", "label": "Customer Segments"},
    {"key": "channel", "label": "Channels"},
]
CANVAS_FINANCIAL = [
    {"key": "cost_structure", "label": "Cost Structure"},
    {"key": "revenue", "label": "Revenue Streams"},
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

LLM_PURPOSES = [
    "extract_ideas",
    "generate_segment_tasks",
    "analyze_segment_run",
    "generate_launch_strategy",
]
LLM_STATUSES = ["ok", "error"]

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
