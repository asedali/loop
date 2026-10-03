"""
Per-user LLM call quota.

Every LLM call costs money and burns a shared provider account, so a public
sign-up URL is all it takes for one user to exhaust the budget. This is a
deliberately blunt monthly cap per account, not a fair-use heuristic.
"""
from . import config
from . import db
from .llm import QuotaExceeded

# A full nine-block validation is 9 blocks x 3 runs x 2 calls (design + verdict)
# = 54 calls, so the default of 500 covers roughly nine complete validations.
DEFAULT_MONTHLY_LIMIT = 500


def monthly_limit() -> int:
    return config.llm_monthly_limit_per_user()


def check(user_id: int) -> None:
    """Raises QuotaExceeded if the user is out of calls for the month."""
    limit = monthly_limit()
    if limit == 0:
        return  # explicitly disabled
    used = db.count_llm_calls_this_month(user_id)
    if used >= limit:
        raise QuotaExceeded(
            f"You've used all {limit} AI calls allowed this month. "
            f"The limit resets on the first of next month."
        )
