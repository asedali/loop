"""
Per-request rate limiting for the LLM routes.

The monthly quota in app/quota.py is a *budget*: it bounds spend per account per
month and says nothing about how fast that budget can be burned. So today a
single user can fire requests back to back and tie up a worker thread for the
full duration of a slow provider call — LLM_TIMEOUT is 60s and the transport
retries up to LLM_MAX_ATTEMPTS — while the provider 429s us and the user is
told, misleadingly, that "the AI provider is rate-limiting requests right now".

This module is the brake: a token bucket per (scope, key) pair, checked on the
five routes that call the model and *before* the call, so a rejected request
writes no `llm_calls` row and therefore does not consume the caller's monthly
quota either.

    user scope   one bucket per account — a single user cannot monopolise a worker
    ip scope     one bucket per source address — many accounts behind one address
                 is a signup bot pattern the per-user budget cannot see

Capacity equals the configured per-window number and refills continuously, so a
burst of N calls is allowed and call N+1 inside the window is refused, while a
steady drip at the same average rate is not.

Phase 1's identifier import (M1.2) rides the same bucket under a `prefix`, because
it has the same shape of risk — a user-triggered outbound request that holds a
worker thread — at a much smaller cost. It passes its own limits, and the prefix
keeps the two sets of capacity independent.

Scope, not security control
--------------------------
The buckets live in this process's memory. A multi-instance deployment gets one
bucket set per instance, so the effective ceiling multiplies by the instance
count. Render's free plan is a single instance and the app is not documented for
horizontal scaling, so this matches the deployed topology — but if that changes,
this module needs a shared store (Redis, or a `llm_call_events` table) first.
The monthly quota remains the real budget; this is a throttle in front of it.

Deliberately not keyed on email: the login throttle already covers that, and an
email-keyed bucket would lock out a household sharing one address for an hour.
"""
import math
import threading
import time
from collections import OrderedDict

from . import config

# Ceiling on tracked keys. Every distinct source address gets a bucket, so an
# attacker rotating IPv6 addresses would otherwise grow this map for as long as
# the process lives. Oldest-first eviction bounds it; the eviction key is a
# monotonic timestamp, so "oldest" really is least-recently-refilled.
_MAX_BUCKETS = 10_000

_lock = threading.Lock()
_buckets: "OrderedDict[str, list]" = OrderedDict()


class _Bucket:
    """A token bucket. Mutated only under `_lock`."""

    __slots__ = ("tokens", "updated_at")

    def __init__(self, capacity: float, now: float):
        self.tokens = capacity
        self.updated_at = now

    def refill(self, rate_per_second: float, capacity: float, now: float) -> None:
        elapsed = now - self.updated_at
        if elapsed > 0:
            self.tokens = min(capacity, self.tokens + elapsed * rate_per_second)
            self.updated_at = now

    def available(self) -> bool:
        return self.tokens >= 1.0

    def spend(self) -> None:
        self.tokens -= 1.0

    def retry_after(self, rate_per_second: float) -> int:
        if rate_per_second <= 0:
            return 1
        return max(1, math.ceil((1.0 - self.tokens) / rate_per_second))


def _scopes(user_id, ip, per_user, per_ip, prefix) -> list:
    """The bucket keys this request spends from, in the order they are checked.

    `prefix` keeps the LLM buckets and the identifier-import buckets in one map
    without letting them share capacity: a burst of imports must not spend the
    tokens that were reserved for model calls, and vice versa.
    """
    out = []
    if user_id:
        out.append((f"{prefix}user:{user_id}", per_user))
    if ip:
        out.append((f"{prefix}ip:{ip}", per_ip))
    return out


def check(user_id=None, ip=None, *, per_user=None, per_ip=None, prefix: str = "") -> int | None:
    """Spend one token from every applicable scope.

    Returns None when the request may proceed, or the number of seconds until
    the fullest bucket has a token again.

    Refill happens for all scopes first and nothing is spent until every scope has
    been found to have a token, so a request refused by the IP scope does not also
    burn a per-user token — otherwise one busy IP would drain the per-user budget
    of everyone behind it.
    """
    per_user = config.llm_rate_limit_per_min() if per_user is None else per_user
    per_ip = config.llm_rate_limit_ip_per_min() if per_ip is None else per_ip
    window = config.llm_rate_limit_window_seconds()
    now = time.monotonic()
    live = [(key, limit, limit / window)
            for key, limit in _scopes(user_id, ip, per_user, per_ip, prefix) if limit > 0]
    if not live:
        return None

    with _lock:
        while len(_buckets) >= _MAX_BUCKETS:
            _buckets.popitem(last=False)

        buckets = []
        for key, limit, rate in live:
            bucket = _buckets.get(key)
            if bucket is None:
                bucket = _buckets[key] = _Bucket(float(limit), now)
            bucket.refill(rate, float(limit), now)
            buckets.append((key, bucket, rate))

        wait = 0
        for _key, bucket, rate in buckets:
            if not bucket.available():
                wait = max(wait, bucket.retry_after(rate))
        if wait:
            return wait

        for key, bucket, _rate in buckets:
            bucket.spend()
            _buckets.move_to_end(key)
        return None


def reset() -> None:
    """Forget every bucket. For the test suite, which shares one process."""
    with _lock:
        _buckets.clear()