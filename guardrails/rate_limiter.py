"""
SriniMart Data Agent — Layer 4: Guardrails — Rate Limiter
==========================================================
Per-user rate limiting to prevent runaway query costs and
protect the warehouse from unintentional load spikes.

Limits enforced:
  - Queries per minute  (burst protection)
  - Queries per day     (budget enforcement)
  - Token cost per day  (LLM spend cap)

Rate limits are role-differentiated — analysts get higher limits
than store clerks. Finance users are exempt from query-count limits
but still subject to token cost caps.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class RateLimit:
    queries_per_minute: int
    queries_per_day:    int
    token_cost_per_day_usd: float


# Role-based limits
ROLE_LIMITS: dict[str, RateLimit] = {
    "store_clerk":      RateLimit(queries_per_minute=5,  queries_per_day=50,   token_cost_per_day_usd=2.0),
    "store_manager":    RateLimit(queries_per_minute=10, queries_per_day=100,  token_cost_per_day_usd=5.0),
    "regional_manager": RateLimit(queries_per_minute=15, queries_per_day=200,  token_cost_per_day_usd=10.0),
    "analyst":          RateLimit(queries_per_minute=30, queries_per_day=500,  token_cost_per_day_usd=25.0),
    "finance":          RateLimit(queries_per_minute=20, queries_per_day=300,  token_cost_per_day_usd=50.0),
    "admin":            RateLimit(queries_per_minute=60, queries_per_day=2000, token_cost_per_day_usd=200.0),
}


@dataclass
class UserBucket:
    """Sliding-window counters for a single user."""
    minute_timestamps: list = field(default_factory=list)
    day_query_count:   int = 0
    day_token_cost:    float = 0.0
    day_reset_at:      float = field(default_factory=time.time)


class RateLimiter:
    """
    In-memory rate limiter with sliding window for per-minute limits
    and daily counters for budget enforcement.

    In production: backed by Redis for multi-instance deployments.
    """

    def __init__(self):
        self._buckets: dict[str, UserBucket] = defaultdict(UserBucket)

    def check(self, user_id: str, user_role: str) -> tuple[bool, str]:
        """
        Check if the user is within their rate limits.

        Returns:
            (allowed: bool, reason: str)
            If not allowed, reason explains which limit was hit.
        """
        limit = ROLE_LIMITS.get(user_role, ROLE_LIMITS["analyst"])
        bucket = self._buckets[user_id]
        now = time.time()

        # Reset daily counters if a new day has started
        if now - bucket.day_reset_at > 86400:
            bucket.day_query_count = 0
            bucket.day_token_cost  = 0.0
            bucket.day_reset_at    = now

        # Per-minute sliding window
        cutoff = now - 60
        bucket.minute_timestamps = [t for t in bucket.minute_timestamps if t > cutoff]
        if len(bucket.minute_timestamps) >= limit.queries_per_minute:
            return False, f"Rate limit: maximum {limit.queries_per_minute} queries per minute for your role."

        # Daily query count
        if bucket.day_query_count >= limit.queries_per_day:
            return False, f"Daily limit reached: {limit.queries_per_day} queries per day for your role."

        # Daily token cost
        if bucket.day_token_cost >= limit.token_cost_per_day_usd:
            return False, f"Daily budget reached: ${limit.token_cost_per_day_usd:.2f} LLM spend limit for your role."

        return True, "ok"

    def record(self, user_id: str, token_cost_usd: float = 0.0):
        """Record a completed query against the user's counters."""
        bucket = self._buckets[user_id]
        bucket.minute_timestamps.append(time.time())
        bucket.day_query_count += 1
        bucket.day_token_cost  += token_cost_usd
