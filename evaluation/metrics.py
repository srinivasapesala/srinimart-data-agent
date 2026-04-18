"""
SriniMart Data Agent — Layer 5: Evaluation
===========================================
Tracks answer accuracy, response latency, LLM token cost,
and user satisfaction signals for every query.

Without evaluation, you're flying blind — confident everything works
until a wrong answer reaches a board deck.

Metrics tracked:
  - Accuracy score    : Does the answer match the ground truth?
  - Latency P95       : 95th percentile response time (target: < 5 seconds)
  - Token cost        : LLM spend per query (target: < $0.05 average)
  - PII leakage rate  : Fraction of queries where PII appeared in the answer
  - Reasoning pattern : Which pattern was used — for cost/quality analysis
  - Error rate        : Fraction of queries that failed or returned errors

These metrics feed the continuous improvement loop:
  Collect → Analyze → Identify regressions → Fix → Redeploy
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class QueryRecord:
    """Immutable record of a single completed query."""
    query_id:          str
    user_id:           str
    user_role:         str
    query_text:        str
    reasoning_pattern: str
    answer:            str
    latency_ms:        int
    token_cost_usd:    float
    tool_call_count:   int
    error:             str | None
    pii_detected:      bool
    timestamp:         float = field(default_factory=time.time)


class MetricsCollector:
    """
    In-memory metrics store for SriniMart agent queries.
    In production: streams to Azure Monitor + Grafana dashboard.

    Key SriniMart targets:
      Latency P95     < 5,000 ms
      Token cost avg  < $0.05 / query
      Error rate      < 5%
      PII leakage     0%
    """

    # Latency target in milliseconds
    LATENCY_TARGET_MS  = 5_000
    # Cost target per query in USD
    COST_TARGET_USD    = 0.05

    def __init__(self):
        self._records: list[QueryRecord] = []
        self._pattern_stats: dict = defaultdict(lambda: {"count": 0, "total_cost": 0.0, "total_latency_ms": 0})

    def record(self, state: dict):
        """
        Record metrics from a completed agent state.
        Called at the end of every query by the runtime's evaluate node.
        """
        metrics = state.get("metrics", {})
        rec = QueryRecord(
            query_id          = metrics.get("query_id", "unknown"),
            user_id           = state.get("user_id", ""),
            user_role         = state.get("user_role", ""),
            query_text        = state.get("query", ""),
            reasoning_pattern = state.get("reasoning_pattern", "react"),
            answer            = state.get("answer", ""),
            latency_ms        = metrics.get("latency_ms", 0),
            token_cost_usd    = metrics.get("cost_usd", 0.0),
            tool_call_count   = len(state.get("tool_calls", [])),
            error             = state.get("error"),
            pii_detected      = metrics.get("pii_detected", False),
        )
        self._records.append(rec)
        self._update_pattern_stats(rec)

    def summary(self) -> dict:
        """
        Return a metrics summary across all recorded queries.
        Surfaced on the SriniMart agent operations dashboard.
        """
        if not self._records:
            return {"message": "No queries recorded yet."}

        latencies = [r.latency_ms for r in self._records]
        costs     = [r.token_cost_usd for r in self._records]
        errors    = [r for r in self._records if r.error]
        pii_leaks = [r for r in self._records if r.pii_detected]

        latencies_sorted = sorted(latencies)
        p95_idx = int(len(latencies_sorted) * 0.95)

        return {
            "total_queries":      len(self._records),
            "error_rate_pct":     round(len(errors) / len(self._records) * 100, 2),
            "pii_leakage_rate":   round(len(pii_leaks) / len(self._records) * 100, 4),
            "latency": {
                "p50_ms":         latencies_sorted[len(latencies_sorted) // 2],
                "p95_ms":         latencies_sorted[p95_idx],
                "target_ms":      self.LATENCY_TARGET_MS,
                "within_target":  latencies_sorted[p95_idx] <= self.LATENCY_TARGET_MS,
            },
            "cost": {
                "avg_usd":        round(sum(costs) / len(costs), 4),
                "total_usd":      round(sum(costs), 4),
                "target_usd":     self.COST_TARGET_USD,
                "within_target":  (sum(costs) / len(costs)) <= self.COST_TARGET_USD,
            },
            "by_reasoning_pattern": {
                pattern: {
                    "count":          stats["count"],
                    "avg_cost_usd":   round(stats["total_cost"] / stats["count"], 4) if stats["count"] else 0,
                    "avg_latency_ms": round(stats["total_latency_ms"] / stats["count"]) if stats["count"] else 0,
                }
                for pattern, stats in self._pattern_stats.items()
            },
        }

    def _update_pattern_stats(self, rec: QueryRecord):
        stats = self._pattern_stats[rec.reasoning_pattern]
        stats["count"]            += 1
        stats["total_cost"]       += rec.token_cost_usd
        stats["total_latency_ms"] += rec.latency_ms
