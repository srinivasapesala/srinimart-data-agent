"""
SriniMart Data Agent — Reasoning Pattern Router
================================================
Classifies each incoming query and selects the appropriate reasoning
strategy. Keeps simple queries fast and cheap (ReAct), while routing
complex investigations and multi-step reports to the right pattern.

Patterns supported:
  - ReAct           : Simple data lookups, single-metric questions
  - Plan-and-Execute: Multi-step reports, cross-region comparisons
  - Reflexion       : Root-cause investigations ("why did X drop?")
  - Tree-of-Thought : Strategic decisions, optimization analysis
"""

from enum import Enum


class ReasoningPattern(Enum):
    REACT           = "react"
    PLAN_EXECUTE    = "plan_execute"
    REFLEXION       = "reflexion"
    TREE_OF_THOUGHT = "tree_of_thought"


class IntentClass(Enum):
    SIMPLE_LOOKUP    = "simple_lookup"      # "Show me revenue for Q4"
    MULTI_STEP       = "multi_step"         # "Create a quarterly review with charts"
    INVESTIGATIVE    = "investigative"      # "Why did Southwest revenue drop 30%?"
    STRATEGIC        = "strategic"          # "Should we expand into the Pacific NW?"


class ReasoningRouter:
    """
    Lightweight complexity classifier that routes each query to the
    most cost-effective reasoning pattern.

    Routing logic:
      - Action verbs (show, get, list) + single metric → ReAct
      - Compare / report / review keywords → Plan-and-Execute
      - Why / investigate / root cause → Reflexion
      - Should / recommend / strategy → Tree-of-Thought

    85% of SriniMart production queries route to ReAct,
    keeping average query cost well below budget targets.
    """

    # Keyword heuristics — fast path before LLM classification
    SIMPLE_TRIGGERS      = {"show", "get", "list", "what", "how many", "total"}
    MULTI_STEP_TRIGGERS  = {"compare", "report", "review", "breakdown", "across all"}
    INVESTIGATIVE_TRIGGERS = {"why", "investigate", "reason", "cause", "dropped", "declined"}
    STRATEGIC_TRIGGERS   = {"should", "recommend", "strategy", "expand", "forecast"}

    def classify(self, query: str) -> IntentClass:
        """
        Classify query complexity using keyword heuristics.
        Fast path — no LLM call needed for most queries.
        """
        q = query.lower()

        if any(t in q for t in self.INVESTIGATIVE_TRIGGERS):
            return IntentClass.INVESTIGATIVE

        if any(t in q for t in self.STRATEGIC_TRIGGERS):
            return IntentClass.STRATEGIC

        if any(t in q for t in self.MULTI_STEP_TRIGGERS):
            return IntentClass.MULTI_STEP

        return IntentClass.SIMPLE_LOOKUP

    def select_pattern(self, intent: IntentClass) -> ReasoningPattern:
        """
        Map intent class to reasoning pattern.

        Cost/quality trade-offs:
          ReAct           — Low cost,   Low latency,  Good for 80% of queries
          Plan-and-Execute — Medium cost, Medium latency, Best for coordinated workflows
          Reflexion        — Medium cost, High latency,  Best for investigations
          Tree-of-Thought  — High cost (3-5x), High latency, Reserved for strategic
        """
        mapping = {
            IntentClass.SIMPLE_LOOKUP:  ReasoningPattern.REACT,
            IntentClass.MULTI_STEP:     ReasoningPattern.PLAN_EXECUTE,
            IntentClass.INVESTIGATIVE:  ReasoningPattern.REFLEXION,
            IntentClass.STRATEGIC:      ReasoningPattern.TREE_OF_THOUGHT,
        }
        return mapping[intent]

    def escalate(self, current: ReasoningPattern, iterations: int) -> ReasoningPattern:
        """
        Graceful degradation — if the agent is looping without progress,
        escalate to a more powerful (and more expensive) reasoning pattern.
        Called by the runtime when iteration_count exceeds thresholds.

        Escalation ladder:
          ReAct (stuck at 5 iterations) → Plan-and-Execute
          Plan-and-Execute (stuck)       → Reflexion
          Reflexion (stuck at 3 cycles)  → Surface partial result to user
        """
        escalation_map = {
            ReasoningPattern.REACT:        ReasoningPattern.PLAN_EXECUTE,
            ReasoningPattern.PLAN_EXECUTE: ReasoningPattern.REFLEXION,
            ReasoningPattern.REFLEXION:    None,  # Surface partial result
        }
        return escalation_map.get(current, None)
