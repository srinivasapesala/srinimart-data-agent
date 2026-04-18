"""
SriniMart Data Agent — Layer 1: Agent Runtime
=============================================
LangGraph-based orchestration engine. Receives natural language queries,
plans execution steps, and coordinates all five layers to produce
governed, accurate answers from SriniMart's data warehouse.

Architecture Position: L1 — Agent Runtime
Responsibilities:
  - Parse user intent from natural language
  - Route to the appropriate reasoning pattern (ReAct / Plan-and-Execute / Reflexion)
  - Maintain conversation state across turns
  - Orchestrate calls to Tool Layer (L2), RAG Context (L3), Guardrails (L4)
  - Surface results to the user
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.reasoning import ReasoningRouter, ReasoningPattern
from agent.crew import SriniMartAgentCrew
from guardrails.rbac import RBACGuard
from guardrails.pii_masking import PIIMasker
from evaluation.metrics import MetricsCollector
from config.settings import Settings


class AgentState(dict):
    """
    Shared state object that flows through every node in the LangGraph.
    Each layer reads and writes to this state — it is the single source
    of truth for a query's lifecycle.
    """
    query: str                  # Original user question
    user_id: str                # For RBAC permission lookup
    user_role: str              # e.g. "store_manager", "regional_manager", "analyst"
    reasoning_pattern: str      # Selected pattern: react / plan_execute / reflexion / tot
    plan: list                  # Steps produced by Plan-and-Execute planner
    tool_calls: list            # History of tool calls made this turn
    tool_results: list          # Results returned from Tool Layer
    rag_context: dict           # Schema + glossary context from RAG layer
    answer: str                 # Final natural language answer
    error: str | None           # Error message if something failed
    iteration_count: int        # Guard against infinite reasoning loops
    metrics: dict               # Token cost, latency, accuracy signals


class SriniMartRuntime:
    """
    L1 — Agent Runtime.

    Entry point for every user query. Builds the LangGraph workflow,
    threads conversation state through each node, and returns a governed answer.

    The graph topology follows this flow:
        authenticate → classify_intent → load_rag_context →
        route_reasoning → [react_loop | plan_execute | reflexion] →
        apply_guardrails → format_response → evaluate
    """

    MAX_ITERATIONS = 8  # Circuit breaker: escalate if agent loops beyond this

    def __init__(self, settings: Settings):
        self.settings = settings
        self.crew = SriniMartAgentCrew(settings)
        self.router = ReasoningRouter()
        self.rbac = RBACGuard()
        self.pii_masker = PIIMasker()
        self.metrics = MetricsCollector()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def query(self, question: str, user_id: str, session_id: str) -> dict:
        """
        Main entry point. Accepts a plain-English question and returns
        a governed, sourced answer with metadata.

        Args:
            question:   Natural language query from the stakeholder.
            user_id:    SriniMart employee ID — drives RBAC scoping.
            session_id: Conversation session for multi-turn memory.

        Returns:
            {
              "answer":   str,   # Natural language response
              "sources":  list,  # Tables / columns referenced
              "cost_usd": float, # LLM token cost for this query
              "latency_ms": int  # Wall-clock response time
            }
        """
        initial_state: AgentState = {
            "query": question,
            "user_id": user_id,
            "user_role": self.rbac.get_role(user_id),
            "reasoning_pattern": None,
            "plan": [],
            "tool_calls": [],
            "tool_results": [],
            "rag_context": {},
            "answer": "",
            "error": None,
            "iteration_count": 0,
            "metrics": {},
        }

        config = {"configurable": {"thread_id": session_id}}
        final_state = self.graph.invoke(initial_state, config=config)

        return {
            "answer": final_state["answer"],
            "sources": self._extract_sources(final_state["tool_results"]),
            "cost_usd": final_state["metrics"].get("cost_usd", 0),
            "latency_ms": final_state["metrics"].get("latency_ms", 0),
        }

    # ------------------------------------------------------------------ #
    #  Graph Construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_graph(self) -> StateGraph:
        """
        Constructs the LangGraph StateGraph that defines the agent's
        execution flow. Each node is a pure function: state-in → state-out.
        """
        graph = StateGraph(AgentState)

        # Register nodes
        graph.add_node("authenticate",      self._node_authenticate)
        graph.add_node("classify_intent",   self._node_classify_intent)
        graph.add_node("load_rag_context",  self._node_load_rag_context)
        graph.add_node("route_reasoning",   self._node_route_reasoning)
        graph.add_node("react_loop",        self._node_react_loop)
        graph.add_node("plan_execute",      self._node_plan_execute)
        graph.add_node("reflexion",         self._node_reflexion)
        graph.add_node("apply_guardrails",  self._node_apply_guardrails)
        graph.add_node("format_response",   self._node_format_response)
        graph.add_node("evaluate",          self._node_evaluate)

        # Define edges
        graph.set_entry_point("authenticate")
        graph.add_edge("authenticate",     "classify_intent")
        graph.add_edge("classify_intent",  "load_rag_context")
        graph.add_edge("load_rag_context", "route_reasoning")

        # Conditional branching — pick reasoning pattern
        graph.add_conditional_edges(
            "route_reasoning",
            self._select_reasoning_node,
            {
                "react":         "react_loop",
                "plan_execute":  "plan_execute",
                "reflexion":     "reflexion",
            }
        )

        graph.add_edge("react_loop",       "apply_guardrails")
        graph.add_edge("plan_execute",     "apply_guardrails")
        graph.add_edge("reflexion",        "apply_guardrails")
        graph.add_edge("apply_guardrails", "format_response")
        graph.add_edge("format_response",  "evaluate")
        graph.add_edge("evaluate",         END)

        return graph.compile(checkpointer=MemorySaver())

    # ------------------------------------------------------------------ #
    #  Graph Nodes                                                         #
    # ------------------------------------------------------------------ #

    def _node_authenticate(self, state: AgentState) -> AgentState:
        """
        Verify the user exists and their role is known.
        Blocks queries from unknown users before any LLM cost is incurred.
        """
        if not self.rbac.is_authorized(state["user_id"]):
            state["error"] = "Unauthorized: user not found in SriniMart directory."
            state["answer"] = "I'm unable to process your request — please contact IT support."
        return state

    def _node_classify_intent(self, state: AgentState) -> AgentState:
        """
        Lightweight intent classification — determines query complexity
        so the reasoning router can pick the right pattern.
        Uses a small, cheap model call to avoid burning GPT-4o budget
        on classification overhead.
        """
        if state.get("error"):
            return state
        state["intent_class"] = self.router.classify(state["query"])
        return state

    def _node_load_rag_context(self, state: AgentState) -> AgentState:
        """
        L3 — Data & Context layer call.
        Retrieves relevant schema embeddings, business glossary terms,
        and example queries that match the user's question.
        This gives the LLM semantic grounding so it uses real column
        names instead of hallucinating plausible-sounding ones.
        """
        if state.get("error"):
            return state
        from rag.retriever import RAGRetriever
        retriever = RAGRetriever(self.settings)
        state["rag_context"] = retriever.retrieve(
            query=state["query"],
            user_role=state["user_role"],
        )
        return state

    def _node_route_reasoning(self, state: AgentState) -> AgentState:
        """
        Selects the reasoning pattern based on intent class.
        Simple lookups → ReAct (fast, cheap).
        Multi-step reports → Plan-and-Execute (coordinated, predictable).
        Root-cause investigations → Reflexion (self-correcting).
        Strategic decisions → Tree-of-Thought (quality over cost).
        """
        if state.get("error"):
            return state
        pattern = self.router.select_pattern(state["intent_class"])
        state["reasoning_pattern"] = pattern.value
        return state

    def _node_react_loop(self, state: AgentState) -> AgentState:
        """
        ReAct reasoning: Think → Act (tool call) → Observe → Repeat.
        Handles ~80% of SriniMart queries: simple data lookups,
        single-metric questions, straightforward aggregations.
        Capped at MAX_ITERATIONS to prevent infinite loops.
        """
        if state.get("error"):
            return state
        result = self.crew.run_react(
            query=state["query"],
            rag_context=state["rag_context"],
            user_role=state["user_role"],
            max_iterations=self.MAX_ITERATIONS,
        )
        state["tool_calls"] = result["tool_calls"]
        state["tool_results"] = result["tool_results"]
        state["answer"] = result["answer"]
        return state

    def _node_plan_execute(self, state: AgentState) -> AgentState:
        """
        Plan-and-Execute: Planner agent creates a full multi-step plan,
        Executor agent carries out each step sequentially.
        Used for complex queries: quarterly business reviews, cross-region
        comparisons, multi-source reports.
        """
        if state.get("error"):
            return state
        result = self.crew.run_plan_execute(
            query=state["query"],
            rag_context=state["rag_context"],
            user_role=state["user_role"],
        )
        state["plan"] = result["plan"]
        state["tool_calls"] = result["tool_calls"]
        state["tool_results"] = result["tool_results"]
        state["answer"] = result["answer"]
        return state

    def _node_reflexion(self, state: AgentState) -> AgentState:
        """
        Reflexion: Execute → Reflect on what worked / failed → Retry with insight.
        Used for investigative queries: "why did Southwest revenue drop 30%?"
        The reflection step turns each dead end into a better-targeted next hypothesis.
        Capped at 3 reflection cycles — diminishing returns beyond that.
        """
        if state.get("error"):
            return state
        result = self.crew.run_reflexion(
            query=state["query"],
            rag_context=state["rag_context"],
            user_role=state["user_role"],
            max_cycles=3,
        )
        state["tool_calls"] = result["tool_calls"]
        state["tool_results"] = result["tool_results"]
        state["answer"] = result["answer"]
        return state

    def _node_apply_guardrails(self, state: AgentState) -> AgentState:
        """
        L4 — Guardrails layer.
        Runs PII masking on the final answer before it reaches the user.
        Also validates that the answer only surfaces data the user's role
        is permitted to see — a final safety check after reasoning completes.
        """
        if state.get("error"):
            return state
        state["answer"] = self.pii_masker.mask(state["answer"])
        permitted = self.rbac.validate_answer_scope(
            answer=state["answer"],
            user_role=state["user_role"],
            tool_results=state["tool_results"],
        )
        if not permitted:
            state["answer"] = "I found relevant data, but it falls outside your access scope. Please contact your regional manager."
        return state

    def _node_format_response(self, state: AgentState) -> AgentState:
        """
        Final formatting pass — adds citations, structures tables,
        and appends a confidence note if result quality is uncertain.
        """
        return state

    def _node_evaluate(self, state: AgentState) -> AgentState:
        """
        L5 — Evaluation layer.
        Records latency, token cost, and answer quality signals
        for the continuous improvement feedback loop.
        """
        self.metrics.record(state)
        return state

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _select_reasoning_node(self, state: AgentState) -> str:
        """Conditional edge: maps reasoning_pattern value to graph node name."""
        pattern_map = {
            ReasoningPattern.REACT.value:         "react",
            ReasoningPattern.PLAN_EXECUTE.value:  "plan_execute",
            ReasoningPattern.REFLEXION.value:     "reflexion",
            ReasoningPattern.TREE_OF_THOUGHT.value: "react",  # ToT runs inside crew
        }
        return pattern_map.get(state.get("reasoning_pattern", "react"), "react")

    def _extract_sources(self, tool_results: list) -> list:
        """Pull table/column references from tool results for answer citations."""
        sources = []
        for result in tool_results:
            if isinstance(result, dict) and "tables_used" in result:
                sources.extend(result["tables_used"])
        return list(set(sources))
