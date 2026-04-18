# SriniMart Data Agent — Architecture Decisions

## Why Five Layers?

A single chatbot wrapper connected to a database breaks within a week:

1. **Hallucinated SQL** — the LLM invents column names (`sales.revenue`) that don't exist (`sales.total_amount`)
2. **No access control** — a store clerk asks about payroll and gets the full company payroll
3. **Untracked quality** — nobody knows if answers are correct until a wrong number reaches a board deck

Five distinct layers, each with one responsibility, prevent all three failure modes.

---

## Layer Decisions

### L1 — Why LangGraph for orchestration?

LangGraph was chosen over simpler frameworks because SriniMart's core workflow is a stateful pipeline with conditional branching and retry loops — exactly what graph-based orchestration handles best.

The graph model makes the agent's execution path explicit and traceable via LangSmith. When something goes wrong, you can replay the graph node-by-node to find the failure point.

Alternative considered: a single prompt chain. Rejected because it can't handle retry loops, conditional reasoning patterns, or multi-turn conversation state across sessions.

### L1 — Why CrewAI for the agent team?

CrewAI's role-based model (Planner / Analyst / Validator / Writer) makes agent responsibilities explicit and independently testable. Switching from a monolithic single-agent approach to a four-role team reduced debugging time significantly — failures are now attributable to a specific role.

Alternative considered: LangGraph for everything. Rejected because defining agent roles and team dynamics in pure graph nodes was awkward and verbose. CrewAI's role abstraction solved this cleanly.

### L2 — Why MCP for tool integration?

The Model Context Protocol provides a standardized interface for tool integration that works across agent frameworks. All 50+ SriniMart data sources are wrapped as MCP servers — this means any future framework can use any tool without adapting each connection individually.

On-demand loading (rather than loading all schemas at startup) drops the initial context cost from ~15,000 tokens to ~500 tokens, freeing the context window for reasoning.

### L2 — Why few-shot examples on every tool?

Providing 2-3 usage examples per tool schema dropped SQL formatting errors from 25% to 3%. The examples act as implicit documentation that the LLM internalizes — they show the expected parameter format in a way that a text description alone doesn't convey.

Cost: ~200-400 tokens per tool. Savings: thousands of tokens in retry loops.

### L3 — Why FAISS + Azure Cognitive Search (hybrid)?

Schema retrieval needs both semantic understanding (FAISS dense embeddings catch synonyms like "revenue" → `total_amount`) and exact keyword matching (Azure Cognitive Search catches precise column names like `fiscal_quarter`). Neither alone is sufficient — hybrid retrieval covers both cases.

### L4 — Why dual-check guardrails (pre-query + post-query)?

Pre-query scope injection (adding WHERE clauses) prevents over-fetching at the database — it's the primary control. Post-query answer validation is the safety net that catches edge cases in complex multi-step queries where the scope injection might not cover every sub-query.

Two independent checks means a single failure in either layer doesn't expose out-of-scope data.

### L5 — Why track per-reasoning-pattern metrics?

Different reasoning patterns have very different cost profiles:
- ReAct: low cost, low latency (right for 80% of queries)
- Plan-and-Execute: medium cost, medium latency
- Reflexion: medium-high cost, high latency
- Tree-of-Thought: 3-5x cost multiplier

Without per-pattern metrics, cost spikes are invisible. With them, you can see if queries are being misrouted to expensive patterns and fix the complexity classifier.

---

## Reasoning Pattern Selection

The complexity router uses keyword heuristics for speed (no LLM call needed for classification):

| Query contains | → Pattern | Example |
|----------------|-----------|---------|
| show / get / list + single metric | ReAct | "Show me Q4 revenue" |
| compare / report / review | Plan-and-Execute | "Create a quarterly business review" |
| why / investigate / dropped | Reflexion | "Why did Southwest revenue drop 30%?" |
| should / recommend / strategy | Tree-of-Thought | "Should we expand into the Pacific NW?" |

The heuristic correctly classifies 85% of SriniMart production queries. The remaining 15% fall back to ReAct (safe default) or are escalated if they loop without progress.

---

## Programmatic Tool Calling

When the agent detects a pattern like "check X across all stores," it writes a loop rather than making 300 sequential LLM decisions:

```python
# Without programmatic calling: 300 LLM round-trips
for store in stores:
    llm.decide_next_step(store)  # 300 × think-act-observe = 10 minutes, 400K tokens

# With programmatic calling: 1 LLM decision, 300 tool executions
results = tool_registry.batch_query_stores(store_ids=all_store_ids, metric="revenue", period="Q3-2024")
# → 15 seconds, ~2K tokens
```

The LLM writes the loop once. The tool executes it. No per-iteration reasoning overhead.

---

## Token Budget

For a 128K context window, SriniMart's budget allocation:

| Component | Tokens | Notes |
|-----------|--------|-------|
| System prompt + instructions | ~2,000 | Fixed overhead |
| Active tool schemas (4-5 tools) | ~500 | On-demand loading only |
| RAG context (schema + glossary + examples) | ~1,500 | Top-k retrieval, not full catalog |
| Conversation history | ~10,000 | Multi-turn sessions |
| Tool call results | ~5,000 | Paginated (25 rows default) |
| Reasoning space | Remaining | Never consume so much context the agent can't think |

Total: ~19,000 tokens per query on average — well within budget, leaving ample reasoning space.
