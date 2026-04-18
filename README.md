# 🛒 SriniMart Data Agent

> **Self-service analytics for SriniMart's 300-store retail network.**  
> Ask questions in plain English. Get governed, accurate answers from the data warehouse. No SQL required.

---

## The Problem We Solved

SriniMart's data team was spending **70% of their time** writing SQL for business stakeholders — regional managers, product owners, and store operators — instead of doing actual analysis.

The solution: a production-grade **Data Agent** that understands SriniMart's data, enforces access controls, and delivers accurate answers at self-service scale.

**Results after deployment:**
| Metric | Before | After |
|--------|--------|-------|
| Query cost per interaction | Baseline | **↓ 70%** |
| Average response time | ~10 min (analyst queue) | **↓ 60%** |
| SQL formatting error rate | 25% | **3%** |
| Analyst time on ad-hoc SQL | 70% | **< 20%** |

---

## Architecture Overview

This agent is built on a **five-layer architecture**. Each layer has a single responsibility — strip one out and the system fails in production.

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Stakeholder                        │
│         "Show me Q3 revenue by region"                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  L1  AGENT RUNTIME                                          │
│  LangGraph orchestration · ReAct loop · Conversation state  │
│  Multi-agent team (CrewAI): Planner · Analyst · Validator   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  L2  TOOL LAYER                                             │
│  schema_discovery · query_execute · aggregate · calculate   │
│  api_fetch · export_csv · slack_notify                      │
│  MCP Protocol — on-demand loading across 50+ data sources   │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  L3  DATA & CONTEXT (RAG)                                   │
│  Schema embeddings · Business glossary · Query examples     │
│  "revenue" → sales.total_amount  |  "Q3" → Jul–Sep          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  L4  GUARDRAILS                                             │
│  RBAC · PII masking · Rate limiting · Approval workflows    │
│  Store managers see own store. Regional managers see region. │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│  L5  EVALUATION                                             │
│  Accuracy scoring · Latency P95 · Token cost · Drift detect │
│  Target: correct answers · <5s response · cost per query    │
└─────────────────────────────────────────────────────────────┘
```

---

## Advanced Tool-Use Patterns

Beyond basic function calling, this agent uses six production patterns to scale across 50+ data sources:

| Pattern | Problem Solved | Result |
|---------|---------------|--------|
| **Tool Search** | 50+ tools overwhelm context window | 85% context reduction, +40% selection accuracy |
| **Programmatic Calling** | Sequential queries too slow/expensive | 200 store queries: 10 min → 15 sec, 400K → 2K tokens |
| **Few-Shot Examples** | Wrong SQL format / parameters | Error rate: 25% → 3% |
| **Agent Skills** | Repeated capability packages | Reusable skill bundles per domain |
| **MCP On-Demand Loading** | Startup context overload | 500-token startup cost regardless of tool count |
| **Token-Efficient Design** | Wasteful schemas & results | Paginated, field-filtered responses |

---

## Reasoning Patterns

The agent routes each query to the right reasoning strategy:

```
Simple lookup  →  ReAct          (think → act → observe, low cost)
Multi-step     →  Plan-and-Execute (full plan upfront, then execute)
Investigation  →  Reflexion       (act → reflect → retry, self-improving)
Strategic      →  Tree-of-Thought  (explore 3 branches, prune to best)
```

A lightweight **complexity router** classifies each query and picks the pattern automatically — 80% of queries use ReAct, keeping costs low.

---

## Framework Stack

| Layer | Framework | Why |
|-------|-----------|-----|
| Orchestration | **LangGraph** | Stateful graph workflows, retry loops, LangSmith tracing |
| Agent Team | **CrewAI** | Role-based Planner / Analyst / Validator / Writer agents |
| Tool Protocol | **MCP** | Standardized interface across all 50+ data sources |
| LLM | **Azure OpenAI (GPT-4o)** | Enterprise SLA, Azure compliance |
| Vector Store | **FAISS + Azure Cognitive Search** | Schema embeddings + hybrid retrieval |

---

## Project Structure

```
srinimart-data-agent/
├── agent/                  # L1 — Agent Runtime
│   ├── runtime.py          # LangGraph orchestration entry point
│   ├── crew.py             # CrewAI multi-agent team definition
│   └── reasoning.py        # Reasoning pattern router
├── tools/                  # L2 — Tool Layer
│   ├── registry.py         # Tool registry & MCP on-demand loader
│   ├── data_tools.py       # schema_discovery, query_execute, aggregate
│   ├── compute_tools.py    # calculate, validate, compare
│   └── output_tools.py     # export_csv, slack_notify, format_results
├── rag/                    # L3 — Data & Context
│   ├── embeddings.py       # Schema + glossary embedding pipeline
│   ├── retriever.py        # Hybrid dense + keyword retrieval
│   └── glossary.py         # SriniMart business glossary
├── guardrails/             # L4 — Guardrails
│   ├── rbac.py             # Role-based access control
│   ├── pii_masking.py      # PII detection and redaction
│   └── rate_limiter.py     # Per-user rate limiting
├── evaluation/             # L5 — Evaluation
│   ├── scorer.py           # Answer accuracy scoring
│   ├── metrics.py          # Latency, cost, satisfaction tracking
│   └── drift_detector.py   # Schema & answer drift detection
├── config/
│   └── settings.py         # Environment configuration
├── docs/
│   └── architecture.md     # Detailed architecture decisions
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/srinivasapesala/srinimart-data-agent.git
cd srinimart-data-agent
pip install -r requirements.txt
cp config/settings.example.py config/settings.py   # add your API keys
python agent/runtime.py
```

---

## Built By

**Srinivasa Pesala** — GenAI & Data Engineering Lead  
[linkedin.com/in/srinivasa-pesala](https://www.linkedin.com/in/sayantan-bhattacharya) · psrinvas570000@gmail.com
