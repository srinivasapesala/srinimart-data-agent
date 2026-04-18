"""
SriniMart Data Agent — CrewAI Multi-Agent Team
===============================================
Defines the four specialized agents that collaborate to answer
SriniMart stakeholder queries. Each agent has a clear role,
goal, and backstory — CrewAI's role-based model keeps responsibilities
explicit and the system debuggable.

Agent Roles:
  - Planner   : Decomposes complex queries into step-by-step plans
  - Analyst   : Executes SQL, calls APIs, processes data
  - Validator : Checks answer correctness and data integrity
  - Writer    : Formats the final response for the stakeholder

This three-framework approach — LangGraph for orchestration,
CrewAI for team composition, MCP for tool integration — reduced
the SriniMart codebase by 40% vs a single-framework approach.
"""

from crewai import Agent, Crew, Task, Process
from tools.registry import ToolRegistry


class SriniMartAgentCrew:
    """
    Multi-agent team for SriniMart data queries.

    The crew operates in three modes depending on the reasoning pattern
    selected by the runtime:
      - run_react()         : Single-agent loop (Analyst only, fast path)
      - run_plan_execute()  : Planner → Analyst → Validator → Writer pipeline
      - run_reflexion()     : Analyst → Validator (critic) → Analyst retry loop
    """

    def __init__(self, settings):
        self.settings = settings
        self.tool_registry = ToolRegistry(settings)

    # ------------------------------------------------------------------ #
    #  Agent Definitions                                                   #
    # ------------------------------------------------------------------ #

    def _make_planner(self) -> Agent:
        """
        Planner Agent — creates a complete step-by-step execution plan
        before any data retrieval begins. Produces structured plans that
        the Analyst can execute sequentially.

        Uses a smaller, cheaper model — planning is reasoning-heavy
        but doesn't need the full power of GPT-4o.
        """
        return Agent(
            role="Data Query Planner",
            goal=(
                "Decompose complex SriniMart stakeholder questions into a clear, "
                "ordered sequence of data retrieval and analysis steps. Each step "
                "should be independently executable by the Analyst agent."
            ),
            backstory=(
                "You are a senior data architect who has mapped SriniMart's entire "
                "data warehouse schema. You know which tables join to which, where "
                "the business calendar quirks live, and how to break any stakeholder "
                "question into a set of precise, answerable sub-queries."
            ),
            tools=[],  # Planner reasons only — no tool access
            verbose=False,
            llm=self.settings.planner_llm,
        )

    def _make_analyst(self) -> Agent:
        """
        Analyst Agent — the primary data executor. Runs schema discovery,
        builds and validates SQL queries, calls APIs, and aggregates results.
        Has access to the full tool set.
        """
        return Agent(
            role="SriniMart Data Analyst",
            goal=(
                "Execute data queries accurately against SriniMart's warehouse. "
                "Always discover the schema before building SQL. Validate queries "
                "before execution. Return structured, sourced results."
            ),
            backstory=(
                "You are a precise, methodical data analyst who has worked with "
                "SriniMart's retail data for years. You never guess column names — "
                "you always look them up first. You know that 'revenue' means "
                "sales.total_amount, that regions map to stores.sales_region, "
                "and that fiscal Q3 runs July through September."
            ),
            tools=self.tool_registry.get_analyst_tools(),
            verbose=True,
            llm=self.settings.analyst_llm,
        )

    def _make_validator(self) -> Agent:
        """
        Validator Agent — quality gate between raw results and the final response.
        Checks for data integrity issues, hallucinated values, and scope violations
        before the Writer formats the answer for the stakeholder.
        """
        return Agent(
            role="Data Quality Validator",
            goal=(
                "Review query results for accuracy, completeness, and access-scope "
                "compliance. Flag any values that look like hallucinations, "
                "data outliers, or permission boundary violations."
            ),
            backstory=(
                "You are SriniMart's data quality lead. You've caught wrong answers "
                "before they reached board decks. You cross-reference results against "
                "known benchmarks, check that row counts are plausible, and ensure "
                "no PII or out-of-scope data has leaked into the response."
            ),
            tools=self.tool_registry.get_validator_tools(),
            verbose=False,
            llm=self.settings.analyst_llm,
        )

    def _make_writer(self) -> Agent:
        """
        Writer Agent — formats validated results into clear, business-friendly
        natural language. Adds context, highlights key insights, and structures
        the response appropriately for the stakeholder's role.
        """
        return Agent(
            role="Business Response Writer",
            goal=(
                "Transform validated data results into clear, concise answers "
                "that SriniMart stakeholders — not just data analysts — can act on. "
                "Lead with the key insight, then support with data."
            ),
            backstory=(
                "You communicate data insights to executives, regional managers, "
                "and store operators daily. You know that a store manager wants to "
                "know if their numbers are up or down — not the SQL that produced it."
            ),
            tools=[],  # Writer synthesizes — no tool access needed
            verbose=False,
            llm=self.settings.analyst_llm,
        )

    # ------------------------------------------------------------------ #
    #  Execution Modes                                                     #
    # ------------------------------------------------------------------ #

    def run_react(self, query: str, rag_context: dict,
                  user_role: str, max_iterations: int) -> dict:
        """
        ReAct mode — single Analyst agent in a think→act→observe loop.
        Used for ~80% of SriniMart queries: simple lookups and single-metric questions.
        Fast, cheap, sufficient for most use cases.
        """
        analyst = self._make_analyst()
        task = Task(
            description=self._build_analyst_prompt(query, rag_context, user_role),
            agent=analyst,
            expected_output="A structured answer with data values and source tables cited.",
        )
        crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        return self._parse_result(result)

    def run_plan_execute(self, query: str, rag_context: dict, user_role: str) -> dict:
        """
        Plan-and-Execute mode — Planner creates a full step-by-step plan,
        Analyst executes each step, Validator checks the result,
        Writer produces the final stakeholder response.

        Used for complex queries: quarterly reviews, cross-region comparisons,
        multi-source reports.
        """
        planner  = self._make_planner()
        analyst  = self._make_analyst()
        validator = self._make_validator()
        writer   = self._make_writer()

        plan_task = Task(
            description=f"Create a step-by-step plan to answer this SriniMart query: {query}\n\nContext:\n{rag_context}",
            agent=planner,
            expected_output="An ordered list of 3-8 concrete analysis steps.",
        )
        execute_task = Task(
            description=self._build_analyst_prompt(query, rag_context, user_role),
            agent=analyst,
            expected_output="Raw data results with source citations.",
            context=[plan_task],
        )
        validate_task = Task(
            description="Validate the query results for accuracy, completeness, and access-scope compliance.",
            agent=validator,
            expected_output="Validation report: PASS or FAIL with specific findings.",
            context=[execute_task],
        )
        write_task = Task(
            description="Write a clear, business-friendly response for the stakeholder based on validated results.",
            agent=writer,
            expected_output="Final natural language answer with key insight leading.",
            context=[validate_task],
        )

        crew = Crew(
            agents=[planner, analyst, validator, writer],
            tasks=[plan_task, execute_task, validate_task, write_task],
            process=Process.sequential,
            verbose=False,
        )
        result = crew.kickoff()
        return self._parse_result(result)

    def run_reflexion(self, query: str, rag_context: dict,
                      user_role: str, max_cycles: int = 3) -> dict:
        """
        Reflexion mode — Analyst executes, Validator critiques and identifies
        what failed, Analyst retries with the reflection insight.

        Used for investigative queries: root-cause analysis, anomaly investigation.
        Each reflection cycle transforms a dead end into a better hypothesis.
        Capped at max_cycles (default 3) — diminishing returns beyond that.
        """
        analyst   = self._make_analyst()
        validator = self._make_validator()
        all_tool_calls, all_results = [], []

        for cycle in range(max_cycles):
            execute_task = Task(
                description=self._build_reflexion_prompt(query, rag_context, user_role, cycle, all_results),
                agent=analyst,
                expected_output="Data results and a summary of what was found or ruled out.",
            )
            reflect_task = Task(
                description=(
                    "Critique the analyst's results. Identify: what hypothesis was tested, "
                    "what was ruled out, and what the most promising next hypothesis is. "
                    "If a root cause has been identified with confidence, say RESOLVED."
                ),
                agent=validator,
                expected_output="Reflection: what failed, what was learned, next hypothesis. Or RESOLVED.",
                context=[execute_task],
            )
            crew = Crew(agents=[analyst, validator], tasks=[execute_task, reflect_task], process=Process.sequential)
            result = crew.kickoff()
            parsed = self._parse_result(result)
            all_tool_calls.extend(parsed.get("tool_calls", []))
            all_results.append(parsed)

            if "RESOLVED" in parsed.get("answer", ""):
                break

        return {
            "tool_calls": all_tool_calls,
            "tool_results": [r.get("tool_results", []) for r in all_results],
            "answer": all_results[-1].get("answer", ""),
        }

    # ------------------------------------------------------------------ #
    #  Prompt Builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_analyst_prompt(self, query: str, rag_context: dict, user_role: str) -> str:
        glossary = rag_context.get("glossary_terms", {})
        schema   = rag_context.get("relevant_tables", [])
        examples = rag_context.get("example_queries", [])

        return f"""
You are answering this SriniMart stakeholder question:
"{query}"

User role: {user_role}
Only return data this role is permitted to see.

Relevant schema:
{schema}

Business glossary (use these mappings — do not guess column names):
{glossary}

Example queries for reference:
{examples}

Steps:
1. Call schema_discovery to confirm table and column names.
2. Build a SQL query using only confirmed column names.
3. Call validate_sql before executing.
4. Execute the query.
5. If results need aggregation or math, use the calculate tool — do not compute in your head.
6. Return the answer with the tables and columns you used as citations.
"""

    def _build_reflexion_prompt(self, query: str, rag_context: dict,
                                 user_role: str, cycle: int, prior_results: list) -> str:
        prior_summary = "\n".join(
            f"Cycle {i+1}: {r.get('answer', '')[:200]}"
            for i, r in enumerate(prior_results)
        ) if prior_results else "No prior attempts."

        return f"""
Investigative query (cycle {cycle + 1}):
"{query}"

User role: {user_role}

Prior investigation results:
{prior_summary}

Based on what has been ruled out, form a new hypothesis and test it.
Be specific about what you are checking and why.

{self._build_analyst_prompt(query, rag_context, user_role)}
"""

    def _parse_result(self, crew_result) -> dict:
        """Normalize CrewAI output into a consistent dict."""
        return {
            "answer": str(crew_result),
            "tool_calls": [],   # populated by tool_registry callbacks
            "tool_results": [],
        }
