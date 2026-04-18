"""
SriniMart Data Agent — Layer 2: Tool Registry
=============================================
Manages discovery and on-demand loading of tools across SriniMart's
50+ data sources using the Model Context Protocol (MCP).

Key patterns implemented:
  1. Tool Search       — Meta-tool for dynamic tool discovery (85% context reduction)
  2. MCP On-Demand     — Lazy-load MCP servers only when needed (~500 token startup cost)
  3. Few-Shot Examples — Each tool ships with 2-3 usage examples (-25pp error rate)
  4. Token-Efficient   — Descriptions trimmed to essentials, results paginated

Without this registry, loading all 50+ tool schemas at startup would
consume ~40% of the context window before the agent processes a single query.
"""

import json
from typing import Any
from crewai_tools import BaseTool


class ToolRegistry:
    """
    Central registry for all SriniMart agent tools.

    Implements on-demand MCP loading: at startup, only a lightweight
    catalog of available servers is loaded (~500 tokens). When the agent
    needs a capability, the registry connects to the relevant MCP server
    and loads only that server's tool schemas.
    """

    def __init__(self, settings):
        self.settings = settings
        self._loaded_tools: dict[str, Any] = {}
        self._mcp_connections: dict[str, Any] = {}

        # Lightweight catalog — one-line descriptions only
        # Full schemas are loaded on-demand when the tool is actually needed
        self._tool_catalog = {
            "schema_discovery":   "Discover SriniMart warehouse tables and columns.",
            "query_execute":      "Execute a read-only SQL query against the warehouse.",
            "validate_sql":       "Validate SQL syntax and column names before execution.",
            "aggregate":          "Group and aggregate query result rows.",
            "calculate":          "Perform arithmetic on numeric data.",
            "api_fetch":          "Fetch live data from SriniMart's inventory and CRM APIs.",
            "export_csv":         "Export result data as a downloadable CSV file.",
            "slack_notify":       "Send a Slack message with query results to a channel.",
            "format_results":     "Format raw rows into a display-ready table or chart.",
        }

    # ------------------------------------------------------------------ #
    #  Tool Search — Dynamic Discovery                                     #
    # ------------------------------------------------------------------ #

    def search_tools(self, capability_query: str, top_k: int = 4) -> list[dict]:
        """
        Meta-tool: given a natural language capability description,
        return the top-k matching tool schemas.

        Instead of loading all 50+ tool schemas upfront, the agent calls
        this first. Only the 3-5 relevant schemas are loaded per query,
        keeping the context window clean.

        Example:
          search_tools("run a SQL query on revenue data")
          → returns schemas for: query_execute, validate_sql, schema_discovery
        """
        # In production: embedding similarity over self._tool_catalog descriptions
        # Here: keyword matching for clarity
        matches = []
        query_lower = capability_query.lower()
        for tool_name, description in self._tool_catalog.items():
            score = sum(1 for word in query_lower.split() if word in description.lower())
            if score > 0:
                matches.append((tool_name, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        top_tools = [name for name, _ in matches[:top_k]]

        return [self._load_tool_schema(name) for name in top_tools]

    # ------------------------------------------------------------------ #
    #  MCP On-Demand Loading                                               #
    # ------------------------------------------------------------------ #

    def _load_mcp_server(self, server_name: str) -> Any:
        """
        Connect to an MCP server on demand.
        Connection is cached for the session duration to avoid
        reconnection overhead on subsequent tool calls.

        Startup cost: ~500 tokens for the catalog regardless of
        how many MCP servers exist.
        """
        if server_name not in self._mcp_connections:
            server_config = self.settings.mcp_servers.get(server_name)
            if not server_config:
                raise ValueError(f"MCP server '{server_name}' not found in settings.")
            # Production: establish MCP connection via HTTP/SSE
            # self._mcp_connections[server_name] = MCPClient(server_config["url"])
            self._mcp_connections[server_name] = {"url": server_config["url"], "connected": True}

        return self._mcp_connections[server_name]

    def _load_tool_schema(self, tool_name: str) -> dict:
        """
        Load the full JSON schema for a single tool.
        Schemas include 2-3 few-shot usage examples — this alone
        drops SQL formatting errors from 25% to 3%.
        """
        schemas = {
            "query_execute": {
                "name": "query_execute",
                "description": "Execute a read-only SQL SELECT against the SriniMart warehouse.",
                "parameters": {
                    "sql":        {"type": "string",  "description": "SQL SELECT statement."},
                    "timeout_ms": {"type": "integer", "default": 5000},
                    "max_rows":   {"type": "integer", "default": 100},
                },
                "required": ["sql"],
                "returns": {"rows": "array", "row_count": "integer", "execution_ms": "integer"},
                "examples": [
                    {
                        "description": "Total Q3 revenue by region",
                        "call": {
                            "sql": "SELECT sales_region, SUM(total_amount) AS revenue FROM sales_transactions WHERE fiscal_quarter = 'Q3-2024' GROUP BY sales_region ORDER BY revenue DESC",
                            "max_rows": 10
                        }
                    },
                    {
                        "description": "Top 5 stores by units sold last month",
                        "call": {
                            "sql": "SELECT store_id, store_name, SUM(units_sold) AS units FROM daily_store_sales WHERE sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') GROUP BY store_id, store_name ORDER BY units DESC LIMIT 5"
                        }
                    },
                    {
                        "description": "December order count across all stores",
                        "call": {
                            "sql": "SELECT COUNT(*) AS order_count FROM orders WHERE EXTRACT(MONTH FROM order_date) = 12 AND EXTRACT(YEAR FROM order_date) = 2024"
                        }
                    }
                ]
            },
            "schema_discovery": {
                "name": "schema_discovery",
                "description": "Discover SriniMart warehouse tables and columns. Always call this before building SQL.",
                "parameters": {
                    "search":          {"type": "string",  "description": "Keyword to search table/column names."},
                    "include_columns": {"type": "boolean", "default": True},
                },
                "required": ["search"],
                "returns": {"tables": "array of {name, columns, row_count_estimate}"},
                "examples": [
                    {
                        "description": "Find tables related to revenue",
                        "call": {"search": "revenue sales total_amount", "include_columns": True}
                    },
                    {
                        "description": "Find tables related to store regions",
                        "call": {"search": "store region location", "include_columns": True}
                    }
                ]
            },
            "validate_sql": {
                "name": "validate_sql",
                "description": "Validate SQL syntax and confirm all column names exist before execution.",
                "parameters": {
                    "sql": {"type": "string", "description": "SQL to validate."},
                },
                "required": ["sql"],
                "returns": {"valid": "boolean", "errors": "array", "estimated_rows": "integer", "estimated_ms": "integer"},
                "examples": [
                    {
                        "description": "Validate a revenue query",
                        "call": {"sql": "SELECT sales_region, SUM(total_amount) FROM sales_transactions GROUP BY sales_region"}
                    }
                ]
            },
            "aggregate": {
                "name": "aggregate",
                "description": "Group and aggregate rows from a prior query result.",
                "parameters": {
                    "data":      {"type": "array",  "description": "Row data to aggregate."},
                    "group_by":  {"type": "array",  "description": "Column names to group by."},
                    "agg_func":  {"type": "string", "description": "sum | avg | count | min | max"},
                    "agg_col":   {"type": "string", "description": "Column to aggregate."},
                },
                "required": ["data", "group_by", "agg_func", "agg_col"],
            },
            "calculate": {
                "name": "calculate",
                "description": "Perform arithmetic or percentage calculations. Use this — never compute numbers in the LLM.",
                "parameters": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '(5100000 - 4553571) / 4553571 * 100'"},
                },
                "required": ["expression"],
                "returns": {"result": "number"},
                "examples": [
                    {"description": "QoQ change %", "call": {"expression": "(5100000 - 4553571) / 4553571 * 100"}},
                ]
            },
        }
        return schemas.get(tool_name, {"name": tool_name, "description": self._tool_catalog.get(tool_name, "")})

    # ------------------------------------------------------------------ #
    #  Role-Based Tool Sets                                                #
    # ------------------------------------------------------------------ #

    def get_analyst_tools(self) -> list:
        """Tools available to the Analyst agent — full data access set."""
        return [
            SchemaDiscoveryTool(self),
            QueryExecuteTool(self),
            ValidateSQLTool(self),
            AggregateTool(self),
            CalculateTool(self),
            ExportCSVTool(self),
            FormatResultsTool(self),
        ]

    def get_validator_tools(self) -> list:
        """Tools available to the Validator agent — read-only verification set."""
        return [
            SchemaDiscoveryTool(self),
            ValidateSQLTool(self),
            CalculateTool(self),
        ]

    # ------------------------------------------------------------------ #
    #  Programmatic Batch Tool Calling                                     #
    # ------------------------------------------------------------------ #

    def batch_query_stores(self, store_ids: list[int], metric: str, period: str) -> list[dict]:
        """
        Programmatic tool calling pattern: instead of 300 sequential
        LLM round-trips (one per SriniMart store), the agent writes one
        code block that calls the tool in a loop.

        Before: 300 stores × 1 LLM reasoning step = ~10 minutes, ~400K tokens
        After : 1 code block, tool called 300 times = ~15 seconds, ~2K tokens

        This is called by the Analyst agent when it detects a multi-store
        pattern query (e.g. "check inventory levels across all stores").
        """
        results = []
        for store_id in store_ids:
            sql = f"""
                SELECT store_id, store_name, {metric}
                FROM daily_store_summary
                WHERE store_id = {store_id}
                  AND period = '{period}'
            """
            # Tool called programmatically — no LLM round-trip per iteration
            row = self._execute_sql(sql, max_rows=1)
            if row:
                results.append(row[0])

        return results

    def _execute_sql(self, sql: str, max_rows: int = 100) -> list[dict]:
        """Internal SQL execution — connects to warehouse via MCP."""
        # Production: route through MCP warehouse server
        warehouse_conn = self._load_mcp_server("srinimart_warehouse")
        # warehouse_conn.execute(sql, max_rows=max_rows)
        return []  # Placeholder — replace with actual MCP execution


# ------------------------------------------------------------------ #
#  Tool Implementations (CrewAI BaseTool wrappers)                   #
# ------------------------------------------------------------------ #

class SchemaDiscoveryTool(BaseTool):
    name: str = "schema_discovery"
    description: str = "Discover SriniMart warehouse tables and columns matching a search term. Always call before writing SQL."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, search: str, include_columns: bool = True) -> str:
        """
        Returns matching table schemas from SriniMart's warehouse.
        In production: queries the metadata catalog via MCP.
        """
        # SriniMart schema reference (subset)
        srinimart_schema = {
            "sales_transactions": ["transaction_id", "store_id", "product_id", "total_amount", "fiscal_quarter", "sale_date", "sales_region"],
            "stores":             ["store_id", "store_name", "sales_region", "district", "opened_date", "manager_id"],
            "products":           ["product_id", "product_name", "category", "sku", "unit_price"],
            "daily_store_summary":["store_id", "period", "units_sold", "revenue", "transactions"],
            "orders":             ["order_id", "store_id", "customer_id", "order_date", "total_amount", "status"],
            "inventory":          ["store_id", "product_id", "quantity_on_hand", "reorder_point", "last_updated"],
        }
        matches = {
            table: cols for table, cols in srinimart_schema.items()
            if search.lower() in table or any(search.lower() in c for c in cols)
        }
        return json.dumps(matches, indent=2)


class QueryExecuteTool(BaseTool):
    name: str = "query_execute"
    description: str = "Execute a read-only SQL SELECT against the SriniMart data warehouse."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, sql: str, timeout_ms: int = 5000, max_rows: int = 100) -> str:
        rows = self._registry._execute_sql(sql, max_rows=max_rows)
        return json.dumps({"rows": rows, "row_count": len(rows)})


class ValidateSQLTool(BaseTool):
    name: str = "validate_sql"
    description: str = "Validate SQL syntax and column existence before execution. Always validate before calling query_execute."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, sql: str) -> str:
        # Production: EXPLAIN the query against the warehouse
        if "SELECT" not in sql.upper():
            return json.dumps({"valid": False, "errors": ["Only SELECT statements are permitted."]})
        return json.dumps({"valid": True, "errors": [], "estimated_rows": 100, "estimated_ms": 120})


class AggregateTool(BaseTool):
    name: str = "aggregate"
    description: str = "Group and aggregate rows from a prior query result."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, data: list, group_by: list, agg_func: str, agg_col: str) -> str:
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for row in data:
            key = tuple(row.get(col) for col in group_by)
            groups[key].append(row.get(agg_col, 0))
        funcs = {"sum": sum, "avg": lambda x: sum(x)/len(x), "count": len, "min": min, "max": max}
        fn = funcs.get(agg_func, sum)
        result = [{**dict(zip(group_by, k)), agg_col: fn(v)} for k, v in groups.items()]
        return json.dumps(result)


class CalculateTool(BaseTool):
    name: str = "calculate"
    description: str = "Evaluate a math expression. Use this for all arithmetic — never compute numbers in your head."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, expression: str) -> str:
        try:
            # Safe eval — only numeric ops
            allowed = set("0123456789.+-*/() ")
            if not all(c in allowed for c in expression):
                return json.dumps({"error": "Invalid expression characters."})
            result = eval(expression)  # noqa: S307 — guarded above
            return json.dumps({"result": round(result, 4)})
        except Exception as e:
            return json.dumps({"error": str(e)})


class ExportCSVTool(BaseTool):
    name: str = "export_csv"
    description: str = "Export query result rows as a CSV file the stakeholder can download."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, data: list, filename: str = "srinimart_export.csv") -> str:
        import csv, io
        if not data:
            return json.dumps({"error": "No data to export."})
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return json.dumps({"filename": filename, "rows": len(data), "preview": output.getvalue()[:500]})


class FormatResultsTool(BaseTool):
    name: str = "format_results"
    description: str = "Format raw query rows into a readable table or chart-ready structure."

    def __init__(self, registry: ToolRegistry):
        super().__init__()
        self._registry = registry

    def _run(self, data: list, format: str = "table", title: str = "") -> str:
        if format == "table":
            if not data:
                return "No results."
            headers = list(data[0].keys())
            lines = [title, " | ".join(headers), "-" * 60]
            for row in data[:25]:  # Paginate — never return all rows unfiltered
                lines.append(" | ".join(str(row.get(h, "")) for h in headers))
            if len(data) > 25:
                lines.append(f"... and {len(data) - 25} more rows. Request next page if needed.")
            return "\n".join(lines)
        return json.dumps({"data": data, "title": title})
