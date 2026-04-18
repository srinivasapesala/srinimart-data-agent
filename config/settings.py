"""
SriniMart Data Agent — Configuration
=====================================
Centralizes all environment-dependent settings.
Copy this file to settings_local.py and fill in your credentials.
Never commit API keys to version control.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """
    All configuration for the SriniMart Data Agent.
    Loaded from environment variables with sensible defaults.
    """

    # ------------------------------------------------------------------ #
    #  LLM Configuration                                                   #
    # ------------------------------------------------------------------ #

    # Azure OpenAI — primary LLM for the Analyst and Writer agents
    azure_openai_endpoint:   str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""))
    azure_openai_api_key:    str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY", ""))
    azure_openai_deployment: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"))

    # Planner uses a smaller model to keep planning cost low
    planner_deployment: str = field(default_factory=lambda: os.getenv("PLANNER_DEPLOYMENT", "gpt-4o-mini"))

    @property
    def analyst_llm(self):
        """LLM instance for Analyst, Validator, and Writer agents."""
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            deployment_name=self.azure_openai_deployment,
            temperature=0,       # Deterministic outputs for data queries
            max_tokens=2048,
        )

    @property
    def planner_llm(self):
        """Smaller, cheaper LLM for the Planner agent."""
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            deployment_name=self.planner_deployment,
            temperature=0,
            max_tokens=1024,
        )

    # ------------------------------------------------------------------ #
    #  Vector Store / RAG                                                  #
    # ------------------------------------------------------------------ #

    # FAISS index path for local schema embeddings
    faiss_index_path: str = field(default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/schema_index.faiss"))

    # Azure Cognitive Search for hybrid retrieval (production)
    azure_search_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_ENDPOINT", ""))
    azure_search_key:      str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_KEY", ""))
    azure_search_index:    str = field(default_factory=lambda: os.getenv("AZURE_SEARCH_INDEX", "srinimart-schema"))

    # ------------------------------------------------------------------ #
    #  MCP Server Registry                                                 #
    # ------------------------------------------------------------------ #

    # Each data source is wrapped as an MCP server.
    # The tool registry loads these on-demand — only when actually needed.
    mcp_servers: dict = field(default_factory=lambda: {
        "srinimart_warehouse": {
            "url":         os.getenv("MCP_WAREHOUSE_URL", "http://localhost:8001/mcp"),
            "description": "SriniMart Azure SQL Data Warehouse — primary analytics store",
            "auth":        os.getenv("MCP_WAREHOUSE_TOKEN", ""),
        },
        "srinimart_inventory_api": {
            "url":         os.getenv("MCP_INVENTORY_URL", "http://localhost:8002/mcp"),
            "description": "SriniMart real-time inventory API — live stock levels",
            "auth":        os.getenv("MCP_INVENTORY_TOKEN", ""),
        },
        "srinimart_crm": {
            "url":         os.getenv("MCP_CRM_URL", "http://localhost:8003/mcp"),
            "description": "SriniMart CRM — customer segments and loyalty data (PII-masked)",
            "auth":        os.getenv("MCP_CRM_TOKEN", ""),
        },
        "srinimart_finance": {
            "url":         os.getenv("MCP_FINANCE_URL", "http://localhost:8004/mcp"),
            "description": "SriniMart financial systems — revenue targets, budgets (finance role only)",
            "auth":        os.getenv("MCP_FINANCE_TOKEN", ""),
        },
    })

    # ------------------------------------------------------------------ #
    #  Warehouse Connection                                                #
    # ------------------------------------------------------------------ #

    warehouse_connection_string: str = field(
        default_factory=lambda: os.getenv(
            "SRINIMART_WAREHOUSE_CONN",
            "mssql+pyodbc:///?odbc_connect=..."  # Replace with actual connection string
        )
    )

    # ------------------------------------------------------------------ #
    #  Agent Behaviour                                                     #
    # ------------------------------------------------------------------ #

    max_react_iterations:     int   = 8      # Circuit breaker for ReAct loops
    max_reflexion_cycles:     int   = 3      # Max self-correction rounds
    max_plan_steps:           int   = 10     # Max steps in a Plan-and-Execute plan
    query_timeout_seconds:    int   = 30     # Hard timeout per query
    default_page_size:        int   = 25     # Rows returned per tool call
    token_budget_per_query:   int   = 8_000  # Soft token budget — alert if exceeded

    # ------------------------------------------------------------------ #
    #  Observability                                                       #
    # ------------------------------------------------------------------ #

    langsmith_api_key:   str  = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project:   str  = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "srinimart-data-agent"))
    enable_tracing:      bool = field(default_factory=lambda: os.getenv("ENABLE_TRACING", "true").lower() == "true")

    prometheus_port:     int  = field(default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090")))
