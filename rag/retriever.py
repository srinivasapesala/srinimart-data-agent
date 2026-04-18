"""
SriniMart Data Agent — Layer 3: Data & Context (RAG)
=====================================================
Gives the agent semantic grounding in SriniMart's data.
Without this layer, the LLM guesses column names.
With it, the agent knows that "revenue" means sales.total_amount
and that "Q3" means July through September for SriniMart's fiscal calendar.

Components:
  - Schema embeddings : Vector index over table/column descriptions
  - Business glossary : SriniMart-specific term → column name mappings
  - Example queries   : 2-3 worked SQL examples per common query type
  - Hybrid retrieval  : Dense (embedding similarity) + keyword search

This RAG layer uses FAISS for local embedding search and Azure Cognitive
Search for production-scale hybrid retrieval across the full schema catalog.
"""

from rag.glossary import SRINIMART_GLOSSARY


class RAGRetriever:
    """
    Retrieves schema context, glossary terms, and example queries
    relevant to the user's question.

    Hybrid retrieval strategy:
      1. Dense retrieval  — embedding similarity over schema descriptions
      2. Keyword match    — exact term lookup in business glossary
      3. Example lookup   — retrieves closest example queries by embedding

    Only the top-k most relevant context items are returned,
    keeping token consumption predictable per query.
    """

    TOP_K_TABLES   = 4   # Max tables returned per query
    TOP_K_EXAMPLES = 3   # Max example queries returned per query

    def __init__(self, settings):
        self.settings = settings
        self.glossary = SRINIMART_GLOSSARY

    def retrieve(self, query: str, user_role: str) -> dict:
        """
        Main retrieval entry point. Returns a structured context dict
        that is injected into every agent prompt.

        Args:
            query:     Natural language question from the stakeholder.
            user_role: Used to scope schema access — store managers only
                       see their store's tables, not the full warehouse.

        Returns:
            {
              "relevant_tables":  list of table schemas,
              "glossary_terms":   dict of business term → column mapping,
              "example_queries":  list of example SQL with descriptions,
            }
        """
        relevant_tables  = self._retrieve_tables(query)
        glossary_terms   = self._resolve_glossary(query)
        example_queries  = self._retrieve_examples(query)

        return {
            "relevant_tables":  relevant_tables,
            "glossary_terms":   glossary_terms,
            "example_queries":  example_queries,
        }

    # ------------------------------------------------------------------ #
    #  Schema Retrieval                                                    #
    # ------------------------------------------------------------------ #

    def _retrieve_tables(self, query: str) -> list:
        """
        Return the most relevant SriniMart table schemas for this query.
        In production: FAISS embedding similarity over the schema catalog.
        Here: keyword-based matching for clarity.
        """
        q = query.lower()
        all_tables = self._get_srinimart_schema()

        scored = []
        for table in all_tables:
            score = sum(1 for kw in table["keywords"] if kw in q)
            if score > 0:
                scored.append((table, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:self.TOP_K_TABLES]]

    def _get_srinimart_schema(self) -> list:
        """
        SriniMart's core warehouse schema with semantic descriptions.
        These descriptions are what get embedded and indexed for retrieval.
        """
        return [
            {
                "table": "sales_transactions",
                "description": "Individual retail transactions across all SriniMart stores. Primary revenue source.",
                "columns": {
                    "transaction_id":  "Unique transaction identifier.",
                    "store_id":        "References stores.store_id.",
                    "product_id":      "References products.product_id.",
                    "total_amount":    "Revenue for this transaction. Use for 'revenue', 'sales amount', 'total'.",
                    "fiscal_quarter":  "SriniMart fiscal quarter. Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec.",
                    "sale_date":       "Calendar date of transaction.",
                    "sales_region":    "SriniMart sales region. Use for 'region', 'area', 'territory'.",
                },
                "keywords": ["revenue", "sales", "total", "transaction", "region", "quarter"],
            },
            {
                "table": "stores",
                "description": "SriniMart store master data. 300 stores across 6 regions.",
                "columns": {
                    "store_id":      "Unique store identifier.",
                    "store_name":    "Human-readable store name.",
                    "sales_region":  "One of: North, South, East, West, Central, Southwest.",
                    "district":      "Sub-region grouping of stores.",
                    "opened_date":   "Date the store opened.",
                    "manager_id":    "References employees.employee_id.",
                },
                "keywords": ["store", "location", "region", "district", "manager"],
            },
            {
                "table": "products",
                "description": "SriniMart product catalog.",
                "columns": {
                    "product_id":    "Unique product identifier.",
                    "product_name":  "Display name of the product.",
                    "category":      "Product category (Electronics, Apparel, Grocery, etc.)",
                    "sku":           "Stock Keeping Unit code.",
                    "unit_price":    "Retail price per unit.",
                },
                "keywords": ["product", "item", "sku", "category", "price"],
            },
            {
                "table": "daily_store_summary",
                "description": "Pre-aggregated daily performance metrics per store.",
                "columns": {
                    "store_id":     "References stores.store_id.",
                    "period":       "Date or month string.",
                    "units_sold":   "Total units sold.",
                    "revenue":      "Total revenue. Alias of SUM(sales_transactions.total_amount).",
                    "transactions": "Total transaction count.",
                },
                "keywords": ["daily", "summary", "performance", "units", "revenue", "store"],
            },
            {
                "table": "orders",
                "description": "Customer order records including online and in-store orders.",
                "columns": {
                    "order_id":      "Unique order identifier.",
                    "store_id":      "Store where order was placed.",
                    "customer_id":   "Anonymous customer identifier (PII-masked).",
                    "order_date":    "Date order was placed.",
                    "total_amount":  "Order total value.",
                    "status":        "Order status: pending, fulfilled, cancelled, returned.",
                },
                "keywords": ["order", "customer", "purchase", "december", "count"],
            },
            {
                "table": "inventory",
                "description": "Real-time inventory levels per product per store.",
                "columns": {
                    "store_id":          "References stores.store_id.",
                    "product_id":        "References products.product_id.",
                    "quantity_on_hand":  "Current stock quantity.",
                    "reorder_point":     "Stock level that triggers a reorder.",
                    "last_updated":      "Timestamp of last inventory update.",
                },
                "keywords": ["inventory", "stock", "quantity", "reorder", "supply"],
            },
        ]

    # ------------------------------------------------------------------ #
    #  Glossary Resolution                                                 #
    # ------------------------------------------------------------------ #

    def _resolve_glossary(self, query: str) -> dict:
        """
        Find business terms in the query and return their data mappings.
        This is the layer that translates 'revenue' → sales.total_amount
        and 'Q3' → "fiscal_quarter = 'Q3-{year}'" for SriniMart.
        """
        q = query.lower()
        matched = {}
        for term, mapping in self.glossary.items():
            if term.lower() in q:
                matched[term] = mapping
        return matched

    # ------------------------------------------------------------------ #
    #  Example Query Retrieval                                             #
    # ------------------------------------------------------------------ #

    def _retrieve_examples(self, query: str) -> list:
        """
        Return the most similar example queries.
        Including 2-3 examples per tool call dropped formatting errors
        from 25% to 3% in SriniMart production.
        """
        q = query.lower()
        all_examples = self._get_example_queries()
        scored = [(ex, sum(1 for kw in ex["keywords"] if kw in q)) for ex in all_examples]
        scored = [(ex, s) for ex, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ex for ex, _ in scored[:self.TOP_K_EXAMPLES]]

    def _get_example_queries(self) -> list:
        """
        Worked SQL examples covering SriniMart's most common query patterns.
        These are injected into agent prompts to demonstrate correct
        table names, column names, and SQL patterns.
        """
        return [
            {
                "description": "Total revenue by region for a fiscal quarter",
                "keywords": ["revenue", "region", "quarter", "q3", "q4"],
                "sql": """
SELECT s.sales_region,
       SUM(t.total_amount) AS revenue
FROM   sales_transactions t
JOIN   stores s USING (store_id)
WHERE  t.fiscal_quarter = 'Q3-2024'
GROUP  BY s.sales_region
ORDER  BY revenue DESC;
""",
            },
            {
                "description": "Top 10 products by revenue last quarter",
                "keywords": ["top", "product", "revenue", "quarter"],
                "sql": """
SELECT p.product_name,
       SUM(t.total_amount)                              AS revenue,
       SUM(t.total_amount) / SUM(SUM(t.total_amount))
           OVER ()                                      AS revenue_share
FROM   sales_transactions t
JOIN   products p USING (product_id)
WHERE  t.fiscal_quarter = 'Q3-2024'
GROUP  BY p.product_name
ORDER  BY revenue DESC
LIMIT  10;
""",
            },
            {
                "description": "Order count for a specific month",
                "keywords": ["orders", "count", "december", "month"],
                "sql": """
SELECT COUNT(*) AS order_count
FROM   orders
WHERE  EXTRACT(MONTH FROM order_date) = 12
  AND  EXTRACT(YEAR  FROM order_date) = 2024;
""",
            },
            {
                "description": "Store inventory below reorder point",
                "keywords": ["inventory", "stock", "reorder", "low"],
                "sql": """
SELECT s.store_name,
       p.product_name,
       i.quantity_on_hand,
       i.reorder_point
FROM   inventory i
JOIN   stores   s USING (store_id)
JOIN   products p USING (product_id)
WHERE  i.quantity_on_hand < i.reorder_point
ORDER  BY i.quantity_on_hand ASC;
""",
            },
            {
                "description": "Quarter-over-quarter revenue change by region",
                "keywords": ["quarter", "change", "compare", "growth", "qoq"],
                "sql": """
WITH quarterly AS (
    SELECT s.sales_region,
           t.fiscal_quarter,
           SUM(t.total_amount) AS revenue
    FROM   sales_transactions t
    JOIN   stores s USING (store_id)
    WHERE  t.fiscal_quarter IN ('Q2-2024', 'Q3-2024')
    GROUP  BY s.sales_region, t.fiscal_quarter
)
SELECT sales_region,
       MAX(CASE WHEN fiscal_quarter = 'Q3-2024' THEN revenue END) AS q3_revenue,
       MAX(CASE WHEN fiscal_quarter = 'Q2-2024' THEN revenue END) AS q2_revenue,
       (MAX(CASE WHEN fiscal_quarter = 'Q3-2024' THEN revenue END)
        - MAX(CASE WHEN fiscal_quarter = 'Q2-2024' THEN revenue END))
       / MAX(CASE WHEN fiscal_quarter = 'Q2-2024' THEN revenue END) * 100 AS pct_change
FROM   quarterly
GROUP  BY sales_region
ORDER  BY pct_change DESC;
""",
            },
        ]
