"""
SriniMart Business Glossary
============================
Maps business language to exact warehouse column names and values.
Injected into every agent prompt so the LLM uses real column names
instead of hallucinating plausible-sounding ones.

This glossary is the difference between an agent that works
and one that invents column names with 25% failure rate.
"""

SRINIMART_GLOSSARY: dict = {

    # Revenue & financial terms
    "revenue":          "sales_transactions.total_amount  (SUM for aggregates)",
    "sales":            "sales_transactions.total_amount",
    "total sales":      "SUM(sales_transactions.total_amount)",
    "gross revenue":    "sales_transactions.total_amount  (no deductions applied at this layer)",
    "order value":      "orders.total_amount",
    "average order value": "AVG(orders.total_amount)",

    # Time & calendar terms
    "Q1":               "fiscal_quarter = 'Q1-{year}'  (January – March)",
    "Q2":               "fiscal_quarter = 'Q2-{year}'  (April – June)",
    "Q3":               "fiscal_quarter = 'Q3-{year}'  (July – September)",
    "Q4":               "fiscal_quarter = 'Q4-{year}'  (October – December)",
    "last quarter":     "fiscal_quarter = prior quarter relative to CURRENT_DATE",
    "last month":       "sale_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
    "year to date":     "sale_date >= DATE_TRUNC('year', CURRENT_DATE)",

    # Geography
    "region":           "stores.sales_region  (values: North, South, East, West, Central, Southwest)",
    "territory":        "stores.sales_region",
    "area":             "stores.sales_region",
    "district":         "stores.district  (sub-region grouping below sales_region)",
    "Southwest":        "stores.sales_region = 'Southwest'",
    "North":            "stores.sales_region = 'North'",
    "South":            "stores.sales_region = 'South'",
    "East":             "stores.sales_region = 'East'",
    "West":             "stores.sales_region = 'West'",
    "Central":          "stores.sales_region = 'Central'",

    # Product terms
    "top products":     "ORDER BY SUM(total_amount) DESC LIMIT N",
    "category":         "products.category  (values: Electronics, Apparel, Grocery, Home, Sports)",
    "SKU":              "products.sku",
    "item":             "products.product_name",

    # Store & operations
    "store":            "stores.store_id / stores.store_name",
    "location":         "stores.store_name + stores.sales_region",
    "all stores":       "no WHERE filter on store_id  (300 stores total)",
    "manager":          "employees.employee_id via stores.manager_id",

    # Inventory
    "in stock":         "inventory.quantity_on_hand > 0",
    "out of stock":     "inventory.quantity_on_hand = 0",
    "low stock":        "inventory.quantity_on_hand < inventory.reorder_point",
    "inventory level":  "inventory.quantity_on_hand",

    # Order terms
    "order count":      "COUNT(*) FROM orders",
    "orders placed":    "COUNT(order_id) FROM orders",
    "cancelled orders": "orders.status = 'cancelled'",
    "returns":          "orders.status = 'returned'",

    # Role-scoped access hints
    "my store":         "FILTER: store_id = {user_store_id}  (store manager scope)",
    "my region":        "FILTER: sales_region = {user_region}  (regional manager scope)",
}
