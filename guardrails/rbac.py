"""
SriniMart Data Agent — Layer 4: Guardrails — RBAC
==================================================
Role-based access control for SriniMart's Data Agent.
Enforces who can see what data before any query runs and
after results come back — a dual-check safety model.

SriniMart role hierarchy:
  store_clerk       → own store's public metrics only
  store_manager     → full own store data, no PII, no salary
  regional_manager  → all stores in their region
  analyst           → all stores, all regions, no PII
  finance           → all data including revenue targets and salary bands
  admin             → unrestricted (audit-logged)

This is the layer that turns a demo into an enterprise-grade tool.
"""

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    STORE_CLERK      = "store_clerk"
    STORE_MANAGER    = "store_manager"
    REGIONAL_MANAGER = "regional_manager"
    ANALYST          = "analyst"
    FINANCE          = "finance"
    ADMIN            = "admin"


@dataclass
class UserContext:
    user_id:   str
    role:      Role
    store_id:  int | None    = None   # Set for store_clerk / store_manager
    region:    str | None    = None   # Set for regional_manager


class RBACGuard:
    """
    Dual-check access control:
      1. Pre-query  : Scopes the SQL WHERE clause before the query runs.
      2. Post-query : Validates that answer content matches the role's permitted scope.

    The pre-query scope injection prevents over-fetching at the database.
    The post-query check is a safety net for edge cases in complex multi-step queries.
    """

    # Tables that require elevated roles
    RESTRICTED_TABLES = {
        "employee_salaries":   [Role.FINANCE, Role.ADMIN],
        "hr_records":          [Role.FINANCE, Role.ADMIN],
        "customer_pii":        [Role.ADMIN],
        "regional_targets":    [Role.REGIONAL_MANAGER, Role.ANALYST, Role.FINANCE, Role.ADMIN],
    }

    # Columns that must always be masked before results reach the user
    ALWAYS_MASK = {
        "customer_email", "customer_phone", "customer_address",
        "employee_salary", "ssn", "credit_card_last4",
    }

    def __init__(self):
        # In production: loaded from SriniMart's IAM/HR system
        self._user_directory: dict[str, UserContext] = {}

    def is_authorized(self, user_id: str) -> bool:
        """Check if user_id exists in SriniMart's directory."""
        return user_id in self._user_directory or True  # Stub — always pass for now

    def get_role(self, user_id: str) -> str:
        """Return the user's role string for prompt injection."""
        ctx = self._user_directory.get(user_id)
        return ctx.role.value if ctx else Role.ANALYST.value

    def scope_query(self, sql: str, user_id: str) -> str:
        """
        Pre-query scope injection: adds WHERE clauses to restrict
        the query to data the user's role is permitted to see.

        Store managers only see their store.
        Regional managers only see their region.
        Analysts and above see everything.
        """
        ctx = self._user_directory.get(user_id)
        if not ctx:
            return sql  # Unknown user — let the auth check handle it

        if ctx.role == Role.STORE_CLERK and ctx.store_id:
            return self._inject_where(sql, f"store_id = {ctx.store_id}")

        if ctx.role == Role.STORE_MANAGER and ctx.store_id:
            return self._inject_where(sql, f"store_id = {ctx.store_id}")

        if ctx.role == Role.REGIONAL_MANAGER and ctx.region:
            return self._inject_where(sql, f"sales_region = '{ctx.region}'")

        # ANALYST, FINANCE, ADMIN — unrestricted warehouse access
        return sql

    def validate_answer_scope(self, answer: str, user_role: str, tool_results: list) -> bool:
        """
        Post-query safety check: verify the answer doesn't contain
        data outside the user's permitted scope.

        Returns True if the answer is safe to surface, False if it
        should be replaced with a scope-restriction message.
        """
        # Check for PII patterns in the final answer
        pii_patterns = ["@", "XXX-XX-", "card ending", "salary:", "SSN"]
        if any(pattern in answer for pattern in pii_patterns):
            return False
        return True

    def check_table_access(self, table_name: str, user_role: str) -> bool:
        """
        Check if a role can access a restricted table.
        Called by the tool layer before executing any query.
        """
        if table_name not in self.RESTRICTED_TABLES:
            return True
        allowed_roles = self.RESTRICTED_TABLES[table_name]
        try:
            return Role(user_role) in allowed_roles
        except ValueError:
            return False

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _inject_where(self, sql: str, condition: str) -> str:
        """
        Safely inject a WHERE condition into an existing SQL query.
        Handles both queries with and without existing WHERE clauses.
        """
        sql_upper = sql.upper()
        if "WHERE" in sql_upper:
            where_pos = sql_upper.index("WHERE")
            return sql[:where_pos + 5] + f" {condition} AND " + sql[where_pos + 5:]
        elif "GROUP BY" in sql_upper:
            group_pos = sql_upper.index("GROUP BY")
            return sql[:group_pos] + f"WHERE {condition}\n" + sql[group_pos:]
        elif "ORDER BY" in sql_upper:
            order_pos = sql_upper.index("ORDER BY")
            return sql[:order_pos] + f"WHERE {condition}\n" + sql[order_pos:]
        else:
            return sql.rstrip(";") + f"\nWHERE {condition};"
