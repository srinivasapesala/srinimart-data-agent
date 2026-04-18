"""
SriniMart Data Agent — Layer 4: Guardrails — PII Masking
=========================================================
Detects and redacts personally identifiable information (PII)
from agent answers before they are surfaced to the user.

PII categories handled:
  - Customer emails, phone numbers, addresses
  - Employee salary figures
  - Credit card numbers (last 4 digits)
  - Social Security Numbers

Applied as the final step before format_response —
a backstop that catches PII regardless of which query path produced it.
"""

import re


class PIIMasker:
    """
    Regex-based PII detection and redaction.

    In production: augmented with a dedicated PII detection model
    (e.g. Azure AI Language PII detection) for higher recall on
    free-text fields like customer notes and product descriptions.
    """

    # PII patterns — ordered from most to least specific
    PII_PATTERNS = [
        # Email addresses
        (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL REDACTED]"),

        # US phone numbers (various formats)
        (re.compile(r"\b(\+1[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"), "[PHONE REDACTED]"),

        # Social Security Numbers
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),

        # Credit card patterns (last-4 references)
        (re.compile(r"\bcard ending (?:in )?\d{4}\b", re.IGNORECASE), "[CARD REDACTED]"),
        (re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}\b"), "[CARD REDACTED]"),

        # Salary figures (heuristic: dollar amounts over $10K with "salary" context)
        (re.compile(r"\bsalary[:\s]+\$[\d,]+\b", re.IGNORECASE), "[SALARY REDACTED]"),

        # US ZIP codes embedded in addresses (5-digit or ZIP+4)
        (re.compile(r"\b\d{5}(?:-\d{4})?\b"), "[ZIP REDACTED]"),
    ]

    # Column names that should never appear in agent responses
    BLOCKED_COLUMNS = {
        "customer_email", "customer_phone", "customer_address",
        "employee_salary", "ssn", "tax_id", "date_of_birth",
        "credit_card_last4", "bank_account",
    }

    def mask(self, text: str) -> str:
        """
        Apply all PII patterns to the text.
        Returns the redacted version safe to surface to the user.
        """
        if not text:
            return text

        result = text
        for pattern, replacement in self.PII_PATTERNS:
            result = pattern.sub(replacement, result)

        return result

    def scan(self, text: str) -> list[str]:
        """
        Detect PII in text without redacting.
        Returns a list of PII categories found.
        Used by the evaluation layer to track PII leakage rate.
        """
        found = []
        label_map = {
            0: "email",
            1: "phone",
            2: "ssn",
            3: "credit_card",
            4: "credit_card",
            5: "salary",
            6: "zip_code",
        }
        for idx, (pattern, _) in enumerate(self.PII_PATTERNS):
            if pattern.search(text):
                found.append(label_map.get(idx, "unknown_pii"))

        return list(set(found))

    def has_pii(self, text: str) -> bool:
        """Quick check — returns True if any PII pattern matches."""
        return any(pattern.search(text) for pattern, _ in self.PII_PATTERNS)
