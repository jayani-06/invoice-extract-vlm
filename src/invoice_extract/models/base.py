"""Baseline model interface. No model is implemented yet — this defines the
contract so the eval harness has something concrete to call once one is,
and so the interface itself is reviewable before any specific model's
quirks leak into it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class InvoiceExtractor(Protocol):
    """Anything that turns one document image into a canonical-schema dict
    (see schema/invoice_schema.json) implements this."""

    def predict(self, image_paths: list[Path]) -> dict[str, Any]:
        """image_paths: one path per page, in order. Returns a dict matching
        CanonicalInvoiceDocument (missing/uncertain fields as null, per
        schema.md's null-vs-absent convention) — doc_id/source are filled in
        by the caller, not the model.
        """
        ...


class NullBaseline:
    """Trivial baseline: predicts every field as null / empty. Useful only
    as a lower-bound sanity check for the harness itself (a real model
    should beat this on every metric, obviously) — run it via
    scripts/run_null_baseline.py to smoke-test the harness against a
    non-trivial gold set before any real model exists.
    """

    def predict(self, image_paths: list[Path]) -> dict[str, Any]:
        return {
            "parties": {"vendor": {}},
            "invoice_meta": {"invoice_number": None, "invoice_date": None},
            "line_items": [],
            "tax_lines": [],
            "totals": {"grand_total": None},
        }
