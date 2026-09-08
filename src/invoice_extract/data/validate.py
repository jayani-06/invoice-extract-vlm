"""Validate canonical-schema JSON documents, via jsonschema (structural) and
pydantic (typed, used by the rest of the codebase).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

from invoice_extract.data.models import CanonicalInvoiceDocument

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schema" / "invoice_schema.json"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def validate_jsonschema(doc: dict, schema: dict | None = None) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    schema = schema or load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    return [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]


def validate_pydantic(doc: dict) -> list[str]:
    try:
        CanonicalInvoiceDocument.model_validate(doc)
        return []
    except Exception as exc:  # pydantic.ValidationError, but keep it broad for CLI use
        return [str(exc)]


def check_totals_consistency(doc: dict, tolerance: float = 0.02) -> list[str]:
    """Soft check (warnings, not schema errors): totals should roughly add up."""
    warnings: list[str] = []
    totals = doc.get("totals", {})
    line_items = doc.get("line_items", [])
    tax_lines = doc.get("tax_lines", [])

    if totals.get("subtotal") is not None and line_items:
        computed = sum(
            (li.get("line_total") or 0) for li in line_items if li.get("line_total") is not None
        )
        expected = (
            totals["subtotal"]
            + (totals.get("tax_total") or 0)
            - (totals.get("discount_total") or 0)
        )
        if computed and abs(computed - expected) > tolerance * max(abs(expected), 1):
            warnings.append(
                f"sum(line_items.line_total)={computed:.2f} vs subtotal+tax-discount={expected:.2f}"
            )

    if tax_lines and totals.get("tax_total") is not None:
        tax_sum = sum(t.get("amount", 0) for t in tax_lines)
        if abs(tax_sum - totals["tax_total"]) > tolerance * max(abs(totals["tax_total"]), 1):
            warnings.append(
                f"sum(tax_lines.amount)={tax_sum:.2f} vs totals.tax_total={totals['tax_total']:.2f}"
            )

    return warnings


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    paths = [Path(p) for p in argv] or sorted((SCHEMA_PATH.parent / "examples").glob("*.json"))
    schema = load_schema()

    exit_code = 0
    for path in paths:
        doc = json.loads(path.read_text())
        errors = validate_jsonschema(doc, schema) + validate_pydantic(doc)
        warnings = check_totals_consistency(doc)

        if errors:
            exit_code = 1
            print(f"FAIL {path}")
            for e in errors:
                print(f"  error: {e}")
        else:
            print(f"OK   {path}")
        for w in warnings:
            print(f"  warn: {w}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
