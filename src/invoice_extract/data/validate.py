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


def drop_impermissible_nulls(doc: dict, schema: dict | None = None) -> dict:
    """Remove optional keys whose value is None where the schema forbids null.

    The canonical schema distinguishes "present and null" from "absent" (see
    schema.md). A pydantic `model_dump()` cannot: every unset optional field
    comes out as None, and for a property typed `object` or `string` with no
    `"null"` in its type, `"key": null` is a validation error while omitting
    the key is correct.

    This has now bitten three times, each time surfacing as a plausible wrong
    number rather than a crash:

    - `pages[0].dpi` -- FATURA2 records no dpi, so every one of its 50
      predictions failed schema validation and read in a report as the
      extractor emitting malformed documents.
    - `annotator` and `currency` on SROIE -- all 652 documents were skipped at
      conversion, which would have read as "the mirror is unusable".

    Pruning is driven entirely by the schema and recurses through nested
    objects and arrays, because the nulls do: `pages[0].dpi` and
    `parties.buyer` are both nested, and a top-level-only version of this
    helper left them behind. A **required** key is never removed, and a key the
    schema permits to be null is left alone -- for those the null IS the
    information, and dropping it would turn "we looked and there is nothing"
    into "we did not look".
    """
    schema = schema or load_schema()
    return _prune(doc, schema, schema)


def _resolve(node: dict, root: dict) -> dict:
    """Follow a local `$ref` to its definition.

    Without this the pruner cannot see the type of anything declared by
    reference, which in this schema is every party: `parties.buyer` is
    `{"$ref": "#/$defs/party"}` with no inline `type`, so a null buyer looked
    permissible and survived. Only local refs are followed -- a remote `$ref`
    would mean fetching a schema over the network to decide whether to delete
    a key, which is not a thing this should do.
    """
    seen: set[str] = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if not ref.startswith("#/") or ref in seen:
            return node
        seen.add(ref)
        target: object = root
        for part in ref[2:].split("/"):
            if not isinstance(target, dict) or part not in target:
                return node
            target = target[part]
        if not isinstance(target, dict):
            return node
        node = target
    return node


def _types(node: dict) -> set[str]:
    declared = node.get("type")
    if declared is None:
        return set()
    return {declared} if isinstance(declared, str) else set(declared)


def _prune(value, node: dict, root: dict):
    if not isinstance(node, dict):
        return value
    node = _resolve(node, root)

    if isinstance(value, dict):
        properties = node.get("properties") or {}
        required = set(node.get("required") or [])
        out = {}
        for key, child in value.items():
            child_schema = _resolve(properties.get(key) or {}, root)
            if child is None and key not in required:
                allowed = _types(child_schema)
                if allowed and "null" not in allowed:
                    continue
            out[key] = _prune(child, child_schema, root)
        return out

    if isinstance(value, list):
        item_schema = node.get("items") or {}
        return [_prune(v, item_schema, root) for v in value]

    return value


def validate_pydantic(doc: dict) -> list[str]:
    try:
        CanonicalInvoiceDocument.model_validate(doc)
        return []
    except Exception as exc:  # pydantic.ValidationError, but keep it broad for CLI use
        return [str(exc)]


def check_totals_consistency(doc: dict, tolerance: float = 0.02) -> list[str]:
    """Soft check (warnings, not schema errors): totals should roughly add up.

    Convention (see schema/schema.md): `line_total` is the taxable value of a
    row -- quantity x unit_price, net of that row's discount, BEFORE tax -- and
    `subtotal` is the sum of those. Discount is therefore already netted into
    `subtotal` and must NOT be subtracted again here; doing so was an earlier
    bug that fired on every document carrying a discount.
    """
    warnings: list[str] = []
    totals = doc.get("totals", {})
    line_items = doc.get("line_items", [])
    tax_lines = doc.get("tax_lines", [])

    if totals.get("subtotal") is not None and line_items:
        computed = sum(
            (li.get("line_total") or 0) for li in line_items if li.get("line_total") is not None
        )
        expected = totals["subtotal"]
        if computed and abs(computed - expected) > tolerance * max(abs(expected), 1):
            warnings.append(f"sum(line_items.line_total)={computed:.2f} vs subtotal={expected:.2f}")

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
