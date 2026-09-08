"""Evaluation harness CLI: compare a predictions JSONL against a gold JSONL,
both in canonical schema shape, and print/save a metrics report.

This is deliberately written before any model exists (per the project plan)
so the metric definitions are locked in and reviewable independent of any
particular baseline's quirks. `make eval-smoke` runs it against
tests/fixtures/{gold,pred}.jsonl as a fast sanity check.

Usage:
    python -m invoice_extract.eval.harness \\
        --gold path/to/gold.jsonl --pred path/to/pred.jsonl \\
        --schema schema/invoice_schema.json [--report-out report.json] [--by-source]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from invoice_extract.data.validate import load_schema, validate_jsonschema
from invoice_extract.eval.metrics import aggregate, compare_documents


def read_jsonl(path: Path) -> dict[str, dict]:
    docs = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs[doc["doc_id"]] = doc
    return docs


def print_report(agg, title: str) -> None:
    print(f"\n=== {title} (n={agg.n_docs} docs) ===")
    print(f"Document critical-fields accuracy : {agg.doc_critical_accuracy:.3f}")
    print(f"Macro field accuracy              : {agg.macro_field_accuracy:.3f}")
    print(f"Micro field accuracy              : {agg.micro_field_accuracy:.3f}")
    li = agg.line_items_micro
    print(
        f"Line items  P/R/F1                : {li.precision:.3f} / {li.recall:.3f} / {li.f1:.3f}  (gold={li.n_gold}, pred={li.n_pred}, matched={li.n_matched})"
    )
    if agg.tax_lines.n:
        print(
            f"Tax lines accuracy                : {agg.tax_lines.accuracy:.3f}  (n={agg.tax_lines.n})"
        )
    print("\nPer-field accuracy (n, acc, avg_score):")
    for path, fr in sorted(agg.per_field.items()):
        print(f"  {path:<40} n={fr.n:<4} acc={fr.accuracy:.3f} avg_score={fr.avg_score:.3f}")


def to_serializable(agg) -> dict:
    return {
        "n_docs": agg.n_docs,
        "doc_critical_accuracy": agg.doc_critical_accuracy,
        "macro_field_accuracy": agg.macro_field_accuracy,
        "micro_field_accuracy": agg.micro_field_accuracy,
        "line_items": {
            "precision": agg.line_items_micro.precision,
            "recall": agg.line_items_micro.recall,
            "f1": agg.line_items_micro.f1,
            "n_gold": agg.line_items_micro.n_gold,
            "n_pred": agg.line_items_micro.n_pred,
            "n_matched": agg.line_items_micro.n_matched,
        },
        "tax_lines_accuracy": agg.tax_lines.accuracy if agg.tax_lines.n else None,
        "per_field": {
            path: {"n": fr.n, "accuracy": fr.accuracy, "avg_score": fr.avg_score}
            for path, fr in agg.per_field.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument(
        "--schema", type=Path, default=None, help="Defaults to schema/invoice_schema.json"
    )
    parser.add_argument("--report-out", type=Path, default=None, help="Write the JSON report here")
    parser.add_argument(
        "--by-source", action="store_true", help="Also break the report down by source.dataset"
    )
    parser.add_argument("--skip-schema-validation", action="store_true")
    args = parser.parse_args(argv)

    gold_docs = read_jsonl(args.gold)
    pred_docs = read_jsonl(args.pred)

    schema = load_schema() if args.schema is None else json.loads(args.schema.read_text())

    missing_pred = set(gold_docs) - set(pred_docs)
    extra_pred = set(pred_docs) - set(gold_docs)
    if missing_pred:
        print(
            f"WARNING: {len(missing_pred)} gold doc(s) have no prediction (counted as fully wrong): {sorted(missing_pred)[:5]}...",
            file=sys.stderr,
        )
    if extra_pred:
        print(
            f"WARNING: {len(extra_pred)} prediction(s) have no matching gold doc_id, ignored: {sorted(extra_pred)[:5]}...",
            file=sys.stderr,
        )

    if not args.skip_schema_validation:
        for name, docs in [("gold", gold_docs), ("pred", pred_docs)]:
            for doc_id, doc in docs.items():
                errors = validate_jsonschema(doc, schema)
                if errors:
                    print(
                        f"WARNING: {name} doc {doc_id} fails schema validation: {errors[:3]}",
                        file=sys.stderr,
                    )

    empty_doc = {"doc_id": "", "line_items": [], "tax_lines": []}
    doc_reports = [
        compare_documents(gold_docs[doc_id], pred_docs.get(doc_id, empty_doc))
        for doc_id in gold_docs
    ]

    overall = aggregate(doc_reports)
    print_report(overall, "OVERALL")

    if args.by_source:
        by_source: dict[str, list] = {}
        for doc_id, dr in zip(gold_docs, doc_reports, strict=True):
            dataset = gold_docs[doc_id].get("source", {}).get("dataset", "unknown")
            by_source.setdefault(dataset, []).append(dr)
        for dataset, reports in sorted(by_source.items()):
            print_report(aggregate(reports), f"SOURCE={dataset}")

    if args.report_out:
        args.report_out.write_text(json.dumps(to_serializable(overall), indent=2))
        print(f"\nWrote report to {args.report_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
