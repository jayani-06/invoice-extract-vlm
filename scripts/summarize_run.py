#!/usr/bin/env python
"""Score one or more extraction runs and emit the Phase 2 comparison table.

`run_extraction.py` writes gold.jsonl / pred.jsonl / run.json. This turns any
set of those directories into the headline metrics and a per-field side-by-side,
using the same `HEADER_FIELDS` definition and the same aggregation as
`run_extraction_ablation.py`, so a number produced here is directly comparable
to the ones already in docs/phase2_extraction.md.

It exists because the VLM runs on a GPU elsewhere and the baseline runs locally,
so the two halves of the comparison arrive separately and have to be scored by
one piece of code rather than two.

Every run's `run.json` is carried into the output, which is what keeps a
perfect-OCR ceiling figure from being tabulated beside a genuine end-to-end one
without the difference being visible.

Usage:
    python scripts/summarize_run.py \
        --run rules=reports/phase2/rules_vlmset_gst_in_synthetic \
        --run vlm=reports/phase2/vlm_gst \
        --title "gst_in_synthetic (n=50)" \
        --out reports/phase2/comparison_gst.md

    # Restrict to a split, to get a held-out-only reading from the same run
    python scripts/summarize_run.py --run vlm=reports/phase2/vlm_gst --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.eval.metrics import aggregate, compare_documents  # noqa: E402

#: Identical to run_extraction_ablation.HEADER_FIELDS. Duplicated rather than
#: imported because that module runs extraction on import-time-heavy deps, and a
#: summariser should not need an OCR engine installed to format a table.
HEADER_FIELDS = [
    "parties.vendor.name",
    "parties.vendor.gstin",
    "invoice_meta.invoice_number",
    "invoice_meta.invoice_date",
    "totals.grand_total",
    "totals.subtotal",
]


def fmt(value: float, places: int = 3) -> str:
    return "n/a" if value != value else f"{value:.{places}f}"


def read_jsonl(path: Path) -> dict[str, dict]:
    docs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            docs[doc["doc_id"]] = doc
    return docs


def header_accuracy(agg) -> float:
    scores = [agg.per_field[f].accuracy for f in HEADER_FIELDS if f in agg.per_field]
    scores = [s for s in scores if s == s]
    return sum(scores) / len(scores) if scores else float("nan")


def score(run_dir: Path, split: str | None) -> dict:
    gold = read_jsonl(run_dir / "gold.jsonl")
    pred = read_jsonl(run_dir / "pred.jsonl")

    if split:
        gold = {d: g for d, g in gold.items() if (g.get("source") or {}).get("split") == split}
        if not gold:
            raise SystemExit(f"{run_dir}: no documents in split {split!r}")

    # A document with no prediction is a fully missed document, not an absent
    # one. Dropping it would shrink the denominator and flatter the extractor.
    empty = {"doc_id": "", "line_items": [], "tax_lines": []}
    reports = [compare_documents(gold[d], pred.get(d, empty)) for d in gold]
    agg = aggregate(reports)

    meta = {}
    run_json = run_dir / "run.json"
    if run_json.exists():
        meta = json.loads(run_json.read_text(encoding="utf-8"))

    return {
        "run_dir": str(run_dir),
        "split": split,
        "n_docs": agg.n_docs,
        "header_accuracy": header_accuracy(agg),
        "critical_fields": agg.doc_critical_accuracy,
        "macro_field_accuracy": agg.macro_field_accuracy,
        "micro_field_accuracy": agg.micro_field_accuracy,
        "line_items_f1": agg.line_items_micro.f1,
        "line_items_gold": agg.line_items_micro.n_gold,
        "tax_lines_accuracy": agg.tax_lines.accuracy,
        "per_field": {f: r.accuracy for f, r in sorted(agg.per_field.items())},
        "is_ocr_ceiling": meta.get("is_ocr_ceiling"),
        "vision_only": meta.get("vision_only"),
        "parse_failures": meta.get("parse_failures"),
        "model": meta.get("replay_model") or meta.get("extractor"),
        "mean_ms_per_doc": meta.get("mean_ms_per_doc"),
    }


def markdown(scored: dict[str, dict], title: str) -> str:
    names = list(scored)
    lines = [f"### {title}", ""]

    lines += [
        "| Extractor | n | Header acc | Critical-fields | Macro field acc | Line-item F1 "
        "| Parse failures | Measurement |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        s = scored[name]
        # Spelled out on every row. A ceiling figure and an end-to-end figure
        # in one table with no column distinguishing them is how a ceiling gets
        # quoted as a result.
        if s["is_ocr_ceiling"] is True:
            kind = "perfect-OCR ceiling"
        elif s["vision_only"] is True:
            kind = "end-to-end (pixels only)"
        elif s["is_ocr_ceiling"] is False:
            kind = "end-to-end"
        else:
            kind = "unknown"
        lines.append(
            f"| `{name}` | {s['n_docs']} | {fmt(s['header_accuracy'])} "
            f"| {fmt(s['critical_fields'])} | {fmt(s['macro_field_accuracy'])} "
            f"| {fmt(s['line_items_f1'])} "
            f"| {s['parse_failures'] if s['parse_failures'] is not None else 'n/a'} | {kind} |"
        )

    lines += ["", "**Header fields**", ""]
    lines += ["| Field | " + " | ".join(f"`{n}`" for n in names) + " |"]
    lines += ["|" + "---|" * (len(names) + 1)]
    for field in HEADER_FIELDS:
        cells = [fmt(scored[n]["per_field"].get(field, float("nan"))) for n in names]
        lines.append(f"| `{field}` | {' | '.join(cells)} |")

    every = sorted({f for s in scored.values() for f in s["per_field"]})
    other = [f for f in every if f not in HEADER_FIELDS]
    if other:
        lines += ["", "**Every other scored field**", ""]
        lines += ["| Field | " + " | ".join(f"`{n}`" for n in names) + " |"]
        lines += ["|" + "---|" * (len(names) + 1)]
        for field in other:
            cells = [fmt(scored[n]["per_field"].get(field, float("nan"))) for n in names]
            lines.append(f"| `{field}` | {' | '.join(cells)} |")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="A run to score. Repeatable; column order follows the flag order.",
    )
    ap.add_argument("--split", default=None, help="Score only documents in this source split")
    ap.add_argument("--title", default="Extraction comparison")
    ap.add_argument("--out", type=Path, default=None, help="Write the markdown here")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args(argv)

    scored: dict[str, dict] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        if not path:
            print(f"--run needs NAME=DIR, got {spec!r}", file=sys.stderr)
            return 2
        run_dir = Path(path)
        if not (run_dir / "pred.jsonl").exists():
            print(f"No pred.jsonl in {run_dir}", file=sys.stderr)
            return 1
        scored[name] = score(run_dir, args.split)

    table = markdown(scored, args.title + (f" -- split={args.split}" if args.split else ""))
    print(table)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(scored, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
