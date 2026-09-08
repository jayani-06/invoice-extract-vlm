#!/usr/bin/env python
"""Phase 2 headline experiment: does the VLM earn its VRAM, and does the OCR
hint help?

Sweeps {extractor} x {degradation profile}, scoring every cell with the Phase 0
metrics, and emits a markdown table plus results.json. Header fields and line
items are reported separately because the Phase 2 exit criteria are stated
separately (>= 90% F1 header, >= 80% line items) and because they fail for
different reasons -- header extraction is a reading problem, line items are a
table-structure problem.

The comparisons that matter:

- **rules vs. VLM.** A label-anchored regex baseline is what a competent
  engineer builds in an afternoon. If the VLM does not clearly beat it, the
  model is not earning its place in the pipeline and the report should say so.
- **vision-only vs. hybrid.** Refs [3] and [7] predict that feeding OCR text
  alongside the image wins. Predicting is not measuring.
- **clean vs. degraded.** Ref [3]'s claim is that a multi-stage pipeline
  (preprocess -> OCR -> VLM) beats handing a raw scan to a VLM. The profile
  axis is what tests that.

Usage:
    python scripts/run_extraction_ablation.py --corpus data/processed/gst_in_synthetic \\
        --extractors null rules --profiles clean medium heavy --ocr-source gold \\
        --limit 50 --out reports/phase2_ablation

    # With a GPU (Colab/Kaggle), add the model rows:
    python scripts/run_extraction_ablation.py --corpus data/processed/gst_in_synthetic \\
        --extractors rules vlm:qwen2-vl-2b --vlm-modes hybrid vision-only \\
        --ocr-source engine --ocr-engine paddle --out reports/phase2_ablation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402
from invoice_extract.eval.metrics import aggregate, compare_documents  # noqa: E402
from invoice_extract.models.postprocess import finalize  # noqa: E402
from invoice_extract.models.registry import get_extractor  # noqa: E402
from invoice_extract.reading_order import reading_order_text  # noqa: E402

#: Scored separately from line items, because the exit criteria are separate.
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


def header_accuracy(agg) -> float:
    """Mean per-field accuracy over the header fields present in the report."""
    scores = [agg.per_field[f].accuracy for f in HEADER_FIELDS if f in agg.per_field]
    scores = [s for s in scores if s == s]
    return sum(scores) / len(scores) if scores else float("nan")


def markdown_table(rows: list[dict], profiles: list[str], metric: str, order: list[str]) -> str:
    lookup = {(r["extractor"], r["profile"]): r for r in rows}
    lines = [f"| Extractor | {' | '.join(profiles)} |", "|" + "---|" * (len(profiles) + 1)]
    for name in order:
        cells = []
        for profile in profiles:
            row = lookup.get((name, profile))
            cells.append(fmt(row[metric]) if row else "-")
        lines.append(f"| `{name}` | {' | '.join(cells)} |")
    return "\n".join(lines)


def build_extractor(name: str, mode: str, ocr_engine, preprocess):
    kwargs: dict = {}
    if name.startswith("vlm"):
        kwargs = {
            "use_ocr_hint": mode == "hybrid",
            "ocr_engine": ocr_engine,
            "preprocess": preprocess,
        }
    elif name == "rules" and ocr_engine is not None:
        kwargs = {"ocr_engine": ocr_engine, "preprocess": preprocess}
    return get_extractor(name, **kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--extractors", nargs="+", default=["null", "rules"])
    ap.add_argument("--profiles", nargs="+", default=["clean", "light", "medium", "heavy"])
    ap.add_argument(
        "--vlm-modes",
        nargs="+",
        default=["hybrid"],
        choices=["hybrid", "vision-only"],
        help="VLM rows are run once per mode, so the OCR hint becomes an ablation axis",
    )
    ap.add_argument("--ocr-source", default="gold", choices=["gold", "engine"])
    ap.add_argument("--ocr-engine", default="tesseract")
    ap.add_argument("--preprocess-config", default="full(no-binarize)")
    ap.add_argument("--limit", type=int, default=None, help="Documents per cell")
    ap.add_argument("--out", type=Path, default=Path("reports/phase2_ablation"))
    args = ap.parse_args(argv)

    manifest_path = args.corpus / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}; run `make corpus-full` first", file=sys.stderr)
        return 1
    manifest = [json.loads(ln) for ln in manifest_path.read_text().splitlines() if ln.strip()]

    # Gold text does not vary with degradation -- the profiles move word boxes,
    # not words. Sweeping profiles for a text-only extractor under --ocr-source
    # gold would print several identical columns and imply robustness that was
    # never measured.
    text_only = all(not name.startswith("vlm") for name in args.extractors)
    if args.ocr_source == "gold" and text_only and len(args.profiles) > 1:
        print(
            "Note: --ocr-source gold gives identical text at every degradation profile, "
            f"and none of {args.extractors} reads pixels. Collapsing to a single profile "
            "('clean') rather than printing duplicate columns. Use --ocr-source engine, "
            "or include a vlm:* extractor, to make the profile axis mean something.",
            file=sys.stderr,
        )
        args.profiles = ["clean"]

    ocr_engine = preprocess = None
    if args.ocr_source == "engine":
        from invoice_extract.ocr.base import OcrEngineUnavailableError, get_engine
        from invoice_extract.preprocess.pipeline import Pipeline, get_config

        try:
            ocr_engine = get_engine(args.ocr_engine)
        except OcrEngineUnavailableError as exc:
            print(f"\nOCR engine unavailable:\n{exc}\n", file=sys.stderr)
            return 3
        preprocess = Pipeline(get_config(args.preprocess_config))

    # Expand VLM entries across the requested modes so the hint is an axis.
    plan: list[tuple[str, str, str]] = []  # (row label, extractor name, mode)
    for name in args.extractors:
        if name.startswith("vlm"):
            for mode in args.vlm_modes:
                plan.append((f"{name}[{mode}]", name, mode))
        else:
            plan.append((name, name, "n/a"))

    schema = load_schema()
    results: list[dict] = []
    started = time.time()
    total_cells = len(plan) * len(args.profiles)
    cell = 0

    for label, name, mode in plan:
        try:
            extractor = build_extractor(name, mode, ocr_engine, preprocess)
        except Exception as exc:
            print(f"Skipping {label}: {exc}", file=sys.stderr)
            continue

        for profile in args.profiles:
            cell += 1
            rows = [r for r in manifest if r["profile"] == profile]
            if args.limit:
                rows = rows[: args.limit]
            if not rows:
                continue

            reports = []
            parse_failures = schema_failures = arithmetic_flags = 0
            elapsed_ms = 0.0

            for row in rows:
                doc_id = row["doc_id"]
                gold = json.loads(
                    (args.corpus / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8")
                )
                image = args.corpus / row["image_path"]
                pages = [
                    {
                        "page_index": 0,
                        "image_path": row["image_path"],
                        "width": row["width"],
                        "height": row["height"],
                        "dpi": row.get("dpi"),
                    }
                ]

                t0 = time.perf_counter()
                try:
                    if args.ocr_source == "gold":
                        gold_words = json.loads(
                            (args.corpus / row["ocr_gold_path"]).read_text(encoding="utf-8")
                        )["words"]
                        text = reading_order_text(
                            gold_words, lambda w: tuple(w["bbox"]), lambda w: w["text"]
                        )
                        if hasattr(extractor, "predict_from_text"):
                            raw = extractor.predict_from_text(text)
                        else:
                            raw = extractor.generate([image], ocr_text=text).parsed or {}
                    else:
                        raw = extractor.predict([image])
                except Exception as exc:
                    print(
                        f"  {label}/{profile}/{doc_id}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    raw = {}
                elapsed_ms += (time.perf_counter() - t0) * 1000

                if not raw:
                    parse_failures += 1
                pred, consistency = finalize(raw, doc_id, gold["source"], pages)
                if consistency.problems:
                    arithmetic_flags += 1
                if validate_jsonschema(pred, schema):
                    schema_failures += 1
                reports.append(compare_documents(gold, pred))

            agg = aggregate(reports)
            record = {
                "extractor": label,
                "profile": profile,
                "n_docs": agg.n_docs,
                "header_accuracy": header_accuracy(agg),
                "doc_critical_accuracy": agg.doc_critical_accuracy,
                "macro_field_accuracy": agg.macro_field_accuracy,
                "line_item_f1": agg.line_items_micro.f1,
                "line_item_precision": agg.line_items_micro.precision,
                "line_item_recall": agg.line_items_micro.recall,
                "tax_line_accuracy": agg.tax_lines.accuracy if agg.tax_lines.n else float("nan"),
                "parse_failure_rate": parse_failures / max(len(rows), 1),
                "schema_failure_rate": schema_failures / max(len(rows), 1),
                "arithmetic_flag_rate": arithmetic_flags / max(len(rows), 1),
                "mean_ms_per_doc": elapsed_ms / max(len(rows), 1),
                "per_field": {
                    f: agg.per_field[f].accuracy for f in HEADER_FIELDS if f in agg.per_field
                },
            }
            results.append(record)
            print(
                f"[{cell}/{total_cells}] {label:<26} {profile:<7} "
                f"header={fmt(record['header_accuracy'])} "
                f"lineF1={fmt(record['line_item_f1'])} "
                f"critical={fmt(record['doc_critical_accuracy'])} "
                f"{record['mean_ms_per_doc']:.0f}ms/doc",
                file=sys.stderr,
            )

    if not results:
        print("No cells were evaluated.", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "corpus": str(args.corpus),
                "ocr_source": args.ocr_source,
                "ocr_engine": args.ocr_engine if args.ocr_source == "engine" else None,
                "is_ocr_ceiling": args.ocr_source == "gold",
                "elapsed_s": round(time.time() - started, 1),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    order = [label for label, _, _ in plan if any(r["extractor"] == label for r in results)]
    md = ["# Phase 2 extraction ablation", ""]
    if args.ocr_source == "gold":
        md += [
            "> **OCR ceiling, not an end-to-end result.** `--ocr-source gold` feeds the "
            "renderer's ground-truth text instead of running an OCR engine, so these "
            "numbers show what each extractor achieves *given perfect OCR*. Real "
            "end-to-end scores will be lower. Re-run with `--ocr-source engine` once an "
            "OCR engine is installed.",
            ">",
            "> **The degradation-profile axis is inert here.** Gold text is identical at "
            "every severity (degradation moves the word boxes, not the words), so a "
            "text-only extractor scores the same on `heavy` as on `clean`. Only a real "
            "OCR engine, or a VLM reading the pixels, makes the profile axis meaningful.",
            "",
        ]
    md += [
        f"- Corpus: `{args.corpus}` | documents per cell: {results[0]['n_docs']}",
        f"- Wall clock: {round(time.time() - started, 1)}s",
        "",
        "## Header-field accuracy (exit criterion: >= 0.90)",
        "",
        markdown_table(results, args.profiles, "header_accuracy", order),
        "",
        "## Line-item F1 (exit criterion: >= 0.80)",
        "",
        markdown_table(results, args.profiles, "line_item_f1", order),
        "",
        "## Document critical-fields accuracy (all of vendor/number/date/total correct)",
        "",
        markdown_table(results, args.profiles, "doc_critical_accuracy", order),
        "",
        "## Macro field accuracy (all scalar fields)",
        "",
        markdown_table(results, args.profiles, "macro_field_accuracy", order),
        "",
        "## Parse failure rate (no usable output at all)",
        "",
        markdown_table(results, args.profiles, "parse_failure_rate", order),
        "",
        "## Mean latency per document (ms)",
        "",
        markdown_table(results, args.profiles, "mean_ms_per_doc", order),
        "",
        "## Per-header-field accuracy",
        "",
    ]
    for profile in args.profiles:
        subset = [r for r in results if r["profile"] == profile]
        if not subset:
            continue
        md += [
            f"### profile = {profile}",
            "",
            "| Extractor | " + " | ".join(f.split(".")[-1] for f in HEADER_FIELDS) + " |",
            "|" + "---|" * (len(HEADER_FIELDS) + 1),
        ]
        for row in subset:
            cells = [fmt(row["per_field"].get(f, float("nan"))) for f in HEADER_FIELDS]
            md.append(f"| `{row['extractor']}` | {' | '.join(cells)} |")
        md.append("")

    (args.out / "ablation.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {args.out / 'ablation.md'} and {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
