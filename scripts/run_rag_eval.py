#!/usr/bin/env python
"""Phase 3 headline experiment: does retrieval categorise, and does it flag?

Two separate evaluations, reported separately because they answer different
questions and fail in different ways.

**Categorisation.** Accuracy and macro-F1 over the chart of accounts, plus the
abstention rate. Abstentions are reported as their own column rather than
scored as errors: a system that says "I do not know" on 10% of lines and is
right on the rest is more useful to an accounts team than one that guesses on
everything, and collapsing the two hides exactly that trade-off. Coverage
(share of lines given a category) and accuracy-on-covered are both shown.

**Anomaly detection.** Precision and recall against injected ground truth, with
a per-kind breakdown. The breakdown matters: catching every duplicate invoice
number while missing every inflated total is a very different system from one
that catches 60% of each, and a single recall number cannot tell them apart.

Usage:
    python scripts/run_rag_eval.py
    python scripts/run_rag_eval.py --k 7 --out reports/phase3_rag
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.rag.anomaly import detect_anomalies, normalize_invoice_number  # noqa: E402
from invoice_extract.rag.categorize import categorize_line_item  # noqa: E402
from invoice_extract.rag.store import LedgerEntry, VendorStats  # noqa: E402
from invoice_extract.rag.taxonomy import account_name  # noqa: E402

#: Which injected anomaly each flag code is evidence for. A document counts as
#: detected when any flag maps to a kind that was actually injected.
#: Scenarios deliberately held out: no detector was written for them. They are
#: scored like any other, and a zero is reported as a zero.
HELD_OUT_KINDS = {"line_item_padding", "gstin_state_contradicts_address"}

FLAG_TO_KIND = {
    "duplicate_invoice_number": "duplicate_invoice_number",
    "unusual_amount_for_vendor": "inflated_total",
    "invalid_gst_rate": "invalid_gst_rate",
    "future_invoice_date": "future_invoice_date",
    "invalid_gstin": "corrupt_gstin",
    "arithmetic_mismatch": "inflated_total",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def f1(precision: float, recall: float) -> float:
    if precision != precision or recall != recall or (precision + recall) == 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def fmt(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ledger", type=Path, default=Path("data/processed/ledger"))
    ap.add_argument("--k", type=int, default=5, help="Neighbours retrieved per line item")
    ap.add_argument("--out", type=Path, default=Path("reports/phase3_rag"))
    ap.add_argument(
        "--retriever",
        default="tfidf",
        choices=["tfidf", "dense"],
        help="tfidf = character n-grams (the measured floor). dense = sentence "
        "embeddings, which is what the Phase 3 plan called for.",
    )
    ap.add_argument(
        "--categoriser",
        default="vote",
        choices=["vote", "rag"],
        help="vote = similarity-weighted neighbour vote. rag = an LLM reading the "
        "retrieved context and deciding, which is the generation step the plan "
        "describes and the vote is the control for.",
    )
    ap.add_argument("--generator-model", default=None, help="Override the LLM checkpoint")
    ap.add_argument(
        "--today",
        default="2026-09-20",
        help="Reference date for the future-date rule. Fixed by default so the "
        "result does not drift as the calendar moves.",
    )
    args = ap.parse_args(argv)

    if not (args.ledger / "ledger.json").exists():
        print(f"No ledger at {args.ledger}. Run scripts/build_ledger.py first.", file=sys.stderr)
        return 1

    ledger = load(args.ledger / "ledger.json")
    evaluation = load(args.ledger / "eval.json")
    today = date.fromisoformat(args.today)

    entries = [LedgerEntry(**e) for e in ledger["entries"]]

    from invoice_extract.rag.embed import EmbeddingModelUnavailableError, get_retriever

    try:
        retriever = get_retriever(args.retriever)
    except EmbeddingModelUnavailableError as exc:
        # Never fall back silently. A run that quietly used TF-IDF while the
        # report said "dense" is the most expensive kind of mistake this
        # project can make, and it has made the equivalent one before.
        print(f"Retriever unavailable: {exc}", file=sys.stderr)
        return 3

    generator = None
    if args.categoriser == "rag":
        from invoice_extract.rag.generate import GeneratorUnavailableError, get_generator

        kwargs = {"model_name": args.generator_model} if args.generator_model else {}
        try:
            generator = get_generator("local", **kwargs)
            # Load now rather than on the first line item, so an unavailable
            # model fails before the run reports partial numbers.
            generator._ensure_loaded()
        except GeneratorUnavailableError as exc:
            print(f"Generator unavailable: {exc}", file=sys.stderr)
            return 3

    t0 = time.perf_counter()
    retriever.index(entries)
    index_ms = (time.perf_counter() - t0) * 1000

    stats = VendorStats.build(
        entries,
        doc_totals={},  # filled below; needs one list per vendor
    )
    for row in ledger["doc_totals"]:
        if row.get("vendor") and row.get("grand_total") is not None:
            stats.totals.setdefault(row["vendor"], []).append(row["grand_total"])

    # The ledger is the history duplicates are checked against.
    # Duplicate detection needs the invoice numbers of the ledger documents,
    # which the line-item entries do not carry -- read them from the corpus.
    seen_numbers: dict[str, str] = {}
    ledger_doc_ids = {e.doc_id for e in entries}
    corpus_dir = Path("data/processed/gst_in_synthetic/canonical")
    for doc_id in ledger_doc_ids:
        path = corpus_dir / f"{doc_id}.json"
        if not path.exists():
            continue
        number = normalize_invoice_number(
            (load(path).get("invoice_meta") or {}).get("invoice_number")
        )
        if number:
            seen_numbers.setdefault(number, doc_id)

    # ------------------------------------------------------ categorisation
    n_items = n_correct = n_abstain = 0
    per_class_tp: Counter = Counter()
    per_class_fp: Counter = Counter()
    per_class_fn: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    cat_ms = 0.0

    for record in evaluation["records"]:
        doc = record["document"]
        vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        for item, truth in zip(
            doc.get("line_items") or [], record["line_item_truth"], strict=False
        ):
            gold = truth["category"]
            t = time.perf_counter()
            if generator is not None:
                from invoice_extract.rag.generate import categorize_line_item_rag

                decision = categorize_line_item_rag(
                    item, retriever, generator, vendor=vendor, k=args.k, vendor_stats=stats
                )
            else:
                decision = categorize_line_item(
                    item, retriever, vendor=vendor, k=args.k, vendor_stats=stats
                )
            cat_ms += (time.perf_counter() - t) * 1000

            n_items += 1
            confusion[gold][decision.category] += 1
            if decision.abstained:
                n_abstain += 1
                per_class_fn[gold] += 1
                continue
            if decision.category == gold:
                n_correct += 1
                per_class_tp[gold] += 1
            else:
                per_class_fp[decision.category] += 1
                per_class_fn[gold] += 1

    covered = n_items - n_abstain
    coverage = covered / n_items if n_items else float("nan")
    accuracy_all = n_correct / n_items if n_items else float("nan")
    accuracy_covered = n_correct / covered if covered else float("nan")

    classes = sorted(set(per_class_tp) | set(per_class_fp) | set(per_class_fn))
    per_class = {}
    for c in classes:
        tp, fp, fn = per_class_tp[c], per_class_fp[c], per_class_fn[c]
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        per_class[c] = {"precision": p, "recall": r, "f1": f1(p, r), "support": tp + fn}
    macro_f1 = (
        sum(v["f1"] for v in per_class.values() if v["f1"] == v["f1"]) / len(per_class)
        if per_class
        else float("nan")
    )

    # --------------------------------------------------- anomaly detection
    doc_tp = doc_fp = doc_fn = 0
    per_kind = defaultdict(lambda: {"injected": 0, "detected": 0})
    flag_counts: Counter = Counter()
    anom_ms = 0.0
    false_positive_examples = []

    for record in evaluation["records"]:
        doc = record["document"]
        injected = set(record["anomalies"])
        t = time.perf_counter()
        report = detect_anomalies(
            doc, retriever, stats, today=today, seen_invoice_numbers=seen_numbers
        )
        anom_ms += (time.perf_counter() - t) * 1000

        for flag in report.flags:
            flag_counts[flag.code] += 1

        # Only severity >= medium counts as "this document was flagged": the
        # low-severity signals (novel item, round total) are context for a
        # reviewer, not an accusation, and counting them would make the false
        # positive rate meaningless.
        actionable = [f for f in report.flags if f.severity in ("high", "medium")]
        raised_kinds = {FLAG_TO_KIND.get(f.code) for f in actionable} - {None}

        for kind in injected:
            per_kind[kind]["injected"] += 1
            if kind in raised_kinds:
                per_kind[kind]["detected"] += 1

        if injected and raised_kinds & injected:
            doc_tp += 1
        elif injected:
            doc_fn += 1
        elif actionable:
            doc_fp += 1
            if len(false_positive_examples) < 5:
                false_positive_examples.append(
                    {"doc_id": record["doc_id"], "flags": [f.code for f in actionable]}
                )

    precision = doc_tp / (doc_tp + doc_fp) if (doc_tp + doc_fp) else float("nan")
    recall = doc_tp / (doc_tp + doc_fn) if (doc_tp + doc_fn) else float("nan")
    n_clean = sum(1 for r in evaluation["records"] if not r["anomalies"])
    fp_rate = doc_fp / n_clean if n_clean else float("nan")

    # ------------------------------------------------------------- report
    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        "ledger_entries": len(entries),
        "ledger_vendors": len(stats.counts),
        "eval_documents": len(evaluation["records"]),
        "eval_line_items": n_items,
        "descriptions_varied": evaluation.get("variants"),
        "k": args.k,
        "index_ms": round(index_ms, 1),
        "categorization": {
            "accuracy_all_items": accuracy_all,
            "accuracy_on_covered": accuracy_covered,
            "retriever": args.retriever,
            "categoriser": args.categoriser,
            "generator_model": (
                getattr(generator, "model_name", None) if generator is not None else None
            ),
            "coverage": coverage,
            "abstentions": n_abstain,
            "macro_f1": macro_f1,
            "per_class": per_class,
            "ms_per_item": cat_ms / max(n_items, 1),
        },
        "anomaly": {
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "false_positive_rate_on_clean": fp_rate,
            "documents_with_anomalies": doc_tp + doc_fn,
            "clean_documents": n_clean,
            "per_kind": {k: dict(v) for k, v in per_kind.items()},
            "flag_counts": dict(flag_counts),
            "false_positive_examples": false_positive_examples,
            "ms_per_document": anom_ms / max(len(evaluation["records"]), 1),
        },
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        "# Phase 3 — RAG enrichment",
        "",
        f"- Ledger: **{len(entries)} line items** from {len(stats.counts)} vendors",
        f"- Evaluation: **{len(evaluation['records'])} documents**, {n_items} line items",
        f"- Descriptions rewritten in the eval set: **{evaluation.get('variants')}** "
        "(so retrieval cannot succeed by exact string match)",
        f"- Retrieval: TF-IDF char 3–5 grams, k={args.k}, index built in {index_ms:.0f} ms",
        "",
        "## Categorisation",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy (all line items) | **{fmt(accuracy_all)}** |",
        f"| Accuracy (only the ones it answered) | **{fmt(accuracy_covered)}** |",
        f"| Coverage (share it answered) | {fmt(coverage)} |",
        f"| Abstentions | {n_abstain} / {n_items} |",
        f"| Macro F1 | {fmt(macro_f1)} |",
        f"| Latency | {cat_ms / max(n_items, 1):.2f} ms/item |",
        "",
        "| Account | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for c in classes:
        v = per_class[c]
        md.append(
            f"| {account_name(c)} [{c}] | {fmt(v['precision'])} | {fmt(v['recall'])} "
            f"| {fmt(v['f1'])} | {v['support']} |"
        )
    md += [
        "",
        "Abstentions are counted as recall misses, never as wrong answers. A line the "
        "system declines to categorise goes to a reviewer who clears it in seconds; a "
        "confidently wrong account code propagates into the ledger silently.",
        "",
        "## Anomaly detection",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | **{fmt(precision)}** |",
        f"| Recall | **{fmt(recall)}** |",
        f"| F1 | {fmt(f1(precision, recall))} |",
        f"| False-positive rate on clean documents | {fmt(fp_rate)} |",
        f"| Latency | {anom_ms / max(len(evaluation['records']), 1):.2f} ms/document |",
        "",
        "| Injected anomaly | Caught | Injected |",
        "|---|---|---|",
    ]
    for kind, counts in sorted(per_kind.items()):
        label = f"`{kind}`" + (" *(held out)*" if kind in HELD_OUT_KINDS else "")
        md.append(f"| {label} | {counts['detected']} | {counts['injected']} |")
    md += [
        "",
        "Rows marked *held out* are scenarios no detector was written for. They "
        "are included precisely because a detector evaluated only against its "
        "own design brief measures nothing, and a zero there is reported as a "
        "zero rather than quietly excluded.",
    ]
    md += [
        "",
        "Only `high` and `medium` severity count as flagging a document. The `low` "
        "signals (novel item, suspiciously round total) are context for a reviewer "
        "rather than an accusation; counting them would make the false-positive rate "
        "meaningless.",
        "",
        "### All flags raised",
        "",
        "| Flag | Count |",
        "|---|---|",
    ]
    for code, count in flag_counts.most_common():
        md.append(f"| `{code}` | {count} |")

    (args.out / "rag_eval.md").write_text("\n".join(md), encoding="utf-8")

    print(
        f"Categorisation  accuracy {fmt(accuracy_all)} (covered {fmt(accuracy_covered)}), "
        f"coverage {fmt(coverage)}, macro-F1 {fmt(macro_f1)}"
    )
    print(
        f"Anomaly         P {fmt(precision)} / R {fmt(recall)}, " f"FP rate on clean {fmt(fp_rate)}"
    )
    print(f"\nWrote {args.out / 'rag_eval.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
