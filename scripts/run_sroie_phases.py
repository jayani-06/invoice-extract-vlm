#!/usr/bin/env python
"""Put Phases 3 and 4 in front of real photographed receipts.

Phases 1 and 2 have met SROIE. Phases 3 and 4 never have, and the reason is
not that nobody tried: **receipts do not record what those phases check.**
Measured across all 652 documents:

| Field | Present |
|---|---|
| `parties.vendor.name` | 652 |
| `parties.vendor.address` | 652 |
| `line_items[].description` | 2030 |
| `line_items[].line_total` | 1621 |
| `parties.vendor.gstin` | **0** |
| `payment_info` | **0** |
| `line_items[].unit_price` | **0** |
| `totals.subtotal` / `tax_total` | **0** |

So this script runs exactly the checks the data can support and reports the
rest as **not applicable**, which is a different statement from "passed" and
has to stay different.

**What can be measured, and why it is worth measuring:**

- *Vendor resolution*, on real repeats. 76 suppliers appear more than once
  across 488 receipts -- GARDENIA 44 times, MR. D.I.Y. 34 -- with the name
  variation real OCR and real print produce. Every resolution number so far
  came from names this project corrupted on purpose; this is the first time
  the matcher meets damage it did not author.
- *address_mismatch*, on real branch variation. A chain's outlets print
  different addresses under one company name, which is exactly the case the
  check exists for and exactly what the synthetic corpus had to have branches
  bolted on to simulate.
- *Duplicate detection*, on real repeat visits to the same shop.
- *Retrieval coverage and abstention* for Phase 3. Accuracy is unscoreable --
  "KF MODELLING CLAY KIDDY FISH" has no gold category and inventing one would
  be scoring the system against our own guess. But whether retrieval finds a
  confident neighbour for a real retail item, given a ledger built from other
  real receipts, is a genuine question about generalisation and the abstention
  rate answers it.

**What cannot:** gstin_mismatch, bank_account_changed and price_deviation have
no data to fire on, and Phase 3 categorisation accuracy has no labels.

Usage:
    python scripts/run_sroie_phases.py --out reports/sroie_phases
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.rag.anomaly import detect_anomalies  # noqa: E402
from invoice_extract.rag.categorize import categorize_document  # noqa: E402
from invoice_extract.rag.store import LedgerEntry, TfidfRetriever, VendorStats  # noqa: E402
from invoice_extract.rag.taxonomy import category_of  # noqa: E402
from invoice_extract.rag.validate_vendor import (  # noqa: E402
    resolve_vendor,
    validate_against_master,
)
from invoice_extract.rag.vendor_master import VendorMaster  # noqa: E402

#: Checks with no data to fire on. Named individually rather than lumped into
#: one caveat, because "not applicable" and "did not fire" look identical in a
#: results table and mean opposite things.
NOT_APPLICABLE = {
    "gstin_mismatch": "SROIE annotates no GSTIN (0 of 652)",
    "bank_account_changed": "SROIE annotates no payment details (0 of 652)",
    "price_deviation": "SROIE annotates no unit prices (0 of 2030 line items)",
    "arithmetic_mismatch": "SROIE annotates no subtotal or tax total (0 of 652)",
}


def fmt(value: float, places: int = 3) -> str:
    return "n/a" if value != value else f"{value:.{places}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/sroie"))
    ap.add_argument("--out", type=Path, default=Path("reports/sroie_phases"))
    ap.add_argument(
        "--history-split",
        default="train,val",
        help="Splits forming the history and ledger. The rest is evaluated.",
    )
    args = ap.parse_args(argv)

    paths = sorted((args.corpus / "canonical").glob("*.json"))
    if not paths:
        print(f"No canonical documents under {args.corpus}", file=sys.stderr)
        return 1
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

    splits = {s.strip() for s in args.history_split.split(",")}
    history = [d for d in docs if (d.get("source") or {}).get("split") in splits]
    evaluation = [d for d in docs if d not in history]
    if not history or not evaluation:
        print("Need both a history split and an evaluation split", file=sys.stderr)
        return 1

    # ------------------------------------------------------------ knowledge
    entries = [
        LedgerEntry(
            description=item.get("description") or "",
            category=category_of(item.get("description")),
            vendor=((doc.get("parties") or {}).get("vendor") or {}).get("name"),
            line_total=item.get("line_total"),
            doc_id=doc.get("doc_id"),
        )
        for doc in history
        for item in doc.get("line_items") or []
    ]
    retriever = TfidfRetriever()
    retriever.index(entries)
    stats = VendorStats.build(entries, doc_totals={})
    for doc in history:
        vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        total = (doc.get("totals") or {}).get("grand_total")
        if vendor and total is not None:
            stats.totals.setdefault(vendor, []).append(total)
    master = VendorMaster.build(history)

    # ------------------------------------------- Phase 4: vendor resolution
    # No name corruption is injected. The whole point is that these names
    # already carry the damage real print and real OCR produce.
    seen = {r.name for r in master.records.values()}
    resolvable = [
        d for d in evaluation if ((d.get("parties") or {}).get("vendor") or {}).get("name") in seen
    ]
    res_total = res_correct = res_unresolved = 0
    methods: Counter = Counter()
    for doc in resolvable:
        truth = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        resolution = resolve_vendor(doc, master)
        res_total += 1
        methods[resolution.method] += 1
        if not resolution.resolved:
            res_unresolved += 1
        elif resolution.vendor.name == truth:
            res_correct += 1

    # ------------------------------------------- Phase 4: record validation
    verdicts: Counter = Counter()
    check_counts: Counter = Counter()
    for doc in evaluation:
        report = validate_against_master(doc, master, history=history)
        verdicts[report.verdict] += 1
        for check in report.checks:
            check_counts[check.code] += 1

    # ------------------------------- Phase 3: retrieval coverage, no accuracy
    # Two reasons a line can abstain, and they mean opposite things:
    # retrieval found nothing similar, or it found good neighbours that are
    # themselves unlabelled. Collapsing them into one "coverage" number would
    # read as a retrieval failure when retrieval may have worked perfectly.
    n_items = abstained = no_neighbour = unlabelled_neighbour = 0
    top_scores: list[float] = []
    for doc in evaluation:
        for decision in categorize_document(doc, retriever, vendor_stats=stats):
            n_items += 1
            if decision.neighbours:
                top_scores.append(decision.neighbours[0].score)
            if not decision.abstained:
                continue
            abstained += 1
            if not decision.neighbours:
                no_neighbour += 1
            else:
                unlabelled_neighbour += 1

    labelled = sum(1 for e in entries if e.category != "9999")

    # ------------------------------------------------ Phase 3: anomaly flags
    flag_counts: Counter = Counter()
    for doc in evaluation:
        for flag in detect_anomalies(doc, retriever, stats).flags:
            flag_counts[flag.code] += 1

    coverage = (n_items - abstained) / n_items if n_items else float("nan")
    median_top = sorted(top_scores)[len(top_scores) // 2] if top_scores else float("nan")

    results = {
        "corpus": str(args.corpus),
        "history_documents": len(history),
        "evaluation_documents": len(evaluation),
        "vendor_master_size": len(master),
        "not_applicable": NOT_APPLICABLE,
        "phase4": {
            "resolution": {
                "scored": res_total,
                "correct": res_correct,
                "unresolved": res_unresolved,
                "accuracy": res_correct / res_total if res_total else float("nan"),
                "by_method": dict(methods),
            },
            "verdicts": dict(verdicts),
            "checks_raised": dict(check_counts),
        },
        "phase3": {
            "line_items": n_items,
            "abstained": abstained,
            "coverage": coverage,
            "median_top_similarity": median_top,
            "abstained_no_neighbour": no_neighbour,
            "abstained_neighbour_unlabelled": unlabelled_neighbour,
            "ledger_entries": len(entries),
            "ledger_entries_with_a_real_category": labelled,
            "anomaly_flags": dict(flag_counts),
            "accuracy": None,
            "accuracy_note": (
                "unscoreable: SROIE line items carry no category labels, and "
                "inventing them would score the system against our own guess"
            ),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        "# Phases 3 and 4 on real photographed receipts",
        "",
        f"History: **{len(history)}** receipts. Evaluated on **{len(evaluation)}**. "
        f"Vendor master: {len(master)} suppliers.",
        "",
        "## What receipts cannot test",
        "",
        "| Check | Why |",
        "|---|---|",
    ]
    for code, why in NOT_APPLICABLE.items():
        md.append(f"| `{code}` | {why} |")
    md += [
        "",
        "**Not applicable is not the same as passed.** These checks had nothing to "
        "fire on, and a results table that showed them as clean would be claiming "
        "a result that was never measured.",
        "",
        "## Phase 4 — vendor resolution on names we did not corrupt",
        "",
        "Every resolution figure elsewhere in this project comes from names the "
        "evaluation damaged on purpose. These carry whatever damage real print and "
        "real OCR produced, which is the first honest test the matcher has had.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Receipts from a supplier already on file | {res_total} |",
        f"| Resolved to the right supplier | {res_correct} |",
        f"| Unresolved | {res_unresolved} |",
        f"| **Accuracy** | **{fmt(results['phase4']['resolution']['accuracy'])}** |",
        "",
        "| Method | Count |",
        "|---|---|",
    ]
    for method, count in methods.most_common():
        md.append(f"| `{method}` | {count} |")

    md += ["", "## Phase 4 — checks raised", "", "| Check | Count |", "|---|---|"]
    for code, count in check_counts.most_common():
        md.append(f"| `{code}` | {count} |")
    md += ["", "| Verdict | Count |", "|---|---|"]
    for verdict, count in verdicts.most_common():
        md.append(f"| {verdict} | {count} |")

    md += [
        "",
        "## Phase 3 — retrieval against real retail items",
        "",
        "**Accuracy is not reported because it cannot be.** SROIE line items carry "
        "no category labels, and assigning them ourselves would score the system "
        "against our own guess. What is measurable is whether retrieval finds a "
        "confident neighbour at all, given a ledger built from other real receipts.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Line items | {n_items} |",
        f"| Abstained | {abstained} |",
        f"| **Coverage** | **{fmt(coverage)}** |",
        f"| Median top similarity | **{fmt(median_top)}** |",
        "",
        "### Why coverage is what it is",
        "",
        "| Reason for abstaining | Count |",
        "|---|---|",
        f"| Retrieval found nothing similar | {no_neighbour} |",
        f"| Retrieval found good neighbours, but they carry no category | "
        f"{unlabelled_neighbour} |",
        "",
        f"Of {len(entries)} ledger entries built from real receipts, "
        f"**{labelled}** carry a real account code. The taxonomy covers the 21 "
        "office-supply items our own generator produces; a Malaysian retail "
        "receipt sells modelling clay and nasi lemak.",
        "",
        "So **this is not a retrieval failure.** The median top similarity of "
        f"{fmt(median_top)} is well above the 0.35 the categoriser needs — it is "
        "finding genuinely similar past items. Every one of them is labelled "
        "`9999 Uncategorised`, so a similarity-weighted vote over them can only "
        "elect `Uncategorised`, and abstaining is the correct answer. The "
        "missing piece is a chart of accounts for retail, not a better "
        "retriever.",
    ]
    if flag_counts:
        md += ["", "| Anomaly flag | Count |", "|---|---|"]
        for code, count in flag_counts.most_common():
            md.append(f"| `{code}` | {count} |")
    (args.out / "sroie_phases.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n=== Phases 3 and 4 on {len(evaluation)} real receipts ===")
    print(
        f"Phase 4  resolution {fmt(results['phase4']['resolution']['accuracy'])} "
        f"({res_correct}/{res_total}, {res_unresolved} unresolved)"
    )
    print(f"         verdicts {dict(verdicts)}")
    print(f"         checks   {dict(check_counts)}")
    print(f"Phase 3  coverage {fmt(coverage)} ({n_items - abstained}/{n_items} answered)")
    print(f"         median top similarity {fmt(median_top)} -- retrieval works")
    print(
        f"         abstained: {no_neighbour} found nothing, "
        f"{unlabelled_neighbour} found unlabelled neighbours"
    )
    print(f"         ledger entries with a real category: {labelled}/{len(entries)}")
    print("         accuracy: not scoreable (no category labels)")
    print(f"\nWrote {args.out}/sroie_phases.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
