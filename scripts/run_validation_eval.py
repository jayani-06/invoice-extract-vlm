#!/usr/bin/env python
"""Phase 4 headline experiment: vendor resolution and record validation.

Builds a vendor master from the historical split, then runs the evaluation
split against it — once clean, to measure the false-positive rate, and once
with injected vendor-fraud scenarios, to measure detection.

Two metrics, reported separately:

**Vendor resolution accuracy** — did we identify the right supplier? Every
downstream check is conditional on this, so a resolution error is not one
wrong answer, it is a whole invoice validated against the wrong record.
Resolution is also tested under *name corruption* (OCR damage, abbreviation,
reordering), because that is what arrives from the upstream pipeline, not
clean names.

**Validation precision and recall** — against injected ground truth, broken
down by fraud type. The breakdown matters: catching every redirected bank
account while missing every duplicate payment is a different system from one
that catches 70% of each.

The false-positive rate is measured on *untouched* invoices. It is the number
that decides whether anyone keeps using the system: a control that flags
honest invoices gets switched off, and then it catches nothing at all.

Usage:
    python scripts/run_validation_eval.py
    python scripts/run_validation_eval.py --corrupt-names --out reports/phase4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.rag.validate_vendor import (  # noqa: E402
    FAIL,
    PASS,
    REVIEW,
    resolve_vendor,
    validate_against_master,
)
from invoice_extract.rag.vendor_master import VendorMaster  # noqa: E402

#: Scenarios that are detectable *from the invoice and the vendor record*.
#:
#: `impostor_vendor` is deliberately absent. A shell company submitting its
#: first invoice is, on the page, identical to a legitimate new supplier
#: submitting theirs: no record, no history, nothing to compare. The system
#: raises `unknown_vendor` for both, so counting it as fraud detection would
#: be circular -- it would score 100% by construction while telling you
#: nothing. It is reported separately as an onboarding outcome instead.
FRAUD_KINDS = (
    "redirected_bank_account",
    "gstin_mismatch",
    "price_gouging",
    "duplicate_payment",
)

#: Scenarios deliberately NOT designed for, kept as a permanent probe against
#: circularity. Phase 3 carries the same construct and both of its held-out
#: scenarios score zero; without one here, Phase 4's recall of 1.000 only says
#: that four checks catch the four scenarios they were written to catch.
#:
#: These two are chosen because each is blind in a *different* way from the
#: designed four, not merely harder:
#:
#: - `invoice_splitting` is a pattern across invoices. Every individual
#:   document is impeccable -- known vendor, contract price, matching bank,
#:   arithmetic reconciles -- and the fraud exists only in the sequence. The
#:   one check that looks across invoices, `possible_duplicate_payment`, keys
#:   on an identical total, which splitting specifically avoids.
#: - `quantity_inflation` exploits an asymmetry in what gets validated. The
#:   unit price is left at exactly the contract rate and every total is
#:   recomputed consistently, so `price_deviation` has nothing to object to and
#:   the arithmetic reconciles. The lie is the quantity, and there is no
#:   quantity-conformance check -- the project validates what a vendor charges
#:   per unit but never how many units it claims to have delivered.
#:
#: Expect 0/n on both. Closing them and re-reporting 100% would reproduce the
#: circularity they exist to expose, so they are reported separately and left
#: unfixed. See docs/phase4_validation.md.
HELD_OUT_KINDS = (
    "invoice_splitting",
    "quantity_inflation",
)

#: The ceiling `invoice_splitting` aims to sit under. Arbitrary -- this corpus
#: has no approval policy -- but it has to be a number for the scenario to be
#: constructible. Set between the 25th and 50th percentile of grand totals
#: (16,440 and 49,452), so enough invoices are large enough to be worth
#: splitting. At the median only one case was constructible in a 50-document
#: evaluation, which is too few to report 0/n from.
APPROVAL_THRESHOLD = 25_000.0

#: Which check is evidence for which injected scenario.
CHECK_TO_KIND = {
    "bank_account_changed": "redirected_bank_account",
    "gstin_mismatch": "gstin_mismatch",
    "price_deviation": "price_gouging",
    "possible_duplicate_payment": "duplicate_payment",
    "unknown_vendor": "impostor_vendor",
}

#: How an OCR'd or re-keyed vendor name actually arrives.
NAME_CORRUPTIONS = (
    lambda n, r: n.upper(),
    lambda n, r: n.replace(" ", ""),
    lambda n, r: n.replace("o", "0").replace("l", "1"),
    lambda n, r: " ".join(reversed(n.split())),
    lambda n, r: n[: max(4, len(n) - r.randint(1, 3))],
    lambda n, r: n.replace("&", "and"),
)


def inject(doc: dict, kind: str, rng: random.Random, master: VendorMaster, history: list[dict]):
    """Apply one fraud scenario. Returns None when the document cannot carry it."""
    vendor = (doc.get("parties") or {}).get("vendor") or {}
    payment = doc.setdefault("payment_info", {}) or {}
    record = master.get(vendor.get("name") or "")

    if kind == "redirected_bank_account":
        if not record or not record.bank_accounts or record.invoice_count < 2:
            return None
        payment["account_number"] = str(rng.randint(10**13, 10**14 - 1))

    elif kind == "gstin_mismatch":
        if not record or not record.gstins or record.invoice_count < 2:
            return None
        other = [g for r in master.records.values() for g in r.gstins if g not in record.gstins]
        if not other:
            return None
        vendor["gstin"] = rng.choice(other)

    elif kind == "price_gouging":
        if not record:
            return None
        items = doc.get("line_items") or []
        if not items:
            return None
        item = rng.choice(items)
        if item.get("unit_price") is None:
            return None
        item["unit_price"] = round(float(item["unit_price"]) * rng.uniform(2.5, 5.0), 2)

    elif kind == "duplicate_payment":
        candidates = [
            h
            for h in history
            if ((h.get("parties") or {}).get("vendor") or {}).get("name") == vendor.get("name")
            and (h.get("totals") or {}).get("grand_total") is not None
        ]
        if not candidates:
            return None
        donor = rng.choice(candidates)
        doc.setdefault("totals", {})["grand_total"] = donor["totals"]["grand_total"]
        doc.setdefault("invoice_meta", {})["invoice_date"] = donor["invoice_meta"]["invoice_date"]

    elif kind == "invoice_splitting":
        # One purchase broken into pieces that each sit under an approval
        # threshold. The total is reduced to just below it and a sibling
        # invoice number is minted, so the pair reads as two ordinary small
        # purchases. Nothing on either document is wrong.
        if not record:
            return None
        totals = doc.get("totals") or {}
        grand = totals.get("grand_total")
        if grand is None or grand < APPROVAL_THRESHOLD:
            return None
        share = round(APPROVAL_THRESHOLD * rng.uniform(0.88, 0.98), 2)
        scale = share / grand
        totals["grand_total"] = share
        for key in ("subtotal", "tax_total", "discount_total", "shipping", "round_off"):
            if totals.get(key) is not None:
                totals[key] = round(float(totals[key]) * scale, 2)
        for item in doc.get("line_items") or []:
            for key in ("line_total", "tax_amount"):
                if item.get(key) is not None:
                    item[key] = round(float(item[key]) * scale, 2)
            # Quantity shrinks, unit price does not: the contract rate is
            # untouched, so price conformance has nothing to object to.
            if item.get("quantity") is not None and item.get("unit_price"):
                item["quantity"] = max(1, int(round(float(item["quantity"]) * scale)))
        meta = doc.setdefault("invoice_meta", {})
        if meta.get("invoice_number"):
            meta["invoice_number"] = f"{meta['invoice_number']}-A"

    elif kind == "quantity_inflation":
        # Charge the right rate for units that were never delivered. Unit price
        # is untouched, so price conformance passes; everything downstream is
        # recomputed from the inflated quantity, so the arithmetic reconciles.
        if not record:
            return None
        items = [
            i
            for i in (doc.get("line_items") or [])
            if i.get("quantity") and i.get("unit_price") and i.get("line_total") is not None
        ]
        if not items:
            return None
        item = rng.choice(items)
        factor = rng.randint(3, 6)
        before = float(item["line_total"])
        item["quantity"] = float(item["quantity"]) * factor
        item["line_total"] = round(float(item["unit_price"]) * float(item["quantity"]), 2)
        if item.get("discount"):
            item["line_total"] = round(item["line_total"] - float(item["discount"]), 2)
        delta = item["line_total"] - before

        rate = item.get("tax_rate")
        totals = doc.setdefault("totals", {})
        if totals.get("subtotal") is not None:
            totals["subtotal"] = round(float(totals["subtotal"]) + delta, 2)
        tax_delta = 0.0
        if rate:
            tax_delta = round(delta * float(rate) / 100.0, 2)
            if item.get("tax_amount") is not None:
                item["tax_amount"] = round(float(item["tax_amount"]) + tax_delta, 2)
            # Spread across the components actually shown (CGST+SGST, or IGST).
            lines = [t for t in (doc.get("tax_lines") or []) if t.get("amount") is not None]
            for tax_line in lines:
                tax_line["amount"] = round(
                    float(tax_line["amount"]) + tax_delta / len(lines),
                    2,
                )
            if totals.get("tax_total") is not None:
                totals["tax_total"] = round(float(totals["tax_total"]) + tax_delta, 2)
        if totals.get("grand_total") is not None:
            totals["grand_total"] = round(float(totals["grand_total"]) + delta + tax_delta, 2)

    elif kind == "impostor_vendor":
        # A supplier nobody has ever billed with -- the classic shell-company
        # invoice. Name and GSTIN both unknown, so neither resolves.
        vendor["name"] = (
            f"{rng.choice(['Apex', 'Crown', 'Zenith', 'Prime'])} "
            f"{rng.choice(['Solutions', 'Ventures', 'Holdings'])} "
            f"{rng.randint(100, 999)}"
        )
        vendor["gstin"] = None

    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/gst_in_synthetic"))
    ap.add_argument("--history-size", type=int, default=100)
    ap.add_argument("--fraud-rate", type=float, default=0.4)
    ap.add_argument(
        "--held-out-rate",
        type=float,
        default=0.5,
        help="Rate at which HELD_OUT_KINDS are injected among the documents not "
        "already carrying a designed-for scenario. High on purpose: the probe "
        "needs enough cases to say 0/n credibly.",
    )
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--out", type=Path, default=Path("reports/phase4_validation"))
    ap.add_argument(
        "--corrupt-names",
        action="store_true",
        default=True,
        help="Damage vendor names before resolution, as the OCR stage would.",
    )
    ap.add_argument("--clean-names", dest="corrupt_names", action="store_false")
    args = ap.parse_args(argv)

    paths = sorted((args.corpus / "canonical").glob("*.json"))
    if not paths:
        print(f"No documents in {args.corpus / 'canonical'}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rng.shuffle(docs)
    history, evaluation = docs[: args.history_size], docs[args.history_size :]

    # The supplier file is entitled to its own record of trading names; the
    # generator writes it beside the corpus. Loading it is what separates
    # "the master knows this company bills under two names" from "the master
    # thinks they are two companies with half the history each".
    aliases: dict[str, str] = {}
    aliases_path = args.corpus / "vendor_aliases.json"
    if aliases_path.exists():
        aliases = json.loads(aliases_path.read_text(encoding="utf-8")).get("aliases") or {}

    master = VendorMaster.build(history, aliases=aliases)

    # ---------------------------------------------------- vendor resolution
    res_total = res_correct = res_unresolved = 0
    res_by_method: Counter = Counter()
    res_wrong = []

    for doc in evaluation:
        truth = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        if truth not in master:
            continue  # cannot resolve a vendor with no record; measured separately
        probe = json.loads(json.dumps(doc))
        if args.corrupt_names:
            corrupt = rng.choice(NAME_CORRUPTIONS)
            probe["parties"]["vendor"]["name"] = corrupt(truth, rng)
            # Drop the GSTIN for half of them, forcing name-only resolution --
            # otherwise GSTIN carries every case and the name matching is never
            # actually exercised.
            if rng.random() < 0.5:
                probe["parties"]["vendor"]["gstin"] = None

        resolution = resolve_vendor(probe, master)
        res_total += 1
        res_by_method[resolution.method] += 1
        if not resolution.resolved:
            res_unresolved += 1
            if len(res_wrong) < 5:
                res_wrong.append(
                    {
                        "truth": truth,
                        "probe": probe["parties"]["vendor"]["name"],
                        "outcome": "unresolved",
                    }
                )
        elif resolution.vendor.name == truth:
            res_correct += 1
        elif len(res_wrong) < 5:
            res_wrong.append(
                {
                    "truth": truth,
                    "probe": probe["parties"]["vendor"]["name"],
                    "outcome": f"resolved to {resolution.vendor.name}",
                }
            )

    res_accuracy = res_correct / res_total if res_total else float("nan")

    # -------------------------------------------------- validation accuracy
    clean_flagged = 0
    clean_total = 0
    new_supplier = 0
    tp = fp = fn = 0
    held_tp = held_fn = 0
    per_kind = defaultdict(lambda: {"injected": 0, "detected": 0})
    verdicts: Counter = Counter()
    fp_examples = []

    # Held-out scenarios are drawn from their own pool at their own rate, so
    # adding them cannot change how often a designed-for scenario is injected.
    # Sharing one pool would move the designed-for recall as a side effect of
    # adding a probe, and then the two readings would not be comparable.
    for doc in evaluation:
        probe = json.loads(json.dumps(doc))
        injected: set[str] = set()
        if rng.random() < args.fraud_rate:
            kind = rng.choice(FRAUD_KINDS)
            if inject(probe, kind, rng, master, history) is not None:
                injected.add(kind)
        elif rng.random() < args.held_out_rate:
            kind = rng.choice(HELD_OUT_KINDS)
            if inject(probe, kind, rng, master, history) is not None:
                injected.add(kind)

        report = validate_against_master(probe, master, history=history)
        verdicts[report.verdict] += 1
        raised = {CHECK_TO_KIND.get(c.code) for c in report.checks} - {None}

        for kind in injected:
            per_kind[kind]["injected"] += 1
            if kind in raised:
                per_kind[kind]["detected"] += 1

        if injected:
            # A held-out case is credited only if a check *attributable to it*
            # fires. Counting "the document was flagged for some reason" would
            # let an unrelated check take credit, which is exactly the kind of
            # accidental pass this probe exists to rule out.
            caught = bool(raised & injected)
            if injected & set(HELD_OUT_KINDS):
                held_tp += caught
                held_fn += not caught
            elif caught:
                tp += 1
            else:
                fn += 1
        else:
            clean_total += 1
            codes = {c.code for c in report.checks}
            if codes == {"unknown_vendor"}:
                # Correct behaviour, not a false alarm: an AP team genuinely
                # should look at the first invoice from a supplier with no
                # record. Counted separately so it cannot mask or inflate the
                # fraud false-positive rate.
                new_supplier += 1
            elif report.verdict != PASS:
                clean_flagged += 1
                fp += 1
                if len(fp_examples) < 5:
                    fp_examples.append(
                        {
                            "doc_id": probe.get("doc_id"),
                            "verdict": report.verdict,
                            "checks": sorted(codes),
                        }
                    )

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    # The honest overall figure. The designed-for recall above answers "do the
    # four checks catch the four scenarios they were written for", which is a
    # weaker question than it looks.
    held_total = held_tp + held_fn
    recall_all = (tp + held_tp) / (tp + fn + held_total) if (tp + fn + held_total) else float("nan")
    held_recall = held_tp / held_total if held_total else float("nan")
    # Denominator excludes new suppliers: flagging those is the intended
    # behaviour, and leaving them in would make this metric measure how many
    # first-time vendors the split happens to contain.
    scored_clean = clean_total - new_supplier
    fp_rate = clean_flagged / scored_clean if scored_clean else float("nan")

    def f(v: float) -> str:
        return "n/a" if v != v else f"{v:.3f}"

    results = {
        "history_documents": len(history),
        "evaluation_documents": len(evaluation),
        "vendor_master_size": len(master),
        "trading_names_known": len(aliases),
        "names_corrupted": args.corrupt_names,
        "resolution": {
            "accuracy": res_accuracy,
            "resolved_correctly": res_correct,
            "unresolved": res_unresolved,
            "total": res_total,
            "by_method": dict(res_by_method),
            "errors": res_wrong,
        },
        "validation": {
            "precision": precision,
            "recall": recall,
            "recall_designed_for_only": recall,
            "recall_including_held_out": recall_all,
            "held_out": {
                "kinds": list(HELD_OUT_KINDS),
                "injected": held_total,
                "detected": held_tp,
                "recall": held_recall,
            },
            "false_positive_rate_on_clean": fp_rate,
            "clean_documents": clean_total,
            "clean_from_known_vendors": scored_clean,
            "clean_flagged": clean_flagged,
            "new_supplier_reviews": new_supplier,
            "per_kind": {k: dict(v) for k, v in per_kind.items()},
            "verdicts": dict(verdicts),
            "false_positive_examples": fp_examples,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        "# Phase 4 — vendor validation",
        "",
        f"- Vendor master built from **{len(history)} historical invoices** "
        f"covering {len(master)} suppliers",
        f"- Evaluated on **{len(evaluation)} held-out invoices**",
        f"- Vendor names corrupted before resolution: **{args.corrupt_names}**",
        "",
        "## Vendor resolution",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | **{f(res_accuracy)}** |",
        f"| Resolved correctly | {res_correct} / {res_total} |",
        f"| Unresolved | {res_unresolved} |",
        "",
        "| Method | Count |",
        "|---|---|",
    ]
    for method, count in res_by_method.most_common():
        md.append(f"| `{method}` | {count} |")
    md += [
        "",
        "Every downstream check is conditional on resolution, so an error here is "
        "not one wrong answer — it validates a whole invoice against the wrong "
        "supplier's record.",
        "",
        "## Validation",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | **{f(precision)}** |",
        f"| Recall, designed-for scenarios only | **{f(recall)}** |",
        f"| Recall, including held-out scenarios | **{f(recall_all)}** |",
        f"| False-positive rate (clean, known vendor) | **{f(fp_rate)}** |",
        f"| Clean invoices wrongly flagged | {clean_flagged} / {scored_clean} |",
        f"| New-supplier reviews (correct, not errors) | {new_supplier} |",
        "",
        "| Injected scenario | Caught | Injected |",
        "|---|---|---|",
    ]
    for kind in FRAUD_KINDS:
        counts = per_kind.get(kind, {"injected": 0, "detected": 0})
        md.append(f"| `{kind}` | {counts['detected']} | {counts['injected']} |")
    md += [
        "",
        "### Held-out scenarios",
        "",
        "Not designed for, and kept unfixed on purpose. The recall above these is "
        "the answer to *do the four checks catch the four things they were written "
        "for*, which is a weaker question than it looks. These two are the answer "
        "to *what does this system not see*.",
        "",
        "| Held-out scenario | Caught | Injected |",
        "|---|---|---|",
    ]
    for kind in HELD_OUT_KINDS:
        counts = per_kind.get(kind, {"injected": 0, "detected": 0})
        md.append(f"| `{kind}` | **{counts['detected']}** | {counts['injected']} |")
    md += [
        "",
        "`invoice_splitting` is a pattern across invoices and every single document "
        "in it is impeccable. `quantity_inflation` exploits the project validating "
        "what a vendor charges per unit but never how many units it claims to have "
        "delivered. Both are ordinary AP fraud.",
    ]
    md += [
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for verdict in (PASS, REVIEW, FAIL):
        md.append(f"| `{verdict}` | {verdicts.get(verdict, 0)} |")

    (args.out / "validation.md").write_text("\n".join(md), encoding="utf-8")

    print(
        f"Resolution   accuracy {f(res_accuracy)}  ({res_correct}/{res_total}, "
        f"{res_unresolved} unresolved)"
    )
    print(
        f"Validation   P {f(precision)} / R {f(recall)} designed-for, "
        f"{f(recall_all)} incl. held-out ({held_tp}/{held_total}), "
        f"FP rate (clean, known vendor) {f(fp_rate)}"
    )
    print(
        f"             {new_supplier} new-supplier reviews (correct behaviour, "
        f"not counted as errors)"
    )
    print(f"\nWrote {args.out / 'validation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
