#!/usr/bin/env python
"""Build the historical ledger Phase 3 retrieves against, and an evaluation set.

Phase 3 needs something Phase 0 never generated: a *history*. Categorisation by
retrieval is meaningless without past invoices to retrieve, and the statistical
anomaly signal needs a per-vendor amount distribution.

So the 150 synthetic documents are split:

  ledger  (default 100)  past invoices, categorised, indexed
  eval    (default  50)  "new" invoices to categorise and screen

**Why the evaluation set gets rewritten descriptions.** The generator emits
exactly ten item descriptions. If the evaluation set used them verbatim,
retrieval would be exact string matching and would score ~100% while proving
nothing. Real vendors write the same item differently every time -- `A4 PAPER
500SH`, `Copier paper A4 500`, `A-4 COPIER PAPER (500SHEETS)`. The variants
here reproduce that: abbreviation, case changes, word reordering, punctuation
loss, and OCR-style character confusion. The ledger keeps canonical spellings;
the evaluation set does not. That gap is the thing being measured.

Anomalies are injected into a configurable share of the evaluation set, with
ground truth recorded, so precision and recall are measurable rather than
asserted.

Usage:
    python scripts/build_ledger.py --out data/processed/ledger
    python scripts/build_ledger.py --anomaly-rate 0.2 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.rag.taxonomy import category_of  # noqa: E402

#: Assigned Indian GST state codes, mirrored from the generator. Kept as a
#: literal rather than imported so this script does not depend on
#: scripts/generate_synthetic_gst_invoices.py importing cleanly.
GST_STATE_CODES = ("29", "33", "27", "07", "19", "24")

#: Surface rewrites a real vendor might apply to the same item.
ABBREVIATIONS = {
    "Copier Paper": "COPIER PPR",
    "sheets": "SH",
    "Ballpoint Pens": "BP PENS",
    "box of": "bx",
    "Printer Toner Cartridge": "TONER CART",
    "Whiteboard Marker": "WB MARKER",
    "pack of": "pk",
    "Wireless Mouse": "WRLS MOUSE",
    "Laptop Bag": "LAPTOP BG",
    "USB-C Hub": "USBC HUB",
    "LED Desk Lamp": "LED DSK LAMP",
    "Office Chair": "OFC CHAIR",
    "Steel Filing Cabinet": "STL FILE CAB",
}

#: Character confusions an OCR engine actually makes, so some variants carry
#: the kind of damage the upstream stages leave behind.
OCR_CONFUSIONS = {"O": "0", "l": "1", "I": "1", "S": "5", "B": "8"}


def vary(description: str, rng: random.Random) -> str:
    """Rewrite a description the way a different vendor's system would."""
    text = description
    style = rng.random()

    if style < 0.35:
        for long, short in ABBREVIATIONS.items():
            if long.lower() in text.lower():
                text = text.replace(long, short)
                break
        text = text.upper()
    elif style < 0.55:
        text = text.replace("(", "").replace(")", "").upper()
    elif style < 0.70:
        words = text.replace("(", "").replace(")", "").split()
        rng.shuffle(words)
        text = " ".join(words)
    elif style < 0.85:
        text = text.lower().replace(" ", "-")
    else:
        chars = list(text)
        for _ in range(rng.randint(1, 2)):
            positions = [i for i, c in enumerate(chars) if c in OCR_CONFUSIONS]
            if positions:
                i = rng.choice(positions)
                chars[i] = OCR_CONFUSIONS[chars[i]]
        text = "".join(chars)

    return text.strip()


ANOMALY_KINDS = (
    "duplicate_invoice_number",
    "inflated_total",
    "invalid_gst_rate",
    "future_invoice_date",
    "corrupt_gstin",
    # --- Scenarios the detectors were NOT designed for -------------------
    # Added deliberately after Phase 3 and 4 both scored perfectly against
    # anomalies we had written the detectors to catch. A detector evaluated
    # only on its own design brief measures nothing. These two are held out:
    # if they are missed, that is the honest result and gets reported.
    "line_item_padding",
    "gstin_state_contradicts_address",
)


def inject(doc: dict, kind: str, rng: random.Random, donor: dict | None) -> dict | None:
    """Corrupt a document in one specific, recorded way.

    Returns None when the document cannot carry that anomaly, so the caller
    records only what was actually injected -- an anomaly we claimed to add but
    did not would silently depress recall.
    """
    totals = doc.get("totals") or {}
    meta = doc.get("invoice_meta") or {}

    if kind == "duplicate_invoice_number":
        if not donor or not (donor.get("invoice_meta") or {}).get("invoice_number"):
            return None
        meta["invoice_number"] = donor["invoice_meta"]["invoice_number"]

    elif kind == "inflated_total":
        if totals.get("grand_total") is None:
            return None
        # Large enough to clear the z-threshold against vendor history.
        totals["grand_total"] = round(totals["grand_total"] * rng.uniform(12, 30), 2)

    elif kind == "invalid_gst_rate":
        taxes = doc.get("tax_lines") or []
        if not taxes:
            return None
        taxes[0]["rate"] = rng.choice([7.0, 11.5, 19.0, 23.0])

    elif kind == "future_invoice_date":
        if not meta.get("invoice_date"):
            return None
        year = int(str(meta["invoice_date"])[:4])
        meta["invoice_date"] = f"{year + 3}{str(meta['invoice_date'])[4:]}"

    elif kind == "line_item_padding":
        # One line's quantity x unit_price no longer matches its line_total,
        # but subtotal and grand_total are adjusted so the INVOICE still
        # reconciles. Document-level arithmetic passes; only a per-line check
        # can see it. This is how padding actually hides.
        items = doc.get("line_items") or []
        if not items:
            return None
        item = rng.choice(items)
        if item.get("line_total") is None or item.get("quantity") is None:
            return None
        uplift = round(item["line_total"] * rng.uniform(0.3, 0.8), 2)
        item["line_total"] = round(item["line_total"] + uplift, 2)
        totals_block = doc.setdefault("totals", {})
        for key in ("subtotal", "grand_total"):
            if totals_block.get(key) is not None:
                totals_block[key] = round(totals_block[key] + uplift, 2)

    elif kind == "gstin_state_contradicts_address":
        # First two GSTIN digits are the state code. A Karnataka address with
        # a Maharashtra GSTIN is either a keying error or a fabricated number.
        # Nothing in the pipeline currently cross-checks the two.
        vendor = (doc.get("parties") or {}).get("vendor") or {}
        gstin = vendor.get("gstin")
        if not gstin or len(gstin) != 15:
            return None
        others = [c for c in GST_STATE_CODES if c != gstin[:2]]
        if not others:
            return None
        vendor["gstin"] = rng.choice(others) + gstin[2:]

    elif kind == "corrupt_gstin":
        vendor = (doc.get("parties") or {}).get("vendor") or {}
        gstin = vendor.get("gstin")
        if not gstin or len(gstin) != 15:
            return None
        # Flip the check digit to something else in the charset.
        charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        vendor["gstin"] = gstin[:14] + rng.choice(charset.replace(gstin[14], ""))

    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/gst_in_synthetic"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/ledger"))
    ap.add_argument("--ledger-size", type=int, default=100)
    ap.add_argument("--anomaly-rate", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument(
        "--no-variants",
        action="store_true",
        help="Keep canonical descriptions in the eval set. Retrieval then becomes "
        "exact matching and scores ~100%% -- useful only as a sanity check.",
    )
    args = ap.parse_args(argv)

    paths = sorted((args.corpus / "canonical").glob("*.json"))
    if not paths:
        print(f"No documents in {args.corpus / 'canonical'}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rng.shuffle(docs)

    ledger_docs = docs[: args.ledger_size]
    eval_docs = docs[args.ledger_size :]

    # --- ledger: canonical descriptions, human-assigned categories ---------
    entries = []
    doc_totals = []
    for doc in ledger_docs:
        vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        doc_totals.append(
            {"vendor": vendor, "grand_total": (doc.get("totals") or {}).get("grand_total")}
        )
        for item in doc.get("line_items") or []:
            category = category_of(item.get("description"))
            entries.append(
                {
                    "description": item.get("description"),
                    "category": category,
                    "vendor": vendor,
                    "unit_price": item.get("unit_price"),
                    "quantity": item.get("quantity"),
                    "line_total": item.get("line_total"),
                    "hsn_sac_code": item.get("hsn_sac_code"),
                    "doc_id": doc.get("doc_id"),
                }
            )

    uncategorised = sum(1 for e in entries if e["category"] == "9999")
    if uncategorised:
        print(
            f"WARNING: {uncategorised}/{len(entries)} ledger items have no ground-truth "
            "category; extend CANONICAL_ITEM_CATEGORY in rag/taxonomy.py",
            file=sys.stderr,
        )

    # --- eval set: varied descriptions, recorded truth, injected anomalies --
    eval_records = []
    injected_counter = {k: 0 for k in ANOMALY_KINDS}
    for doc in eval_docs:
        doc = json.loads(json.dumps(doc))  # deep copy; never mutate the corpus
        truth = []
        for item in doc.get("line_items") or []:
            canonical = item.get("description")
            truth.append({"canonical": canonical, "category": category_of(canonical)})
            if not args.no_variants and canonical:
                item["description"] = vary(canonical, rng)

        anomalies = []
        if rng.random() < args.anomaly_rate:
            kind = rng.choice(ANOMALY_KINDS)
            donor = ledger_docs[rng.randrange(len(ledger_docs))]
            if inject(doc, kind, rng, donor) is not None:
                anomalies.append(kind)
                injected_counter[kind] += 1

        eval_records.append(
            {
                "doc_id": doc["doc_id"],
                "document": doc,
                "line_item_truth": truth,
                "anomalies": anomalies,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ledger.json").write_text(
        json.dumps({"entries": entries, "doc_totals": doc_totals}, indent=1), encoding="utf-8"
    )
    (args.out / "eval.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "variants": not args.no_variants,
                "anomaly_rate": args.anomaly_rate,
                "records": eval_records,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    n_anom = sum(1 for r in eval_records if r["anomalies"])
    print(f"Ledger : {len(entries)} line items from {len(ledger_docs)} documents")
    print(
        f"         {len({e['vendor'] for e in entries})} vendors, "
        f"{len({e['category'] for e in entries})} categories"
    )
    print(
        f"Eval   : {len(eval_records)} documents, "
        f"{sum(len(r['line_item_truth']) for r in eval_records)} line items"
    )
    print(f"         descriptions rewritten: {not args.no_variants}")
    print(f"         anomalies injected: {n_anom} documents")
    for kind, count in injected_counter.items():
        if count:
            print(f"           {kind}: {count}")
    print(f"\nWrote {args.out}/ledger.json and {args.out}/eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
