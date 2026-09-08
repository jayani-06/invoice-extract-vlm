#!/usr/bin/env python
"""Generate synthetic Indian GST invoices directly in canonical-schema JSON.

No external data or network access required (stdlib `random` only) — this
is the fastest way to get realistic GST-specific fields (GSTIN, HSN/SAC,
CGST/SGST vs IGST splits) well represented, and doubles as an end-to-end
smoke test of the schema (every generated doc is validated before being
written).

This produces canonical JSON only, not rendered images — wiring up an HTML
template + a renderer (weasyprint/playwright) to also emit a matching
invoice image is the natural next step once you're ready to feed these into
a VLM baseline; the `pages[0].image_path` field is written as a placeholder
path so the schema stays satisfiable in the meantime.

Usage:
    python scripts/generate_synthetic_gst_invoices.py --count 150 --out data/processed/gst_in_synthetic
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402
from invoice_extract.models.postprocess import gstin_check_digit  # noqa: E402

STATE_CODES = {"KA": "29", "TN": "33", "MH": "27", "DL": "07", "WB": "19", "GJ": "24"}
CITIES = [
    ("Bengaluru", "Karnataka", "KA"),
    ("Chennai", "Tamil Nadu", "TN"),
    ("Mumbai", "Maharashtra", "MH"),
    ("New Delhi", "Delhi", "DL"),
    ("Kolkata", "West Bengal", "WB"),
    ("Ahmedabad", "Gujarat", "GJ"),
]
COMPANY_SUFFIXES = [
    "Traders",
    "Enterprises",
    "Industries",
    "Textiles",
    "Electronics",
    "Pvt Ltd",
    "& Co",
]
FIRST_WORDS = [
    "Sundar",
    "Ramesh",
    "Krishna",
    "Ganesh",
    "Lakshmi",
    "Vishal",
    "Anand",
    "Priya",
    "Meera",
    "Arjun",
]
ITEMS = [
    ("A4 Copier Paper (500 sheets)", "4802", "ream", (150, 350)),
    ("Ballpoint Pens (box of 50)", "9608", "box", (200, 400)),
    ("Laptop Bag", "4202", "unit", (800, 2000)),
    ("USB-C Cable 1m", "8544", "unit", (100, 300)),
    ("Office Chair", "9401", "unit", (3000, 9000)),
    ("LED Desk Lamp", "9405", "unit", (500, 1500)),
    ("Printer Toner Cartridge", "8443", "unit", (1500, 4500)),
    ("Whiteboard Marker (pack of 10)", "9608", "pack", (150, 300)),
    ("Steel Filing Cabinet", "9403", "unit", (4000, 12000)),
    ("Wireless Mouse", "8471", "unit", (400, 1200)),
]
GST_RATES = [5.0, 12.0, 18.0, 28.0]


def gen_gstin(rng: random.Random, state_code: str) -> str:
    """Generate a GSTIN with a *correct* mod-36 check digit.

    The check digit is computed, not drawn at random. A random 15th character
    is wrong ~35/36 of the time, which would make the GSTIN checksum validator
    fire on essentially every document in the corpus -- destroying a signal
    that Phase 3 anomaly detection genuinely wants, since a real invoice with a
    bad GSTIN checksum is a meaningful red flag.
    """
    pan = (
        "".join(rng.choices(string.ascii_uppercase, k=5))
        + "".join(rng.choices(string.digits, k=4))
        + rng.choice(string.ascii_uppercase)
    )
    entity_code = rng.choice("123456789")
    first_14 = f"{state_code}{pan}{entity_code}Z"
    return first_14 + (gstin_check_digit(first_14) or "0")


def gen_party(rng: random.Random) -> tuple[dict, str]:
    city, state, code = rng.choice(CITIES)
    name = f"{rng.choice(FIRST_WORDS)} {rng.choice(COMPANY_SUFFIXES)}"
    house_no = rng.randint(1, 200)
    party = {
        "name": name,
        "address": f"{house_no} {rng.choice(['MG Road', 'Anna Salai', 'Park Street', 'Ring Road', 'Station Road'])}, {city}, {state} {rng.randint(100000, 699999)}",
        "gstin": gen_gstin(rng, STATE_CODES[code]),
        "phone": f"+91-{rng.randint(70000,99999)}{rng.randint(10000,99999)}",
        "email": f"billing@{name.lower().split()[0]}.example",
    }
    return party, code


def gen_document(rng: random.Random, doc_id: str, split: str) -> dict:
    vendor, vendor_state = gen_party(rng)
    buyer, buyer_state = gen_party(rng)
    interstate = vendor_state != buyer_state

    n_items = rng.randint(1, 5)
    rate = rng.choice(GST_RATES)
    line_items = []
    subtotal = 0.0
    for i in range(n_items):
        desc, hsn, unit, price_range = rng.choice(ITEMS)
        qty = rng.randint(1, 20)
        unit_price = round(rng.uniform(*price_range), 2)
        discount = round(unit_price * qty * rng.choice([0, 0, 0, 0.02, 0.05]), 2)
        line_subtotal = round(unit_price * qty - discount, 2)
        tax_amount = round(line_subtotal * rate / 100, 2)
        line_total = round(line_subtotal + tax_amount, 2)
        subtotal += line_subtotal
        line_items.append(
            {
                "line_no": i + 1,
                "description": desc,
                "hsn_sac_code": hsn,
                "quantity": qty,
                "unit": unit,
                "unit_price": unit_price,
                "discount": discount,
                "tax_rate": rate,
                "tax_amount": tax_amount,
                "line_total": line_total,
            }
        )

    subtotal = round(subtotal, 2)
    tax_total = round(sum(li["tax_amount"] for li in line_items), 2)
    if interstate:
        tax_lines = [{"type": "IGST", "rate": rate, "amount": tax_total}]
    else:
        half = round(tax_total / 2, 2)
        tax_lines = [
            {"type": "CGST", "rate": rate / 2, "amount": half},
            {"type": "SGST", "rate": rate / 2, "amount": round(tax_total - half, 2)},
        ]

    grand_total_raw = subtotal + tax_total
    grand_total = round(grand_total_raw, 2)
    round_off = round(grand_total - grand_total_raw, 2)

    invoice_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 240))
    due_date = invoice_date + timedelta(days=rng.choice([15, 30, 45]))

    return {
        "schema_version": "1.0.0",
        "doc_id": doc_id,
        "source": {
            "dataset": "gst_in_synthetic",
            "original_id": f"{doc_id}.json",
            "split": split,
            "license": "internal-synthetic",
        },
        "language": "en",
        "currency": "INR",
        "document_type": "invoice",
        "pages": [
            {
                "page_index": 0,
                "image_path": f"data/processed/gst_in_synthetic/images/{doc_id}.png",
                "width": 1654,
                "height": 2339,
                "dpi": 200,
            }
        ],
        "parties": {"vendor": vendor, "buyer": buyer},
        "invoice_meta": {
            "invoice_number": f"INV-{invoice_date.year}-{rng.randint(1000, 9999)}",
            "invoice_date": invoice_date.isoformat(),
            "due_date": due_date.isoformat(),
            "po_number": f"PO-{rng.randint(1000, 9999)}",
            "payment_terms": f"Net {(due_date - invoice_date).days}",
        },
        "line_items": line_items,
        "tax_lines": tax_lines,
        "totals": {
            "subtotal": subtotal,
            "discount_total": round(sum(li["discount"] for li in line_items), 2),
            "tax_total": tax_total,
            "shipping": 0.0,
            "round_off": round_off,
            "grand_total": grand_total,
            "amount_in_words": None,
        },
        "payment_info": {
            "bank_name": rng.choice(
                ["HDFC Bank", "ICICI Bank", "State Bank of India", "Axis Bank"]
            ),
            "account_number": "".join(rng.choices(string.digits, k=14)),
            "ifsc_or_swift": "".join(rng.choices(string.ascii_uppercase, k=4))
            + "0"
            + "".join(rng.choices(string.digits, k=6)),
            "upi_id": None,
            "mode": None,
            "terms": None,
        },
        "grounding": [],
        "annotator": {
            "annotated_by": "synthetic-generator",
            "verified": True,
            "notes": "Auto-generated, not real data.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("data/processed/gst_in_synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits", type=str, default="train:0.8,val:0.1,test:0.1")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    canonical_dir = args.out / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    split_ratios = [(s.split(":")[0], float(s.split(":")[1])) for s in args.splits.split(",")]
    schema = load_schema()

    n_ok = 0
    for i in range(args.count):
        r = rng.random()
        cum = 0.0
        split = split_ratios[-1][0]
        for name, ratio in split_ratios:
            cum += ratio
            if r <= cum:
                split = name
                break
        doc_id = f"gst_synth_{i:04d}"
        doc = gen_document(rng, doc_id, split)
        errors = validate_jsonschema(doc, schema)
        if errors:
            print(f"SKIP {doc_id}: {errors}", file=sys.stderr)
            continue
        (canonical_dir / f"{doc_id}.json").write_text(json.dumps(doc, indent=2))
        n_ok += 1

    print(f"Wrote {n_ok}/{args.count} synthetic GST invoices to {canonical_dir}")
    return 0 if n_ok == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
