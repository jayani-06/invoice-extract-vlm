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
#: The catalogue vendors bill for.
#:
#: The original ten descriptions covered only three of the six accounts in the
#: chart, and every one of them was unambiguous from its words alone -- which
#: is why Phase 3 reported macro-F1 0.996. That number measured a ten-way
#: problem with no hard cases, not categorisation.
#:
#: The additions below are chosen to be *genuinely confusable*, because the
#: hard cases in real accounts payable are near-neighbours rather than unseen
#: vocabulary:
#:
#:   "Software Subscription (annual)"  5040 Professional Services
#:   "Laptop Docking Station"          5020 IT Equipment
#:     -- both are "IT spend"; one is a service and one is hardware.
#:
#:   "Courier Charges (per consignment)"  5060 Travel & Transport
#:   "Freight Forwarding Service"         5060 Travel & Transport
#:     -- same account, almost no shared characters: the case that separates a
#:        character-n-gram retriever from a semantic one.
#:
#:   "Annual Maintenance Contract - HVAC"  5050 Facilities & Utilities
#:   "Air Conditioner Unit 1.5T"           5030 Furniture & Fixtures
#:     -- the same subject matter booked to different accounts depending on
#:        whether it is an asset or its upkeep.
#:
#:   "Legal Retainer (monthly)"      5040 Professional Services
#:   "Contract Labour - Housekeeping" 5050 Facilities & Utilities
#:     -- both are bought-in labour, split by what the labour is for.
#:
#: Expect Phase 3's figures to fall. That is the point: a lower honest number
#: against a harder label space is the deliverable, and tuning thresholds to
#: recover 0.996 would put the old defect back.
ITEMS = [
    # 5010 Office Supplies
    ("A4 Copier Paper (500 sheets)", "4802", "ream", (150, 350)),
    ("Ballpoint Pens (box of 50)", "9608", "box", (200, 400)),
    ("Printer Toner Cartridge", "8443", "unit", (1500, 4500)),
    ("Whiteboard Marker (pack of 10)", "9608", "pack", (150, 300)),
    # 5020 IT Equipment
    ("Laptop Bag", "4202", "unit", (800, 2000)),
    ("USB-C Cable 1m", "8544", "unit", (100, 300)),
    ("Wireless Mouse", "8471", "unit", (400, 1200)),
    ("Laptop Docking Station", "8471", "unit", (3500, 9000)),
    # 5030 Furniture & Fixtures
    ("Office Chair", "9401", "unit", (3000, 9000)),
    ("LED Desk Lamp", "9405", "unit", (500, 1500)),
    ("Steel Filing Cabinet", "9403", "unit", (4000, 12000)),
    ("Air Conditioner Unit 1.5T", "8415", "unit", (28000, 52000)),
    # 5040 Professional Services
    ("Software Subscription (annual)", "9983", "licence", (4000, 36000)),
    ("Legal Retainer (monthly)", "9982", "month", (15000, 60000)),
    ("Statutory Audit Fee", "9982", "engagement", (25000, 90000)),
    # 5050 Facilities & Utilities
    ("Annual Maintenance Contract - HVAC", "9987", "year", (18000, 65000)),
    ("Contract Labour - Housekeeping", "9985", "month", (8000, 30000)),
    ("Electricity Charges", "9969", "month", (6000, 42000)),
    # 5060 Travel & Transport
    ("Courier Charges (per consignment)", "9968", "consignment", (120, 900)),
    ("Freight Forwarding Service", "9965", "shipment", (2500, 18000)),
    ("Diesel Fuel", "2710", "litre", (90, 110)),
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


#: Identity is a property of the *company*, not of an individual invoice.
#: Re-rolling GSTIN, address and bank details per invoice -- the original
#: behaviour -- meant a vendor appearing on four invoices carried four
#: different GSTINs and four different bank accounts. That makes a vendor
#: master impossible to build, and makes bank-detail-change detection (the
#: strongest fraud signal in the Phase 4 plan) fire on literally every
#: invoice. Parties are therefore minted once and reused.
_PARTY_REGISTRY: dict[str, tuple[dict, str, dict]] = {}

#: Every address a company bills from, and the trading name it sometimes uses.
#:
#: Three Phase 4 limitations came from this being absent:
#:
#: - `address_mismatch` never fired. One address per supplier meant every
#:   invoice matched the record at 100% similarity, so the check never executed
#:   and its 75% threshold was never validated.
#: - No vendor aliasing. A real supplier master carries trading names; matching
#:   on the legal name alone made resolution easier than it is.
#: - No legitimate bank-detail changes. Every account change in the corpus was
#:   fraud, so `bank_account_changed` had no honest counter-example and its
#:   precision was measured against a population containing none.
#:
#: NOTE the branch addresses are per *company*, not per invoice. Re-rolling an
#: address per invoice would make `address_mismatch` fire on everything, which
#: is the same defect as the vendor identity bug recorded in
#: docs/limitations.md -- a check that fires always carries no information.
_SITES: dict[str, list[str]] = {}
_TRADING_NAMES: dict[str, str] = {}

#: Companies that legitimately change bank account partway through their
#: history, with no fraud involved. `bank_account_changed` SHOULD flag these
#: for review -- that is correct behaviour -- but they must exist so the
#: false-positive rate is measured against a population that contains them.
_BANK_CHANGES: dict[str, dict] = {}


def reset_site_registry() -> None:
    _SITES.clear()
    _TRADING_NAMES.clear()
    _BANK_CHANGES.clear()


def _street(rng: random.Random, city: str, state: str) -> str:
    road = rng.choice(
        ["MG Road", "Anna Salai", "Park Street", "Ring Road", "Station Road", "Industrial Estate"]
    )
    return f"{rng.randint(1, 200)} {road}, {city}, {state} {rng.randint(100000, 699999)}"


def _trading_name(rng: random.Random, legal: str) -> str | None:
    """A shortened or abbreviated form the company also bills under.

    Not a corruption -- those are injected at evaluation time to simulate OCR.
    This is a second *legitimate* name, which resolution has to handle without
    treating it as a different supplier.
    """
    words = legal.split()
    if len(words) < 2 or rng.random() > 0.35:
        return None
    return rng.choice(
        [
            f"{words[0]} {words[-1][:3]}.",
            words[0],
            f"{''.join(w[0] for w in words)} {words[-1]}",
        ]
    )


def reset_party_registry() -> None:
    """Clear the registry. Called per generation run so a fixed seed keeps
    producing a fixed corpus rather than depending on process state."""
    _PARTY_REGISTRY.clear()
    reset_site_registry()


def gen_bank_details(rng: random.Random, name: str) -> dict:
    """Banking details, minted once per company."""
    return {
        "bank_name": rng.choice(
            ["State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank"]
        ),
        "account_number": str(rng.randint(10**13, 10**14 - 1)),
        "ifsc_or_swift": "".join(rng.choices(string.ascii_uppercase, k=4))
        + "0"
        + str(rng.randint(100000, 999999)),
    }


#: A vendor's price for an item is a contract rate, not a fresh random draw.
#: Drawing uniformly per invoice gave the same (vendor, item) pair a 57% price
#: spread, so no contract rate existed to validate an invoice against -- which
#: made Phase 4's price-conformance check meaningless. Real quoted prices move
#: a few percent between invoices, so that is what is modelled.
_PRICE_REGISTRY: dict[tuple[str, str], float] = {}

#: How much a vendor's price for one item drifts invoice to invoice.
PRICE_JITTER = 0.04


def reset_price_registry() -> None:
    _PRICE_REGISTRY.clear()


def vendor_price(
    rng: random.Random, vendor: str, description: str, price_range: tuple[float, float]
) -> float:
    """This vendor's price for this item, stable across invoices."""
    key = (vendor, description)
    if key not in _PRICE_REGISTRY:
        _PRICE_REGISTRY[key] = rng.uniform(*price_range)
    base = _PRICE_REGISTRY[key]
    return round(base * rng.uniform(1 - PRICE_JITTER, 1 + PRICE_JITTER), 2)


#: Invoice numbers are a per-issuer sequence, not a random draw. Drawing from
#: randint(1000, 9999) over 150 documents produced ~4 birthday-paradox
#: collisions, and duplicate-invoice-number detection correctly flagged them --
#: which read as false positives and dropped Phase 3 anomaly precision to 0.40.
#: The detector was right; the data was wrong.
_INVOICE_SEQUENCE: dict[str, int] = {}
_VENDOR_INDEX: dict[str, int] = {}


def reset_invoice_sequence() -> None:
    _INVOICE_SEQUENCE.clear()
    _VENDOR_INDEX.clear()


def next_invoice_number(vendor: str, year: int) -> str:
    """`INV-<year>-<vendor index><sequence>`, unique by construction.

    The vendor part is an incrementing index, not `hash(vendor)`: Python
    randomises string hashing per process, so a hash-derived prefix would both
    collide across vendors and change between runs, breaking reproducibility
    from a fixed seed.
    """
    if vendor not in _VENDOR_INDEX:
        _VENDOR_INDEX[vendor] = len(_VENDOR_INDEX) + 1
    key = f"{vendor}:{year}"
    _INVOICE_SEQUENCE[key] = _INVOICE_SEQUENCE.get(key, 0) + 1
    return f"INV-{year}-{_VENDOR_INDEX[vendor]:03d}{_INVOICE_SEQUENCE[key]:03d}"


def gen_party(rng: random.Random) -> tuple[dict, str]:
    city, state, code = rng.choice(CITIES)
    name = f"{rng.choice(FIRST_WORDS)} {rng.choice(COMPANY_SUFFIXES)}"

    # Same company name -> same identity, every time.
    if name in _PARTY_REGISTRY:
        party, party_code, _ = _PARTY_REGISTRY[name]
        return dict(party), party_code

    # Registered office first, then 0-2 branches. Minted once per company and
    # reused, so an address is a property of the supplier rather than of the
    # invoice -- see the note on _SITES.
    registered = _street(rng, city, state)
    sites = [registered]
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        branch_city, branch_state, _branch_code = rng.choice(CITIES)
        sites.append(_street(rng, branch_city, branch_state))
    _SITES[name] = sites

    trading = _trading_name(rng, name)
    if trading:
        _TRADING_NAMES[name] = trading

    party = {
        "name": name,
        "address": registered,
        "gstin": gen_gstin(rng, STATE_CODES[code]),
        "phone": f"+91-{rng.randint(70000,99999)}{rng.randint(10000,99999)}",
        "email": f"billing@{name.lower().split()[0]}.example",
    }
    _PARTY_REGISTRY[name] = (party, code, gen_bank_details(rng, name))

    # One supplier in roughly twelve genuinely changes bank account partway
    # through its life. Recorded here so the switch is deterministic for a
    # given seed rather than depending on invoice ordering.
    if rng.random() < 0.08:
        _BANK_CHANGES[name] = gen_bank_details(rng, name + "-new")

    return dict(party), code


def gen_document(rng: random.Random, doc_id: str, split: str, index: int = 0) -> dict:
    vendor, vendor_state = gen_party(rng)
    buyer, buyer_state = gen_party(rng)
    interstate = vendor_state != buyer_state

    legal_name = vendor["name"]

    # Bill from one of the company's sites. Weighted towards the registered
    # office, because that is what a supplier file would mostly contain and a
    # uniform draw would make a branch address the common case.
    sites = _SITES.get(legal_name) or [vendor["address"]]
    vendor["address"] = sites[0] if rng.random() < 0.7 else rng.choice(sites)

    # Some invoices arrive under the trading name. This is a legitimate second
    # name, not an OCR corruption -- resolution has to join it to the same
    # supplier rather than treat it as a new one.
    trading = _TRADING_NAMES.get(legal_name)
    if trading and rng.random() < 0.3:
        vendor["name"] = trading

    # A legitimate bank change takes effect from the midpoint of the corpus.
    # `bank_account_changed` SHOULD flag these -- it is correct behaviour on
    # evidence this thin -- but they have to exist for the false-positive rate
    # to mean anything.
    bank = dict(_PARTY_REGISTRY[legal_name][2])
    changed = _BANK_CHANGES.get(legal_name)
    if changed and index >= 75:
        bank = dict(changed)

    n_items = rng.randint(1, 5)
    rate = rng.choice(GST_RATES)
    line_items = []
    subtotal = 0.0
    for i in range(n_items):
        desc, hsn, unit, price_range = rng.choice(ITEMS)
        qty = rng.randint(1, 20)
        unit_price = vendor_price(rng, vendor["name"], desc, price_range)
        discount = round(unit_price * qty * rng.choice([0, 0, 0, 0.02, 0.05]), 2)
        # `line_total` is the TAXABLE VALUE: quantity x unit_price, net of the
        # line discount, BEFORE tax. This is the conventional meaning of the
        # "Amount" column on a GST invoice, and it is what makes the printed
        # page internally consistent -- the Subtotal is the sum of the Amount
        # column. A tax-inclusive line_total (the earlier behaviour) rendered
        # an invoice whose Amount column did not add up to its own Subtotal,
        # and left Phase 3 unable to use line-item arithmetic as a signal.
        line_total = round(unit_price * qty - discount, 2)
        tax_amount = round(line_total * rate / 100, 2)
        subtotal += line_total
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
            "invoice_number": next_invoice_number(vendor["name"], invoice_date.year),
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
        # Banking details belong to the vendor, so they are looked up rather
        # than regenerated. An invoice whose account number differs from the
        # vendor's known one is exactly the fraud signal Phase 4 looks for,
        # and that signal only exists if the normal case is stable.
        "payment_info": {
            **bank,
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
    reset_party_registry()
    reset_price_registry()
    reset_invoice_sequence()
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
        doc = gen_document(rng, doc_id, split, index=i)
        errors = validate_jsonschema(doc, schema)
        if errors:
            print(f"SKIP {doc_id}: {errors}", file=sys.stderr)
            continue
        (canonical_dir / f"{doc_id}.json").write_text(json.dumps(doc, indent=2))
        n_ok += 1

    # The supplier file's own record of trading names. A real master carries
    # these; without them a company billing under two names produces two
    # records with half the history each, so identity checks see a thinner
    # record than exists. VendorMaster.build(documents, aliases=...) reads it.
    #
    # Written as a sidecar rather than inferred, because inferring it means
    # fuzzy-merging names -- which would also merge the genuinely distinct
    # suppliers whose names are near-identical.
    alias_map = {alias: legal for legal, alias in _TRADING_NAMES.items()}
    aliases_path = args.out / "vendor_aliases.json"
    aliases_path.write_text(
        json.dumps(
            {
                "note": (
                    "Trading name -> legal entity, as a supplier master would "
                    "record them. Not ground truth about the invoices; it is "
                    "reference data the vendor master is entitled to have."
                ),
                "aliases": alias_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    multi_site = sum(1 for sites in _SITES.values() if len(sites) > 1)
    print(f"Wrote {n_ok}/{args.count} synthetic GST invoices to {canonical_dir}")
    print(
        f"  {len(_SITES)} suppliers, {multi_site} with branch addresses, "
        f"{len(alias_map)} trading names, {len(_BANK_CHANGES)} legitimate bank changes"
    )
    print(f"  wrote {aliases_path}")
    return 0 if n_ok == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
