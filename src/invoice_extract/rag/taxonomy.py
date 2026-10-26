"""Chart of accounts, and the ground-truth mapping from item to category.

This is the label set Phase 3 categorisation is scored against. It is kept
small and flat on purpose: a real chart of accounts has hundreds of codes and
a hierarchy, but the research question here is whether *retrieval over
historical line items* can assign a category at all, and a 400-way label
problem on a 412-item corpus would measure data sparsity instead.

`CANONICAL_ITEM_CATEGORY` is ground truth, not a classifier. The synthetic
generator emits exactly ten item descriptions, and each maps to one account.
Anything that is not one of those ten -- a description variant, a FATURA2 row,
a real invoice -- has to be *inferred*, which is the actual task.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    code: str
    name: str
    description: str


#: Deliberately expense-side only. The pipeline reads purchase invoices, so
#: revenue and balance-sheet accounts would never be assigned and would only
#: inflate the apparent label space.
CHART_OF_ACCOUNTS: dict[str, Account] = {
    "5010": Account("5010", "Office Supplies", "Consumables: paper, pens, toner"),
    "5020": Account("5020", "IT Equipment", "Computers, peripherals, accessories"),
    "5030": Account("5030", "Furniture & Fixtures", "Desks, chairs, lighting, storage"),
    "5040": Account("5040", "Professional Services", "Consulting, legal, audit, contract labour"),
    "5050": Account("5050", "Facilities & Utilities", "Rent, power, water, maintenance"),
    "5060": Account("5060", "Travel & Transport", "Freight, courier, fuel, travel"),
    "9999": Account("9999", "Uncategorised", "Needs human review"),
}

UNCATEGORISED = "9999"

#: Ground truth for the generator's catalogue. Must stay in step with ITEMS in
#: scripts/generate_synthetic_gst_invoices.py -- tests/test_rag.py asserts
#: every generated description has a category here, because a description
#: missing from this map silently becomes UNCATEGORISED gold and the
#: categoriser is then scored against the wrong answer.
#:
#: The confusable pairs are deliberate and are documented beside ITEMS. Each
#: line below that carries a comment is one the words alone do not settle.
CANONICAL_ITEM_CATEGORY: dict[str, str] = {
    "A4 Copier Paper (500 sheets)": "5010",
    "Ballpoint Pens (box of 50)": "5010",
    "Printer Toner Cartridge": "5010",
    "Whiteboard Marker (pack of 10)": "5010",
    "Wireless Mouse": "5020",
    "Laptop Bag": "5020",
    "USB-C Cable 1m": "5020",
    "Laptop Docking Station": "5020",  # vs "Software Subscription" -> 5040
    "LED Desk Lamp": "5030",
    "Office Chair": "5030",
    "Steel Filing Cabinet": "5030",
    "Air Conditioner Unit 1.5T": "5030",  # the asset; its upkeep is 5050
    "Software Subscription (annual)": "5040",  # IT spend, but a service
    "Legal Retainer (monthly)": "5040",  # bought-in labour, vs 5050 below
    "Statutory Audit Fee": "5040",
    "Annual Maintenance Contract - HVAC": "5050",  # upkeep of the 5030 asset
    "Contract Labour - Housekeeping": "5050",  # bought-in labour, vs 5040
    "Electricity Charges": "5050",
    "Courier Charges (per consignment)": "5060",  # no shared chars with...
    "Freight Forwarding Service": "5060",  # ...this, same account
    "Diesel Fuel": "5060",
}


def category_of(description: str | None) -> str:
    """Ground-truth category for an exact canonical description.

    Returns `UNCATEGORISED` for anything else -- including the description
    *variants* used to evaluate retrieval. That is intentional: if this
    resolved variants too, the evaluation would be comparing a classifier
    against itself.
    """
    if not description:
        return UNCATEGORISED
    return CANONICAL_ITEM_CATEGORY.get(description.strip(), UNCATEGORISED)


def account_name(code: str) -> str:
    account = CHART_OF_ACCOUNTS.get(code)
    return account.name if account else "Unknown"


def is_real_category(code: str) -> bool:
    return code in CHART_OF_ACCOUNTS and code != UNCATEGORISED


def all_categories() -> list[str]:
    return [c for c in CHART_OF_ACCOUNTS if c != UNCATEGORISED]
