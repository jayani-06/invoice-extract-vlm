"""Flag invoices that need a human to look at them.

Three independent signal families, deliberately kept separate rather than
collapsed into one score:

- **Rule** — things that are wrong on their face: a future invoice date, a GST
  rate that is not a real GST rate, a duplicate invoice number, arithmetic that
  does not add up.
- **Statistical** — an amount far outside what this vendor has billed before.
- **Semantic** — an item unlike anything in the ledger, or a vendor never seen.

They are separate because they fail differently and a reviewer responds to them
differently. A duplicate invoice number is near-certain fraud or a double-entry;
an unusual amount is a question. Averaging those into "risk: 0.62" throws away
the only information that makes the flag actionable.

Severity is deliberately coarse -- `high` / `medium` / `low`. A continuous risk
score invites a threshold nobody can justify, and this corpus cannot calibrate
one honestly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from invoice_extract.models.postprocess import check_arithmetic, validate_gstin
from invoice_extract.rag.store import Retriever, VendorStats

#: Statutory Indian GST slabs. A rate outside these is either an extraction
#: error or a non-compliant invoice; both want a human.
VALID_GST_RATES = {0.0, 0.25, 3.0, 5.0, 12.0, 18.0, 28.0}

#: Half-slabs, because CGST and SGST each carry half the combined rate.
VALID_SPLIT_RATES = VALID_GST_RATES | {r / 2 for r in VALID_GST_RATES}

#: |z| beyond this is "unlike anything this vendor has billed".
AMOUNT_Z_THRESHOLD = 3.0

#: Below this similarity to anything in the ledger, an item is novel.
NOVEL_ITEM_SIMILARITY = 0.25

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Flag:
    code: str
    severity: str
    message: str
    signal: str  # "rule" | "statistical" | "semantic"
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "signal": self.signal,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class AnomalyReport:
    doc_id: str
    flags: list[Flag] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.flags

    @property
    def max_severity(self) -> str | None:
        if not self.flags:
            return None
        return max(self.flags, key=lambda f: SEVERITY_ORDER[f.severity]).severity

    def by_signal(self) -> dict[str, list[Flag]]:
        out: dict[str, list[Flag]] = {}
        for flag in self.flags:
            out.setdefault(flag.signal, []).append(flag)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "clean": self.clean,
            "max_severity": self.max_severity,
            "n_flags": len(self.flags),
            "flags": [f.as_dict() for f in self.flags],
        }


_NON_ALNUM = re.compile(r"[^0-9A-Za-z]")


def normalize_invoice_number(value: Any) -> str | None:
    """Canonical form for duplicate detection.

    Vendors re-issue the same invoice as `INV-2026-0012`, `inv 2026 0012`,
    `INV/2026/0012`. Comparing raw strings would miss every one of those.
    """
    if not value:
        return None
    text = _NON_ALNUM.sub("", str(value)).upper()
    return text or None


# ------------------------------------------------------------------- rules


def _parse_iso(value: Any) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def rule_flags(
    doc: dict[str, Any],
    *,
    today: date | None = None,
    seen_invoice_numbers: dict[str, str] | None = None,
) -> list[Flag]:
    flags: list[Flag] = []
    meta = doc.get("invoice_meta") or {}
    today = today or date.today()

    invoice_date = _parse_iso(meta.get("invoice_date"))
    due_date = _parse_iso(meta.get("due_date"))

    if invoice_date and invoice_date > today:
        flags.append(
            Flag(
                "future_invoice_date",
                "high",
                f"Invoice dated {invoice_date} is in the future",
                "rule",
                {"invoice_date": str(invoice_date), "today": str(today)},
            )
        )
    if invoice_date and due_date and due_date < invoice_date:
        flags.append(
            Flag(
                "due_before_invoice",
                "medium",
                f"Due date {due_date} precedes the invoice date {invoice_date}",
                "rule",
                {"invoice_date": str(invoice_date), "due_date": str(due_date)},
            )
        )

    number = meta.get("invoice_number")
    if number and seen_invoice_numbers is not None:
        # Normalise on lookup, because the history is keyed normalised. Using
        # the raw string here meant this check could never fire -- it scored
        # 0/2 on injected duplicates until the mismatch was found.
        previous = seen_invoice_numbers.get(normalize_invoice_number(number) or "")
        if previous and previous != doc.get("doc_id"):
            flags.append(
                Flag(
                    "duplicate_invoice_number",
                    "high",
                    f"Invoice number {number} was already seen on {previous}",
                    "rule",
                    {"invoice_number": number, "first_seen_on": previous},
                )
            )

    for i, tax in enumerate(doc.get("tax_lines") or []):
        rate = tax.get("rate")
        if rate is not None and float(rate) not in VALID_SPLIT_RATES:
            flags.append(
                Flag(
                    "invalid_gst_rate",
                    "medium",
                    f"{tax.get('type')} at {rate}% is not a statutory GST rate",
                    "rule",
                    {"index": i, "type": tax.get("type"), "rate": rate},
                )
            )

    gstin = ((doc.get("parties") or {}).get("vendor") or {}).get("gstin")
    if gstin:
        problems = validate_gstin(gstin)
        if problems:
            flags.append(
                Flag(
                    "invalid_gstin",
                    "high",
                    f"Vendor GSTIN {gstin} failed validation: {problems[0]}",
                    "rule",
                    {"gstin": gstin, "problems": problems},
                )
            )

    consistency = check_arithmetic(doc)
    if consistency.problems:
        flags.append(
            Flag(
                "arithmetic_mismatch",
                "high",
                f"Totals do not reconcile: {consistency.problems[0]}",
                "rule",
                {"problems": consistency.problems},
            )
        )

    return flags


# ------------------------------------------------------------- statistical


def statistical_flags(doc: dict[str, Any], stats: VendorStats) -> list[Flag]:
    flags: list[Flag] = []
    vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
    grand_total = (doc.get("totals") or {}).get("grand_total")

    z = stats.amount_zscore(vendor, grand_total)
    if z is not None and abs(z) > AMOUNT_Z_THRESHOLD:
        direction = "above" if z > 0 else "below"
        flags.append(
            Flag(
                "unusual_amount_for_vendor",
                "high" if abs(z) > 5 else "medium",
                f"Total {grand_total:,.2f} is {abs(z):.1f} standard deviations "
                f"{direction} this vendor's history",
                "statistical",
                {"vendor": vendor, "amount": grand_total, "z_score": round(z, 2)},
            )
        )

    # A round number at scale is a weak but real fraud indicator: fabricated
    # amounts cluster on round figures far more than real ones do. Low
    # severity on its own; it earns attention when it co-occurs.
    if (
        isinstance(grand_total, (int, float))
        and grand_total >= 10_000
        and abs(grand_total - round(grand_total, -3)) < 1e-6
    ):
        flags.append(
            Flag(
                "suspiciously_round_total",
                "low",
                f"Total {grand_total:,.2f} is an exact round figure",
                "statistical",
                {"amount": grand_total},
            )
        )

    return flags


# ---------------------------------------------------------------- semantic


def semantic_flags(doc: dict[str, Any], retriever: Retriever, stats: VendorStats) -> list[Flag]:
    flags: list[Flag] = []
    vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")

    if vendor and not stats.known_vendor(vendor):
        # Low, not medium. Businesses onboard suppliers constantly, so a first
        # invoice from a new vendor is routine. Treating it as actionable put
        # 12 flags on clean documents in the first Phase 3 run and dropped
        # anomaly precision to 0.41 on its own. It is context a reviewer should
        # see, and it earns attention only when it co-occurs with a real
        # signal -- which is what `escalate_combinations` handles.
        flags.append(
            Flag(
                "unknown_vendor",
                "low",
                f"No prior invoices from {vendor!r}",
                "semantic",
                {"vendor": vendor},
            )
        )

    for i, item in enumerate(doc.get("line_items") or []):
        description = (item.get("description") or "").strip()
        if not description:
            continue
        matches = retriever.search(description, k=1)
        best = matches[0].score if matches else 0.0
        if best < NOVEL_ITEM_SIMILARITY:
            flags.append(
                Flag(
                    "novel_line_item",
                    "low",
                    f"Line {i + 1} {description!r} is unlike anything previously purchased",
                    "semantic",
                    {"index": i, "description": description, "best_similarity": round(best, 3)},
                )
            )

    return flags


# ------------------------------------------------------------------ facade


#: Weak signals that mean little alone but are meaningful together. A first
#: invoice from an unknown vendor is routine; a first invoice from an unknown
#: vendor for a suspiciously round five-figure sum is the classic shape of
#: invoice fraud. Escalating only on co-occurrence keeps the individual signals
#: quiet enough to stay switched on.
ESCALATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("unknown_vendor", "suspiciously_round_total"),
    ("unknown_vendor", "unusual_amount_for_vendor"),
    ("unknown_vendor", "novel_line_item"),
)


def escalate_combinations(flags: list[Flag]) -> list[Flag]:
    """Add a medium flag where two low signals co-occur meaningfully."""
    present = {f.code for f in flags}
    extra = []
    for a, b in ESCALATION_PAIRS:
        if a in present and b in present:
            extra.append(
                Flag(
                    f"combined_{a}_{b}",
                    "medium",
                    f"{a} and {b} together on one document",
                    "semantic",
                    {"signals": [a, b]},
                )
            )
    return flags + extra


def detect_anomalies(
    doc: dict[str, Any],
    retriever: Retriever,
    stats: VendorStats,
    *,
    today: date | None = None,
    seen_invoice_numbers: dict[str, str] | None = None,
) -> AnomalyReport:
    """Run all three signal families over one document."""
    flags = (
        rule_flags(doc, today=today, seen_invoice_numbers=seen_invoice_numbers)
        + statistical_flags(doc, stats)
        + semantic_flags(doc, retriever, stats)
    )
    flags = escalate_combinations(flags)
    flags.sort(key=lambda f: -SEVERITY_ORDER[f.severity])
    return AnomalyReport(doc_id=doc.get("doc_id", "<unknown>"), flags=flags)
