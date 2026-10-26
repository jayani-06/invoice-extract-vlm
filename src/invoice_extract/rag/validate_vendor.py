"""Phase 4: resolve an invoice to a supplier, then validate it against them.

Two steps, deliberately separate.

**Resolution** answers "who sent this?". It is its own step because every
downstream check is conditional on it: validating an invoice against the wrong
supplier's record produces confident nonsense, and the honest failure mode is
to say the vendor is unknown rather than to pick the closest name.

**Validation** answers "is this consistent with what we know about them?" and
returns a verdict with evidence, never a silent correction.

The verdict is three-valued on purpose:

- `PASS`    — nothing to look at
- `REVIEW`  — something is off; a human decides
- `FAIL`    — do not pay this without an investigation

A two-valued verdict forces every soft signal into one bucket or the other.
Most real findings are genuinely "ask someone", and collapsing them into
`FAIL` produces an alert stream nobody reads — which is the practical failure
mode of these systems, not missed fraud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from invoice_extract.models.postprocess import normalize_gstin
from invoice_extract.rag.vendor_master import VendorMaster, VendorRecord, rate_key

#: Name similarity above which two names are the same company.
NAME_MATCH_THRESHOLD = 88.0

#: And below which they are definitely not. Between the two is ambiguous, and
#: ambiguity resolves to "unknown vendor" rather than a guess.
NAME_REJECT_THRESHOLD = 70.0

#: A vendor needs at least this much history before a "changed bank account"
#: or "new GSTIN" claim means anything. Below it, we have not seen enough to
#: know what normal looks like, and alerting anyway trains reviewers to
#: dismiss alerts.
MIN_HISTORY_FOR_IDENTITY_CHECK = 2

#: Deviation from this vendor's own contract price. Their quoted price moves
#: only a few percent between invoices, so a quarter is already a lot.
PRICE_DEVIATION_REVIEW = 0.25
PRICE_DEVIATION_FAIL = 1.00

#: Deviation from the pooled market price, used when this vendor has too little
#: history of their own. Suppliers legitimately differ on price, so the band has
#: to be far wider or ordinary competitive variation reads as fraud.
MARKET_DEVIATION_REVIEW = 0.60
MARKET_DEVIATION_FAIL = 1.50

#: Invoices to the same vendor for the same amount within this many days are
#: a likely double-submission.
DUPLICATE_WINDOW_DAYS = 45

PASS, REVIEW, FAIL = "PASS", "REVIEW", "FAIL"
_VERDICT_RANK = {PASS: 0, REVIEW: 1, FAIL: 2}


# ----------------------------------------------------------------- resolution


def normalize_company_name(name: str) -> str:
    """Case-fold, expand `&`, and flatten punctuation to spaces.

    rapidfuzz is **case-sensitive by default**, which is not obvious and was a
    real defect here: "KRISHNA TEXTILES" scored 18.75 against "Krishna
    Textiles" and failed to resolve. Vendor names arrive upper-cased from OCR
    constantly, so this cost roughly a third of all resolutions.

    Legal suffixes (`Pvt Ltd`, `& Co`) are deliberately **kept**. Stripping
    them looks tidy and is wrong: "Arjun & Co" and "Arjun Pvt Ltd" are
    different companies that both reduce to "arjun", and merging them routed a
    clean invoice to the wrong supplier record -- which then mismatched on
    bank account, GSTIN and address at once, turning a correct invoice into a
    confident FAIL. The suffix is often the only thing distinguishing two real
    suppliers.
    """
    text = (name or "").lower().replace("&", " and ")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def name_similarity(a: str, b: str) -> float:
    """Similarity between two company names, 0-100.

    Takes the max of token-sort and plain ratio. Token-sort handles reordering
    ("Co Krishna Traders"); plain ratio handles lost spacing
    ("KrishnaTraders"), where token-sort sees a single token matching nothing.
    Neither alone covers what the OCR stage actually produces.
    """
    x, y = normalize_company_name(a), normalize_company_name(b)
    if not x or not y:
        return 0.0
    return max(float(fuzz.token_sort_ratio(x, y)), float(fuzz.ratio(x, y)))


@dataclass
class Resolution:
    vendor: VendorRecord | None
    method: str  # "gstin" | "exact_name" | "fuzzy_name" | "unresolved"
    score: float = 0.0
    candidates: list[tuple[str, float]] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.vendor is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "vendor": self.vendor.name if self.vendor else None,
            "method": self.method,
            "score": round(self.score, 2),
            "candidates": [(n, round(s, 2)) for n, s in self.candidates[:3]],
        }


def resolve_vendor(doc: dict[str, Any], master: VendorMaster) -> Resolution:
    """Identify which supplier sent this invoice.

    GSTIN first: it is the one globally unique identifier on an Indian invoice,
    so an exact hit beats any amount of name similarity -- including when the
    name is misspelled, abbreviated, or the OCR mangled it.
    """
    vendor = (doc.get("parties") or {}).get("vendor") or {}
    name = (vendor.get("name") or "").strip()
    gstin = normalize_gstin(vendor.get("gstin"))

    if gstin:
        record = master.by_gstin(gstin)
        if record:
            # Resolving on GSTIN alone silently redirects an invoice carrying
            # another supplier's GSTIN to *that* supplier -- whose record then
            # matches perfectly, so the mismatch can never be detected. It
            # scored 0/3 on injected GSTIN swaps. When the name points
            # somewhere else, report the conflict instead of picking a side.
            # Compare canonical entities, not billing names: an invoice under
            # a registered trading name is the same supplier, and reading it as
            # a GSTIN/name conflict would flag every aliased invoice as fraud.
            if name and master.canonical_name(name) != record.name:
                by_name = master.get(name)
                if by_name is None:
                    scored_names = [(c, name_similarity(name, c)) for c in master.names()]
                    best = max(scored_names, key=lambda kv: kv[1], default=(None, 0.0))
                    by_name = master.get(best[0]) if best[1] >= NAME_MATCH_THRESHOLD else None
                if by_name is not None and by_name.name != record.name:
                    return Resolution(
                        by_name,
                        "conflict",
                        0.0,
                        [(by_name.name, 100.0), (record.name, 100.0)],
                    )
            return Resolution(record, "gstin", 100.0)

    if not name:
        return Resolution(None, "unresolved", 0.0)

    if name in master:
        return Resolution(master.get(name), "exact_name", 100.0)

    scored = sorted(
        ((candidate, name_similarity(name, candidate)) for candidate in master.names()),
        key=lambda kv: -kv[1],
    )
    if scored and scored[0][1] >= NAME_MATCH_THRESHOLD:
        return Resolution(master.get(scored[0][0]), "fuzzy_name", scored[0][1], scored)

    return Resolution(None, "unresolved", scored[0][1] if scored else 0.0, scored)


# ----------------------------------------------------------------- validation


@dataclass(frozen=True)
class Check:
    code: str
    verdict: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "verdict": self.verdict,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass
class ValidationReport:
    doc_id: str
    resolution: Resolution
    checks: list[Check] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """The worst individual verdict wins. Averaging would let one FAIL be
        diluted by a handful of passes, which is the opposite of what a
        control is for."""
        if not self.checks:
            return PASS
        return max((c.verdict for c in self.checks), key=lambda v: _VERDICT_RANK[v])

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == FAIL]

    @property
    def reviews(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == REVIEW]

    def explain(self) -> str:
        if self.verdict == PASS:
            who = self.resolution.vendor.name if self.resolution.vendor else "unknown vendor"
            return f"PASS — consistent with {who}"
        worst = self.failures or self.reviews
        return f"{self.verdict} — " + "; ".join(c.message for c in worst[:3])

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "verdict": self.verdict,
            "resolution": self.resolution.as_dict(),
            "checks": [c.as_dict() for c in self.checks],
            "explanation": self.explain(),
        }


def _days_between(a: str | None, b: str | None) -> int | None:
    from datetime import datetime

    try:
        return abs(
            (datetime.strptime(str(a), "%Y-%m-%d") - datetime.strptime(str(b), "%Y-%m-%d")).days
        )
    except (TypeError, ValueError):
        return None


def validate_against_master(
    doc: dict[str, Any],
    master: VendorMaster,
    *,
    history: list[dict[str, Any]] | None = None,
) -> ValidationReport:
    """Run every vendor-record check over one invoice."""
    resolution = resolve_vendor(doc, master)
    checks: list[Check] = []
    vendor_block = (doc.get("parties") or {}).get("vendor") or {}
    payment = doc.get("payment_info") or {}

    if not resolution.resolved:
        checks.append(
            Check(
                "unknown_vendor",
                REVIEW,
                f"No supplier record matches {vendor_block.get('name')!r}",
                {
                    "name": vendor_block.get("name"),
                    "best_candidate": resolution.candidates[0] if resolution.candidates else None,
                },
            )
        )
        return ValidationReport(doc.get("doc_id", "<unknown>"), resolution, checks)

    record = resolution.vendor
    assert record is not None
    enough_history = record.invoice_count >= MIN_HISTORY_FOR_IDENTITY_CHECK

    # The name and the GSTIN point at different suppliers. On a real invoice
    # that is either a serious keying error or someone pasting a legitimate
    # company's tax number onto their own invoice.
    if resolution.method == "conflict":
        checks.append(
            Check(
                "gstin_mismatch",
                FAIL,
                f"GSTIN on this invoice belongs to a different supplier than the "
                f"name {record.name!r}",
                {
                    "invoice_gstin": normalize_gstin(vendor_block.get("gstin")),
                    "name_resolves_to": record.name,
                    "known_gstins": sorted(record.gstins),
                },
            )
        )

    # --- bank account -------------------------------------------------------
    # The highest-value check in the whole pipeline. Invoice-redirection fraud
    # works by submitting a genuine-looking invoice from a real supplier with
    # the attacker's account number on it; nothing else on the page changes.
    account = payment.get("account_number")
    if (
        account
        and record.bank_accounts
        and enough_history
        and str(account).strip() not in record.bank_accounts
    ):
        checks.append(
            Check(
                "bank_account_changed",
                FAIL,
                f"Bank account {account} has never been used by {record.name}",
                {
                    "invoice_account": str(account),
                    "known_accounts": sorted(record.bank_accounts),
                    "vendor_invoice_count": record.invoice_count,
                },
            )
        )

    # --- GSTIN --------------------------------------------------------------
    gstin = normalize_gstin(vendor_block.get("gstin"))
    if gstin and record.gstins and enough_history and gstin not in record.gstins:
        checks.append(
            Check(
                "gstin_mismatch",
                FAIL,
                f"GSTIN {gstin} does not match the one on file for {record.name}",
                {"invoice_gstin": gstin, "known_gstins": sorted(record.gstins)},
            )
        )

    # --- address ------------------------------------------------------------
    address = (vendor_block.get("address") or "").strip()
    if address and record.addresses and enough_history and address not in record.addresses:
        best = max(
            (float(fuzz.token_sort_ratio(address, known)) for known in record.addresses),
            default=0.0,
        )
        # Addresses are reformatted constantly, so only a large divergence is
        # worth a reviewer's time -- and it is REVIEW, never FAIL.
        if best < 75.0:
            checks.append(
                Check(
                    "address_mismatch",
                    REVIEW,
                    f"Billing address differs from the one on file for {record.name}",
                    {"invoice_address": address, "best_similarity": round(best, 1)},
                )
            )

    # --- contracted rates ---------------------------------------------------
    for i, item in enumerate(doc.get("line_items") or []):
        key = rate_key(item.get("description"))
        price = item.get("unit_price")
        if not key or price is None:
            continue

        rate, source = master.rate_for(record, key)
        if rate is None or source == "none":
            continue

        # A vendor's own contract price barely moves, so a small deviation is
        # already meaningful. Pooled market prices legitimately vary between
        # suppliers, and by different amounts per item -- so that band is taken
        # from the spread the item actually shows rather than assumed flat.
        #
        # The flat percentages remain as floors. An item every supplier bills
        # at the same price has no spread, and without a floor any rounding
        # would read as an anomaly.
        if source == "vendor":
            review_at, fail_at = PRICE_DEVIATION_REVIEW, PRICE_DEVIATION_FAIL
        else:
            review_at = rate.band(MARKET_DEVIATION_REVIEW)
            fail_at = max(MARKET_DEVIATION_FAIL, review_at * 2.5)

        deviation = rate.deviation(float(price))
        if abs(deviation) >= fail_at:
            verdict = FAIL
        elif abs(deviation) >= review_at:
            verdict = REVIEW
        else:
            continue

        against = "this vendor's usual" if source == "vendor" else "the market"
        # Say which band was applied, so a reviewer can tell a tight-priced
        # item from a loosely-priced one without opening the ledger.
        band_note = (
            ""
            if source == "vendor"
            else f" (band {review_at:.0%}, from {rate.observations} observations)"
        )
        checks.append(
            Check(
                "price_deviation",
                verdict,
                f"Line {i + 1} at {price:,.2f} is {deviation:+.0%} against "
                f"{against} {rate.median_unit_price:,.2f}{band_note}",
                {
                    "index": i,
                    "description": item.get("description"),
                    "unit_price": float(price),
                    "reference_price": rate.median_unit_price,
                    "reference_source": source,
                    "deviation": round(deviation, 4),
                    "observations": rate.observations,
                    "review_band": round(review_at, 4),
                    "observed_spread": round(rate.observed_spread, 4),
                },
            )
        )

    # --- duplicate payment --------------------------------------------------
    total = (doc.get("totals") or {}).get("grand_total")
    invoice_date = (doc.get("invoice_meta") or {}).get("invoice_date")
    if total is not None and history:
        for past in history:
            past_vendor = ((past.get("parties") or {}).get("vendor") or {}).get("name")
            if past_vendor != record.name or past.get("doc_id") == doc.get("doc_id"):
                continue
            past_total = (past.get("totals") or {}).get("grand_total")
            if past_total is None or abs(float(past_total) - float(total)) > 0.01:
                continue
            gap = _days_between(invoice_date, (past.get("invoice_meta") or {}).get("invoice_date"))
            if gap is not None and gap <= DUPLICATE_WINDOW_DAYS:
                checks.append(
                    Check(
                        "possible_duplicate_payment",
                        FAIL,
                        f"Same vendor and amount ({total:,.2f}) as {past.get('doc_id')} "
                        f"{gap} days apart",
                        {
                            "matching_doc_id": past.get("doc_id"),
                            "amount": float(total),
                            "days_apart": gap,
                        },
                    )
                )
                break

    return ValidationReport(doc.get("doc_id", "<unknown>"), resolution, checks)
