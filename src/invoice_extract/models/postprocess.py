"""Deterministic repair and validation of a model's raw extraction output.

A VLM returns plausible-looking JSON; it does not return *canonical* JSON. It
writes dates as "12/01/2026", amounts as "Rs. 1,234.56", quantities as "12
nos", and occasionally a field the schema has never heard of. Fixing that with
rules is strictly better than asking the model to be more careful, because
rules are deterministic, testable, and free.

Two principles run through this module:

1. **Normalise silently, never invent.** A value that cannot be parsed becomes
   `None`, not a guess. A wrong value is far more damaging downstream than a
   missing one -- Phase 3 can retrieve context for a missing field, but it will
   happily enrich and validate a confidently wrong one.

2. **Flag arithmetic, never fix it.** If `subtotal + tax != total`, that is
   either an extraction error or a genuinely malformed invoice, and those need
   opposite responses. Silently rewriting the total to make the sum work would
   destroy exactly the signal Phase 3's anomaly detection is built to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ------------------------------------------------------------------- dates

# Ordered by specificity. Indian invoices are day-first, which is also the
# ISO-adjacent reading, so ambiguous d/m/Y is resolved day-first by default.
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%d-%m-%y",
    "%d/%m/%y",
    "%d.%m.%y",
    "%d-%b-%Y",
    "%d %b %Y",
    "%d-%B-%Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d %Y",
    "%B %d %Y",
    "%Y%m%d",
]
_MONTH_FIRST_FORMATS = ["%m-%d-%Y", "%m/%d/%Y", "%m-%d-%y", "%m/%d/%y"]

_DATE_CLEAN = re.compile(r"[^0-9A-Za-z/.\- ]")


def normalize_date(value: Any, day_first: bool = True) -> str | None:
    """Parse a date into ISO `YYYY-MM-DD`, or return None if it cannot be read.

    `day_first` decides the genuinely ambiguous case (03/04/2026). It defaults
    to True because this project's primary corpus is Indian GST invoices; a
    US-sourced corpus would want it False. Unambiguous inputs (a day > 12, or a
    named month) are parsed correctly either way.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    text = _DATE_CLEAN.sub(" ", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    formats = _DATE_FORMATS if day_first else _MONTH_FIRST_FORMATS + _DATE_FORMATS
    if day_first:
        formats = _DATE_FORMATS + _MONTH_FIRST_FORMATS

    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        # A two-digit year in an invoice context is this century, not 1970.
        if parsed.year < 1970:
            parsed = parsed.replace(year=parsed.year + 100)
        return parsed.strftime("%Y-%m-%d")
    return None


# ----------------------------------------------------------------- amounts

_CURRENCY_SYMBOLS = "₹$€£¥"
_AMOUNT_STRIP = re.compile(rf"[{_CURRENCY_SYMBOLS}]|(?i:rs\.?|inr|usd|eur|gbp)")
_AMOUNT_CHARS = re.compile(r"[^0-9.,\-()]")

CURRENCY_BY_SYMBOL = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
CURRENCY_BY_WORD = {
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupees": "INR",
    "usd": "USD",
    "eur": "EUR",
    "gbp": "GBP",
    "jpy": "JPY",
}


def parse_amount(value: Any) -> float | None:
    """Parse a monetary or numeric value, or return None.

    Handles currency symbols and codes, thousands separators in both the
    1,234.56 and 1.234,56 conventions, and accounting-style negatives in
    parentheses. The separator convention is inferred from which character
    appears last, which is correct for every well-formed number and is the only
    signal available without knowing the locale.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = _AMOUNT_STRIP.sub(" ", str(value))
    text = _AMOUNT_CHARS.sub("", text).strip()
    if not text:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if text.startswith("-"):
        negative = True
        text = text[1:]
    text = text.replace("-", "")
    if not text:
        return None

    last_dot, last_comma = text.rfind("."), text.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        # Whichever comes last is the decimal separator.
        if last_comma > last_dot:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif last_comma >= 0:
        # A lone comma is a decimal point only when it is not in a thousands
        # position: "1,50" is 1.5, but "1,500" is 1500.
        tail = text[last_comma + 1 :]
        text = text.replace(",", "." if len(tail) != 3 else "")

    try:
        amount = float(text)
    except ValueError:
        return None
    return -amount if negative else amount


def parse_quantity(value: Any) -> float | None:
    """Quantities arrive with units attached ("12 nos", "2.5 kg")."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d[\d,]*\.?\d*", str(value))
    return parse_amount(match.group()) if match else None


def parse_rate(value: Any) -> float | None:
    """Tax rates arrive as "18%", "18.0", or "@18 %"."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return parse_quantity(str(value).replace("%", " "))


def detect_currency(*values: Any) -> str | None:
    """Infer an ISO-4217 code from any currency marker seen in the raw text."""
    for value in values:
        if value is None:
            continue
        text = str(value)
        for symbol, code in CURRENCY_BY_SYMBOL.items():
            if symbol in text:
                return code
        for word in re.findall(r"[A-Za-z.]+", text.lower()):
            if word in CURRENCY_BY_WORD:
                return CURRENCY_BY_WORD[word]
    return None


# ------------------------------------------------------------------- GSTIN

GSTIN_RE = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
_GSTIN_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# State codes 01-38 plus 97 (other territory) and 99 (centre) are assigned.
_VALID_STATE_CODES = {f"{i:02d}" for i in range(1, 39)} | {"97", "99"}


def normalize_gstin(value: Any) -> str | None:
    """Strip spacing and case. Returns None only if nothing plausible remains."""
    if value is None:
        return None
    text = re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()
    return text or None


def gstin_check_digit(first_14: str) -> str | None:
    """Compute the GSTIN check character (the standard mod-36 algorithm)."""
    if len(first_14) != 14 or any(c not in _GSTIN_CHARSET for c in first_14):
        return None
    base = len(_GSTIN_CHARSET)
    factor = 2
    total = 0
    for char in reversed(first_14):
        product = _GSTIN_CHARSET.index(char) * factor
        factor = 1 if factor == 2 else 2
        total += product // base + product % base
    return _GSTIN_CHARSET[(base - total % base) % base]


def validate_gstin(value: Any) -> list[str]:
    """Return a list of problems with a GSTIN (empty = structurally valid).

    Checked in three independent layers -- format, state code, checksum -- and
    reported separately, because they fail for different reasons. A checksum
    failure on an otherwise well-formed GSTIN usually means one OCR character
    was misread; a format failure usually means the wrong field was extracted.
    """
    gstin = normalize_gstin(value)
    if gstin is None:
        return ["gstin is empty"]

    problems: list[str] = []
    if len(gstin) != 15:
        return [f"gstin has {len(gstin)} characters, expected 15"]
    if not GSTIN_RE.match(gstin):
        problems.append("gstin does not match the expected format")
    if gstin[:2] not in _VALID_STATE_CODES:
        problems.append(f"gstin state code {gstin[:2]!r} is not an assigned code")

    expected = gstin_check_digit(gstin[:14])
    if expected is not None and expected != gstin[14]:
        problems.append(f"gstin checksum is {gstin[14]!r}, expected {expected!r}")
    return problems


# -------------------------------------------------------------- arithmetic


@dataclass
class ConsistencyReport:
    """Arithmetic problems found in an extracted document.

    Deliberately separate from the document itself: these are *flags for
    review*, not corrections. Phase 3's anomaly detection consumes them, and
    Phase 5's review UI surfaces them.
    """

    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __bool__(self) -> bool:  # truthy when there is something to look at
        return bool(self.problems)


def check_arithmetic(doc: dict[str, Any], tolerance: float = 0.02) -> ConsistencyReport:
    """Verify the internal arithmetic of an extracted invoice.

    Tolerance is relative, because invoices legitimately round at several
    points (per-line tax, then a whole-invoice round-off), and an absolute
    epsilon would fire constantly on large totals while missing real errors on
    small ones.
    """
    problems: list[str] = []
    totals = doc.get("totals") or {}
    line_items = doc.get("line_items") or []
    tax_lines = doc.get("tax_lines") or []

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= tolerance * max(abs(a), abs(b), 1.0)

    line_sum = sum(li["line_total"] for li in line_items if li.get("line_total") is not None)
    subtotal = totals.get("subtotal")
    if line_items and subtotal is not None and line_sum and not close(line_sum, subtotal):
        problems.append(f"sum(line_items)={line_sum:.2f} != subtotal={subtotal:.2f}")

    tax_sum = sum(t["amount"] for t in tax_lines if t.get("amount") is not None)
    tax_total = totals.get("tax_total")
    if tax_lines and tax_total is not None and not close(tax_sum, tax_total):
        problems.append(f"sum(tax_lines)={tax_sum:.2f} != tax_total={tax_total:.2f}")

    grand = totals.get("grand_total")
    if grand is not None and subtotal is not None:
        # `subtotal` is the net taxable value: discounts are already netted into
        # it via each row's `line_total`, so subtracting `discount_total` here
        # would remove them twice. See schema/schema.md for the convention.
        expected = (
            subtotal
            + (tax_total if tax_total is not None else tax_sum)
            + (totals.get("shipping") or 0.0)
            + (totals.get("round_off") or 0.0)
        )
        if not close(expected, grand):
            problems.append(
                f"subtotal+tax+shipping+round_off={expected:.2f} != grand_total={grand:.2f}"
            )

    for i, li in enumerate(line_items):
        qty, price, total = li.get("quantity"), li.get("unit_price"), li.get("line_total")
        if (
            qty is not None
            and price is not None
            and total is not None
            # Expected: quantity x unit_price, net of the row discount, before
            # tax. The tax-inclusive form is still accepted because plenty of
            # real invoices print it that way, and flagging every one of those
            # would drown the genuine errors.
            and not close(qty * price - (li.get("discount") or 0.0), total)
            and not close(qty * price, total)
            and not close(qty * price + (li.get("tax_amount") or 0.0), total)
        ):
            problems.append(
                f"line_items[{i}]: quantity*unit_price-discount="
                f"{qty * price - (li.get('discount') or 0.0):.2f} != line_total={total:.2f}"
            )

    return ConsistencyReport(problems)


# ------------------------------------------------------- document coercion

_PARTY_KEYS = {"name", "address", "tax_id", "gstin", "phone", "email"}
_LINE_ITEM_NUMERIC = {
    "quantity": parse_quantity,
    "unit_price": parse_amount,
    "discount": parse_amount,
    "tax_rate": parse_rate,
    "tax_amount": parse_amount,
    "line_total": parse_amount,
}
_TOTALS_KEYS = [
    "subtotal",
    "discount_total",
    "tax_total",
    "shipping",
    "round_off",
    "grand_total",
]
_VALID_TAX_TYPES = {"CGST", "SGST", "IGST", "CESS", "VAT", "GST", "SALES_TAX", "OTHER"}


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _coerce_party(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out = {k: _clean_str(v) for k, v in raw.items() if k in _PARTY_KEYS}

    if out.get("gstin"):
        gstin = normalize_gstin(out["gstin"])
        # Keep anything structurally GSTIN-shaped, even with a bad check digit
        # -- that is a real anomaly signal Phase 3 wants. Drop anything else: a
        # model that writes an address into `gstin` has extracted the wrong
        # field, and passing it through produces a schema-invalid document,
        # which breaks this pipeline's one hard guarantee. Observed for real:
        # Qwen2-VL-2B put a 39-character US address here on a FATURA2 invoice.
        out["gstin"] = gstin if gstin and GSTIN_RE.match(gstin) else None

    return {k: v for k, v in out.items() if v is not None}


def coerce_document(raw: dict[str, Any], day_first: bool = True) -> dict[str, Any]:
    """Turn a model's raw JSON into the canonical shape.

    Unknown keys are dropped rather than passed through: the schema is
    `extra="forbid"`, so a single hallucinated field would fail validation for
    the whole document and cost us every field the model got right.

    `doc_id`, `source` and `pages` are the caller's to fill in -- the model
    never sees them and must not invent them.
    """
    raw = raw if isinstance(raw, dict) else {}

    parties_raw = raw.get("parties") if isinstance(raw.get("parties"), dict) else {}
    parties: dict[str, Any] = {"vendor": _coerce_party(parties_raw.get("vendor"))}
    for optional in ("buyer", "ship_to"):
        party = _coerce_party(parties_raw.get(optional))
        if party:
            parties[optional] = party

    meta_raw = raw.get("invoice_meta") if isinstance(raw.get("invoice_meta"), dict) else {}
    invoice_meta = {
        "invoice_number": _clean_str(meta_raw.get("invoice_number")),
        "invoice_date": normalize_date(meta_raw.get("invoice_date"), day_first),
        "due_date": normalize_date(meta_raw.get("due_date"), day_first),
        "po_number": _clean_str(meta_raw.get("po_number")),
        "payment_terms": _clean_str(meta_raw.get("payment_terms")),
    }

    line_items = []
    for item in raw.get("line_items") or []:
        if not isinstance(item, dict):
            continue
        coerced: dict[str, Any] = {
            "description": _clean_str(item.get("description")),
            "hsn_sac_code": _clean_str(item.get("hsn_sac_code")),
            "unit": _clean_str(item.get("unit")),
        }
        line_no = parse_quantity(item.get("line_no"))
        if line_no is not None:
            coerced["line_no"] = int(line_no)
        for key, parser in _LINE_ITEM_NUMERIC.items():
            coerced[key] = parser(item.get(key))
        # A row with nothing on it is a hallucinated row, not an empty one.
        if any(v is not None for k, v in coerced.items() if k != "line_no"):
            line_items.append({k: v for k, v in coerced.items() if v is not None})

    tax_lines = []
    for tax in raw.get("tax_lines") or []:
        if not isinstance(tax, dict):
            continue
        amount = parse_amount(tax.get("amount"))
        if amount is None:
            continue  # `amount` is required by the schema
        tax_type = (_clean_str(tax.get("type")) or "OTHER").upper().replace(" ", "_")
        entry: dict[str, Any] = {
            "type": tax_type if tax_type in _VALID_TAX_TYPES else "OTHER",
            "amount": amount,
        }
        rate = parse_rate(tax.get("rate"))
        if rate is not None:
            entry["rate"] = rate
        tax_lines.append(entry)

    totals_raw = raw.get("totals") if isinstance(raw.get("totals"), dict) else {}
    totals: dict[str, Any] = {k: parse_amount(totals_raw.get(k)) for k in _TOTALS_KEYS}
    totals["amount_in_words"] = _clean_str(totals_raw.get("amount_in_words"))

    # Derive tax_total from its printed components when the document shows the
    # individual taxes but no combined line -- which is the norm on Indian GST
    # invoices, where CGST and SGST are printed separately and never summed.
    # This is arithmetic over values that were read off the page, not a guess:
    # it fires only when every component carries an amount.
    if totals["tax_total"] is None and tax_lines:
        totals["tax_total"] = round(sum(t["amount"] for t in tax_lines), 2)

    totals = {k: v for k, v in totals.items() if v is not None or k == "grand_total"}

    doc: dict[str, Any] = {
        "schema_version": "1.0.0",
        "document_type": (
            "receipt" if str(raw.get("document_type", "")).lower() == "receipt" else "invoice"
        ),
        "parties": parties,
        "invoice_meta": invoice_meta,
        "line_items": line_items,
        "tax_lines": tax_lines,
        "totals": totals,
    }

    currency = _clean_str(raw.get("currency"))
    currency = (currency or "").upper() if currency else None
    if not (currency and len(currency) == 3 and currency.isalpha()):
        currency = detect_currency(
            totals_raw.get("grand_total"), totals_raw.get("subtotal"), raw.get("currency")
        )
    if currency:
        doc["currency"] = currency

    payment_raw = raw.get("payment_info") if isinstance(raw.get("payment_info"), dict) else {}
    payment = {
        k: _clean_str(payment_raw.get(k))
        for k in ("bank_name", "account_number", "ifsc_or_swift", "upi_id", "mode", "terms")
    }
    payment = {k: v for k, v in payment.items() if v is not None}
    if payment:
        doc["payment_info"] = payment

    return doc


def finalize(
    raw: dict[str, Any],
    doc_id: str,
    source: dict[str, Any],
    pages: list[dict[str, Any]],
    day_first: bool = True,
) -> tuple[dict[str, Any], ConsistencyReport]:
    """Coerce, attach caller-owned identity fields, and run the arithmetic check.

    Returns the document and its consistency report separately: the report is
    review metadata, and writing it into the document would put a field in the
    prediction that the gold annotation does not have.
    """
    doc = coerce_document(raw, day_first=day_first)
    doc["doc_id"] = doc_id
    doc["source"] = source
    doc["pages"] = pages
    return doc, check_arithmetic(doc)
