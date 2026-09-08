"""OCR + regex baseline extractor.

This is the honest floor a VLM has to clear. Label-anchored regex over OCR text
is what a competent engineer builds in an afternoon without any model, and
refs [1] and [6] are clear that it works acceptably on consistent layouts and
falls apart on unseen ones. Reporting VLM numbers without it would leave the
obvious question -- "does the model earn its 4GB of VRAM?" -- unanswered.

The design is deliberately unheroic. Every rule is label-anchored (find
"Invoice No", take what follows) rather than position-anchored, because
position rules would be tuned to our own renderer and would flatter this
baseline on our corpus while collapsing on real documents. Line-item parsing
uses a generic "row with several numbers" heuristic for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from invoice_extract.models.postprocess import (
    parse_amount,
    parse_quantity,
    parse_rate,
)

# A money-shaped token: optional currency marker, digits with separators.
_MONEY = re.compile(r"(?:[₹$€£]|\bRs\.?|\bINR\b)?\s*-?\d[\d,]*\.?\d{0,2}\b")
_GSTIN_ANYWHERE = re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?:\+\d{1,3}[-\s]?)?\b\d{5}[-\s]?\d{5}\b|\b\d{10}\b")
_DATE_TOKEN = re.compile(
    r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}\b"
    r"|\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}\b"
)

# Label -> canonical field. Longest labels first so "Invoice Date" is not
# consumed by the "Invoice" prefix of "Invoice No".
_META_LABELS: list[tuple[str, str]] = [
    (r"invoice\s*(?:no|number|#)", "invoice_number"),
    (r"bill\s*(?:no|number)", "invoice_number"),
    (r"invoice\s*date", "invoice_date"),
    (r"due\s*date", "due_date"),
    (r"date\s*of\s*issue", "invoice_date"),
    (r"p\.?o\.?\s*(?:no|number)", "po_number"),
    (r"payment\s*terms", "payment_terms"),
    (r"\bterms\b", "payment_terms"),
]

_TOTAL_LABELS: list[tuple[str, str]] = [
    (r"grand\s*total", "grand_total"),
    (r"(?:net|total)\s*(?:amount\s*)?payable", "grand_total"),
    (r"\btotal\b", "grand_total"),
    (r"sub\s*-?\s*total", "subtotal"),
    (r"taxable\s*(?:value|amount)", "subtotal"),
    (r"round\s*-?\s*off", "round_off"),
    (r"\bdiscount\b", "discount_total"),
    (r"shipping|freight|delivery\s*charge", "shipping"),
    (r"tax\s*total|total\s*tax", "tax_total"),
]

_TAX_TYPES = ["CGST", "SGST", "IGST", "CESS", "VAT", "GST"]

_NOISE_LINES = re.compile(
    r"^\s*(tax\s+invoice|invoice|receipt|bill\s+of\s+supply|original\s+for\s+recipient)\s*$",
    re.IGNORECASE,
)


def _last_amount(text: str) -> float | None:
    """Rightmost money-shaped token on a line.

    Invoice rows put the label on the left and the value on the right, so the
    last number is the value far more often than the first -- and the first is
    frequently part of the label itself ("CGST @ 9%").
    """
    matches = [m.group() for m in _MONEY.finditer(text) if any(c.isdigit() for c in m.group())]
    return parse_amount(matches[-1]) if matches else None


def _is_standalone(line: str, match: re.Match) -> bool:
    """Is this number a column value rather than part of a word?

    A digit run glued to letters belongs to the description -- the "4" of
    "A4 Copier Paper", the "500" of "(500 sheets)" -- and stripping it would
    corrupt the field we are trying to read. Column values are whitespace
    delimited (a trailing "%" still counts as delimited).
    """
    text = match.group()
    if text != text.lstrip():
        # `_MONEY` allows leading whitespace, so a match that starts with space
        # is delimited by construction.
        before_ok = True
    else:
        # Whitespace specifically, not merely "non-alphanumeric": the "500" of
        # "(500 sheets)" is preceded by a bracket, and it is description text.
        before_ok = match.start() == 0 or line[match.start() - 1].isspace()
    after = line[match.end() : match.end() + 1]
    return before_ok and (after == "" or after.isspace() or after in "%,)")


def _value_after_label(line: str, label_pattern: str) -> str | None:
    """Text following a label on the same line.

    The label pattern is wrapped in a non-capturing group and the value is
    captured by name. Without the wrap, an alternating pattern such as
    `\\bifsc\\b|\\bswift\\b` binds the trailing value expression to only its
    last branch; without the name, any capturing group inside a label pattern
    would shift the value's index.
    """
    match = re.search(rf"(?:{label_pattern})\s*[:\-]?\s*(?P<value>.+)$", line, re.IGNORECASE)
    if not match:
        return None
    value = match.group("value").strip(" :-\t")
    return value or None


class RuleBasedExtractor:
    """Label-anchored regex extraction over OCR text.

    Implements the `InvoiceExtractor` protocol when constructed with an OCR
    engine; `predict_from_text` is the engine-free entry point used by tests
    and by the runner when OCR text has already been computed.
    """

    name = "rules"

    def __init__(self, ocr_engine: Any = None, preprocess: Any = None) -> None:
        self.ocr_engine = ocr_engine
        self.preprocess = preprocess

    # ------------------------------------------------------------- protocol

    def predict(self, image_paths: list[Path]) -> dict[str, Any]:
        if self.ocr_engine is None:
            raise RuntimeError(
                "RuleBasedExtractor needs an OCR engine to read an image. "
                "Construct it with ocr_engine=..., or call predict_from_text()."
            )
        import numpy as np
        from PIL import Image

        texts = []
        for path in image_paths:
            image = np.array(Image.open(path).convert("L"))
            if self.preprocess is not None:
                image = self.preprocess(image).image
            texts.append(self.ocr_engine.recognize(image).text)
        return self.predict_from_text("\n".join(texts))

    # ------------------------------------------------------------- the rules

    def predict_from_text(self, text: str) -> dict[str, Any]:
        lines = [ln.strip() for ln in (text or "").splitlines()]
        lines = [ln for ln in lines if ln]

        return {
            "document_type": self._document_type(text),
            "parties": self._parties(lines),
            "invoice_meta": self._meta(lines),
            "line_items": self._line_items(lines),
            "tax_lines": self._tax_lines(lines),
            "totals": self._totals(lines),
            "payment_info": self._payment_info(lines),
        }

    def _document_type(self, text: str) -> str:
        head = (text or "")[:400].lower()
        return "receipt" if "receipt" in head and "invoice" not in head else "invoice"

    def _parties(self, lines: list[str]) -> dict[str, Any]:
        gstins = [m.group() for ln in lines for m in _GSTIN_ANYWHERE.finditer(ln)]

        # The vendor name is the first substantive line: letterheads lead with
        # it, and the only things above it are document-type banners.
        vendor_name = None
        for line in lines[:8]:
            if _NOISE_LINES.match(line) or _GSTIN_ANYWHERE.search(line):
                continue
            if sum(c.isalpha() for c in line) >= 4:
                vendor_name = line
                break

        vendor: dict[str, Any] = {"name": vendor_name}
        if gstins:
            vendor["gstin"] = gstins[0]

        # The address is the run of lines between the vendor name and the first
        # labelled field, joined back together -- letterhead addresses wrap
        # across two or three lines and only make sense reassembled.
        if vendor_name is not None:
            start = lines.index(vendor_name) + 1
            address_lines: list[str] = []
            for entry in lines[start : start + 4]:
                if re.search(
                    r"\b(gstin|phone|tel|mobile|email|invoice|bill\s*to|date)\b",
                    entry,
                    re.IGNORECASE,
                ) or _EMAIL.search(entry):
                    break
                if sum(c.isalnum() for c in entry) >= 3:
                    address_lines.append(entry)
            if address_lines:
                vendor["address"] = " ".join(address_lines)
        for line in lines[:15]:
            if not vendor.get("email"):
                email = _EMAIL.search(line)
                if email:
                    vendor["email"] = email.group()
            if not vendor.get("phone") and re.search(r"phone|tel|mobile", line, re.IGNORECASE):
                phone = _PHONE.search(line)
                if phone:
                    vendor["phone"] = phone.group()

        parties: dict[str, Any] = {"vendor": vendor}

        # The buyer block follows a "Bill To" / "Ship To" marker.
        for i, line in enumerate(lines):
            if re.search(r"\b(bill\s*to|buyer|customer|billed\s*to)\b", line, re.IGNORECASE):
                block = lines[i + 1 : i + 5]
                buyer: dict[str, Any] = {}
                for entry in block:
                    if _GSTIN_ANYWHERE.search(entry):
                        buyer["gstin"] = _GSTIN_ANYWHERE.search(entry).group()
                    elif not buyer.get("name") and sum(c.isalpha() for c in entry) >= 3:
                        buyer["name"] = entry
                    elif buyer.get("name") and not buyer.get("address"):
                        buyer["address"] = entry
                if buyer:
                    parties["buyer"] = buyer
                break

        return parties

    def _meta(self, lines: list[str]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "invoice_number": None,
            "invoice_date": None,
            "due_date": None,
            "po_number": None,
            "payment_terms": None,
        }
        for line in lines:
            for pattern, field_name in _META_LABELS:
                if meta.get(field_name) is not None:
                    continue
                if not re.search(pattern, line, re.IGNORECASE):
                    continue
                value = _value_after_label(line, pattern)
                if not value:
                    continue
                if field_name.endswith("date"):
                    date_token = _DATE_TOKEN.search(value)
                    meta[field_name] = date_token.group() if date_token else None
                else:
                    meta[field_name] = value.split()[0] if field_name.endswith("number") else value
        return meta

    def _totals(self, lines: list[str]) -> dict[str, Any]:
        totals: dict[str, Any] = {}
        for line in lines:
            for pattern, field_name in _TOTAL_LABELS:
                if field_name in totals:
                    continue
                if re.search(pattern, line, re.IGNORECASE):
                    amount = _last_amount(line)
                    if amount is not None:
                        totals[field_name] = amount
                    break
        totals.setdefault("grand_total", None)
        return totals

    def _tax_lines(self, lines: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line in lines:
            for tax_type in _TAX_TYPES:
                if tax_type in seen or not re.search(rf"\b{tax_type}\b", line, re.IGNORECASE):
                    continue
                amount = _last_amount(line)
                if amount is None:
                    continue
                entry: dict[str, Any] = {"type": tax_type, "amount": amount}
                rate = re.search(r"@?\s*(\d{1,2}(?:\.\d+)?)\s*%", line)
                if rate:
                    entry["rate"] = parse_rate(rate.group(1))
                out.append(entry)
                seen.add(tax_type)
        return out

    def _line_items(self, lines: list[str]) -> list[dict[str, Any]]:
        """Rows carrying several numbers, between the table header and the totals.

        This is where a rules baseline is genuinely weak, and that weakness is
        the point: it is the gap a VLM is supposed to close.
        """
        start, end = 0, len(lines)
        for i, line in enumerate(lines):
            if re.search(
                r"\bdescription\b|\bparticulars\b|\bitem\b", line, re.IGNORECASE
            ) and re.search(r"\bamount\b|\bqty\b|\bquantity\b|\brate\b", line, re.IGNORECASE):
                start = i + 1
                break
        for i in range(start, len(lines)):
            if re.search(
                r"sub\s*-?\s*total|grand\s*total|taxable\s*value", lines[i], re.IGNORECASE
            ):
                end = i
                break

        items: list[dict[str, Any]] = []
        for row_index, line in enumerate(lines[start:end], start=1):
            # Work with match *spans*, not the matched strings. Deleting the
            # string "1" from a row would also gut "18%" and "12,204.22"; a
            # span deletion touches only the token that was actually matched.
            # Only whitespace-delimited numbers are column values. A digit run
            # glued to other characters belongs to the description -- the "4"
            # of "A4 Copier Paper", the "500" of "(500 sheets)" -- and removing
            # it would corrupt the very field we are trying to read.
            matches = [
                m
                for m in _MONEY.finditer(line)
                if any(c.isdigit() for c in m.group()) and _is_standalone(line, m)
            ]
            if len(matches) < 3:
                continue

            # A leading small integer at the very start of the row is the row
            # index, not data. Only strip it when it matches this row's
            # position, so a genuine leading quantity survives.
            line_no = None
            if matches and matches[0].start() <= 2:
                head = parse_quantity(matches[0].group())
                if head is not None and head == float(row_index):
                    line_no = row_index
                    matches = matches[1:]
            if len(matches) < 2:
                continue

            description = line
            for match in reversed(matches):
                description = description[: match.start()] + " " + description[match.end() :]
            if line_no is not None:
                description = re.sub(r"^\s*\d+\s*", "", description)
            description = re.sub(r"[%|]", " ", description)
            description = re.sub(r"\s{2,}", " ", description).strip(" .,-|")
            if not description or sum(c.isalpha() for c in description) < 3:
                continue

            values = [parse_amount(m.group()) for m in matches]
            values = [v for v in values if v is not None]
            if len(values) < 2:
                continue

            item: dict[str, Any] = {"description": description, "line_total": values[-1]}
            if line_no is not None:
                item["line_no"] = line_no

            rate_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", line)
            if rate_match:
                item["tax_rate"] = parse_rate(rate_match.group(1))
                # The rate is one of the numbers we collected; drop it so it is
                # not mistaken for a quantity or a price.
                rate_value = parse_rate(rate_match.group(1))
                values = [v for v in values if v != rate_value] or values

            hsn = re.search(r"\b\d{4,8}\b", description) or re.search(r"\b\d{4,8}\b", line)
            if hsn and parse_amount(hsn.group()) not in (item.get("line_total"),):
                item["hsn_sac_code"] = hsn.group()
                values = [v for v in values if v != parse_amount(hsn.group())] or values

            # Remaining columns, left to right, are conventionally
            # quantity -> rate -> amount.
            if len(values) >= 3:
                item["quantity"] = values[-3]
                item["unit_price"] = values[-2]
            elif len(values) == 2:
                item["unit_price"] = values[-2]

            items.append(item)

        return items

    def _payment_info(self, lines: list[str]) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for line in lines:
            for pattern, key in (
                (r"\bbank\b", "bank_name"),
                (r"a\s*/?\s*c\s*(?:no|number)|account\s*(?:no|number)", "account_number"),
                (r"\bifsc\b|\bswift\b", "ifsc_or_swift"),
                (r"\bupi\b", "upi_id"),
            ):
                if key in info or not re.search(pattern, line, re.IGNORECASE):
                    continue
                value = _value_after_label(line, pattern)
                if value:
                    info[key] = value.split()[0] if key != "bank_name" else value
        return info


class NullExtractor:
    """Predicts nothing. The absolute floor, and a check that the harness is
    not somehow scoring above zero on an empty prediction."""

    name = "null"

    def predict(self, image_paths: list[Path]) -> dict[str, Any]:
        return {
            "parties": {"vendor": {}},
            "invoice_meta": {"invoice_number": None, "invoice_date": None},
            "line_items": [],
            "tax_lines": [],
            "totals": {"grand_total": None},
        }

    def predict_from_text(self, text: str) -> dict[str, Any]:
        return self.predict([])
