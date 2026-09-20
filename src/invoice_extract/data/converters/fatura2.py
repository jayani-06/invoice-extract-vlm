"""FATURA2 (Hugging Face mirror) -> canonical schema converter.

Source: `mathieu1256/FATURA2-invoices` on the Hugging Face Hub, CC-BY-4.0.
10,000 synthetic invoices across 50 layout templates, from Limam, Dhiaf and
Kessentini (arXiv:2311.11856). This is the mirror rather than the original
Zenodo release because Zenodo was returning 504s; the parquet mirror carries
the same images plus token/bbox/tag annotations in a single file.

**The tag vocabulary is not shipped with the parquet.** `ner_tags` are bare
integers with no `ClassLabel` names in the file metadata, so the map below was
inferred by sampling every tag across all 1,400 test documents and reading the
tokens it covers. That audit is reproducible via
`scripts/convert_fatura2.py --audit-tags`, and the evidence is recorded in
`TAG_EVIDENCE` so a future release with shifted ids is caught rather than
silently mis-mapped.

**What this dataset can and cannot evaluate.** Tag 10 is a single placeholder
token, `"table"`, covering the whole line-item region: FATURA2 does *not*
tokenise line items. So this corpus supports header-field evaluation (vendor,
invoice number, invoice date, due date, buyer, total) and nothing at the line
level. Any line-item metric computed against it would be measuring an empty
gold list, which is why the converter writes `line_items: []` and the runner
must report header fields separately.
"""

from __future__ import annotations

import re
from typing import Any

from invoice_extract.models.postprocess import normalize_date, parse_amount

#: Inferred tag -> meaning. See module docstring for how this was established.
TAG_TOTAL = 1
TAG_TOTAL_IN_WORDS = 2
TAG_INVOICE_DATE = 3
TAG_DUE_DATE = 4
TAG_BUYER = 5
TAG_VENDOR = 6
TAG_BILL_TO = 8
TAG_SHIP_TO = 9
TAG_TABLE = 10
TAG_LOGO = 11
TAG_INVOICE_NUMBER = 12
TAG_TERMS = 13

#: Recorded so a tag-vocabulary shift in a future release is detectable rather
#: than silent: (tag, a token that must plausibly appear under it).
TAG_EVIDENCE: dict[int, tuple[str, ...]] = {
    TAG_TOTAL: ("TOTAL",),
    TAG_TOTAL_IN_WORDS: ("words",),
    TAG_INVOICE_DATE: ("Date",),
    TAG_DUE_DATE: ("Due",),
    TAG_BUYER: ("Bill", "to"),
    TAG_VENDOR: (),  # vendor name is free text, no fixed label token
    TAG_BILL_TO: ("BILL_TO",),
    TAG_SHIP_TO: ("SHIP_TO",),
    TAG_TABLE: ("table",),
    TAG_LOGO: ("logo",),
    TAG_INVOICE_NUMBER: ("INVOICE", "Invoice"),
    TAG_TERMS: ("Terms",),
}

#: FATURA prints the field label inside the same tagged span as its value
#: ("TOTAL 441.14 EUR"), so labels have to be stripped before the value is
#: usable. Matching is case-insensitive and punctuation-only tokens go too.
LABEL_TOKENS = {
    "invoice",
    "inv",
    "no",
    "no.",
    "num",
    "number",
    "#",
    ":",
    "-",
    "date",
    "due",
    "total",
    "amount",
    "bill",
    "to",
    "billed",
    "ship",
    "shipped",
    "bill_to",
    "ship_to",
    "sold",
    "customer",
    "client",
    "in",
    "words",
    "payable",
    "net",
    "grand",
    "sub",
    "subtotal",
    "balance",
    "of",
}

_DATE_LIKE = re.compile(
    r"\d{1,4}[-/.]\w{2,9}[-/.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+\w{3,9}\s+\d{4}"
)
_MONEY_LIKE = re.compile(r"^-?[\d,]+\.?\d*$")
_CURRENCY_TOKENS = {"EUR", "USD", "GBP", "INR", "JPY", "$", "€", "£", "₹"}


def _clean(tokens: list[str]) -> list[str]:
    """Drop label words and bare punctuation, keeping value tokens in order."""
    out = []
    for token in tokens:
        stripped = token.strip()
        if not stripped:
            continue
        if stripped.lower().strip(":#.,") in LABEL_TOKENS:
            continue
        if all(not c.isalnum() for c in stripped):
            continue
        out.append(stripped)
    return out


def _tokens_for(tags: list[int], tokens: list[str], wanted: int) -> list[str]:
    return [t for tag, t in zip(tags, tokens, strict=True) if tag == wanted]


def _boxes_for(tags: list[int], boxes: list[list[int]], wanted: int) -> list[list[int]]:
    return [b for tag, b in zip(tags, boxes, strict=True) if tag == wanted]


def _merge_box(boxes: list[list[int]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        float(min(b[0] for b in boxes)),
        float(min(b[1] for b in boxes)),
        float(max(b[2] for b in boxes)),
        float(max(b[3] for b in boxes)),
    ]


def _first_date(tokens: list[str]) -> str | None:
    for token in tokens:
        if _DATE_LIKE.search(token):
            normalized = normalize_date(token)
            if normalized:
                return normalized
    # Some layouts split a date across tokens ("15", "Jan", "2014").
    joined = " ".join(tokens)
    match = _DATE_LIKE.search(joined)
    return normalize_date(match.group()) if match else None


def _amount_and_currency(tokens: list[str]) -> tuple[float | None, str | None]:
    amount = None
    currency = None
    for token in tokens:
        upper = token.upper()
        if upper in _CURRENCY_TOKENS:
            currency = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}.get(upper, upper)
        elif amount is None and _MONEY_LIKE.match(token.replace(",", "")):
            amount = parse_amount(token)
    return amount, currency


def convert_record(
    record: dict[str, Any],
    *,
    doc_id: str,
    image_path: str,
    width: int,
    height: int,
    split: str = "test",
) -> dict[str, Any]:
    """Convert one FATURA2 row into a canonical-schema document.

    `record` needs `ner_tags`, `tokens` and `bboxes` (parallel lists, as the
    parquet stores them).
    """
    tags = list(record["ner_tags"])
    tokens = [str(t) for t in record["tokens"]]
    boxes = [list(b) for b in record["bboxes"]]

    vendor_tokens = _clean(_tokens_for(tags, tokens, TAG_VENDOR))
    # Buyer appears under either tag depending on the template.
    buyer_raw = _tokens_for(tags, tokens, TAG_BUYER) or _tokens_for(tags, tokens, TAG_BILL_TO)
    buyer_tokens = _clean(buyer_raw)
    ship_tokens = _clean(_tokens_for(tags, tokens, TAG_SHIP_TO))

    number_tokens = _clean(_tokens_for(tags, tokens, TAG_INVOICE_NUMBER))
    total_tokens = _tokens_for(tags, tokens, TAG_TOTAL)
    grand_total, currency = _amount_and_currency(total_tokens)

    invoice_date = _first_date(_tokens_for(tags, tokens, TAG_INVOICE_DATE))
    due_date = _first_date(_tokens_for(tags, tokens, TAG_DUE_DATE))
    words_tokens = _clean(_tokens_for(tags, tokens, TAG_TOTAL_IN_WORDS))

    def party(name_tokens: list[str]) -> dict[str, Any]:
        """First token run is the name; the remainder is the address.

        FATURA's buyer block is name-then-address with no separator, so the
        split is heuristic: the name runs until the first token that looks like
        a street number, which is how these templates are laid out.
        """
        if not name_tokens:
            return {}
        cut = len(name_tokens)
        for i, token in enumerate(name_tokens):
            if i > 0 and token.isdigit():
                cut = i
                break
        name = " ".join(name_tokens[:cut]) or None
        address = " ".join(name_tokens[cut:]) or None
        out: dict[str, Any] = {}
        if name:
            out["name"] = name
        if address:
            out["address"] = address
        return out

    parties: dict[str, Any] = {"vendor": party(vendor_tokens) or {"name": None}}
    if buyer_tokens:
        parties["buyer"] = party(buyer_tokens)
    if ship_tokens:
        parties["ship_to"] = party(ship_tokens)

    grounding = []
    for tag, path in (
        (TAG_VENDOR, "parties.vendor.name"),
        (TAG_INVOICE_NUMBER, "invoice_meta.invoice_number"),
        (TAG_INVOICE_DATE, "invoice_meta.invoice_date"),
        (TAG_DUE_DATE, "invoice_meta.due_date"),
        (TAG_TOTAL, "totals.grand_total"),
    ):
        merged = _merge_box(_boxes_for(tags, boxes, tag))
        if merged:
            grounding.append(
                {"field_path": path, "page_index": 0, "bbox": merged, "confidence": 1.0}
            )

    doc: dict[str, Any] = {
        "schema_version": "1.0.0",
        "doc_id": doc_id,
        "source": {
            "dataset": "fatura",
            "original_id": str(record.get("id", doc_id)),
            "split": split,
            "license": "CC-BY-4.0",
        },
        "language": "en",
        "document_type": "invoice",
        "pages": [{"page_index": 0, "image_path": image_path, "width": width, "height": height}],
        "parties": parties,
        "invoice_meta": {
            "invoice_number": " ".join(number_tokens) or None,
            "invoice_date": invoice_date,
            "due_date": due_date,
            "po_number": None,
            "payment_terms": None,
        },
        # FATURA2 does not tokenise the line-item table (tag 10 is one
        # placeholder token), so there is genuinely nothing to put here.
        "line_items": [],
        "tax_lines": [],
        "totals": {
            "grand_total": grand_total,
            "amount_in_words": " ".join(words_tokens) or None,
        },
        "grounding": grounding,
        "annotator": {
            "annotated_by": "FATURA2 (synthetic, template-generated)",
            "verified": False,
            "notes": "Header fields only; line items are not annotated in this release.",
        },
    }
    if currency:
        doc["currency"] = currency
    return doc
