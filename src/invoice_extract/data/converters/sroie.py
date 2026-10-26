"""SROIE -> canonical schema converter.

**The first real documents in this project.** Every other corpus here is
generated: ours by our own renderer, FATURA2 by someone else's. SROIE is 652
photographs of actual Malaysian retail receipts -- creased, skewed, unevenly
lit, thermal-printed, shot on a phone. Nothing in this pipeline had met one
before.

Two annotation shapes are supported, because the mirrors differ:

- **`convert`** takes the flat 4-field key-value file (`company`, `date`,
  `address`, `total`) that the Kaggle mirrors ship. Everything else in the
  canonical schema is `null` by design (see schema.md, "null vs absent").
- **`convert_tokens`** takes labelled OCR lines with boxes, which the
  `arvindrajan92/sroie_document_understanding` mirror provides (MIT). That
  carries `line_description` and `line_total` as well, so SROIE documents get
  **real line items** -- the annotation FATURA2 lacks entirely.

One thing to be careful about when reading any Phase 1 number from this corpus:
**SROIE's annotations are line-level, not word-level.** Our synthetic corpus has
a box per word. CER and WER are unaffected (text is text), but box-IoU
detection metrics computed here measure line detection, and must not be
compared against the word-level detection figures elsewhere in this project.
The manifest records `annotation_granularity` so a runner cannot confuse them.
"""

from __future__ import annotations

import re

from invoice_extract.data.models import (
    CanonicalInvoiceDocument,
    InvoiceMeta,
    Page,
    Parties,
    Party,
    Source,
    Totals,
)

DATE_FORMATS = [
    "%d %b %Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%d %B %Y",
    "%d.%m.%Y",
    "%d-%b-%Y",
    "%d %b %y",
    "%d/%m/%y",
]


def normalize_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    import datetime

    text = (raw_date or "").strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Real receipt date lines are nothing like a date string. Observed in
    # SROIE: "DATE : 18/01/2018", "BILL DATE : 20/12/2017",
    # "12-01-19 21:13 SH01 ZK09", "ORD #50 -REG #19- 18/01/2018 10:27:03",
    # "05 MAR 2018 18:24". Only 33% of them parse as a whole string, which is
    # the sharpest illustration in this project of how much messier real
    # documents are than generated ones.
    #
    # Scanning inside the line is justified HERE and would not be in general:
    # this function is called on text SROIE has annotated as the date, so the
    # annotation is the evidence that a date is present. Running the same scan
    # over an arbitrary line would invent dates out of document and register
    # numbers -- "REG #19-" is exactly the shape that would fall for.
    return _extract_date(text)


#: Day-first, because SROIE is Malaysian and so is its convention. Only used
#: when the numbers themselves cannot settle it.
_NUMERIC = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b")
_ISO = re.compile(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_NAMED = re.compile(r"\b(\d{1,2})[\s-]+([A-Za-z]{3,9})[\s-]+(\d{2,4})\b")

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def _year(raw: str) -> int:
    """Expand a two-digit year. SROIE spans 2017-2019, and a receipt is never
    from the 1900s, so the pivot is not a close call."""
    value = int(raw)
    if value >= 100:
        return value
    return 2000 + value if value < 70 else 1900 + value


def _build(year: int, month: int, day: int) -> str | None:
    import datetime

    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        return None


def _extract_date(text: str) -> str | None:
    """Find a date inside an annotated date line."""
    iso = _ISO.search(text)
    if iso:
        return _build(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))

    named = _NAMED.search(text)
    if named:
        month = _MONTHS.get(named.group(2)[:3].lower())
        if month:
            return _build(_year(named.group(3)), month, int(named.group(1)))

    numeric = _NUMERIC.search(text)
    if numeric:
        a, b, c = (int(numeric.group(i)) for i in (1, 2, 3))
        year = _year(numeric.group(3))
        # Let the numbers disambiguate where they can: a first field above 12
        # must be the day, a second field above 12 must be the day. Only when
        # both are <= 12 is the order genuinely ambiguous, and then the
        # Malaysian day-first convention applies. "12/28/2017" resolves
        # month-first on this rule, which is correct.
        if a > 12:
            return _build(year, b, a)
        if b > 12:
            return _build(year, a, b)
        return _build(year, b, a)

    return None  # unparseable -> null, per schema convention, rather than guessing


def normalize_total(raw_total: str | None) -> float | None:
    if not raw_total:
        return None
    cleaned = re.sub(r"[^\d.]", "", raw_total)
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def convert(
    kv: dict, doc_id: str, split: str, image_path: str, width: int, height: int
) -> CanonicalInvoiceDocument:
    """kv: the parsed key-value dict from a SROIE `*.txt` entities file,
    i.e. {"company": ..., "date": ..., "address": ..., "total": ...}.
    """
    return CanonicalInvoiceDocument(
        doc_id=doc_id,
        source=Source(dataset="sroie", original_id=doc_id, split=split),
        document_type="receipt",
        pages=[Page(page_index=0, image_path=image_path, width=width, height=height)],
        parties=Parties(vendor=Party(name=kv.get("company"), address=kv.get("address"))),
        invoice_meta=InvoiceMeta(invoice_number=None, invoice_date=normalize_date(kv.get("date"))),
        line_items=[],
        totals=Totals(grand_total=normalize_total(kv.get("total"))),
    )


# ----------------------------------------------------- labelled-token mirror


def polygon_bbox(polygon) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a 4-point polygon.

    SROIE boxes are quadrilaterals because the receipts are photographed, not
    scanned: a creased receipt's text line is genuinely not a rectangle. The
    rest of this project's geometry is axis-aligned, so the bounds are taken
    here and the fact that information is lost is recorded rather than hidden.
    """
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def group_line_items(tokens: list[dict]) -> list[dict]:
    """Pair each `line_description` with the `line_total` that belongs to it.

    **A thermal receipt is not an invoice table.** The first version of this
    paired on same-row vertical overlap, which is how an invoice lays out a
    line item -- description left, amount right, one row. Real receipts
    overwhelmingly print the amount on the line *below* the description:

        ST-PRIVILEGE CARD/GD INDAH     centre y = 551.0
                               10.00   centre y = 573.5

    With a half-line-height tolerance that pairing found 37% of amounts, and
    the missing 63% went straight into line-item F1 as the extractor's fault.

    So a description claims the nearest unclaimed amount at or below its own
    baseline, stopping before the next description begins. That covers both
    layouts -- a same-row amount is still "at" the description -- without a
    description ever reaching past its successor to steal a row.

    A description with no amount is still a line item with a null total.
    Dropping it would quietly shrink the line-item count and make extraction
    recall look better than it is.
    """
    descriptions = [t for t in tokens if t.get("label") == "line_description"]
    totals = [t for t in tokens if t.get("label") == "line_total"]

    def centre_y(token: dict) -> float:
        y0, y1 = token["bbox"][1], token["bbox"][3]
        return (y0 + y1) / 2.0

    def height(token: dict) -> float:
        return max(1.0, token["bbox"][3] - token["bbox"][1])

    ordered = sorted(descriptions, key=centre_y)
    unclaimed = sorted(totals, key=centre_y)
    items: list[dict] = []

    for i, desc in enumerate(ordered, start=1):
        start = centre_y(desc) - height(desc) * 0.6  # a little above, for a same-row amount
        # The zone ends where the next description starts; for the last one,
        # one and a half line heights below, so a trailing amount is caught
        # without sweeping up the subtotal block beneath the items.
        end = centre_y(ordered[i]) if i < len(ordered) else centre_y(desc) + height(desc) * 1.5

        best = next((t for t in unclaimed if start <= centre_y(t) < end), None)
        if best is not None:
            unclaimed.remove(best)

        items.append(
            {
                "line_no": i,
                "description": desc["text"].strip() or None,
                "line_total": normalize_total(best["text"]) if best else None,
            }
        )
    return items


def convert_tokens(
    tokens: list[dict], doc_id: str, split: str, image_path: str, width: int, height: int
) -> CanonicalInvoiceDocument:
    """Build a canonical document from labelled OCR lines.

    `tokens` is a list of `{"text": str, "label": str, "bbox": (x0,y0,x1,y1)}`.
    Labels are SROIE's: company, address, date, total, line_description,
    line_total, other.
    """
    by_label: dict[str, list[dict]] = {}
    for token in tokens:
        by_label.setdefault(token.get("label") or "other", []).append(token)

    def joined(label: str) -> str | None:
        """Concatenate every token carrying a label, in reading order.

        An address spans several lines and each is annotated separately, so
        taking only the first would truncate it to its first line.
        """
        parts = [
            t["text"].strip()
            for t in sorted(by_label.get(label, []), key=lambda t: (t["bbox"][1], t["bbox"][0]))
            if t["text"].strip()
        ]
        return " ".join(parts) or None

    def first(label: str) -> str | None:
        parts = by_label.get(label) or []
        return parts[0]["text"].strip() if parts else None

    return CanonicalInvoiceDocument(
        doc_id=doc_id,
        source=Source(dataset="sroie", original_id=doc_id, split=split),
        document_type="receipt",
        pages=[Page(page_index=0, image_path=image_path, width=width, height=height)],
        # `company` is occasionally annotated on two lines (a trading name
        # above a legal name); joining keeps both rather than picking one.
        parties=Parties(vendor=Party(name=joined("company"), address=joined("address"))),
        invoice_meta=InvoiceMeta(invoice_number=None, invoice_date=normalize_date(first("date"))),
        line_items=group_line_items(tokens),
        totals=Totals(grand_total=normalize_total(first("total"))),
    )
