"""SROIE -> canonical schema converter.

SROIE (task 3, key info extraction) ships one flat key-value file per
receipt with exactly 4 fields: `company`, `date`, `address`, `total`.
Everything else in the canonical schema is left `null` for SROIE documents
by design (see schema.md, "null vs absent") — this is expected, not a bug.
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

DATE_FORMATS = ["%d %b %Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]


def normalize_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    import datetime

    for fmt in DATE_FORMATS:
        try:
            return datetime.datetime.strptime(raw_date.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
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
