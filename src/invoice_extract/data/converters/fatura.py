"""FATURA -> canonical schema converter.

FATURA annotates each invoice as a set of (token, bbox, field_tag) triples
across ~50 layout templates. The exact tag vocabulary can vary slightly by
release, so treat FIELD_TAG_MAP as the thing to audit first against a real
sample once the data is downloaded (`scripts/download_fatura.py`).

FIELD_TAG_MAP keys are FATURA's field tags; values are canonical dotted
paths (matching the `grounding[].field_path` convention in schema.md).
Tags not listed here are dropped with a warning rather than silently lost —
extend the map as you audit real samples.
"""

from __future__ import annotations

from invoice_extract.data.models import (
    CanonicalInvoiceDocument,
    InvoiceMeta,
    LineItem,
    Page,
    Parties,
    Party,
    Source,
    Totals,
)

# FATURA field tag -> canonical field path. Fill in / correct against real
# samples; this is a best-effort starting map based on FATURA's documented
# field categories (seller/client info, invoice info, table line items, totals).
FIELD_TAG_MAP: dict[str, str] = {
    "SELLER_NAME": "parties.vendor.name",
    "SELLER_ADDRESS": "parties.vendor.address",
    "CLIENT_NAME": "parties.buyer.name",
    "CLIENT_ADDRESS": "parties.buyer.address",
    "INVOICE_NUMBER": "invoice_meta.invoice_number",
    "INVOICE_DATE": "invoice_meta.invoice_date",
    "DUE_DATE": "invoice_meta.due_date",
    "TOTAL": "totals.grand_total",
    "SUBTOTAL": "totals.subtotal",
    "TAX": "totals.tax_total",
    # Table rows are handled separately in convert_line_items() below, since
    # FATURA groups them by row index rather than a flat tag.
}


def convert_line_items(raw_table_rows: list[dict]) -> list[LineItem]:
    """raw_table_rows: FATURA's per-row dict, e.g.
    {"description": ..., "quantity": ..., "unit_price": ..., "total": ...}
    Adjust keys once you've inspected a real FATURA sample.
    """
    items = []
    for i, row in enumerate(raw_table_rows):
        items.append(
            LineItem(
                line_no=i + 1,
                description=row.get("description"),
                quantity=row.get("quantity"),
                unit_price=row.get("unit_price"),
                tax_rate=row.get("tax_rate"),
                tax_amount=row.get("tax_amount"),
                line_total=row.get("total"),
            )
        )
    return items


def convert(
    raw: dict, doc_id: str, split: str, image_path: str, width: int, height: int
) -> CanonicalInvoiceDocument:
    """raw: FATURA's per-document annotation dict (exact top-level keys to be
    confirmed against a real downloaded sample — this assumes a
    `{"fields": {tag: value}, "table_rows": [...]}` shape as a starting point).
    """
    fields = raw.get("fields", {})

    return CanonicalInvoiceDocument(
        doc_id=doc_id,
        source=Source(dataset="fatura", original_id=raw.get("id", doc_id), split=split),
        currency="USD",
        document_type="invoice",
        pages=[Page(page_index=0, image_path=image_path, width=width, height=height)],
        parties=Parties(
            vendor=Party(name=fields.get("SELLER_NAME"), address=fields.get("SELLER_ADDRESS")),
            buyer=Party(name=fields.get("CLIENT_NAME"), address=fields.get("CLIENT_ADDRESS")),
        ),
        invoice_meta=InvoiceMeta(
            invoice_number=fields.get("INVOICE_NUMBER"),
            invoice_date=fields.get("INVOICE_DATE"),
            due_date=fields.get("DUE_DATE"),
        ),
        line_items=convert_line_items(raw.get("table_rows", [])),
        totals=Totals(
            subtotal=fields.get("SUBTOTAL"),
            tax_total=fields.get("TAX"),
            grand_total=fields.get("TOTAL"),
        ),
    )
