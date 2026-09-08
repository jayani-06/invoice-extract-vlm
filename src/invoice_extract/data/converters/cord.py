"""CORD -> canonical schema converter.

CORD annotates receipts with a hierarchical category tree
(menu/subtotal/total, each with sub-fields like menu.nm, menu.cnt,
menu.price, sub_total.tax_price, total.total_price, ...). CATEGORY_MAP below
lists the common leaf categories; confirm the exact set against a real
sample (`scripts/download_cord.py`) since CORD's schema has had minor
revisions across releases.
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
    TaxLine,
    Totals,
)

# CORD leaf category -> canonical field. "menu.*" categories are handled
# per-row in convert_line_items() instead, since they repeat per line item.
CATEGORY_MAP: dict[str, str] = {
    "sub_total.subtotal_price": "totals.subtotal",
    "sub_total.tax_price": "totals.tax_total",
    "sub_total.discount_price": "totals.discount_total",
    "total.total_price": "totals.grand_total",
    "total.menuqty_cnt": None,  # informational only, not part of canonical schema
}


def convert_line_items(menu_rows: list[dict]) -> list[LineItem]:
    items = []
    for i, row in enumerate(menu_rows):
        items.append(
            LineItem(
                line_no=i + 1,
                description=row.get("nm"),
                quantity=_to_float(row.get("cnt")),
                unit_price=_to_float(row.get("unitprice")),
                line_total=_to_float(row.get("price")),
                discount=_to_float(row.get("discountprice")),
            )
        )
    return items


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).replace(",", "").strip())
    except ValueError:
        return None


def convert(
    raw: dict, doc_id: str, split: str, image_path: str, width: int, height: int
) -> CanonicalInvoiceDocument:
    """raw: CORD's parsed `ground_truth` dict for one receipt
    (`{"gt_parse": {"menu": [...], "sub_total": {...}, "total": {...}}}`).
    """
    gt = raw.get("gt_parse", raw)
    sub_total = gt.get("sub_total", {})
    total = gt.get("total", {})
    menu = gt.get("menu", [])
    if isinstance(menu, dict):  # CORD emits a single dict, not a list, for single-item receipts
        menu = [menu]

    return CanonicalInvoiceDocument(
        doc_id=doc_id,
        source=Source(dataset="cord", original_id=doc_id, split=split),
        document_type="receipt",
        pages=[Page(page_index=0, image_path=image_path, width=width, height=height)],
        parties=Parties(vendor=Party(name=None)),  # CORD generally omits store name/id in gt_parse
        invoice_meta=InvoiceMeta(invoice_number=None, invoice_date=None),
        line_items=convert_line_items(menu),
        tax_lines=(
            [TaxLine(type="OTHER", amount=_to_float(sub_total.get("tax_price")) or 0.0)]
            if sub_total.get("tax_price")
            else []
        ),
        totals=Totals(
            subtotal=_to_float(sub_total.get("subtotal_price")),
            tax_total=_to_float(sub_total.get("tax_price")),
            discount_total=_to_float(sub_total.get("discount_price")),
            grand_total=_to_float(total.get("total_price")),
        ),
    )
