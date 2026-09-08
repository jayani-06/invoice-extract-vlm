"""Indian GST invoice -> canonical schema converter, for the *real* (hand
annotated) subset. The synthetic subset is generated directly in canonical
form by scripts/generate_synthetic_gst_invoices.py and needs no converter.

Expects a Label Studio JSON export (one task per document) with a
rectangle-labels + textarea setup: each result gives a bbox plus a
transcribed value tagged with the canonical field_path directly, so the
label config's tag names should just BE canonical field paths
(e.g. "parties.vendor.gstin", "line_items[0].description") to avoid a
second mapping table drifting out of sync with schema.md.
"""

from __future__ import annotations

from typing import Any

from invoice_extract.data.models import (
    CanonicalInvoiceDocument,
    Grounding,
    InvoiceMeta,
    Parties,
    Party,
    Source,
    Totals,
)


def _set_by_path(doc: dict[str, Any], path: str, value: Any) -> None:
    """Minimal setter for the dotted/bracket field_path convention used in
    grounding[].field_path, e.g. 'parties.vendor.gstin' or 'line_items[0].description'.
    """
    import re

    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    cursor = doc
    for i, tok in enumerate(tokens):
        last = i == len(tokens) - 1
        if tok.startswith("["):
            idx = int(tok[1:-1])
            while len(cursor) <= idx:
                cursor.append({})
            if last:
                cursor[idx] = value
            else:
                cursor = cursor[idx]
        else:
            if last:
                cursor[tok] = value
            else:
                nxt_is_list = i + 1 < len(tokens) and tokens[i + 1].startswith("[")
                cursor = cursor.setdefault(tok, [] if nxt_is_list else {})


def convert(
    label_studio_task: dict,
    doc_id: str,
    split: str,
    image_path: str,
    width: int,
    height: int,
) -> CanonicalInvoiceDocument:
    skeleton: dict[str, Any] = {
        "schema_version": "1.0.0",
        "doc_id": doc_id,
        "source": Source(dataset="gst_in_real", original_id=doc_id, split=split).model_dump(),
        "currency": "INR",
        "document_type": "invoice",
        "pages": [{"page_index": 0, "image_path": image_path, "width": width, "height": height}],
        "parties": Parties(vendor=Party()).model_dump(),
        "invoice_meta": InvoiceMeta(invoice_number=None, invoice_date=None).model_dump(),
        "line_items": [],
        "tax_lines": [],
        "totals": Totals(grand_total=None).model_dump(),
        "grounding": [],
    }

    for annotation in label_studio_task.get("annotations", []):
        for result in annotation.get("result", []):
            field_path = result.get("from_name")  # label config tag == canonical field_path
            value = result.get("value", {})
            text = (value.get("text") or [None])[0] if "text" in value else None
            if field_path and text is not None:
                _set_by_path(skeleton, field_path, text)
            if field_path and all(k in value for k in ("x", "y", "width", "height")):
                x0 = value["x"] / 100 * width
                y0 = value["y"] / 100 * height
                x1 = x0 + value["width"] / 100 * width
                y1 = y0 + value["height"] / 100 * height
                skeleton["grounding"].append(
                    Grounding(
                        field_path=field_path, page_index=0, bbox=[x0, y0, x1, y1]
                    ).model_dump()
                )

    return CanonicalInvoiceDocument.model_validate(skeleton)
