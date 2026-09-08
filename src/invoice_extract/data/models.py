"""Pydantic v2 models mirroring schema/invoice_schema.json 1:1.

Keep this file and the JSON Schema in lockstep manually — there are only a
handful of fields, and a hand-maintained pair is easier to reason about than
a codegen step at this stage. `tests/test_schema.py` cross-checks that the
example fixture validates against both.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

Dataset = Literal["fatura", "sroie", "cord", "gst_in_real", "gst_in_synthetic", "other"]
Split = Literal["train", "val", "test"]
DocumentType = Literal["invoice", "receipt"]
TaxType = Literal["CGST", "SGST", "IGST", "CESS", "VAT", "GST", "SALES_TAX", "OTHER"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(StrictModel):
    dataset: Dataset
    original_id: str
    split: Split
    license: str | None = None


class Page(StrictModel):
    page_index: int = Field(ge=0)
    image_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    dpi: int | None = Field(default=None, ge=1)


class Party(StrictModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None
    gstin: str | None = None
    phone: str | None = None
    email: str | None = None


class Parties(StrictModel):
    vendor: Party
    buyer: Party | None = None
    ship_to: Party | None = None


class InvoiceMeta(StrictModel):
    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None = None
    po_number: str | None = None
    payment_terms: str | None = None


class LineItem(StrictModel):
    line_no: int | None = None
    description: str | None
    hsn_sac_code: str | None = None
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    discount: float | None = None
    tax_rate: float | None = None
    tax_amount: float | None = None
    line_total: float | None = None


class TaxLine(StrictModel):
    type: TaxType
    rate: float | None = None
    amount: float


class Totals(StrictModel):
    subtotal: float | None = None
    discount_total: float | None = None
    tax_total: float | None = None
    shipping: float | None = None
    round_off: float | None = None
    grand_total: float | None
    amount_in_words: str | None = None


class PaymentInfo(StrictModel):
    bank_name: str | None = None
    account_number: str | None = None
    ifsc_or_swift: str | None = None
    upi_id: str | None = None
    mode: str | None = None
    terms: str | None = None


class Grounding(StrictModel):
    field_path: str
    page_index: int = Field(ge=0)
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Annotator(StrictModel):
    annotated_by: str | None = None
    verified: bool = False
    notes: str | None = None


class CanonicalInvoiceDocument(StrictModel):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    doc_id: str
    source: Source
    language: str = "en"
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    document_type: DocumentType
    pages: list[Page] = Field(min_length=1)
    parties: Parties
    invoice_meta: InvoiceMeta
    line_items: list[LineItem] = Field(default_factory=list)
    tax_lines: list[TaxLine] = Field(default_factory=list)
    totals: Totals
    payment_info: PaymentInfo | None = None
    grounding: list[Grounding] = Field(default_factory=list)
    annotator: Annotator | None = None
