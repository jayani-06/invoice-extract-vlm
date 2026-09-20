# Canonical invoice/receipt schema — v1.0.0

This is the single source of truth for document structure. It is frozen for
this project phase: FATURA, SROIE, CORD, and GST-invoice converters all
target this shape, the annotation tooling (Label Studio export mapping)
targets this shape, and the eval harness only ever reads this shape. If it
needs to change, bump `schema_version` to `1.1.0`/`2.0.0` and add a migration
script under `scripts/migrations/` — never silently reinterpret an existing
version.

Files:
- `invoice_schema.json` — the JSON Schema (draft 2020-12), used for validation
  (`scripts/validate_examples.py`, and at load-time in the eval harness).
- `../src/invoice_extract/data/models.py` — a pydantic v2 model mirroring this
  schema 1:1, for type-safe use in Python (converters, harness, future
  training code).
- `examples/example_invoice.json` — one hand-written example that validates
  against the schema, used as a fixture in tests.

## Design decisions (and why)

1. **Values are decoupled from bounding-box grounding.** Every extracted
   field (`invoice_meta.invoice_number`, `line_items[i].description`, ...) is
   a plain scalar. Bounding boxes live separately in the top-level
   `grounding` array, keyed by a dotted/bracket `field_path` string (e.g.
   `"parties.vendor.gstin"`, `"line_items[2].line_total"`).
   *Why:* most eval metrics (field accuracy, ANLS) only care about values,
   and most VLM baselines won't emit boxes at all in the first pass. Keeping
   grounding optional means the harness and converters for FATURA/SROIE/CORD
   (which do have boxes) and GST invoices (which mostly won't) share the same
   value schema, and grounding-aware metrics can be added later without a
   schema break.

2. **Two tax representations, on purpose.** `line_items[i].tax_rate` /
   `tax_amount` covers the common single-tax-rate-per-line case (SROIE/CORD
   receipts, most Western invoices). The document-level `tax_lines[]` array
   covers India's split-rate GST (CGST+SGST or IGST, sometimes +CESS), where
   tax is a document-level breakdown rather than a strictly per-line one.
   Converters populate whichever applies; `totals.tax_total` should always
   equal the sum of `tax_lines[].amount` when tax_lines is used, or the sum
   of line-level `tax_amount` otherwise. `scripts/validate_examples.py` warns
   (does not hard-fail) on mismatches beyond a small rounding tolerance.

3. **Money conventions (`line_total` and `subtotal`).** These are the two
   fields whose meaning is genuinely ambiguous across real invoices, so this
   project fixes them:

   - `line_items[].line_total` is the **taxable value** of the row:
     `quantity x unit_price`, **net of that row's `discount`**, **before tax**.
     This is the conventional meaning of the "Amount" column on a GST invoice
     and is what makes a printed page internally consistent — the Subtotal is
     the sum of the Amount column.
   - `totals.subtotal` is the **net** taxable value: `sum(line_items[].line_total)`.
     Discounts are therefore **already netted into `subtotal`** and must not be
     subtracted from it again.
   - `totals.discount_total` is `sum(line_items[].discount)`, reported for
     visibility, not as a further deduction.
   - `totals.grand_total` = `subtotal + tax_total + shipping + round_off`.

   The alternative convention — a tax-inclusive `line_total`, so that
   `sum(line_total) == grand_total` — is what the synthetic generator produced
   originally. It was rejected because it renders an invoice whose Amount
   column does not add up to its own Subtotal, and because it leaves
   downstream anomaly detection unable to use line-item arithmetic as a
   signal. `check_totals_consistency` and
   `models.postprocess.check_arithmetic` both enforce the convention above;
   the per-line check still *accepts* a tax-inclusive `line_total` without
   flagging it, because plenty of real vendors print it that way and flagging
   every one would drown the genuine errors.

4. **Null vs. absent.** Required top-level keys (`invoice_meta`, `totals`,
   etc.) must always be present, but individual leaf fields are typed
   `["<type>", "null"]` and should be set to `null` when the value is
   illegible, absent from the document, or not applicable — never omitted.
   This keeps every gold/prediction record structurally identical, which the
   eval harness's field-by-field diff depends on.

5. **`gstin`/`hsn_sac_code` vs. generic `tax_id`.** Indian GST documents get
   dedicated, format-validated fields (`gstin` has a regex for the 15-char
   GSTIN format). Non-Indian datasets (FATURA/SROIE/CORD, mostly synthetic
   Western invoices and receipts) use the generic `tax_id` on the same
   `party` object instead. A document should populate one or the other, not
   both, for a given party.

6. **`source` block for provenance.** Every converted document keeps its
   original dataset name, original id/filename, and split. This is what lets
   you trace a weird prediction back to the source image, and what lets the
   harness report metrics broken down by source dataset (a VLM baseline will
   very likely do better on FATURA's clean layouts than on scanned GST
   invoices — you want that visible, not averaged away).

7. **Dates are ISO 8601 strings (`YYYY-MM-DD`), not datetimes.** Invoices
   don't carry a time component; keeping this a plain pattern-validated
   string (rather than a datetime object) avoids timezone bugs and keeps
   JSON diffing trivial for the exact-match metric.

## Minimal valid document shape

See `examples/example_invoice.json` for a full instance. At minimum a
document needs: `schema_version`, `doc_id`, `source`, `document_type`,
`pages` (>=1), `parties.vendor`, `invoice_meta.invoice_number` +
`invoice_meta.invoice_date` (either may be `null`), `line_items` (may be
`[]` if genuinely a summary-only receipt), and `totals.grand_total`.

## Validating a document

```bash
python scripts/validate_examples.py                 # validates everything under schema/examples/
python -c "
import json, jsonschema
schema = json.load(open('schema/invoice_schema.json'))
doc = json.load(open('schema/examples/example_invoice.json'))
jsonschema.validate(doc, schema)
print('OK')
"
```

## Converting a source dataset into this schema

Each dataset gets one converter module under `src/invoice_extract/data/`:

| Dataset | Native format | Converter |
|---|---|---|
| FATURA | Per-document JSON with token-level bbox + field tags, ~50 layout templates | `converters/fatura.py` |
| SROIE | OCR box/transcript `.txt` + key-value `.txt` (company/date/address/total) | `converters/sroie.py` |
| CORD | COCO-like JSON with hierarchical menu/subtotal/total categories + bbox | `converters/cord.py` |
| GST (real+synthetic) | Manually annotated / synthetically generated JSON or Label Studio export | `converters/gst_in.py` |

Each converter is a pure function `raw_record -> CanonicalInvoiceDocument`
(see `scripts/` for dataset acquisition — converters are stubbed there with
the exact field mapping table, ready to fill in once the raw data is
downloaded).
