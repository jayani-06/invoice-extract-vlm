# Dataset acquisition

Raw and processed data are **never committed** (see `.gitignore`); this repo
only ships the *scripts* to fetch and convert them. Run these from the repo
root, ideally inside the `app` container (`docker compose run --rm app bash`)
so paths line up.

Layout each script produces:

```
data/
  raw/<dataset>/...            # untouched, as downloaded/extracted
  processed/<dataset>/
    images/<doc_id>.png
    canonical/<doc_id>.json    # validated against schema/invoice_schema.json
```

## 1. FATURA (primary, ~10k invoices, ~50 layouts)

FATURA (Limam et al.) is distributed as a research dataset — check current
hosting (Zenodo / a Kaggle mirror / the authors' GitHub) before relying on a
specific URL, since research dataset hosts move. `scripts/download_fatura.py`
is a template: fill in `FATURA_SOURCE_URL` (or use the Kaggle API path) once
you've confirmed the current location, then run:

```bash
python scripts/download_fatura.py --out data/raw/fatura
python scripts/convert_fatura.py --raw data/raw/fatura --out data/processed/fatura
```

`invoice_extract/data/converters/fatura.py` maps FATURA's per-token
bbox+tag annotations to the canonical schema (field tags -> `invoice_meta`/
`line_items`/`totals`, token boxes -> `grounding`).

## 2. SROIE (receipts, key-value fields)

SROIE (ICDAR 2019 Task, via Kaggle or the ICDAR archive) ships OCR boxes plus
a flat key-value file (`company`, `date`, `address`, `total`). Requires a
free Kaggle account + API token (`KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env`).

```bash
python scripts/download_sroie.py --out data/raw/sroie      # uses kaggle CLI
python scripts/convert_sroie.py --raw data/raw/sroie --out data/processed/sroie
```

SROIE only gives 4 fields natively (company/date/address/total) — the
converter (`converters/sroie.py`) fills everything else in the canonical
schema with `null`, which is expected and fine (the schema is designed to
allow this).

## 3. CORD (receipts, hierarchical categories + bbox)

CORD (Consolidated Receipt Dataset, Naver Clova) is on HuggingFace Hub
(`naver-clova-ix/cord-v2`) — needs `HUGGINGFACE_TOKEN` in `.env` only if the
mirror you use is gated; the canonical HF dataset is public.

```bash
python scripts/download_cord.py --out data/raw/cord        # uses `datasets` lib
python scripts/convert_cord.py --raw data/raw/cord --out data/processed/cord
```

CORD's menu/subtotal/total hierarchy maps fairly directly onto
`line_items`/`totals`; see `converters/cord.py` for the exact field-name
mapping table (kept as a module-level dict so it's easy to audit/extend).

## 4. Indian GST invoices (100–200 real/synthetic, for realism)

Two sources, both handled by `converters/gst_in.py`:

- **Real**: a small hand-collected/scanned set (yours to source — vendor
  invoices, redacted as needed). Drop scans in `data/raw/gst_in_real/images/`
  and annotate via the `labelstudio` compose profile
  (`docker compose --profile annotate up labelstudio`); export as JSON and
  run `scripts/convert_gst_in.py --mode real`.
- **Synthetic**: `scripts/generate_synthetic_gst_invoices.py` is a template
  for a synthetic generator (e.g. Faker + a Jinja/HTML invoice template
  rendered to image via `weasyprint`/`playwright`) that emits canonical JSON
  directly (no separate conversion step needed, since you control the
  generator's output format) plus the matching rendered image. This is the
  fastest way to get GST-specific fields (GSTIN, HSN/SAC, CGST/SGST/IGST
  splits) well-represented without waiting on real-world sourcing.

```bash
python scripts/generate_synthetic_gst_invoices.py --count 150 --out data/processed/gst_in_synthetic
python scripts/convert_gst_in.py --mode real --raw data/raw/gst_in_real --out data/processed/gst_in_real
```

## Validating everything after conversion

```bash
python scripts/validate_examples.py data/processed/**/canonical/*.json
```

(Recommended: run this in CI on every PR that touches a converter, on a
small sample, not the full 10k+ — see `configs/` for a suggested sample-size
knob once CI is set up.)
