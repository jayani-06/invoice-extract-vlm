# invoice-extract-vlm

VLM-based invoice/receipt field extraction: canonical annotation schema,
multi-dataset ingestion (FATURA, SROIE, CORD, Indian GST invoices), a
model-agnostic evaluation harness, and a baseline-inference compute plan for
Colab Pro / Kaggle T4s.

## Project phase & ordering

This repo is deliberately sequenced so nothing downstream gets built on
sand:

1. **Schema frozen first** (`schema/`) — every dataset converter, every
   annotation export, and the eval harness all target one canonical JSON
   shape. See `schema/schema.md` for the design rationale.
2. **Eval harness before any model** (`src/invoice_extract/eval/`) — metrics
   are defined and testable (`tests/test_metrics.py`) against synthetic
   fixtures before a single VLM has been run.
3. **Dataset acquisition** (`scripts/`, `data/README.md`) — download +
   convert scripts per dataset, plus a fully-synthetic GST invoice generator
   that needs no external data and doubles as an end-to-end pipeline
   smoke test.
4. **Compute plan** (`configs/compute_plan.md`) — quantization and model-size
   decisions made up front for T4-class GPUs (Colab Pro / Kaggle), so
   notebook work doesn't re-litigate them per run.

## Quickstart

```bash
# Environment
uv venv && source .venv/bin/activate     # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install

# Verify the schema + harness work end to end (no external data needed)
make validate-schema
make eval-smoke
make test

# Generate a batch of synthetic GST invoices (also no external data needed)
python scripts/generate_synthetic_gst_invoices.py --count 150 --out data/processed/gst_in_synthetic

# Docker, if you'd rather not touch the host Python at all
docker compose build
docker compose run --rm app make test
```

See `data/README.md` for FATURA/SROIE/CORD download+convert instructions
(these need dataset credentials/hosting confirmed — see that file) and
`configs/compute_plan.md` + `notebooks/colab_baseline_inference.ipynb` for
running an actual VLM baseline once data is in place.

## Layout

```
schema/                     canonical JSON Schema + docs + example
src/invoice_extract/
  data/models.py            pydantic mirror of the schema
  data/validate.py          jsonschema + pydantic validation, totals-consistency checks
  data/converters/          per-dataset -> canonical converters (fatura/sroie/cord/gst_in)
  render/                   canonical JSON -> page image + word-level ground truth
  degrade/                  scan-artifact simulation (12 transforms, 4 severity profiles)
  preprocess/               8 toggleable OpenCV stages + the ablation pipeline
  ocr/                      engine-agnostic OCR interface (tesseract, paddle, oracle)
  eval/metrics.py           field/line-item metrics (ANLS, numeric tolerance, Hungarian matching)
  eval/ocr_metrics.py       CER, WER, word-box detection P/R/F1
  eval/harness.py           CLI: score predictions.jsonl against gold.jsonl
  models/base.py            baseline model interface (no model implemented yet, by design)
  utils/quantization.py     4-bit NF4 loading helper for T4-class GPUs
scripts/                    download_*, convert_*, generate_synthetic_gst_invoices.py,
                            build_image_corpus.py, run_preprocess_ablation.py
tests/                      pytest suite incl. tests/fixtures/{gold,pred}.jsonl smoke fixtures
configs/compute_plan.md     Colab Pro / Kaggle T4 inference plan, quantization decision
docs/phase1_preprocessing.md  Phase 1 design, findings, and how to run the ablation
notebooks/                  Colab/Kaggle baseline-inference notebook skeleton
docker-compose.yml          app (dev) + optional labelstudio/minio profiles
```

## Phase 1 quickstart

```bash
make corpus          # render 25 synthetic invoices x 4 degradation profiles
make ablation-smoke  # verify the whole chain with no OCR engine installed
make ocr-install     # then install Tesseract or PaddleOCR (see docs/)
make ablation ENGINE=tesseract
```

## Status

**Phase 0 (complete)** — schema frozen, eval harness written and tested,
150 synthetic GST invoices generated.

**Phase 1 (code complete, awaiting an OCR engine)** — rendering, degradation,
preprocessing, OCR abstraction, and CER/WER metrics are all built and tested
(165 tests). No OCR engine is installed on the dev machine yet, so no real
CER number exists and the exit criteria (CER < 8%, >= 5-point improvement from
preprocessing) are **not yet demonstrated**. See `docs/phase1_preprocessing.md`.

**Still open from Phase 0** — confirm FATURA's current hosting location
(`scripts/download_fatura.py`) and set up Kaggle credentials for SROIE; the
converters exist but have not been run against real data.
