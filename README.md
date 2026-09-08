# invoice-extract-vlm

Intelligent invoice and receipt processing: a canonical annotation schema,
multi-dataset ingestion (FATURA, SROIE, CORD, Indian GST invoices), a synthetic
paired image corpus, an OpenCV preprocessing pipeline, an engine-agnostic OCR
layer, VLM-based structured extraction, and a model-agnostic evaluation harness.

VIT BCSE497J Project-I. The research contribution is the RAG enrichment and
vendor-validation stages (Phases 3-4); Phases 0-2 build the substrate they need.

## Project phase and ordering

This repo is deliberately sequenced so nothing downstream gets built on sand:

1. **Schema frozen first** (`schema/`) - every dataset converter, every
   annotation export, and the eval harness all target one canonical JSON shape.
   See `schema/schema.md` for the design rationale.
2. **Eval harness before any model** (`src/invoice_extract/eval/`) - metrics are
   defined and testable against synthetic fixtures before a single model has
   been run, so no model is ever judged by rules written after seeing its output.
3. **Dataset acquisition** (`scripts/`, `data/README.md`) - download and convert
   scripts per dataset, plus a fully-synthetic GST invoice generator that needs
   no external data.
4. **Compute plan** (`configs/compute_plan.md`) - quantization and model-size
   decisions made up front for T4-class GPUs, so notebook work doesn't
   re-litigate them per run.
5. **Paired image corpus before preprocessing** (`render/`, `degrade/`) - the
   synthetic documents are rendered and then degraded, so every noisy page has a
   clean original and pixel-exact word boxes. That pairing is what makes the
   preprocessing ablation measurable.

## Quickstart

```bash
# Environment
uv venv && source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install

# Verify schema + harness end to end (no external data, no OCR engine needed)
make validate-schema
make eval-smoke
make test

# Build the paired image corpus: 150 invoices x 4 degradation profiles
make corpus-full

# Phase 1: preprocessing chain check and the visual QA figure
make ablation-smoke
make figure

# Phase 2: extraction headline table (perfect-OCR ceiling)
make extract-ablation

# With a real OCR engine installed
make ocr-install        # or: make paddle-install
make ablation ENGINE=tesseract
make extract-ablation OCR_SOURCE=engine
```

## Layout

```
schema/                     canonical JSON Schema + docs + example
src/invoice_extract/
  data/models.py            pydantic mirror of the schema
  data/validate.py          jsonschema + pydantic validation, totals consistency
  data/converters/          per-dataset -> canonical (fatura/sroie/cord/gst_in)
  reading_order.py          shared visual-line grouping (renderer + OCR + harness)
  render/                   canonical JSON -> page image + word-level ground truth
  degrade/                  scan-artifact simulation (12 transforms, 4 profiles)
  preprocess/               8 toggleable OpenCV stages + the ablation pipeline
  ocr/                      engine-agnostic OCR (tesseract, paddle, oracle)
  models/base.py            extractor Protocol
  models/postprocess.py     date/amount/GSTIN parsing, arithmetic flags, coercion
  models/prompting.py       extraction schema, prompt, robust JSON recovery
  models/rules.py           OCR+regex and null baselines
  models/vlm.py             VLM extractor, model presets, per-field confidence
  models/registry.py        get_extractor("rules" | "null" | "vlm:<preset>")
  eval/metrics.py           field/line-item metrics (ANLS, tolerance, Hungarian)
  eval/ocr_metrics.py       CER, WER, word-box detection P/R/F1
  eval/harness.py           CLI: score predictions.jsonl against gold.jsonl
  utils/quantization.py     4-bit NF4 loading helper for T4-class GPUs
scripts/                    download_*, convert_*, generate_synthetic_gst_invoices.py,
                            build_image_corpus.py, make_pipeline_figure.py,
                            run_preprocess_ablation.py, run_extraction.py,
                            run_extraction_ablation.py
tests/                      pytest suite (312 tests)
configs/compute_plan.md     Colab / Kaggle T4 inference plan, quantization decision
docs/phase1_preprocessing.md  Phase 1 design, findings, how to run the ablation
docs/phase2_extraction.md     Phase 2 design, results, honest limitations
notebooks/                  Colab/Kaggle baseline-inference notebook skeleton
docker-compose.yml          app (dev) + optional labelstudio/minio profiles
```

## Status

**Phase 0 (complete)** - schema frozen, eval harness written and tested, 150
synthetic GST invoices generated.

**Dataset (built)** - 150 canonical documents x 4 degradation profiles = 600
images, 600 word-level OCR ground-truth files, 62,092 annotated words, 150/150
unique layout combinations, 0 schema violations. Reproducible with
`make corpus-full`; not tracked in git.

**Phase 1 (code complete, awaiting an OCR engine)** - rendering, degradation,
preprocessing, the OCR abstraction and CER/WER metrics are all built and tested.
No OCR engine is installed on the dev machine, so **no real CER number exists**
and the exit criteria (CER < 8%, >= 5-point improvement from preprocessing) are
**not yet demonstrated**. See `docs/phase1_preprocessing.md`.

**Phase 2 (code complete, VLM unrun)** - post-processing, prompting, rules and
null baselines, the VLM wrapper, registry, runner and ablation runner, all
tested. At the perfect-OCR ceiling the **regex baseline already meets both exit
criteria** (header accuracy 0.991, line-item F1 1.000). That is the honest
headline: this corpus cannot distinguish a VLM from a regex, so the model has to
justify itself on real scans, unseen layouts or degraded OCR rather than here.
**The VLM itself has never been run** - there is no GPU on the dev machine, so
`models/vlm.py` is unit-tested through an injected fake model but its generation
call is unverified. See `docs/phase2_extraction.md`.

**Still open**

- **No OCR engine installed.** Everything is measured at a perfect-OCR ceiling.
- **FATURA / SROIE / CORD unconverted.** `scripts/download_fatura.py` is an
  unfilled template with no confirmed URL. Every result so far is on synthetic
  renders, so the layout-generalisation claim has no third-party benchmark.
- **`line_total` convention unsettled.** The generator makes it tax-inclusive,
  so `sum(line_items) != subtotal + tax - discount` and 46 of 150 documents trip
  `check_totals_consistency`. Gold matches what is printed, so extraction
  scoring is unaffected, but Phase 3 anomaly detection cannot use line-item
  arithmetic until this is decided.

## A note on instruments vs. results

Two things in this repo return ground truth rather than measuring anything:
`ocr/oracle.py` and `--ocr-source gold`. They exist so the pipeline can be
tested end to end without an OCR engine, and so ablations have a
perfect-recognition ceiling row. **They are never baselines.** The runners
refuse to use the oracle without `--allow-oracle`, and every report generated
from either is watermarked. Their numbers must not appear in the project report
as OCR or end-to-end results.
