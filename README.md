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

> **Windows:** `make` is not installed with Windows. Use `.\run.ps1 <task>`
> instead — same task names (`.\run.ps1 demo`, `.\run.ps1 test`). Run
> `.\run.ps1 help` to list them.

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

# Third-party benchmark: 50 layouts we did not design
make fatura2 && make fatura2-eval

# Real OCR (PaddleOCR is already installed)
make ablation ENGINE=paddle
make extract-ablation OCR_SOURCE=engine

# The demo (see demo/README.md)
make demo-cache        # once, ~4 min: precomputes OCR + extraction
make demo              # opens the 5-screen walkthrough
```

## Layout

```
schema/                     canonical JSON Schema + docs + example
src/invoice_extract/
  data/models.py            pydantic mirror of the schema
  data/validate.py          jsonschema + pydantic validation, totals consistency
  data/converters/          per-dataset -> canonical (fatura2/sroie/cord/gst_in)
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
                            run_extraction_ablation.py, convert_fatura2.py
tests/                      pytest suite (347 tests)
configs/compute_plan.md     Colab / Kaggle T4 inference plan, quantization decision
docs/phase1_preprocessing.md  Phase 1 design, findings, how to run the ablation
docs/phase2_extraction.md     Phase 2 design, results, honest limitations
demo/                       Streamlit demo app + its README
notebooks/                  Colab notebooks (incl. vlm_demo_outputs.ipynb)
run.ps1                     Windows task runner (mirrors the Makefile)
docker-compose.yml          app (dev) + optional labelstudio/minio profiles
```

## Status

**Phase 0 (complete)** - schema frozen, eval harness written and tested, 150
synthetic GST invoices generated.

**Dataset (built)** - 150 canonical documents x 4 degradation profiles = 600
images, 600 word-level OCR ground-truth files, 62,092 annotated words, 150/150
unique layout combinations, 0 schema violations. Reproducible with
`make corpus-full`; not tracked in git.

**Phase 1 (run; both exit criteria FAILED)** - PaddleOCR 3.7 installed and the
ablation executed on real images. Reported as a result, not deferred:

| Criterion | Outcome |
|---|---|
| CER < 8% on the noisy subset | **FAILED** - best `medium` CER 0.162, `heavy` 0.369 (clean passes at 0.005) |
| Preprocessing improves CER by >= 5 pts | **FAILED** - best delta is -1.2 pts; on `heavy` preprocessing is *worse* |

The interesting part is that the two metric families disagree. On heavily
degraded pages `deskew` raises **word accuracy from 0.273 to 0.479 (+21 pts)**
and detection F1 from 0.294 to 0.528, while nudging CER the wrong way. CER is
text-only - a modern detector reads a skewed page fine - whereas word accuracy
requires the word to be found *in the right place*, which is what Phase 2
extraction actually depends on. Deskewing earns its place; the exit criterion
was written against the wrong metric. Sauvola binarisation is the worst row in
every CER column: PaddleOCR's detector is trained on continuous-tone images and
binarising discards what it uses. See `docs/phase1_preprocessing.md`.

**Phase 2 (code complete, VLM unrun)** - post-processing, prompting, rules and
null baselines, the VLM wrapper, registry, runner and ablation runner, all
tested. The headline is a generalisation gap, measured on a third-party
benchmark:

| Corpus | Header accuracy (rules baseline) |
|---|---|
| Our synthetic invoices (150 docs) | **0.983** |
| FATURA2 (300 docs, 50 unseen layouts) | **0.559** |

A 42-point drop, reproducing FATURA's central claim on our own baseline.
`vendor.name` collapses from 0.900 to **0.000** - worse than predicting nothing.
So the regex baseline saturates *our* corpus, not the task, and there is a
large measurable gap for a VLM to close. **The VLM itself has never been run** -
there is no GPU on this machine, so `models/vlm.py` is unit-tested through an
injected fake model but its generation call is unverified. See
`docs/phase2_extraction.md`.

**Recently closed**

- **OCR engine installed.** PaddleOCR 3.7 + paddlepaddle 3.3.1, pip-only. Two
  workarounds were needed and are documented in `ocr/paddle.py`: oneDNN crashes
  the detector on this CPU (`enable_mkldnn=False`), and word-level boxes are
  opt-in (`return_word_box=True`). A `fast=True` preset swaps to mobile models
  for a 2.6x speedup (~40 s/page vs ~105 s/page, CPU).
- **FATURA2 converted.** 10,000-invoice / 50-layout third-party benchmark
  (CC-BY-4.0) via the Hugging Face parquet mirror, since Zenodo was returning
  504s. `scripts/convert_fatura2.py`. The tag vocabulary ships as bare integers
  with no label names, so it was inferred by audit; `--audit-tags` re-verifies
  it and fails loudly if a future release renumbers them.
- **`line_total` convention settled.** It is the pre-tax taxable value; see
  `schema/schema.md`. Fixing it exposed a second bug - both consistency checks
  subtracted `discount_total` from a `subtotal` that already had it netted in.
  Warnings went from 46/150 and 150/150 to **0/150 and 0/150**.

**Still open**

- **The VLM has never been run.** No GPU here.
- **No real *photographed* documents.** FATURA2 is third-party but still
  synthetic. SROIE (real photographed receipts) and CORD remain unconverted.
- **FATURA2 has no line-item annotations** (its table is one placeholder
  token), so the line-item criterion is verified only on our own corpus.

## A note on instruments vs. results

Two things in this repo return ground truth rather than measuring anything:
`ocr/oracle.py` and `--ocr-source gold`. They exist so the pipeline can be
tested end to end without an OCR engine, and so ablations have a
perfect-recognition ceiling row. **They are never baselines.** The runners
refuse to use the oracle without `--allow-oracle`, and every report generated
from either is watermarked. Their numbers must not appear in the project report
as OCR or end-to-end results.
