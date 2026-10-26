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

## Try it in two commands

```bash
# the five-screen walkthrough: cleanup, OCR, extraction, RAG enrichment, results
streamlit run demo/app.py

# one invoice, end to end, to a PASS / REVIEW / FAIL verdict
python scripts/process_invoice.py \
    --image data/processed/gst_in_synthetic/images/medium/gst_synth_0105.png \
    --ledger data/processed/ledger \
    --history data/processed/gst_in_synthetic
```

`docs/RESULTS.md` has every measured number with its sample size and whether it
is a perfect-OCR ceiling or a genuine end-to-end figure.

## Quickstart

> **Windows:** `make` is not installed with Windows. Use `.\run.ps1 <task>`
> instead — same task names (`.\run.ps1 demo`, `.\run.ps1 test`). Run
> `.\run.ps1 help` to list them.

```bash
# Environment
uv venv && source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,ocr,demo,datasets]"
uv pip install paddleocr paddlepaddle   # Phase 1 OCR engine; see note below
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

# Phase 3: RAG categorisation + anomaly detection
make phase3

# Phase 4: vendor resolution + record validation
make phase4

# Real OCR
make ablation ENGINE=paddle
make extract-ablation OCR_SOURCE=engine

# The demo (see demo/README.md)
make demo-cache        # once, ~4 min: precomputes OCR + extraction
make demo              # opens the 5-screen walkthrough
```

### What each extra is for

The core install runs the schema, the harness, Phase 3 and Phase 4. Everything
that touches pixels or third-party data sits behind an extra, because a missing
one otherwise surfaces as an `ImportError` several minutes into a long run.

| Extra | Needed for |
|---|---|
| `dev` | `make test`, `make lint`, pre-commit |
| `ocr` | the renderer, the degradation profiles, `make corpus-full`, every Phase 1 target |
| `demo` | `make demo` (Streamlit). `make demo-cache` runs without it |
| `datasets` | `make fatura2` (reads the Hugging Face parquet mirror) |
| `vlm` | Colab/Kaggle inference only — do not install locally, it pulls torch |

`paddleocr` and `paddlepaddle` are listed as the `paddle` extra but are better
installed directly: the pinned resolver in `uv` often cannot satisfy them
alongside the rest, and they are the one dependency a CPU-only machine may
legitimately skip (`--engine oracle` substitutes gold text).

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
  rag/taxonomy.py           chart of accounts + ground-truth item categories
  rag/store.py              ledger retrieval (TF-IDF char n-grams) + vendor stats
  rag/categorize.py         retrieval-vote categorisation that abstains
  rag/anomaly.py            rule / statistical / semantic flags
  rag/vendor_master.py      supplier file accreted from past invoices
  rag/validate_vendor.py    vendor resolution + record validation
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
                            run_extraction_ablation.py, convert_fatura2.py,
                            build_ledger.py, run_rag_eval.py,
                            run_validation_eval.py, build_ledger.py
tests/                      pytest suite (347 tests)
configs/compute_plan.md     Colab / Kaggle T4 inference plan, quantization decision
docs/phase1_preprocessing.md  Phase 1 design, findings, how to run the ablation
docs/phase2_extraction.md     Phase 2 design, results, honest limitations
docs/phase3_rag.md            Phase 3 RAG enrichment, results, honest limitations
docs/phase4_validation.md     Phase 4 vendor validation, results, honest limitations
docs/limitations.md           everything the project has NOT established, in one place
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

15 documents per cell, 92.6 min CPU.

| Criterion | Outcome |
|---|---|
| CER < 8% on the noisy subset | **FAILED** - best `medium` CER 0.0812 against a 0.0800 target, short by 0.12 of a point; `heavy` 0.4284 (clean passes at 0.0066) |
| Preprocessing improves CER by >= 5 pts | **FAILED** - it improves CER on `heavy` only, by 1.1 pts; `medium` is 0.3 pts *worse* than raw |

The interesting part is that the two metric families disagree. On heavily
degraded pages the full chain minus binarisation raises **word accuracy from
0.1651 to 0.3062 (+14.1 pts)** while nudging CER the wrong way. CER is
text-only - a modern detector reads a skewed page fine - whereas word accuracy
requires the word to be found *in the right place*, which is what Phase 2
extraction actually depends on. Preprocessing earns its place on degraded
pages; the exit criterion was written against the wrong metric. On *clean*
pages it is a net loss (0.5733 raw vs 0.4842 for the full chain), so it should
be applied on evidence of degradation rather than unconditionally - nothing in
the pipeline makes that decision yet. Sauvola binarisation is the worst row in
every CER column: PaddleOCR's detector is trained on continuous-tone images and
binarising discards what it uses.

**An earlier sweep at n = 4 picked a different winning pipeline in 4 of 6
metric/profile cells.** Every qualitative claim above survived the re-run; none
of the magnitudes did. That is a finding about the experiment, and
`docs/phase1_preprocessing.md` records it rather than quietly replacing the
table.

**Phase 2 (done; VLM run, and it lost)** - post-processing, prompting, rules and
null baselines, the VLM wrapper, registry, runner and ablation runner, all
tested. The headline is a generalisation gap, measured on a third-party
benchmark:

| Corpus | Header accuracy (rules baseline) |
|---|---|
| Our synthetic invoices (150 docs) | **1.000** |
| FATURA2 (300 docs, 50 unseen layouts) | **0.559** |

A 44-point drop, reproducing FATURA's central claim on our own baseline.
`vendor.name` collapses from 1.000 to **0.000** - worse than predicting nothing.
So the regex baseline saturates *our* corpus, not the task, and there is a
large measurable gap for a VLM to close.

**The VLM has now been run, and it did not close the gap.** Qwen2-VL-2B, fp16
on a Colab T4, vision only, 100 documents - 50 of ours and 50 FATURA2, the same
documents the baseline is re-scored on:

| Extractor | Our layouts | FATURA2 | Drop |
|---|---|---|---|
| `rules` (perfect-OCR ceiling) | **1.000** | **0.610** | -39 pts |
| `vlm:qwen2-vl-2b` (end-to-end) | 0.843 | 0.432 | **-41 pts** |

**The VLM is just as layout-sensitive as the regex, and worse everywhere.** A
2B vision-language model losing to hand-written regex on clean synthetic
invoices is the result, and it is kept rather than explained away.

The mechanism the phase proposed is nonetheless real: on FATURA2 the VLM wins
exactly the fields a label anchor cannot reach - `buyer.name` 0.000 -> 0.560,
`payment_terms` 0.000 -> 0.460, `invoice_date` 0.640 -> 0.900 - and is swamped
by being weaker on everything that does have a label. Three caveats all favour
the baseline, chiefly that its figures are a perfect-OCR ceiling while the
VLM's are genuine end-to-end. See `docs/phase2_extraction.md`.

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

**Phase 3 (done; both exit criteria met)** - retrieval-based categorisation and
a three-family anomaly detector over a 100-invoice historical ledger.

| Criterion | Target | Result |
|---|---|---|
| Categorisation macro-F1 | >= 0.85 | **1.000** |
| Anomaly recall | >= 0.90 | **1.000** |
| Anomaly precision | >= 0.70 | **1.000** |

The scores are treated as a warning rather than a win: three accounts and ten
item types is a small label space, and recall is measured against anomalies we
injected ourselves. A sensitivity sweep shows categorisation degrading smoothly
as the ledger shrinks (1.000 at 100 documents, 0.977 at 40, 0.934 at 15), which
is evidence it genuinely retrieves rather than memorises. Two bugs surfaced only
by measuring: `unknown_vendor` rated medium dropped precision to 0.412, and
duplicate detection could never fire because the history was keyed normalised
and looked up raw. See `docs/phase3_rag.md`.

**Phase 4 (done; both exit criteria met)** - a vendor master accreted from 100
historical invoices, then held-out invoices resolved against it and validated.

| Criterion | Target | Result |
|---|---|---|
| Vendor resolution accuracy | >= 0.95 | **1.000** |
| False-positive rate (clean, known vendor) | < 0.10 | **0.000** |

Vendor names are corrupted before resolution (upper-cased, spacing removed,
truncated, reordered) because that is what the OCR stage produces. Three silent
bugs surfaced only by measuring: rapidfuzz is case-sensitive by default, so
"KRISHNA TEXTILES" scored 18.75 against "Krishna Textiles"; stripping legal
suffixes merged "Arjun & Co" with "Arjun Pvt Ltd" and turned a clean invoice
into a confident FAIL; and GSTIN-first resolution hid GSTIN fraud entirely by
routing the invoice to the GSTIN's real owner. See `docs/phase4_validation.md`.

**The corpus could not support Phase 4 as generated.** Vendor identity was
re-rolled per invoice, so no supplier kept one GSTIN or bank account across
their invoices (0/45), and prices swung 57% for the same vendor-item pair.
Bank-account-change detection would have fired on every invoice ever issued.
Identity is now a property of the company, prices are per-(vendor, item) with
+-4% jitter, and invoice numbers are a per-vendor sequence rather than a random
draw that produced ~4 genuine collisions.

**Known gaps are catalogued in `docs/limitations.md`** rather than scattered
across the phase docs. Two anomaly scenarios are deliberately held out and
currently **missed entirely** (0/4 and 0/3), which drops Phase 3 recall from
1.000 to 0.800 — that is the honest measure of a detector, and they are kept
unfixed as a permanent probe.

**Still open**

- **One VLM, one configuration.** 2B vision-only has been run and lost. The
  3B/7B presets and the hybrid image+OCR mode are implemented and unrun, so
  "a VLM cannot close this gap" is *not* established - only that this one
  does not.
- **Phase 5 (API / service layer) not started.**
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
