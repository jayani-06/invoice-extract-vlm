# Phase 1 — Document ingestion & preprocessing

*Objective 1 from the project proposal: "build a document ingestion &
preprocessing module (OpenCV-based OCR pipeline) that reliably handles
scanned, noisy, multi-layout invoices and receipts."*

**Exit criteria:** CER < 8% on the noisy subset, and preprocessing must
improve CER by ≥ 5 absolute points versus the raw degraded page.

---

## Results

PaddleOCR 3.7 (PP-OCRv5 mobile, CPU), 4 documents per cell, 3 degradation
profiles. Reproduce with `make ablation ENGINE=paddle`.

### Normalised CER — preprocessing does not help, and binarisation hurts badly

| Pipeline | clean | medium | heavy |
|---|---|---|---|
| `raw` | **0.0046** | 0.1738 | **0.3687** |
| `gray+border+deskew` | 0.0042 | **0.1618** | 0.4294 |
| `full(no-binarize)` | 0.0061 | 0.2330 | 0.4275 |
| `full+sauvola` | 0.0228 | 0.4300 | 0.4589 |

### Word accuracy (located *and* read correctly) — preprocessing helps a lot

| Pipeline | clean | medium | heavy |
|---|---|---|---|
| `raw` | **0.5810** | 0.3056 | 0.2731 |
| `gray+border+deskew` | 0.5347 | **0.4190** | **0.4792** |
| `full(no-binarize)` | 0.4560 | 0.4074 | 0.4769 |
| `full+sauvola` | 0.5093 | 0.3495 | 0.4167 |

### The two metric families disagree, and that is the finding

On heavily degraded pages, `deskew` **raises word accuracy from 0.273 to 0.479
(+21 points)** and detection F1 from 0.294 to 0.528 — while making CER slightly
*worse*. That is not a contradiction; the metrics measure different things:

- **CER is text-only.** A modern detector reads a skewed page perfectly well;
  the characters come out right even when the page is at 5 degrees.
- **Word accuracy requires the word to be found *in the right place*.** On a
  skewed page the predicted boxes are rotated away from the gold boxes, IoU
  matching fails, and the word scores as missed even though it was read
  correctly.

**Word accuracy is the metric that matters downstream.** Phase 2 extraction
needs to know *which field* a value belongs to, and that is a spatial question
— a correctly-read total in the wrong place is not a usable extraction. So the
honest summary is: deskewing earns its place, and CER alone would have hidden
that.

### Against the exit criteria — both failed

| Criterion | Result |
|---|---|
| CER < 8% on the noisy subset | **FAILED.** Best `medium` CER is 0.162; best `heavy` is 0.369. Clean passes easily at 0.005. |
| Preprocessing improves CER by >= 5 points | **FAILED.** Best CER delta from preprocessing is −1.2 points (medium); on heavy it is +6 points *worse*. |

Stating this plainly rather than reframing: **the pipeline as specified did not
meet its targets.** What it did establish is that the targets were written for
the wrong metric — the criterion should be word accuracy, where preprocessing
delivers +21 points on the hardest pages.

### Why classical preprocessing backfires on a modern detector

The Phase 1 plan assumed a Tesseract-era pipeline, where binarisation and
denoising are prerequisites. PaddleOCR's detector is a CNN trained on
photographs of real documents; it does its own normalisation internally and
expects continuous-tone input. Sauvola binarisation throws away the grayscale
information it was trained to use, which is why `full+sauvola` is the worst row
in every CER column (0.43 on medium, versus 0.17 raw — a 25-point
regression). This is consistent with refs [3] and [7] recommending hybrid
image+text input rather than aggressive cleanup.

### Cost

| Pipeline | preprocessing ms/page |
|---|---|
| `raw` | 0 |
| `gray+border+deskew` | 152 |
| `full(no-binarize)` | 587 |
| `full+sauvola` | 1053 |

OCR itself dominates at ~40 s/page (CPU, mkldnn disabled — see below), so
preprocessing cost is not the constraint; correctness is.

### Caveats on these numbers

- **n = 4 documents per cell.** The whole sweep took 100 minutes on CPU. The
  deskew effect on `heavy` (+21 points word accuracy) is large enough to be
  credible at this n; the smaller CER deltas are not, and should not be quoted
  as precise.
- **Detection metrics penalise tokenisation differences.** PaddleOCR splits
  punctuation into separate tokens (`Salai` `,`), so predicted word counts run
  ~1.8x gold. That depresses precision independently of recognition quality,
  which is part of why absolute detection F1 looks low even on clean pages.
- **`fast=True` models.** PP-OCRv5 mobile, chosen for a 2.6x speedup. The
  larger PP-OCRv6 medium models may rank the pipelines differently.

---

## The problem Phase 1 had to solve first

Phase 0 froze the schema, wrote the eval harness, and generated 150 synthetic
GST invoices — but only as **JSON**. Every `pages[0].image_path` was a
placeholder. Phase 1 is about image preprocessing and OCR, and there were no
images anywhere in the project, nor any OCR ground truth to score against.

The obvious route — download FATURA/SROIE, hand-annotate word boxes — costs
weeks and is squarely off the critical path, because this project's research
contribution is *downstream* of extraction (RAG enrichment and vendor
validation, per the research gap). So Phase 1 takes a different route:

> **Render the synthetic invoices, then degrade them.**

This yields something no real scan corpus can offer: a **paired** corpus. The
same document exists as a pristine render *and* at several degradation
severities, with pixel-exact word boxes at every one. That matters because:

- **Ground truth is free and perfect.** No annotation, no annotator drift.
- **The preprocessing ablation becomes measurable.** With real scans you only
  know OCR was wrong; here you know *which degradation* cost the characters,
  because you hold the clean original of every noisy page.
- **Layout diversity is controllable.** The FATURA finding is that
  template-tuned extractors collapse on unseen layouts, so layout is *sampled*
  across independent axes rather than drawn from a few fixed templates.

The honest limitation is stated up front: **synthetic renders are not real
scans.** Real documents bring stamps, handwriting, staples, folds, logos and
non-Latin script that this generator does not model. Phase 1 therefore
establishes the pipeline and the measurement apparatus; validating it on
FATURA/SROIE/CORD remains a task, and is listed under "What Phase 1 does not
establish" below.

---

## What was built

```
src/invoice_extract/
  render/
    fonts.py              cross-platform TTF resolution; family is a layout axis
    layout.py             drawing primitives that emit word boxes as they draw
    invoice_renderer.py   canonical JSON -> page image + word-level ground truth
  degrade/
    transforms.py         12 scan artifacts + 4 severity profiles
  preprocess/
    stages.py             8 toggleable OpenCV stages
    pipeline.py           composable pipeline + the ablation configurations
  ocr/
    base.py               engine-agnostic interface (OcrWord / OcrResult)
    tesseract.py          Tesseract 5 adapter
    paddle.py             PaddleOCR adapter
    oracle.py             ground-truth instrument (NOT a baseline — see below)
  eval/
    ocr_metrics.py        CER, WER, detection P/R/F1, word accuracy
scripts/
  build_image_corpus.py       canonical JSON -> paired image corpus
  run_preprocess_ablation.py  the Phase 1 headline experiment
  make_pipeline_figure.py     before/after contact sheet (visual QA + report figure)
```

### The rendering pipeline

`InvoiceRenderer` draws an A4 page from a canonical document. Every text-drawing
call records a `WordBox` — the token, its pixel bounding box, and the canonical
field path it came from — so the image and its ground truth are produced in
lockstep and cannot drift apart.

`LayoutStyle.sample()` draws from twelve independent axes (typeface, base point
size, header placement, table rule style, header shading, totals-block side,
HSN column presence, buyer block presence, margins, line spacing, label
punctuation, label case). The corpus therefore covers a combinatorial layout
space; a fixed set of templates would have quietly reproduced the very
weakness FATURA warns about inside our own evaluation.

The renderer also writes real `pages[]` dimensions and a `grounding[]` entry
per canonical field back into each document — which is what **Phase 2 (VLM
extraction)** needs to train and evaluate against.

### Degradation

Twelve artifacts, applied in physical order (printed → photocopied → warped →
lit → sensed → compressed), sampled from four severity profiles:

| | clean | light | medium | heavy |
|---|---|---|---|---|
| geometric | — | ±0.7° skew | ±2.5° skew, perspective, downsample | ±5° skew, strong perspective, heavy downsample |
| photometric | — | mild gradient, grain | shadow, scan border, ink spread | strong shadow, bleed-through, wide border |
| sensor | — | light Gaussian noise, JPEG q70–92 | Gaussian + salt-pepper, JPEG q45–80 | heavy noise + speckle, JPEG q25–60 |

**The load-bearing design decision:** every geometric transform returns the 3×3
homography it applied, and the pipeline composes them. Gold word boxes are
mapped through that composition, so they follow the glyphs through skew,
perspective and rescaling. Without this the ablation would still run and still
print numbers — they would simply be wrong, in a way no amount of staring at a
CER table would reveal. `tests/test_degrade.py` and
`tests/test_preprocess.py::TestGeometryEndToEnd` verify this against the
*pixels*, asserting that mapped boxes still land on ink.

### Preprocessing stages

| Stage | What it does | Why it is separately toggleable |
|---|---|---|
| `grayscale` | channel reduction | baseline |
| `remove_border` | whitens (never crops) dark scanner bands | crop would shift every coordinate |
| `correct_perspective` | 4-point page-quad warp | can catastrophically misfire; opt-in only |
| `deskew` | projection-profile or `minAreaRect` | two methods, measurably different |
| `normalize_illumination` | morphological background division | this is what rescues global thresholding |
| `denoise` | median / NLM / bilateral | which one wins depends on the degradation |
| `upscale` | lifts small text to OCR's comfort zone | Tesseract degrades sharply below ~20 px x-height |
| `binarize` | Otsu / adaptive / Sauvola / none | `none` is a real option — modern engines binarize internally |

Ordering is enforced by the `Pipeline`, not by the config: a configuration
names a *subset* of stages and they always execute in canonical order, so a
config cannot express a physically wrong sequence (binarizing before deskewing,
for instance).

#### Two findings that came out of building it

1. **A scan border defeats deskew unless removed first.** The border is added
   to an already-skewed page, so its edges are axis-aligned while the text is
   not. Those full-width edges dominate the projection profile, which peaks at
   0°, and the estimator reports no skew at all. `remove_border` preceding
   `deskew` is therefore load-bearing, not tidiness.
   (`test_a_scan_border_defeats_deskew_unless_removed_first`)

2. **Projection-profile deskew beats `minAreaRect` on clean skewed text**, and
   the gap widens with page furniture. `minAreaRect` fits a rectangle to the
   whole ink mask; the projection profile keys on the text lines themselves.
   Both are implemented and both are in the ablation, so the claim is measured
   rather than asserted.

### Metrics

- **CER / WER**, page-level edit distance over reading-order text. Reported
  **raw and normalized** (case-folded, punctuation-stripped, whitespace
  collapsed) — raw punishes a stray space, normalized reflects what survives
  into field extraction, and reporting only one lets a preprocessing choice
  look better than it is.
- **Detection P/R/F1 and mean IoU** — were the words *found*.
- **Word accuracy** — found *and* read correctly. This is the metric that
  predicts downstream field-extraction success.
- Aggregates are **macro-averaged per page** (the system is judged invoice by
  invoice, and micro-averaging would let one dense page dominate), with
  **median and p90** alongside the mean because CER distributions are
  long-tailed.

---

## Running it

```bash
# 1. Build the paired image corpus (no OCR engine needed)
make corpus                 # 25 documents x 4 profiles
make corpus-full            # all 150

# 2. Verify the whole chain end to end without any OCR engine
make ablation-smoke         # metrics plumbing
make figure                 # visual QA: degraded input vs each pipeline stage

# 3. Install an OCR engine, then run the real experiment
make ocr-install            # pytesseract wrapper + imaging deps
#   Windows : winget install --id UB-Mannheim.TesseractOCR
#   macOS   : brew install tesseract
#   Debian  : sudo apt install tesseract-ocr
make ablation ENGINE=tesseract

# PaddleOCR needs no system binary and is the plan's primary engine:
make paddle-install
make ablation ENGINE=paddle
```

Output lands in `reports/phase1_ablation/` as `ablation.md` (the tables) and
`results.json` (the raw record).

---

## On the `oracle` engine

`ocr/oracle.py` returns the renderer's own ground truth, optionally with
synthetic character errors. **It is an instrument, not a baseline.** It exists
so that the render → degrade → preprocess → recognize → score chain can be
exercised and unit-tested with no OCR engine installed, and so the ablation has
a perfect-recognition ceiling row that separates "OCR lost it" from
"extraction lost it".

It is fenced off deliberately: `run_preprocess_ablation.py` **refuses to run**
it without an explicit `--allow-oracle`, and any report generated from it
carries a warning banner. Numbers from the oracle must never appear in the
project report as OCR results.

---

## Status

**Done and verified**

- Renderer with sampled layout diversity and pixel-exact word-box ground truth
- 12 degradation transforms, 4 severity profiles, homography-tracked
- 8 preprocessing stages, 14 named pipeline configurations
- Engine-agnostic OCR interface with Tesseract and PaddleOCR adapters
- CER / WER / detection metrics with per-stage delta tables
- Corpus builder, ablation runner and figure generator, all CLI-driven
- 165 tests passing, including geometry verified against pixels rather than
  against the matrices that produced them

**Visual verification** — `reports/figures/pipeline_stages.png` (regenerate with
`make figure`) shows the pipeline on `medium` and `heavy` pages: the scan
border is removed, the page levelled, the shadow gradient flattened, and
Sauvola yields crisp text. This is the report figure; it is also the fastest
way to catch a stage that is quietly destroying the page, which a single CER
number will happily hide.

**OCR engine installed and the ablation run.** PaddleOCR 3.7 + paddlepaddle
3.3.1 (pip only, no system binary). Two workarounds were needed, both
documented in `ocr/paddle.py`:

- **oneDNN crashes the detector on this CPU** — paddlepaddle 3.3.1 raises
  `ConvertPirAttribute2RuntimeAttribute not support` from the oneDNN executor.
  `enable_mkldnn=False` avoids it; it is the default because a crash is worse
  than being slower. This is not model-specific: PP-OCRv5 and v6 both fail.
- **Word boxes are opt-in** — `return_word_box=True` yields real per-word
  geometry. Without it the only option is apportioning a line box across its
  words by character count, which fabricates the geometry that word-level
  detection metrics then measure.

A `fast=True` preset swaps to PP-OCRv5 mobile models: ~40 s/page versus
~105 s/page on CPU.

**Both exit criteria failed** — see Results above. That is reported as a
result, not deferred.

### What Phase 1 does not establish

State these as limitations in the report rather than letting a reviewer find
them:

- Results are on **synthetic renders, not real scans.** No stamps,
  handwriting, folds, logos, or non-Latin script.
- **FATURA / SROIE / CORD are still unconverted.** The converters exist from
  Phase 0 but have not been run, so the layout-generalisation claim is
  currently untested against a third-party benchmark.
- **Multi-page documents are not handled.** The renderer emits one page per
  document; the page-retrieval step from refs [3] and [4] is a Phase 2 concern.
- The oracle's quality-aware error injection is deliberately crude and should
  not be read as a noise model.
