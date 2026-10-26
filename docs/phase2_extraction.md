# Phase 2 — VLM-based structured extraction

*Objective 2a from the project proposal: "develop a lightweight VLM-based
structured extraction module."*

**Exit criteria:** ≥ 90% F1 on header fields (vendor, invoice number, invoice
date, grand total); ≥ 80% F1 on line items.

---

## Result up front

**On our own synthetic corpus** (150 documents, perfect-OCR ceiling):

| Extractor | Header accuracy | Line-item F1 | Critical-fields | Macro field acc |
|---|---|---|---|---|
| `null` (predicts nothing) | 0.000 | n/a | 0.000 | 0.225 |
| `rules` (OCR + regex) | **1.000** | **1.000** | 1.000 | 0.925 |

**On FATURA2** (300 documents, 50 unseen layouts, third-party, CC-BY-4.0):

| Extractor | Header accuracy | Critical-fields |
|---|---|---|
| `null` | 0.083 | 0.000 |
| `rules` | **0.559** | **0.000** |

Per field, the collapse is not uniform — and where it lands is the whole story:

| Field | Synthetic | FATURA2 | Δ |
|---|---|---|---|
| `totals.grand_total` | 1.000 | 1.000 | **0.000** |
| `invoice_meta.invoice_number` | 1.000 | 0.667 | −0.333 |
| `invoice_meta.invoice_date` | 1.000 | 0.570 | −0.430 |
| `parties.vendor.name` | 1.000 | **0.000** | **−1.000** |
| **Header accuracy** | **1.000** | **0.559** | **−0.441** |

### What this means

The regex baseline meets both exit criteria on our corpus and **fails badly on
a third-party one — a 44-point drop in header accuracy.** This reproduces
FATURA's central claim (ref [6]) on our own baseline: template-tuned extraction
collapses on unseen layouts.

Read field by field, the mechanism is clear:

- **`grand_total` transfers perfectly (1.000 → 1.000).** "TOTAL: 441.14 EUR" is
  a genuinely universal pattern, and a label anchor generalises across all 50
  layouts.
- **`invoice_date` and `invoice_number` degrade (−0.43, −0.33)** because the
  label vocabulary widens: "Date" vs "Invoice Date" vs "Date of Issue", and
  date formats our generator never produced.
- **`vendor.name` goes from 1.000 to zero.** This is the sharpest result in the
  phase. Our rule takes the first substantive line of the page, stripped of any
  document-type banner. That is correct on *every* invoice we render and wrong
  on essentially every FATURA layout, where the vendor name sits beside a logo
  block partway down the page. **The `null`
  extractor scores 0.333 on this field** (33% of FATURA2 documents have no
  vendor annotation, and null-vs-null counts as correct), so the rules baseline
  is *worse than predicting nothing* — the clearest possible demonstration that
  a positional heuristic has no transfer.

This **overturns the conclusion Phase 2 would have reached on synthetic data
alone.** The earlier reading — "a regex saturates the task, so the VLM must
justify itself elsewhere" — was an artefact of evaluating on documents we
generated ourselves. On unseen layouts there is a large, measurable gap
(0.559 → the 0.90 exit criterion), and closing it is exactly what a VLM is for.
That is the argument for the model, and it is evidence rather than assertion.

---

## The VLM has now been run, and it loses

Qwen2-VL-2B-Instruct, fp16 on a Colab T4, greedy decoding, **vision only** — no
OCR text in the prompt. 100 documents: 50 from our corpus and 50 from FATURA2,
the identical documents the baseline is re-scored on below. 39.3 s/document.
Reproduce with `notebooks/vlm_eval.ipynb`, then `run_extraction.py --replay`.

### On our own layouts (n = 50)

| Extractor | Header acc | Critical-fields | Macro field acc | Line-item F1 | Parse failures | Measurement |
|---|---|---|---|---|---|---|
| `rules` | **1.000** | **1.000** | **0.919** | **1.000** | 0 | perfect-OCR ceiling |
| `vlm:qwen2-vl-2b` | 0.843 | 0.720 | 0.818 | 0.970 | 2 | end-to-end (pixels only) |

On the 17 genuinely held-out test-split documents alone: rules 1.000 / 0.924,
VLM 0.853 / 0.813. The same answer, so the figure is not an artefact of the
train-split documents the 50 necessarily include.

### On FATURA2 (n = 50)

| Extractor | Header acc | Critical-fields | Macro field acc | Parse failures | Measurement |
|---|---|---|---|---|---|
| `rules` | **0.610** | 0.000 | **0.453** | 0 | perfect-OCR ceiling |
| `vlm:qwen2-vl-2b` | 0.432 | 0.000 | 0.241 | 4 | end-to-end (pixels only) |

### The gap did not close, which is the result

This phase predicted that a VLM would transfer to unseen layouts where a
label-anchored regex cannot. **It did not.**

| | Our layouts | FATURA2 | Drop |
|---|---|---|---|
| `rules` | 1.000 | 0.610 | **−39 points** |
| `vlm:qwen2-vl-2b` | 0.843 | 0.432 | **−41 points** |

The VLM is *just as* layout-sensitive as the regex, and worse everywhere. A 2B
vision-language model losing to hand-written regex on clean synthetic invoices
is the finding, and it is kept rather than explained away.

Counting fields, `rules` wins 21 to 3 on our corpus. On FATURA2 it is much
closer — 4 to 3 with 5 ties — and the three the VLM wins are exactly the ones a
label anchor cannot reach:

| Field | `rules` | `vlm` | Why |
|---|---|---|---|
| `parties.buyer.name` | 0.000 | **0.560** | No stable label precedes it; the model reads it from layout |
| `invoice_meta.payment_terms` | 0.000 | **0.460** | Free text, no anchor vocabulary |
| `invoice_meta.invoice_date` | 0.640 | **0.900** | Handles "Date of Issue" / "Date" variants the regex never saw |

So the mechanism the phase proposed is real — the model *does* read fields that
have no label to anchor on — it is simply swamped by the model being weaker at
everything that does have one, including `grand_total` (1.000 → 0.720) and
`invoice_number` (0.800 → 0.540).

### Three caveats, all of which favour the baseline

1. **This is not yet apples to apples, and the asymmetry helps `rules`.** Every
   baseline figure uses `--ocr-source gold`, a perfect-OCR ceiling. The VLM
   figure is genuine end to end — it read pixels, with no OCR anywhere. A fair
   comparison needs the baseline re-run with `--ocr-source engine`, which is
   open work. Expect the baseline to come down.
2. **The prompt template hard-codes `"currency": "INR"`.** Correct for our
   corpus, wrong for FATURA2, and it will have depressed currency-adjacent
   fields there. It was left alone deliberately: changing it *because the test
   set is known to be non-INR* is tuning against the test split.
3. **6 of 100 responses did not parse** and are scored as fully missed
   documents, not dropped. At 1536 `max_new_tokens` some responses are likely
   truncated mid-object rather than malformed, so part of that 6% is a budget
   problem rather than a model one.

### What this says about the project

The honest reading: **a 2B model is not the answer to the generalisation gap,
and the gap is still open.** The next move is a larger model (`qwen2.5-vl-3b`
in 4-bit, or 7B) and the hybrid image+OCR mode, which is already implemented
and unmeasured — refs [3] and [7] say the OCR hint should help, and on a model
this weak at reading there is a lot of room for it to.

---

## What was built

```
src/invoice_extract/
  reading_order.py          shared visual-line grouping (renderer + OCR + harness)
  models/
    postprocess.py          date/amount/GSTIN parsing, arithmetic checks, coercion
    prompting.py            extraction schema, prompt, robust JSON recovery
    rules.py                RuleBasedExtractor + NullExtractor baselines
    vlm.py                  VlmExtractor, model presets, per-field confidence
    registry.py             get_extractor("rules" | "null" | "vlm:<preset>")
scripts/
  run_extraction.py            one extractor -> gold.jsonl + pred.jsonl
  run_extraction_ablation.py   the Phase 2 headline table
```

### Scoring reuses Phase 0, deliberately

`run_extraction.py` writes prediction and gold JSONL in exactly the shape
`eval/harness.py` already consumes. Phase 2 is therefore scored by metrics that
were frozen *before any model existed* — ANLS for text, relative tolerance for
numbers, Hungarian matching for line items. Writing a new scoring path here
would have let the model be judged by rules written after seeing its output.

### Post-processing: normalise silently, flag arithmetic

Two rules run through `postprocess.py`:

1. **Never invent.** An unparseable value becomes `None`, not a guess. A wrong
   value is worse downstream than a missing one — Phase 3 can retrieve context
   for a gap, but it will happily enrich and validate a confidently wrong
   number.
2. **Flag arithmetic, never fix it.** If `subtotal + tax ≠ total`, that is
   either an extraction error or a genuinely malformed invoice, and those need
   opposite responses. Rewriting the total to make it balance would destroy
   exactly the signal Phase 3's anomaly detection is built to catch.

The one derivation allowed is `tax_total` from the printed tax components, and
only when every component carries an amount. Indian GST invoices print CGST and
SGST separately and never sum them, so this is arithmetic over values read off
the page, not a guess. It is tested both ways: derived when absent, never
overwriting a printed total.

`coerce_document` drops unknown keys rather than passing them through, because
the canonical schema is `extra="forbid"` — one hallucinated field would fail
validation for the whole document and cost every field the model got right.

### Prompting

`EXTRACTION_SCHEMA` is a reduced mirror of the canonical schema, usable
directly as a grammar for constrained decoding. It deliberately excludes
`doc_id`, `source`, `pages` and `grounding`: those are provenance, and a model
that invents them is hallucinating identity metadata.

`extract_json_object` is the most load-bearing piece of non-model code in the
phase. Models wrap JSON in prose, in markdown fences, in single-element lists,
and emit trailing commas. A parse failure scores as a totally missed document,
so a fragile parser would show up in the ablation as a *model quality* problem.
It tries, in decreasing order of confidence: the whole response, a code fence,
the first brace-balanced block (string- and escape-aware, because invoice text
contains braces), then each again with trailing commas stripped.

### The VLM wrapper

Model choice follows `configs/compute_plan.md` rather than re-deciding per
notebook: fp16 below 3B, 4-bit NF4 at or above. `qwen2-vl-2b` is the default.
A test asserts the presets obey that rule, so adding a model that violates the
compute plan fails CI rather than silently OOM-ing a T4.

`use_ocr_hint` is a flag, not a hard-coded choice, because whether hybrid
image+OCR input beats vision-only is exactly what the ablation is meant to
measure. Refs [3] and [7] predict it wins; predicting is not measuring. The OCR
block is placed *after* the instructions and explicitly framed as fallible —
presented as authoritative, a model copies OCR errors verbatim instead of
reading the pixels, which would make hybrid strictly worse on any page the OCR
mangled.

`predict()` returns raw fields without coercing, so the same
`postprocess.finalize` runs over every extractor's output. A difference in the
ablation table is then a difference in the model, not in how carefully each
wrapper cleaned up after itself.

---

## Three defects found and fixed while building this

1. **Reading order was wrong (Phase 1 defect).** `RenderResult.text` grouped
   words by `line_id`, which increments per *draw call*, not per visual line.
   A right-aligned value drawn separately from its label landed on its own
   line, so the gold text read `Invoice No:\nINV-2026-4733` for a row every OCR
   engine reads as one line. Any real engine would have been charged CER and
   WER for pure line-break noise. Fixed by deriving lines from geometry in a
   new shared `reading_order.py`, now used by the renderer, the OCR adapters
   and the ablation harness so all three agree on what a line is.

2. **Every GSTIN in the corpus had an invalid checksum (Phase 0 defect).** The
   synthetic generator drew the 15th character at random, so 295 of 300 GSTINs
   were wrong — the checksum validator would have fired on essentially every
   document, destroying a signal Phase 3 genuinely wants, since a real invoice
   with a bad GSTIN checksum is a meaningful red flag. The generator now
   computes the real mod-36 check digit, and the existing corpus was repaired
   in place. (The commonly-cited placeholder `29AAAAA0000A1Z5` is itself
   invalid; a test pins this so nobody "fixes" the implementation to match it.)

3. **Gold contained fields that appear nowhere on the page.** The generator
   emitted `buyer.phone` and `buyer.email`, which the renderer's compact buyer
   block never draws. An annotation containing values absent from the image is
   unextractable by construction — it charges every model, forever, for not
   hallucinating. The corpus builder now drops any party field the renderer did
   not ground.

---

## Running it

```bash
# Score one extractor (perfect-OCR ceiling; no OCR engine needed)
make extract EXTRACTOR=rules

# The headline table
make extract-ablation

# With a real OCR engine, once one is installed
make extract EXTRACTOR=rules OCR_SOURCE=engine

# With a GPU (Colab / Kaggle) — adds the model rows and makes the
# degradation-profile axis meaningful
python scripts/run_extraction_ablation.py --corpus data/processed/gst_in_synthetic \
    --extractors rules vlm:qwen2-vl-2b --vlm-modes hybrid vision-only \
    --ocr-source engine --ocr-engine paddle --profiles clean medium heavy \
    --out reports/phase2_ablation
```

---

## Status

**Done and verified**

- Post-processing, prompting, rules baseline, null baseline, VLM wrapper,
  extractor registry, runner and ablation runner
- Scoring wired to the Phase 0 harness, not a new one
- **312 tests passing**, ruff and black clean
- Rules baseline meets both exit criteria at the perfect-OCR ceiling

**Not done**

- **The VLM has never actually been run.** `models/vlm.py` is written against
  the transformers API and unit-tested through an injected fake
  model/processor, which covers prompt → response → parse → coerce. It does
  *not* establish that the generation call itself is correct. Expect the first
  Colab/Kaggle run to surface real bugs. No VLM number in this document is a
  measurement, because there isn't one.
- **No end-to-end number.** Every result here is `--ocr-source gold`, a
  perfect-OCR ceiling. No OCR engine is installed, so Phase 1's exit criteria
  (CER < 8%) also remain undemonstrated.

### What Phase 2 does not establish

- **Still nothing about *real* documents.** FATURA2 is a third-party benchmark
  but it is still synthetic — template-generated invoices, no stamps,
  handwriting, folds, or non-Latin script. It establishes layout
  generalisation, not scan realism. SROIE (real photographed receipts) and CORD
  remain unconverted and are the obvious next acquisition.
- **FATURA2 has no line-item annotations.** Tag 10 is a single `table`
  placeholder covering the whole table region, so every converted document has
  `line_items: []`. The line-item exit criterion is therefore verified only on
  our own corpus, where it is least meaningful.
- **The FATURA2 numbers are a perfect-OCR ceiling too.** The converter reuses
  FATURA2's own tokens and boxes as ground-truth text, so no OCR engine ran.
  That transcript is also *partial* — it covers annotated regions only, and the
  line-item table's text is simply absent — which is fine for header fields and
  useless for CER.
- **Nothing about robustness to degradation.** Under `--ocr-source gold` the
  degradation profiles are inert: they move word boxes, not words, so a
  text-only extractor scores identically on `heavy` and `clean`. The runner now
  refuses to print those duplicate columns. Only a real OCR engine, or a VLM
  reading pixels, makes that axis mean anything.
- **Nothing about multi-page documents.** One page per document; the page
  retrieval step from refs [3] and [4] is untouched.
- **The rules baseline is tuned to a renderer we wrote**, and FATURA2 now
  quantifies exactly how much that flattered it: 1.000 on home turf, 0.559 on
  unseen layouts. Treat any number measured only on the synthetic corpus as an
  upper bound, not an estimate.

### Settled: the `line_total` convention

Previously open, now decided and implemented. `line_total` is the **taxable
value** of a row — `quantity x unit_price`, net of that row's discount, before
tax — and `subtotal` is the sum of those. This is written up in
`schema/schema.md`.

Fixing it also uncovered a second bug: both consistency checks subtracted
`discount_total` from a `subtotal` that already had discounts netted into it,
double-counting them. That is why 46 of 150 documents were flagged.

After regenerating the corpus and correcting both checks:

| | before | after |
|---|---|---|
| Phase 0 `check_totals_consistency` warnings | 46 / 150 | **0 / 150** |
| `postprocess.check_arithmetic` flags | 150 / 150 | **0 / 150** |

Phase 3 can now use line-item arithmetic as an anomaly signal, because a flag
means a genuine inconsistency rather than a convention mismatch.
