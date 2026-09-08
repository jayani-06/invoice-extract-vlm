# Phase 2 — VLM-based structured extraction

*Objective 2a from the project proposal: "develop a lightweight VLM-based
structured extraction module."*

**Exit criteria:** ≥ 90% F1 on header fields (vendor, invoice number, invoice
date, grand total); ≥ 80% F1 on line items.

---

## Result up front

| Extractor | Header accuracy | Line-item F1 | Critical-fields | Macro field acc |
|---|---|---|---|---|
| `null` (predicts nothing) | 0.000 | n/a | 0.000 | 0.225 |
| `rules` (OCR + regex) | **0.991** | **1.000** | 0.947 | 0.918 |
| `vlm:qwen2-vl-2b` | *not yet run — no GPU* | | | |

150 documents, clean profile, **perfect-OCR ceiling** (`--ocr-source gold`).
Reproduce with `make extract-ablation`.

**Both exit criteria are met — by the regex baseline.** That is the honest
headline, and it is a more useful finding than a VLM number would have been:

> On clean, synthetic, single-page invoices with perfect OCR, a label-anchored
> regex baseline already saturates the extraction task. The VLM has to justify
> itself somewhere else — on real scans, on unseen layouts, on multi-page
> documents, or on degraded OCR — not on this corpus.

That conclusion also supports the project's own framing: the research
contribution lives in Phases 3–4 (RAG enrichment and vendor validation), not
in extraction. It should be stated plainly in the report rather than buried,
because a reviewer will otherwise ask why a 2B-parameter model is in a
pipeline a regex can already do.

The number is *not* evidence the VLM is unnecessary. It is evidence that **this
corpus cannot tell the two apart**, which is a statement about the corpus. See
"What Phase 2 does not establish".

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

- **Nothing about real documents.** Results are on synthetic renders: no
  stamps, handwriting, folds, logos, or non-Latin script. FATURA / SROIE / CORD
  remain unconverted (`scripts/download_fatura.py` is an unfilled template with
  no confirmed URL), so the layout-generalisation claim has no third-party
  benchmark behind it.
- **Nothing about robustness to degradation.** Under `--ocr-source gold` the
  degradation profiles are inert: they move word boxes, not words, so a
  text-only extractor scores identically on `heavy` and `clean`. The runner now
  refuses to print those duplicate columns. Only a real OCR engine, or a VLM
  reading pixels, makes that axis mean anything.
- **Nothing about multi-page documents.** One page per document; the page
  retrieval step from refs [3] and [4] is untouched.
- **The rules baseline is tuned to a renderer we wrote.** Its rules are
  label-anchored rather than position-anchored specifically to limit this, but
  a baseline evaluated on synthetic data it was developed against is flattered
  regardless. Treat 0.991 as an upper bound on its real-world performance.

### Open question for Phase 3

The generator makes `line_total` **tax-inclusive**
(`qty × unit_price = 55,285.30 → line_total = 58,049.56`), so
`sum(line_items) ≠ subtotal + tax − discount` and 46 of 150 documents trip
Phase 0's `check_totals_consistency`. Gold matches what is printed, so
extraction scoring is unaffected — but Phase 3's anomaly detection cannot use
line-item arithmetic as a signal until the convention is settled. Conventionally
`line_total` is the pre-tax taxable value. Either regenerate with pre-tax line
totals, or document the tax-inclusive convention in `schema/schema.md` and relax
the Phase 0 check.
