# Results

Every number this project has measured, in one place, with what each one does
and does not establish.

Read `limitations.md` alongside this. Nothing here is quoted without its `n`
and without saying whether it is a **ceiling** or an **end-to-end** figure —
the two are not comparable and conflating them is the easiest way to overstate
this system.

---

## Run it

```bash
python scripts/process_invoice.py \
    --image data/processed/gst_in_synthetic/images/medium/gst_synth_0105.png \
    --ledger data/processed/ledger \
    --history data/processed/gst_in_synthetic
```

One image in, a structured document and a verdict out. Stages that could not
run (no ledger, no history) are named in the output rather than passing
silently — a skipped check reported as a clean result is the failure mode this
project is built against.

A real run on a `medium`-degraded page, abbreviated:

```
2. OCR (paddle, mobile models)        165 words in 42.8s
3. extraction    vendor Arjun Pvt Ltd | INV-2026-039002 | 2026-04-10 | 464.81 | 5 lines
4. arithmetic    x sum(line_items)=3021.03 != subtotal=21.03
5. categories    + 5/5 assigned (Office Supplies, Furniture, IT Equipment)
6. anomalies     high  arithmetic_mismatch
7. vendor        resolved Arjun Pvt Ltd (gstin); REVIEW price_deviation x4

  REVIEW   based on all stages
```

That is the system working. OCR misread the subtotal on a noisy page, and
**every downstream stage caught it** rather than booking a wrong number. The
verdict is REVIEW, not PASS.

---

## Phase 1 — preprocessing and OCR

PaddleOCR 3.7 on CPU. 15 documents per cell for the mobile models, 5 for
PP-OCRv6 medium.

**Normalised CER (lower better), best pipeline per profile:**

| Models | clean | medium | heavy |
|---|---|---|---|
| PP-OCRv5 mobile (`fast=True`) | 0.0066 | 0.0812 | 0.4284 |
| PP-OCRv6 medium | **0.0006** | **0.0114** | **0.2486** |

**Word accuracy — located *and* read correctly (higher better), mobile models:**

| Pipeline | clean | medium | heavy | mean |
|---|---|---|---|---|
| `raw` | **0.5733** | 0.4281 | 0.1651 | 0.3888 |
| `full(no-binarize)` | 0.4842 | **0.4918** | **0.3062** | 0.4274 |
| `adaptive` | **0.5733** | 0.4432 | 0.2671 | **0.4279** |

### Against the exit criteria

| Criterion | Outcome |
|---|---|
| CER < 8% on noisy pages | **Met on `clean` and `medium` with v6-medium** (0.0114). **Fails on `heavy`** under every configuration measured — best is 0.2486, three times the target. |
| Preprocessing improves CER by ≥ 5 points | **Fails.** It improves CER on `heavy` only, by 1.1 points, and is neutral-to-worse elsewhere. |

### What Phase 1 actually established

- **The criteria were written against the wrong metric.** Preprocessing is
  worth +14.1 points of word accuracy on `heavy` with mobile models and
  **+22 points** with v6-medium (0.3077 → 0.5304), while moving CER the wrong
  way. Extraction needs to know *which field* a value belongs to, which is
  spatial, so word accuracy is the metric Phase 2 depends on.
- **Preprocessing is a net loss on clean pages** (−8.9 points), which is why
  `adaptive` exists.
- **The pipeline ranking is not a property of the pipeline.** It changed when
  n went 4 → 15 (4 of 6 cells) and again when the recognition model changed
  (3 of 4 cells). Any "pipeline X is best" claim is conditional on both.
- **`adaptive` buys a tie, not best-of-both.** It matches `raw` exactly on
  clean pages — the penalty is gone — but gives up 4.9 points on `medium` and
  3.9 on `heavy`, ties the best fixed config on the mean, and costs ~350 ms
  more. The cause is identified: its `upscale` gate is keyed on Laplacian
  variance, which heavy speckle keeps high, so the gate fails to fire on the
  pages that most need it.
- **Binarisation always hurts.** `full+sauvola` is the worst row in every CER
  column. PaddleOCR's detector is trained on continuous-tone images.

---

## Phase 2 — extraction

All `rules` figures are a **perfect-OCR ceiling** (`--ocr-source gold`, the
renderer's own text). The VLM figures are **genuine end-to-end** — it read
pixels with no OCR anywhere. The asymmetry favours the baseline.

| Corpus | What it is | `rules` ceiling | `rules` end-to-end | n |
|---|---|---|---|---|
| `gst_in_synthetic` | our own renderer | **1.000** | **0.653** | 50 |
| `fatura2` | third-party, 50 synthetic layouts | **0.700** | **0.645** | 50 |
| `sroie` | **652 real photographed receipts** | **0.777** | not measured | 157 |

The two right-hand columns are the single most important thing on this page.
A perfect-OCR ceiling costs the baseline **34.7 points** on our own corpus and
**5.5** on FATURA2, so a ceiling figure is not a small overstatement — on home
turf it is most of the score.

### The generalisation gap, and the finding that complicates it

A regex tuned to our own renderer scores 1.000 at home and 0.700 on someone
else's layouts. That gap is the argument for a model, and it is evidence
rather than assertion.

But **real photographed receipts score higher than FATURA2's synthetic
templates** (0.777 vs 0.700). All three are ceiling numbers from gold text, so
image quality is not in play: what hurts a label-anchored baseline is *layout
diversity*, not degradation. The difficulty of real documents lives in the OCR
stage, which these figures exclude by construction.

**One missing label cost an entire field.** `invoice_date` was null on 157 of
157 receipts because `_META_LABELS` knew "invoice date" and "date of issue"
but not a bare "DATE", which is what receipts print. Adding it (fitted on the
train split, declared) moved SROIE test header accuracy 0.684 → **0.777** and
independently moved FATURA2 0.610 → **0.700**, while our corpus stayed at
1.000. That cross-corpus transfer is what separates a fix from test-set tuning.

### The VLM, compared fairly

Qwen2-VL-2B-Instruct, fp16 on a Colab T4, greedy, **vision only**, 100
documents, 39.3 s each.

Every earlier version of this comparison put the VLM's end-to-end figure
beside the baseline's perfect-OCR ceiling, and concluded the VLM lost on both
corpora. **That conclusion was an artefact of the asymmetry.** With the
baseline re-run through real OCR on the same 50 documents:

| Extractor | ours | FATURA2 | Drop |
|---|---|---|---|
| `rules`, perfect-OCR ceiling | 1.000 | 0.700 | −30 pts |
| `rules`, **end-to-end** | 0.653 | 0.645 | **−0.8 pts** |
| `vlm:qwen2-vl-2b`, **end-to-end** | **0.843** | 0.432 | **−41.1 pts** |

Two things reverse:

**1. The VLM wins on our layouts, by 19 points.** 0.843 against 0.653, and on
whole-document critical fields the gap is 0.720 against 0.000 — the regex gets
*no* document completely right once real OCR is in the loop. The ceiling was
hiding a 34.7-point penalty that only the baseline pays, because the VLM reads
pixels and never touches the OCR stage at all.

**2. The regex is the robust one; the VLM is the layout-sensitive one.** This
is the exact opposite of what the phase predicted. End-to-end, `rules` is
essentially flat across the two corpora (0.653 → 0.645, −0.8 points) while the
VLM collapses by 41. The 30-point "generalisation gap" that motivated the whole
model argument was **mostly a measurement artefact**: it was the gap between
perfect text and real text, not between home layouts and foreign ones.

The mechanism is clear once stated. Real OCR damages the label anchors the
regex depends on — a mangled "Invoice No:" costs the field — so the baseline
falls hard on our corpus. On FATURA2 it was already failing for layout
reasons, so OCR noise adds almost nothing. The VLM, reading pixels, is immune
to the OCR stage and pays instead for layouts it has not seen.

**What this does not establish:** that the VLM is better. It wins on
documents our own renderer made and loses by 21 points on someone else's,
which is its own generalisation problem. 6 of 100 of its responses did not
parse and are scored as fully missed documents. And 2B is the smallest preset
— the 3B/7B presets and the hybrid image+OCR mode are implemented and unrun.

The mechanism the phase proposed is nonetheless real: on FATURA2 the VLM wins
exactly the fields a label anchor cannot reach — `buyer.name` 0.000 → 0.560,
`payment_terms` 0.000 → 0.460, `invoice_date` 0.640 → 0.900 — and is swamped
by being weaker at everything that *has* a label.

6 of 100 responses did not parse and are scored as fully missed documents, not
dropped. "A VLM cannot close this gap" is **not** established: 2B is the
smallest preset, and the 3B/7B presets and the hybrid image+OCR mode are
implemented and unrun.

---

## Phase 3 — RAG enrichment

100 ledger documents, 50 evaluation. TF-IDF character n-grams, weighted vote.

| Metric | Value |
|---|---|
| Categorisation accuracy | 0.993 |
| Coverage (share it answered) | 0.993 |
| Macro F1 | 0.996 |
| Anomaly precision | 1.000 |
| Anomaly recall, designed-for scenarios | 1.000 |
| **Anomaly recall, including held-out** | **0.800** |
| False-positive rate on clean | 0.000 |

**These are pre-hardening numbers.** The label space has since widened from 3
accounts / 10 descriptions to 6 / 21 with deliberately confusable pairs, and
macro-F1 of 0.996 should be expected to fall. The re-run is queued.

### The held-out probe

Two anomaly scenarios were withheld from the design and **both are missed
completely**:

| Held-out scenario | Caught |
|---|---|
| `line_item_padding` (line total inflated, invoice still reconciles) | **0 / 4** |
| `gstin_state_contradicts_address` | **0 / 3** |

Kept unfixed permanently. Closing them and re-reporting 100% would reproduce
the circularity they exist to expose.

### Complete RAG

The vote is retrieval plus aggregation, not RAG. `rag/generate.py` now prompts
a local Qwen2.5-0.5B-Instruct with the retrieved neighbours and their
human-assigned categories, constrained to one account code or `ABSTAIN`;
`rag/embed.py` adds dense retrieval behind the same interface. Both are
verified against real weights. **Their Phase 3 numbers are queued** — every
figure above is a TF-IDF vote number.

The vote is kept as the control: a generator that cannot beat a weighted vote
over the *same* retrieved context has added a model and bought nothing.

---

## Phase 4 — vendor validation

100 history documents, 50 evaluation, vendor names corrupted before resolution
to simulate OCR.

| Metric | Easy corpus | **Hardened corpus** |
|---|---|---|
| Vendor resolution accuracy | 1.000 | **1.000** |
| Fraud precision | 1.000 | **0.750** |
| Recall, designed-for scenarios | 1.000 | 1.000 |
| **Recall, including held-out** | **0.524** | 0.545 |
| False-positive rate on clean | 0.000 | **0.308** |

The right-hand column is the honest one. The 1.000s were flattered by a corpus
with no hard cases: one address per supplier, one name, one bank account
forever. It now has branch addresses (44 of 70 suppliers), trading names (24),
and **legitimate** bank changes (4). Precision fell to 0.750 and the
false-positive rate to 0.308, and **no threshold was adjusted to recover the
old figures** — a lower honest number is the deliverable.

Of the new false positives, one is `address_mismatch` (a check that previously
never fired at all) and the rest are `price_deviation`, because a 21-item
catalogue thins the per-(vendor, item) history. Every one is REVIEW, so nothing
is wrongly blocked — only wrongly queued.

### The held-out probe

| Held-out scenario | Caught |
|---|---|
| `invoice_splitting` (a purchase broken under an approval ceiling) | **0 / 4** |
| `quantity_inflation` (contract unit price, units never delivered) | **0 / 6** |

Each is blind in a different way from the designed four. `invoice_splitting` is
a pattern *across* invoices and every individual document is impeccable;
`quantity_inflation` exploits the project validating what a vendor charges per
unit but never **how many units it claims to have delivered**.

---

## Statistical weight

| Experiment | n |
|---|---|
| Phase 1 ablation, mobile models | 15 documents per cell |
| Phase 1, PP-OCRv6 medium | 5 documents per cell |
| Phase 2, our corpus | 150 (50 in the VLM comparison) |
| Phase 2, FATURA2 | 300 (50 in the VLM comparison) |
| Phase 2, SROIE | 157 test, of 652 |
| Phase 3 | 100 ledger / 50 evaluation |
| Phase 4 | 100 history / 50 evaluation |
| VLM run | 100 documents |
| Tests | 543 passing, 0 skipped |

---

## The three things to take away

1. **Comparing a ceiling against an end-to-end figure can invert a
   conclusion.** For most of this project the headline was "the VLM lost on
   both corpora". Re-running the baseline through real OCR on the same
   documents turned that into "the VLM wins by 19 points on our layouts and
   loses by 21 on someone else's", and turned a 30-point generalisation gap
   into a 0.8-point one. The gap was mostly the gap between perfect text and
   real text. Nothing about either system changed — only the fairness of the
   comparison.

2. **Measuring the wrong thing is worse than measuring badly.** Phase 1's exit
   criteria failed on CER while preprocessing was delivering +22 points of the
   metric extraction actually needs. A thin sample was worse still: at n=4 the
   ranking was wrong in 4 of 6 cells.

3. **A system that catches what it was designed to catch has told you almost
   nothing.** Both held-out probes score zero, and they are kept that way.

   The data has also been wrong repeatedly, and always silently. The
   **generator** three times (random GSTIN check digits, a tax-inclusive
   `line_total`, vendor identity re-rolled per invoice), and the **converters**
   three more (`pages.0.dpi` null failing all 50 FATURA2 predictions,
   `annotator`/`currency` null skipping all 652 SROIE documents, and
   same-row line-item pairing finding 37% of receipt amounts). Not one of
   those raised an error — each produced a plausible number and was found only
   because a later stage depended on it being right.

   That is the strongest argument in this project for building the consumer
   before trusting the producer, and for the 543 tests: the failure mode here
   is never a crash.
