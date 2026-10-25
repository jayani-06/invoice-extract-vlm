# Phase 3 — RAG enrichment

*Objective 2b: "a RAG-based enrichment layer for auto-categorisation and
anomaly flagging."* This is the first half of the project's stated research
contribution — the part the lit review identified as missing from existing
invoice-extraction work.

**Exit criteria:** categorisation macro-F1 ≥ 0.85; anomaly recall ≥ 0.90 at
precision ≥ 0.70.

---

## Results

100 historical invoices indexed, 50 held out for evaluation. Reproduce with
`.\run.ps1 phase3`.

| Criterion | Target | Result | |
|---|---|---|---|
| Categorisation macro-F1 | ≥ 0.85 | **1.000** | met |
| Anomaly recall | ≥ 0.90 | **1.000** | met |
| Anomaly precision | ≥ 0.70 | **1.000** | met |
| False-positive rate on clean documents | — | **0.000** | |

**Both exit criteria are met, and the numbers are too good to take at face
value.** The section below says why, and what the sensitivity sweep shows.

### Categorisation

| Metric | Value |
|---|---|
| Accuracy (all line items) | 1.000 |
| Coverage (share it answered rather than abstained) | 1.000 |
| Macro F1 | 1.000 |
| Latency | ~0.5 ms/item |

### Anomaly detection

| Injected anomaly | Caught |
|---|---|
| `inflated_total` | 5 / 5 |
| `duplicate_invoice_number` | 2 / 2 |
| `invalid_gst_rate` | 1 / 1 |
| `corrupt_gstin` | 1 / 1 |

---

## Why perfect scores are a warning, not a win

Phase 2 established the habit: a baseline that scores 98% on our own corpus and
56% on a third-party one was measuring the corpus, not the method. The same
scepticism applies here, and there are two concrete reasons these numbers
flatter the system.

**The label space is tiny.** The synthetic generator emits ten item
descriptions across three accounts. Real charts of accounts have hundreds of
codes, and the hard cases are the near-neighbours — "software subscription"
versus "IT equipment" — which this corpus does not contain at all.

**The anomalies are the ones we injected.** Recall is measured against five
anomaly types we chose and wrote ourselves. It says the detector catches what
it was built to catch. It says nothing about the anomaly nobody thought of,
which is the one that matters in production.

### What the sensitivity sweep does establish

Shrinking the ledger degrades categorisation in the way a genuine retrieval
system should, which is evidence it is retrieving rather than memorising:

| Configuration | Cat. accuracy | Coverage | Anomaly P | Anomaly R |
|---|---|---|---|---|
| 100-document ledger (baseline) | 1.000 | 1.000 | 1.000 | 1.000 |
| 100-document, 60% anomaly rate | 1.000 | 1.000 | 1.000 | 1.000 |
| 100-document, different seed | 1.000 | 1.000 | 0.947 | 1.000 |
| **40-document ledger** | **0.977** | 0.981 | 0.978 | 1.000 |
| **15-document ledger** | **0.934** | 0.987 | 1.000 | 1.000 |

Less history, worse categorisation — a smooth curve, not a cliff. Anomaly
precision stays ≥ 0.947 and recall at 1.000 throughout.

---

## How it works

### The ledger

Phase 3 needs something Phase 0 never produced: a *history*. Retrieval-based
categorisation is meaningless without past invoices to retrieve, and the
statistical signal needs a per-vendor amount distribution. So
`scripts/build_ledger.py` splits the 150 synthetic documents into 100
historical (indexed, categorised) and 50 evaluation.

**The evaluation set's descriptions are deliberately rewritten.** With only ten
canonical descriptions, an unmodified evaluation set would make retrieval exact
string matching — 100% accuracy that proves nothing. Real vendors write the
same item differently every time, so the builder reproduces that:

| Canonical | Rewritten as |
|---|---|
| `Printer Toner Cartridge` | `TONER CART` |
| `Steel Filing Cabinet` | `STL FILE CAB` |
| `Laptop Bag` | `Bag Laptop` / `laptop-bag` |
| `Office Chair` | `0ffice Chair` *(OCR-style confusion)* |
| `A4 Copier Paper (500 sheets)` | `A4 COPIER PPR (500 SHEETS)` |

125 of 139 evaluation line items carry a rewritten description. That gap is
what the categoriser has to bridge.

### Retrieval

**TF-IDF over character 3–5 grams**, cosine similarity, exact nearest-neighbour
search. The Phase 3 plan called for dense embeddings in a vector store; this
ships the cheap method first for the same reason Phase 2 shipped a regex
baseline before the VLM — a method that runs today establishes the floor a
model has to clear. `Retriever` is a Protocol, so dense embeddings are a new
implementation rather than a rewrite.

Character n-grams rather than word tokens specifically because the variation to
survive is *surface*: `Steel Filing Cabinet` and `STL FILE CAB` share almost no
word tokens and a great many character grams.

### Categorisation

Retrieve the k=5 nearest historical items, weight their votes by similarity,
and take the winner — but only if it clears two thresholds:

- **≥ 50% of the weighted vote**, or the neighbours disagree too much
- **top neighbour ≥ 0.35 similarity**, because five weak neighbours agreeing
  are still five weak neighbours

Otherwise it returns `UNCATEGORISED` with the evidence attached.

**Abstention is a feature, and it is reported separately from accuracy.** A
system that declines 10% of lines and is right on the rest is more useful to an
accounts team than one that guesses on everything — a wrong account code
propagates into the ledger silently, while an abstention lands in a review
queue and is cleared in seconds. Collapsing the two into one accuracy figure
hides exactly that trade-off, so coverage and accuracy-on-covered are both
reported.

Every decision carries the neighbours that produced it:

> Office Supplies [5010] at 94% — nearest past items: `'A4 Copier Paper (500
> sheets)'` (0.71), `'Printer Toner Cartridge'` (0.44)

That explainability is the stated advantage of RAG over a fine-tuned classifier
in ref [8], and it is only real if the evidence survives into the output.

### Anomaly detection

Three signal families, kept **separate rather than fused into one risk score**,
because they fail differently and a reviewer responds to them differently. A
duplicate invoice number is near-certain double-payment; an unusual amount is a
question. Averaging those into "risk: 0.62" discards the only information that
makes the flag actionable.

| Family | Signals |
|---|---|
| **Rule** | future invoice date, due-before-invoice, duplicate invoice number, non-statutory GST rate, invalid GSTIN checksum, arithmetic mismatch |
| **Statistical** | amount z-score against vendor history, suspiciously round total |
| **Semantic** | unknown vendor, line item unlike anything previously purchased |

Severity is coarse — `high` / `medium` / `low`. A continuous risk score invites
a threshold nobody can justify, and this corpus cannot calibrate one honestly.

---

## Two bugs found by measuring

**1. `unknown_vendor` was rated medium, and it destroyed precision.**

First run: precision 0.412, with a 24% false-positive rate on clean documents.
Every false positive was the same flag. Businesses onboard suppliers
constantly, so a first invoice from a new vendor is routine — it is *context*,
not an accusation. Downgrading it to `low` took precision from **0.412 to
1.000** and the false-positive rate to zero.

It still earns attention in combination: `escalate_combinations` raises a
medium flag when an unknown vendor co-occurs with a suspiciously round total or
an unusual amount, which is the actual shape of invoice fraud.

**2. Duplicate detection could never fire.**

It scored 0/2 on injected duplicates. The history was keyed on *normalised*
invoice numbers (`INV-2026-0001` → `INV20260001`) but looked up with the raw
string. Nothing errored; the signal was simply dead. Recall went from 0.778 to
1.000 once the lookup normalised too. A test now pins the punctuation-variant
case directly.

Neither bug would have surfaced from reading the code.

---

## What this does not establish

- **Nothing about real ledgers.** Three accounts, ten item types, 56 vendors,
  all synthetic. A real chart of accounts is hundreds of codes deep and the
  confusable pairs are the whole difficulty.
- **Nothing about unanticipated anomalies.** Recall is against five types we
  injected ourselves.
- **Nothing about scale.** 273 ledger entries. Exact nearest-neighbour search
  is fine here and will need an approximate index at production size.
- **Dense retrieval is untested.** The interface is ready; TF-IDF is what has
  actually been measured. Whether embeddings beat character n-grams on this
  task is an open question, and on evidence so far the baseline is hard to
  beat because it is already at ceiling.
- **No LLM is involved.** The plan describes an LLM assigning the category from
  retrieved context. This retrieves and votes. Adding generation is the obvious
  next step, and the current numbers leave it nothing to improve on *this*
  corpus — which is itself an argument for a harder corpus before a bigger
  model.

## Running it

```powershell
.\run.ps1 phase3              # build the ledger, then evaluate
.\run.ps1 phase3-ledger       # rebuild the split only
.\run.ps1 phase3-eval         # re-score an existing split
```

Underlying commands:

```bash
python scripts/build_ledger.py --ledger-size 100 --anomaly-rate 0.2
python scripts/run_rag_eval.py --k 5
```
