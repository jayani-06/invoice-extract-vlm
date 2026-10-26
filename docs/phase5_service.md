# Phase 5 — service and human review

*Makes the pipeline callable, and makes its abstentions and flags actionable by
a person.* Phases 1–4 ran only as scripts. Several modules were already written
for this phase: `postprocess.py` keeps arithmetic problems as "flags for review,
not corrections … Phase 5's review UI surfaces them", `categorize.py` abstains
so that "Phase 5's reviewer can clear an abstention in seconds", and
`test_pipeline_end_to_end.py` asserts the whole result round-trips through JSON
"because Phase 5 will serve this over HTTP".

**Exit criteria:**

1. A single POST with an invoice image returns a complete result in under 10 s
   on CPU with fast models, *or* the job endpoint returns a handle immediately.
2. Every field in the response validates against the canonical schema.
3. A document that reaches FAIL is not payable without a recorded human
   override, and the override is attributable.
4. Reviewer corrections change the vendor master, demonstrably, with a test.

---

## Results

Windows 11, Python 3.11.9, PaddleOCR 3.7 / paddlepaddle 3.3.1 on CPU, PP-OCRv5
mobile models (`fast=True`), `full(no-binarize)` preprocessing. Measured with
`python scripts/time_service.py`: the real service (`create_app` + FastAPI
`TestClient`), real OCR, held-out corpus pages, knowledge bootstrapped from the
Phase 3 ledger.

| Page | Profile | Wall clock | Preprocess | OCR | Extract | Enrich | Validate | Verdict |
|---|---|---|---|---|---|---|---|---|
| gst_synth_0004 | clean | 52.1 s | 0.93 s | 50.8 s | 31 ms | 62 ms | 3 ms | REVIEW |
| gst_synth_0004 | medium | 50.7 s | 0.91 s | 49.5 s | 7 ms | 42 ms | 12 ms | REVIEW |
| gst_synth_0005 | clean | 47.4 s | 0.90 s | 46.2 s | 13 ms | 46 ms | 5 ms | REVIEW |
| gst_synth_0005 | medium | 46.6 s | 0.89 s | 45.4 s | 12 ms | 47 ms | 5 ms | REVIEW |
| gst_synth_0009 | clean | 40.6 s | 0.90 s | 39.5 s | 11 ms | 37 ms | 4 ms | REVIEW |
| gst_synth_0009 | medium | 42.6 s | 0.98 s | 41.1 s | 11 ms | 46 ms | 9 ms | REVIEW |
| `POST /jobs` (same page as row 1) | | **0.066 s to the handle**; result after 51.2 s | | | | | | |

The OCR engine itself takes a further 31.1 s to load, once, on the first request.
Everything after OCR -- extraction, categorisation, anomaly flags, vendor
validation -- is under 0.1 s per page.

| Criterion | Result | |
|---|---|---|
| 1. < 10 s synchronous, **or** a job handle immediately | Synchronous: 40.6–52.1 s per page, 97–98% of it OCR; `POST /jobs` returned its handle in 0.066 s | **met, by the second branch only** |
| 2. Every response field validates against the canonical schema | `test_every_returned_document_validates_against_the_canonical_schema` checks the served document with the JSON Schema itself, not only the pydantic mirror | **met** — after a defect it found, below |
| 3. FAIL not payable without an attributable override | `accept` on a FAIL is refused (409); `override` requires a reason (422 without); the override decision stores reviewer, reason, previous status and timestamp; 30 tampered documents all route to `blocked` | **met** |
| 4. Corrections change the vendor master | `test_a_corrected_vendor_name_creates_the_supplier_record`: an OCR-garbled first-time supplier is corrected by a reviewer, the master gains the *corrected* name with that invoice's GSTIN and bank account, and the supplier's next invoice resolves to it instead of going to review | **met** |

Test suite: `tests/test_service.py`, 24 tests, self-contained (documents come
from the generator and are rendered in-process; no corpus or OCR engine
needed). Whole suite on Windows: **452 passed, 2 skipped (torch absent; no VLM results merged), 1 deselected (`slow`)**; ruff and black clean.

### Criterion 1 is met only through the job endpoint

Stated plainly: **the synchronous endpoint does not return in under 10 s with
real OCR.** OCR alone is ~40 s per page on this CPU (Phase 1), and everything
after OCR — extraction, categorisation, anomaly flags, vendor validation —
takes well under a second. No service design changes that; only a faster
engine, a GPU, or a smaller page does. The criterion's second branch exists for
exactly this case, and `POST /jobs` meets it. `POST /documents` is kept for
fast engines and tests, and every response carries `timings_ms` per stage so
the cost is visible rather than assumed.

### With real OCR, every page went to review

All six real-OCR pages routed to `REVIEW`; none was auto-accepted and none
falsely reached `FAIL`. The reasons are OCR errors surfacing, not invoice
problems:

- `Line 1 at 7.00 is -100% against the market 6,472.15` — the quantity column
  was read into the unit-price field, and Phase 4's price check caught it.
- `Totals do not reconcile: line_items[0]: quantity*unit_price-discount=270.45 != line_total=893.15`
  — the same misread, caught by the arithmetic check.
- `No supplier record matches 'TAXINVOICE Anand Textiles'` — on a degraded
  page the title merged into the vendor-name line.

That is the system working as designed — an extraction error lands in the queue
with its evidence instead of being paid — but it also says the review queue, not
the auto-accept path, is what a real deployment of the rules extractor would
mostly exercise. With gold text instead of OCR, 15 of 20 held-out documents
auto-accept (the 5 others are genuinely first-time suppliers). Six pages are a
small sample; the direction is clear, the rate is not.

---

## What the service returns

`POST /documents` (or the `result` of a finished job) returns one object:

| Field | From | Notes |
|---|---|---|
| `document` | Phase 2 | **is** `data.models.CanonicalInvoiceDocument` — no second schema |
| `consistency_problems` | Phase 2 `finalize` | arithmetic findings, never corrections |
| `categories[]` | Phase 3 | per line: category or abstention, confidence, votes, the retrieved neighbours, and `explanation` |
| `anomalies` | Phase 3 | flags with signal family and severity |
| `validation` | Phase 4 | resolution (method, candidates), every check with evidence, and the `explain()` text |
| `overall_verdict`, `status`, `payable`, `review_reasons` | Phase 5 routing | see below |
| `ocr`, `timings_ms` | Phase 1 / service | engine, word count, mean confidence, the text read; per-stage milliseconds |

The response models in `service/schemas.py` mirror the `as_dict()` methods of
the Phase 3/4 dataclasses field for field and forbid extra keys, and the
service validates every outgoing result through them *before* storing it — so a
key renamed in a stage fails at the first document, not when someone next
reads the API.

## Routing

| Overall verdict | When | Status | Payable |
|---|---|---|---|
| `FAIL` | Phase 4 returned FAIL | `blocked` | only via `override` |
| `REVIEW` | Phase 4 REVIEW, **or** any line abstained, **or** a high/medium anomaly flag | `in_review` | after `accept` |
| `PASS` | none of the above | `auto_accepted` | yes |

Anomaly flags never escalate to FAIL on their own. Phase 3 measures how
surprising a document is; Phase 4 measures whether it is payable. Mixing the two
scales here would be a policy neither phase evaluated.

`payable` is derived from `status` (`auto_accepted`, `accepted`,
`accepted_with_override`), never stored separately, so the two cannot disagree.

## The review queue

`GET /review/queue` returns open items, each self-contained — a reviewer needs
no second request to decide it:

- **`document` items** (every REVIEW or FAIL document): the page image URL, the
  extracted vendor / meta / totals / payment block, the Phase 4 checks with
  evidence, the anomaly flags, and the `explain()` text.
- **`line_category` items** (every abstained line): the page image URL, the
  extracted line item, the abstention reason, and the retrieved neighbours with
  their categories, vendors and similarity scores.

`GET /review` serves a single-file HTML page over the same API.

### Decision rules

Enforced in `service/review.py` and pinned by tests:

1. **A FAIL can never be accepted by `accept`** (409). The only path to payable
   is `override`, which requires a non-empty reason (422 otherwise) and stores
   the reviewer's name, the reason, the previous status and a timestamp.
2. **A document cannot be accepted while an abstained line is uncategorised**
   (409). An unresolved abstention is an unresolved question.
3. **"Accept" on an abstained line is refused** (409) — there is no category to
   accept; the reviewer must choose one.
4. **Every decision needs a named reviewer** (422 for empty or whitespace).
5. **Decisions are final.** A second decision on a decided document is a 409,
   not an edit.
6. **Nothing is silently repaired.** The machine's extraction is stored as
   extracted. A reviewer's correction produces a separate accepted copy, and
   each difference is its own decision row with old value, new value and
   reviewer. `test_the_service_does_not_repair_the_document` posts an invoice
   with a broken total and asserts it comes back broken, flagged, and not
   auto-accepted.

### Persistence

SQLite via the standard library (`data/service/review.sqlite3`, gitignored —
real invoices may land there). Three tables: documents, review items, and an
**append-only decision log**. Corrected vendor names and categories are the only
non-synthetic labels this project will ever collect, so they are never updated
in place. On restart the service replays every accepted document into the
knowledge base (`test_corrections_survive_a_restart`).

## Feedback into the vendor master and ledger

Every accepted document — auto-accepted or human-accepted, with the reviewer's
corrections applied — is appended to the history the vendor master is built
from, and its categorised lines to the ledger the categoriser retrieves
against. Indices are rebuilt from the full history on each acceptance rather
than patched: at hundreds of rows a rebuild is milliseconds, and incremental
medians and TF-IDF weights would be a second implementation of the same maths.

Rejected and blocked documents are never learned from: a redirected bank
account must not enter the master (`test_a_redirected_bank_account_is_blocked`).
A line that is still uncategorised is not added to the ledger — an abstention
nobody resolved is not a label.

## Two things found only by building it

### Serving the canonical model through FastAPI made it schema-invalid

The first contract test failed. The extracted document validated against
`schema/invoice_schema.json` when it left the pipeline, but the copy FastAPI
served did not: `annotator: None is not of type 'object'`, and four more.
Serialising through the pydantic model emits every optional field at its default,
and the JSON Schema does not accept `null` for those objects. The pydantic
mirror and the JSON Schema disagree about absent-versus-null.

Fixed with `response_model_exclude_unset=True` on every endpoint that returns a
document, so the document is served exactly as extracted. The test checks the
served document with the JSON Schema directly; a test against the pydantic
model alone would have passed.

### One reviewer correction does not teach the categoriser

A reviewer categorised an abstained line ("Floor Polisher" → Facilities). The
next invoice carrying the same item still abstained. The corrected entry *was*
retrieved — as the top neighbour, similarity ≥ 0.8 — but was outvoted by four
weak neighbours at 0.26–0.34 that are similar mainly because the **vendor name
is part of the indexed text** (Phase 3's design, so that vendor context informs
category). Vote share: 47%, under the 50% threshold.

It took **3 reviewed corrections** before the fourth invoice was categorised
without help (`test_an_abstained_line_blocks_acceptance_until_it_is_categorised`
measures and pins this). That is not wrong — the categoriser abstains when its
evidence disagrees, as designed — but it means the feedback loop teaches slowly,
and a reviewer will see the same item in the queue several times. A fix belongs
in Phase 3 (e.g. weighting a near-exact match above the vote, or a similarity
floor relative to the top neighbour). It was not made here: it is a change to
Phase 3's categoriser, would move Phase 3's headline numbers, and needs
re-measuring there rather than slipping in with a service.

## What this does not establish

- **No authentication or authorisation.** The reviewer name on a decision is
  self-asserted: "attributable" here means *recorded*, not *verified*. Anyone
  who can reach the port can override a FAIL under any name. The service binds
  to 127.0.0.1 by default for that reason. Out of scope by instruction; see
  `docs/limitations.md`.
- **Sub-10 s synchronous processing on CPU.** Not achieved and not achievable
  with this OCR engine on this hardware; see above.
- **Throughput or concurrency.** One OCR worker, one process, in-memory job
  registry (lost on restart — the documents themselves are persisted). Nothing
  was load-tested.
- **That reviewers make good decisions.** The queue is built from the evidence a
  reviewer needs; whether it is enough, or whether people rubber-stamp it, needs
  real reviewers.
- **That accreted knowledge improves accuracy.** Tested mechanically (corrections
  reach the master and the ledger and change later results). Not measured as an
  accuracy gain, and accretion from auto-accepted documents means a fraud that
  passes Phase 4 would be learned as normal.
- **Multi-page documents.** One image per document.
- **The VLM path.** The service uses the rules extractor; the VLM has still not
  been run on this machine.
- **Every document is still synthetic.** The service has only ever processed
  generated invoices.

## Running it

```powershell
python -m pip install -e ".[ocr,paddle,service]"
.\run.ps1 serve          # http://127.0.0.1:8000/review  and  /docs
.\run.ps1 service-test
python scripts/time_service.py --pages 6   # exit criterion 1, real OCR
```

```bash
make service-install && make serve
curl -F file=@data/processed/gst_in_synthetic/images/medium/gst_synth_0031.png \
     http://127.0.0.1:8000/jobs
```
