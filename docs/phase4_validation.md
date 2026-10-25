# Phase 4 — vendor validation

*Objective 2b, second half: "RAG-based validation against vendor records."*
With Phase 3 this completes the project's stated research contribution — the
combination the lit review identified as absent from existing invoice-extraction
work, which treats extraction accuracy as the whole problem.

**Exit criteria:** vendor resolution ≥ 95%; false-positive rate < 10% on clean
documents.

---

## Results

Vendor master accreted from 100 historical invoices covering 63 suppliers;
evaluated on 50 held-out invoices with injected fraud. Reproduce with
`.\run.ps1 phase4`.

| Criterion | Target | Result | |
|---|---|---|---|
| Vendor resolution accuracy | ≥ 0.95 | **1.000** | met |
| False-positive rate (clean, known vendor) | < 0.10 | **0.000** | met |
| Fraud recall | — | **1.000** | |
| Fraud precision | — | **1.000** | |

| Injected scenario | Caught |
|---|---|
| `redirected_bank_account` | 3 / 3 |
| `gstin_mismatch` | 4 / 4 |
| `price_gouging` | 5 / 5 |
| `duplicate_payment` | 2 / 2 |

Vendor names are **corrupted before resolution** (upper-cased, spacing removed,
`&`→`and`, truncated, word-reordered) and the GSTIN is dropped from half the
cases, because that is what arrives from the upstream OCR stage. Resolution on
clean names is trivial and would not have exposed the defects below.

### Sensitivity

| Configuration | Resolution | Precision | Recall | FP rate |
|---|---|---|---|---|
| Baseline (100 historical) | 1.000 | 1.000 | 1.000 | 0.000 |
| Different seed | 0.975 | 0.867 | **0.929** | 0.077 |
| 80% fraud rate | 0.975 | 0.964 | 1.000 | 0.077 |
| **50-invoice history** | 0.978 | **0.778** | 1.000 | **0.214** |
| 25-invoice history | 0.968 | 1.000 | 1.000 | 0.000 |
| Clean (uncorrupted) names | 1.000 | 0.900 | 1.000 | 0.032 |

Resolution never drops below 0.968. Precision and recall move around on other
seeds (0.867 and 0.929 at worst), and the **50-invoice** row is the one to read
carefully: precision 0.778 with a 0.214 false-positive rate. With half the
history, more vendors sit just above the two-invoice threshold where identity
checks switch on but the record is still thin — enough history to be checked,
not enough to be right. The 25-invoice row looks clean only because almost
every vendor falls *below* the threshold and is routed to new-supplier review
instead (92 of them).

So the honest reading is that a single configuration hitting 1.000 is not the
result; the result is that the system is reliable with adequate history and
degrades in a specific, explicable way without it.

---

## The corpus could not support this phase as generated

Phase 4 validates an invoice against a stable supplier record. Three properties
have to hold for that to mean anything, and **none of them did**:

| Property | Before | After |
|---|---|---|
| A vendor keeps one GSTIN across invoices | **0 / 45** | 45 / 45 |
| A vendor keeps one bank account | **0 / 45** | 45 / 45 |
| A vendor's price for an item is stable | **57% spread** | 4.5% spread |

The generator re-rolled vendor identity on every invoice, so "Lakshmi
Enterprises" carried four different GSTINs and four different bank accounts
across its four invoices. Bank-account-change detection — the single
highest-value check in the pipeline — would have fired on **every invoice ever
issued**. Prices were drawn uniformly per invoice, so no contract rate existed
to compare against.

Fixing it meant making identity a property of the company rather than of the
document: a `_PARTY_REGISTRY` mints each supplier once and reuses it, and
`vendor_price` holds a per-(vendor, item) base price with ±4% jitter, which is
how a real quoted price behaves.

This is the same class of problem as the Phase 2 `line_total` convention and
the Phase 0 GSTIN checksums: the data model was never exercised hard enough to
reveal it. It only surfaced because a downstream phase actually depended on it.

---

## How it works

### Vendor master

A supplier file is *accreted from past accepted invoices*, which is how real
accounts systems build one — nobody types it in. Each record holds the GSTINs,
addresses, phones, emails, bank accounts and IFSC codes seen, the invoice
count, the amount history, and per-item contract rates.

**The master is only as trustworthy as the history behind it**, so every record
carries `invoice_count` and the identity checks refuse to fire below
`MIN_HISTORY_FOR_IDENTITY_CHECK`. A vendor with one prior invoice has a "known"
bank account in the sense that one coin flip establishes a bias. Alerting anyway
produces the failure that kills these systems in practice: an alert on every
new supplier, which trains the reviewer to click through alerts.

### Resolution, then validation

Two separate steps. Resolution answers *who sent this*, and every downstream
check is conditional on it — validating against the wrong supplier produces
confident nonsense. The honest failure mode is "unknown vendor", not the
closest available name.

Order: exact GSTIN → exact name → fuzzy name (≥ 88 similarity) → unresolved.

### Verdicts are three-valued

`PASS` / `REVIEW` / `FAIL`. A two-valued verdict forces every soft signal into
one bucket or the other, and most real findings are genuinely "ask someone".
Collapsing them into `FAIL` produces an alert stream nobody reads — which is
the practical failure mode of these systems, not missed fraud.

The **worst** individual check decides the document verdict. Averaging would
let one `FAIL` be diluted by a handful of passes, which is the opposite of what
a control is for.

### The checks

| Check | Verdict | Why |
|---|---|---|
| `bank_account_changed` | FAIL | Invoice-redirection fraud changes nothing on the page except the account number |
| `gstin_mismatch` | FAIL | Name and tax number point at different suppliers |
| `possible_duplicate_payment` | FAIL | Same vendor, same amount, within 45 days |
| `price_deviation` | REVIEW / FAIL | Against the vendor's own rate, or the pooled market rate |
| `address_mismatch` | REVIEW | Addresses are reformatted constantly; only a large divergence is worth attention |
| `unknown_vendor` | REVIEW | No record to validate against |

**Price checks use a two-tier reference.** Most (vendor, item) pairs appear
once in a real ledger, so a per-vendor contract rate alone leaves nearly every
line unchecked — it caught 0 of 3 injected price-gouging cases. Falling back to
a *pooled market rate* across all suppliers fixed that (3/3), with a much wider
tolerance (60%/150% versus 25%/100%), because suppliers legitimately differ on
price while a single supplier's own quote barely moves.

---

## Three bugs found only by measuring

**1. rapidfuzz is case-sensitive by default.** `"KRISHNA TEXTILES"` scored
**18.75** against `"Krishna Textiles"` and failed to resolve. OCR produces
upper-case names constantly; this cost roughly a third of all resolutions.
Resolution went from 0.703 to 0.909 once names were normalised.

**2. Stripping legal suffixes merged distinct companies.** Normalising
`"Arjun & Co"` and `"Arjun Pvt Ltd"` both to `"arjun"` looked tidy and was
wrong — they are different suppliers. A clean invoice resolved to the wrong
record and then mismatched on bank account, GSTIN *and* address simultaneously,
turning a correct invoice into a confident `FAIL`. The suffix is often the only
thing distinguishing two real suppliers. Resolution went to **1.000**.

**3. GSTIN-first resolution hid GSTIN fraud entirely.** An invoice carrying
another supplier's GSTIN resolved to *that* supplier, whose record then matched
perfectly — so the mismatch could never be detected. It scored **0/3** on
injected swaps. Now, when the name and the GSTIN disagree, the resolver returns
a `conflict` and the validator raises `gstin_mismatch` rather than silently
picking a side. 2/2.

None of these would have surfaced from reading the code, and none crashed
anything.

---

## A limitation worth stating plainly

**An impostor vendor is indistinguishable from a legitimate new supplier.**

A shell company submitting its first invoice looks exactly like a real supplier
submitting theirs: no record, no history, nothing to compare against. The system
raises `unknown_vendor` for both, correctly.

This matters for how the numbers are read. An earlier version of this evaluation
counted `impostor_vendor` as a detected fraud and scored 6/6 — which is
circular, since it is the same signal, and would have reported 100% detection
while establishing nothing. It is now excluded from the scored fraud set and
reported separately as an **onboarding outcome**.

Likewise, the false-positive rate is computed over clean invoices **from known
vendors only**. Flagging a genuinely new supplier is correct behaviour, not an
error, and including those would make the metric measure how many first-time
vendors the split happens to contain (15 of 38 here) rather than control
quality.

Distinguishing the two requires an onboarding process outside this pipeline:
verifying the supplier exists, that someone authorised them, and that the bank
details came through a channel other than the invoice itself.

## What this does not establish

- **Nothing about real fraud.** Four scenarios, injected by us. It says the
  system catches what it was built to catch, not that it catches what an
  attacker would actually try.
- **63 suppliers, all synthetic.** Real vendor masters have name collisions,
  mergers, subsidiaries sharing a GSTIN, and genuine bank-detail changes with
  paperwork behind them. None of that is here.
- **No vendor-aliasing.** A real master carries known trading names; this
  matches on similarity alone.
- **Duplicate detection is O(n) over history** per invoice. Fine at this scale,
  needs an index at production size.
- **`address_mismatch` is never evaluated**, because the generator gives each
  supplier exactly one address. Its threshold is therefore unvalidated.

## Running it

```powershell
.\run.ps1 phase4
```

```bash
python scripts/run_validation_eval.py --history-size 100 --fraud-rate 0.4
python scripts/run_validation_eval.py --clean-names   # resolution without OCR damage
```
