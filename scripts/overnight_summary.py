#!/usr/bin/env python
"""Collect whatever the overnight run produced into one page.

Reads only from reports/. Anything a run did not produce is reported as
**not measured** rather than omitted, because a missing row that silently
disappears reads as a result that was never needed.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def f(value, places: int = 3) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float) and value != value:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}"
    return str(value)


def header_accuracy(run_dir: Path) -> dict | None:
    """Score a run directory the way summarize_run.py does."""
    gold_path, pred_path = run_dir / "gold.jsonl", run_dir / "pred.jsonl"
    if not gold_path.exists() or not pred_path.exists():
        return None
    sys.path.insert(0, str(ROOT / "src"))
    from invoice_extract.eval.metrics import aggregate, compare_documents

    fields = [
        "parties.vendor.name",
        "parties.vendor.gstin",
        "invoice_meta.invoice_number",
        "invoice_meta.invoice_date",
        "totals.grand_total",
        "totals.subtotal",
    ]

    def read(p: Path) -> dict:
        return {
            json.loads(ln)["doc_id"]: json.loads(ln)
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    gold, pred = read(gold_path), read(pred_path)
    empty = {"doc_id": "", "line_items": [], "tax_lines": []}
    agg = aggregate([compare_documents(gold[d], pred.get(d, empty)) for d in gold])
    scores = [
        agg.per_field[x].accuracy
        for x in fields
        if x in agg.per_field and agg.per_field[x].accuracy == agg.per_field[x].accuracy
    ]
    meta = load(run_dir / "run.json") or {}
    return {
        "n": agg.n_docs,
        "header": sum(scores) / len(scores) if scores else float("nan"),
        "critical": agg.doc_critical_accuracy,
        "macro": agg.macro_field_accuracy,
        "ceiling": meta.get("is_ocr_ceiling"),
    }


out: list[str] = []
w = out.append

w("# Overnight run")
w("")
w(f"Finished {datetime.now():%Y-%m-%d %H:%M}.")
w("")
w("Every number below comes from a file in `reports/`. Anything a run did not")
w("produce says **not measured** rather than being left out.")
w("")

# ----------------------------------------------------------------- corpus
canonical = ROOT / "data/processed/gst_in_synthetic/canonical"
n_docs = len(list(canonical.glob("*.json"))) if canonical.exists() else 0
w("## Corpus")
w("")
w(f"- `gst_in_synthetic`: **{n_docs} invoices** (was 150)")
sroie = ROOT / "data/processed/sroie/canonical"
w(
    f"- `sroie`: {len(list(sroie.glob('*.json'))) if sroie.exists() else 0} real photographed receipts"
)
w("")

# -------------------------------------------------------------- extraction
w("## Phase 2 — extraction")
w("")
w("| Run | n | Header acc | Critical | Macro | Measurement |")
w("|---|---|---|---|---|---|")
for label, rel in [
    ("rules, ceiling (ours)", "phase2/rules_vlmset_gst_in_synthetic"),
    ("rules, end-to-end (ours)", "phase2/rules_e2e_gst_in_synthetic"),
    ("vlm, end-to-end (ours)", "phase2/vlm_gst_in_synthetic"),
    ("rules, ceiling (FATURA2)", "phase2/rules_vlmset_fatura2"),
    ("rules, end-to-end (FATURA2)", "phase2/rules_e2e_fatura2"),
    ("vlm, end-to-end (FATURA2)", "phase2/vlm_fatura2"),
    ("rules, ceiling (SROIE)", "phase2/rules_sroie_test_after"),
    ("**rules, END-TO-END (SROIE, real photos)**", "phase2/rules_sroie_e2e"),
]:
    r = header_accuracy(REPORTS / rel)
    if r is None:
        w(f"| {label} | — | **not measured** | — | — | — |")
        continue
    kind = "perfect-OCR ceiling" if r["ceiling"] else "end-to-end"
    w(
        f"| {label} | {r['n']} | {f(r['header'])} | {f(r['critical'])} "
        f"| {f(r['macro'])} | {kind} |"
    )
w("")

# ------------------------------------------------------------------ router
w("## Router")
w("")
w("| Corpus | rules | vlm | router |")
w("|---|---|---|---|")
for corpus, rel in [
    ("gst_in_synthetic", "phase2/router_gst"),
    ("fatura2", "phase2/router_fatura2"),
]:
    data = load(REPORTS / rel / "results.json")
    if not data:
        w(f"| {corpus} | — | — | **not measured** |")
        continue
    res = data["results"]
    w(
        f"| {corpus} (n={data['held_out_documents']} held out) "
        f"| {f(res['rules']['header_accuracy'])} | {f(res['vlm']['header_accuracy'])} "
        f"| **{f(res['router']['header_accuracy'])}** |"
    )
w("")

# ----------------------------------------------------------------- phase 3
w("## Phase 3 — categorisation and anomalies")
w("")
w("| Path | Accuracy | Macro F1 | Coverage | Anomaly P | R designed-for | R incl. probe |")
w("|---|---|---|---|---|---|---|")
for label, rel in [
    ("weighted vote (control)", "phase3_rag"),
    ("**LLM generation (RAG)**", "phase3_generated"),
]:
    data = load(REPORTS / rel / "results.json")
    if not data:
        w(f"| {label} | **not measured** | — | — | — | — | — |")
        continue
    # The runner writes "categorization" (US). Reading the British
    # spelling printed "not measured" over numbers that were right there,
    # which is the summary lying about its own inputs.
    cat = data.get("categorization") or data.get("categorisation") or {}
    an = data.get("anomaly") or {}
    w(
        f"| {label} | {f(cat.get('accuracy'))} | {f(cat.get('macro_f1'))} "
        f"| {f(cat.get('coverage'))} | {f(an.get('precision'))} "
        f"| {f(an.get('recall_designed_for_only', an.get('recall')))} "
        f"| {f(an.get('recall_including_held_out'))} |"
    )
w("")
probe = (load(REPORTS / "phase3_rag/results.json") or {}).get("anomaly", {}).get("held_out")
if probe:
    w(
        f"Held-out probe: **{probe['detected']}/{probe['injected']}** "
        f"({', '.join(probe['kinds'])}) — deliberately unfixed."
    )
    w("")

# ----------------------------------------------------------------- phase 4
w("## Phase 4 — vendor validation")
w("")
data = load(REPORTS / "phase4_validation/results.json")
if not data:
    w("**not measured**")
else:
    v, res = data["validation"], data["resolution"]
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Resolution accuracy | {f(res['accuracy'])} |")
    w(f"| Precision | {f(v['precision'])} |")
    w(f"| Recall, designed-for | {f(v.get('recall_designed_for_only', v.get('recall')))} |")
    w(f"| Recall, incl. held-out probe | {f(v.get('recall_including_held_out'))} |")
    w(f"| False-positive rate on clean | {f(v['false_positive_rate_on_clean'])} |")
    held = v.get("held_out")
    if held:
        w(f"| Held-out probe | **{held['detected']}/{held['injected']}** (kept unfixed) |")
w("")

# ------------------------------------------------------------- before/after
old4 = load(REPORTS / "_corpus150/phase4_validation/results.json")
if old4 and data:
    w("### Before / after the larger, harder corpus")
    w("")
    w("| Metric | 150 docs | current |")
    w("|---|---|---|")
    a, b = old4["validation"], data["validation"]
    for label, key in [
        ("precision", "precision"),
        ("FP rate on clean", "false_positive_rate_on_clean"),
    ]:
        w(f"| {label} | {f(a.get(key))} | {f(b.get(key))} |")
    w("")

w("---")
w("")
w("## What to check first")
w("")
w("1. The **SROIE end-to-end** row above. That is the only measurement of this")
w("   pipeline on real photographs with no perfect-text ceiling propping it up,")
w("   and it should lead any claim about real-world performance.")
w("2. Whether the **LLM generation** row beats the **weighted vote** row. If it")
w("   does not, the generation step has added a model and bought nothing, and")
w("   that is the honest finding to report.")
w("3. `reports/_overnight/log.txt` for anything that failed.")

print("\n".join(out))
