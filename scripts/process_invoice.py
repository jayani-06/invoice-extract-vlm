#!/usr/bin/env python
"""Run one invoice image through the whole pipeline and print the verdict.

The single entry point. Every other script here measures something; this one
just *uses* the system, which is what you want when the question is "does it
work" rather than "how well does it score".

    image -> triage -> preprocess -> OCR -> reading order -> extract
          -> coerce + validate -> categorise -> anomaly flags
          -> vendor resolution + record validation -> PASS / REVIEW / FAIL

Everything is optional except the image. With no ledger it skips
categorisation and anomaly detection; with no history it skips vendor
validation. What ran and what did not is stated in the output, because a
missing stage reported as a clean result is the failure mode this project
cares most about.

Usage:
    python scripts/process_invoice.py --image path/to/invoice.png

    # the full thing, against the synthetic corpus's own ledger and history
    python scripts/process_invoice.py \
        --image data/processed/gst_in_synthetic/images/medium/gst_synth_0105.png \
        --ledger data/processed/ledger \
        --history data/processed/gst_in_synthetic

    python scripts/process_invoice.py --image x.png --json      # machine-readable
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.validate import validate_jsonschema  # noqa: E402
from invoice_extract.models.postprocess import finalize  # noqa: E402
from invoice_extract.models.registry import get_extractor  # noqa: E402
from invoice_extract.reading_order import reading_order_text  # noqa: E402

GREEN, YELLOW, RED, DIM, BOLD, OFF = (
    "\033[32m",
    "\033[33m",
    "\033[31m",
    "\033[2m",
    "\033[1m",
    "\033[0m",
)
VERDICT_COLOUR = {"PASS": GREEN, "REVIEW": YELLOW, "FAIL": RED}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--ocr-engine", default="paddle")
    ap.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="PP-OCRv5 mobile models, ~40s/page. --accurate uses v6 medium (~100s).",
    )
    ap.add_argument("--accurate", dest="fast", action="store_false")
    ap.add_argument("--preprocess", default="adaptive", help="Pipeline config, or 'adaptive'")
    ap.add_argument("--ledger", type=Path, default=None, help="Enables Phase 3")
    ap.add_argument("--history", type=Path, default=None, help="Corpus whose canonical/ is history")
    ap.add_argument("--history-size", type=int, default=100)
    ap.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense"])
    ap.add_argument("--json", action="store_true", help="Emit JSON only")
    args = ap.parse_args(argv)

    if not args.image.exists():
        print(f"No image at {args.image}", file=sys.stderr)
        return 1

    def log(*parts) -> None:
        if not args.json:
            print(*parts, file=sys.stderr)

    import numpy as np
    from PIL import Image

    from invoice_extract.ocr.base import OcrEngineUnavailableError, get_engine
    from invoice_extract.preprocess.pipeline import make_pipeline

    ran: dict[str, bool] = {}
    timings: dict[str, float] = {}

    # ---------------------------------------------------- triage + preprocess
    t0 = time.perf_counter()
    image = np.array(Image.open(args.image).convert("L"))
    pipeline = make_pipeline(args.preprocess)
    pre = pipeline(image)
    timings["preprocess_ms"] = round(pre.total_ms, 1)
    triage = pre.stage_info.get("triage")
    ran["preprocess"] = True

    log(f"{BOLD}1. page condition{OFF}")
    if triage:
        for stage, why in sorted(triage["reasons"].items()):
            log(f"   {stage:<24} {DIM}{why}{OFF}")
        if not triage["reasons"]:
            log(f"   {DIM}page looks clean; no preprocessing applied{OFF}")
    else:
        log(f"   {DIM}fixed config {args.preprocess!r}{OFF}")

    # ------------------------------------------------------------------- OCR
    try:
        engine = get_engine(args.ocr_engine, fast=args.fast)
    except OcrEngineUnavailableError as exc:
        print(f"OCR engine unavailable: {exc}", file=sys.stderr)
        return 3
    log(f"{BOLD}2. OCR{OFF} ({args.ocr_engine}, {'mobile' if args.fast else 'medium'} models)")
    t1 = time.perf_counter()
    recognised = engine.recognize(pre.image)
    timings["ocr_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    ran["ocr"] = bool(recognised.words)
    log(f"   {len(recognised.words)} words in {timings['ocr_ms'] / 1000:.1f}s")

    text = reading_order_text(recognised.words, lambda w: tuple(w.bbox), lambda w: w.text)

    # -------------------------------------------------------------- extract
    raw = get_extractor("rules").predict_from_text(text)
    pages = [
        {
            "page_index": 0,
            "image_path": str(args.image),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }
    ]
    document, consistency = finalize(
        raw,
        doc_id=args.image.stem,
        source={"dataset": "other", "original_id": args.image.name, "split": "test"},
        pages=pages,
    )
    ran["extraction"] = True
    schema_errors = validate_jsonschema(document)

    log(f"{BOLD}3. extraction{OFF}")
    vendor = (document.get("parties") or {}).get("vendor") or {}
    meta = document.get("invoice_meta") or {}
    totals = document.get("totals") or {}
    for label, value in (
        ("vendor", vendor.get("name")),
        ("GSTIN", vendor.get("gstin")),
        ("invoice no", meta.get("invoice_number")),
        ("date", meta.get("invoice_date")),
        ("grand total", totals.get("grand_total")),
    ):
        log(f"   {label:<12} {value if value is not None else DIM + 'not found' + OFF}")
    log(f"   {'line items':<12} {len(document.get('line_items') or [])}")

    log(f"{BOLD}4. arithmetic and GSTIN{OFF}")
    if consistency.problems:
        for problem in consistency.problems:
            log(f"   {RED}x{OFF} {problem}")
    else:
        log(f"   {GREEN}ok{OFF} totals reconcile")

    result: dict = {
        "document": document,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "consistency_ok": consistency.ok,
        "consistency_problems": list(consistency.problems),
        "triage": triage,
        "timings": timings,
    }

    # ------------------------------------------------------------- Phase 3
    retriever = stats = None
    if args.ledger and (args.ledger / "ledger.json").exists():
        from invoice_extract.rag.categorize import categorize_document
        from invoice_extract.rag.embed import get_retriever
        from invoice_extract.rag.store import LedgerEntry, VendorStats

        ledger = load_json(args.ledger / "ledger.json")
        entries = [LedgerEntry(**e) for e in ledger["entries"]]
        retriever = get_retriever(args.retriever)
        retriever.index(entries)
        stats = VendorStats.build(entries, doc_totals={})
        for row in ledger.get("doc_totals") or []:
            if row.get("vendor") and row.get("grand_total") is not None:
                stats.totals.setdefault(row["vendor"], []).append(row["grand_total"])

        from invoice_extract.rag.anomaly import detect_anomalies

        decisions = categorize_document(document, retriever, vendor_stats=stats)
        anomalies = detect_anomalies(document, retriever, stats)
        ran["categorisation"] = True
        ran["anomaly_detection"] = True
        result["categories"] = [d.as_dict() for d in decisions]
        result["anomalies"] = anomalies.as_dict()

        log(
            f"{BOLD}5. expense categories{OFF} ({args.retriever} retrieval, {len(entries)} ledger entries)"
        )
        for item, decision in zip(document.get("line_items") or [], decisions, strict=False):
            mark = f"{YELLOW}?{OFF}" if decision.abstained else f"{GREEN}+{OFF}"
            log(f"   {mark} {str(item.get('description'))[:38]:<40} {decision.explain()[:70]}")

        log(f"{BOLD}6. anomaly flags{OFF}")
        if anomalies.flags:
            for flag in anomalies.flags:
                colour = RED if flag.severity == "high" else YELLOW
                log(f"   {colour}{flag.severity:<6}{OFF} {flag.code:<28} {flag.message[:60]}")
        else:
            log(f"   {GREEN}none{OFF}")
    else:
        ran["categorisation"] = ran["anomaly_detection"] = False
        log(f"{BOLD}5-6. categories and anomalies{OFF} {DIM}skipped (no --ledger){OFF}")

    # ------------------------------------------------------------- Phase 4
    verdict = None
    if args.history and (args.history / "canonical").exists():
        from invoice_extract.rag.validate_vendor import validate_against_master
        from invoice_extract.rag.vendor_master import VendorMaster

        paths = sorted(glob.glob(str(args.history / "canonical" / "*.json")))
        history = [load_json(Path(p)) for p in paths[: args.history_size]]
        aliases = {}
        alias_path = args.history / "vendor_aliases.json"
        if alias_path.exists():
            aliases = load_json(alias_path).get("aliases") or {}
        master = VendorMaster.build(history, aliases=aliases)
        report = validate_against_master(document, master, history=history)
        ran["vendor_validation"] = True
        verdict = report.verdict
        result["validation"] = report.as_dict()

        log(f"{BOLD}7. vendor record{OFF} ({len(master)} suppliers, {len(aliases)} trading names)")
        res = report.resolution
        log(
            f"   resolved: {res.vendor.name if res.resolved else DIM + 'no match' + OFF}"
            f"  {DIM}({res.method}){OFF}"
        )
        for check in report.checks:
            colour = {"PASS": GREEN, "REVIEW": YELLOW, "FAIL": RED}.get(check.verdict, DIM)
            log(f"   {colour}{check.verdict:<6}{OFF} {check.code:<24} {check.message[:58]}")
    else:
        ran["vendor_validation"] = False
        log(f"{BOLD}7. vendor record{OFF} {DIM}skipped (no --history){OFF}")

    # --------------------------------------------------------------- verdict
    # A stage that did not run must never read as a clean result, so the
    # headline says what was actually checked.
    if verdict is None:
        verdict = "FAIL" if consistency.problems or schema_errors else "REVIEW"
        basis = "arithmetic and schema only"
    else:
        basis = "all stages"
    result["verdict"] = verdict
    result["verdict_basis"] = basis
    result["stages_run"] = ran

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    colour = VERDICT_COLOUR.get(verdict, DIM)
    total_s = time.perf_counter() - t0
    print()
    print(f"{colour}{BOLD}  {verdict}  {OFF}  {DIM}based on {basis}{OFF}")
    skipped = [k for k, v in ran.items() if not v]
    if skipped:
        print(f"  {YELLOW}not checked:{OFF} {', '.join(skipped)}")
    print(f"  {DIM}{total_s:.1f}s total ({timings['ocr_ms'] / 1000:.1f}s OCR){OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
