#!/usr/bin/env python
"""Run an extractor over the image corpus and write predictions + gold JSONL.

The output is deliberately in exactly the shape `invoice_extract.eval.harness`
already consumes, so Phase 2 is scored by the metrics Phase 0 froze before any
model existed. Inventing a second scoring path here would let the model be
judged by rules written after seeing its output.

Usage:
    # Rules baseline over OCR text (needs an OCR engine, or --ocr-source gold)
    python scripts/run_extraction.py --corpus data/processed/gst_in_synthetic \\
        --extractor rules --ocr-source gold --profile clean --out reports/phase2/rules

    # VLM, hybrid image + OCR text (needs a GPU; run on Colab/Kaggle)
    python scripts/run_extraction.py --corpus data/processed/gst_in_synthetic \\
        --extractor vlm:qwen2-vl-2b --ocr-source engine --ocr-engine paddle \\
        --profile medium --out reports/phase2/vlm_hybrid

Then score:
    python -m invoice_extract.eval.harness \\
        --gold reports/phase2/rules/gold.jsonl --pred reports/phase2/rules/pred.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402
from invoice_extract.models.postprocess import finalize  # noqa: E402
from invoice_extract.models.prompting import extract_json_object  # noqa: E402
from invoice_extract.models.registry import get_extractor  # noqa: E402
from invoice_extract.reading_order import reading_order_text  # noqa: E402


def gold_text_for(corpus: Path, row: dict) -> str:
    gold = json.loads((corpus / row["ocr_gold_path"]).read_text(encoding="utf-8"))
    return reading_order_text(gold["words"], lambda w: tuple(w["bbox"]), lambda w: w["text"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--extractor", default="rules")
    ap.add_argument("--profile", default="clean", help="Degradation profile to evaluate on")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--doc-ids",
        type=Path,
        default=None,
        help="Restrict to the documents listed in a selection record written by "
        "export_vlm_payload.py. Comparing two extractors is only meaningful on "
        "identical documents, and --limit does not guarantee that.",
    )
    ap.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Score raw responses from a finished GPU run (vlm_outputs.json) instead "
        "of calling a model. Needs no GPU and no extractor. The point is that a "
        "replay is scored by this same code path, so the VLM and the rules "
        "baseline are judged by identical post-processing and metrics.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--ocr-source",
        default="gold",
        choices=["gold", "engine", "none"],
        help=(
            "Where OCR text comes from. 'engine' runs a real OCR engine. 'gold' uses the "
            "renderer's ground-truth text -- a CEILING, not a baseline: it measures the "
            "extractor given perfect OCR, and is flagged as such in the report."
        ),
    )
    ap.add_argument("--ocr-engine", default="tesseract")
    ap.add_argument("--preprocess-config", default="full(no-binarize)")
    ap.add_argument("--vision-only", action="store_true", help="VLM only: drop the OCR hint")
    ap.add_argument("--day-first", action="store_true", default=True)
    ap.add_argument("--month-first", dest="day_first", action="store_false")
    args = ap.parse_args(argv)

    manifest_path = args.corpus / "manifest.jsonl"
    if not manifest_path.exists():
        print(
            f"No manifest at {manifest_path}. Build the corpus first:\n"
            f"  python scripts/build_image_corpus.py --canonical {args.corpus}/canonical "
            f"--out {args.corpus}",
            file=sys.stderr,
        )
        return 1

    manifest = [json.loads(ln) for ln in manifest_path.read_text().splitlines() if ln.strip()]
    rows = [r for r in manifest if r["profile"] == args.profile]

    if args.doc_ids:
        record = json.loads(args.doc_ids.read_text(encoding="utf-8"))
        wanted: list[str] = []
        for selection in record.get("selections", [record]):
            # One record can cover several corpora; take only the ids that
            # belong to this one, so the same file drives every corpus's run.
            if selection.get("corpus") in (None, args.corpus.name):
                wanted += selection.get("doc_ids") or []
        if not wanted:
            corpora = [s.get("corpus") for s in record.get("selections", [record])]
            print(
                f"{args.doc_ids} selects no documents for corpus {args.corpus.name!r}; "
                f"it covers {corpora}",
                file=sys.stderr,
            )
            return 1
        by_id = {r["doc_id"]: r for r in rows}
        missing = sorted(set(wanted) - set(by_id))
        if missing:
            print(
                f"{len(missing)} selected document(s) are not in this corpus at profile "
                f"{args.profile!r}, e.g. {missing[:3]}",
                file=sys.stderr,
            )
            return 1
        rows = [by_id[d] for d in wanted]

    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"No corpus rows for profile {args.profile!r}", file=sys.stderr)
        return 1

    # A replay reads responses a GPU already produced. Nothing below needs an
    # extractor, OCR engine or model weights.
    replayed: dict[str, dict] = {}
    replay_model = None
    if args.replay:
        payload = json.loads(args.replay.read_text(encoding="utf-8"))
        replayed = payload.get("outputs") or {}
        replay_model = payload.get("model_id") or payload.get("model")
        if not replayed:
            print(f"{args.replay} carries no outputs", file=sys.stderr)
            return 1
        covered = [r for r in rows if r["doc_id"] in replayed]
        if not covered:
            print(
                f"None of the {len(rows)} selected documents appear in {args.replay}",
                file=sys.stderr,
            )
            return 1
        if len(covered) < len(rows):
            # Scoring only the documents the model happened to answer would
            # inflate the result. Say so and keep the full denominator.
            print(
                f"NOTE: {len(rows) - len(covered)} of {len(rows)} selected documents are "
                f"absent from {args.replay.name}. They are scored as misses, not dropped.",
                file=sys.stderr,
            )

        # How the run was actually produced is recorded in the payload, and it
        # overrides the CLI. Left to the `--ocr-source` default, a vision-only
        # GPU run would be stamped `is_ocr_ceiling: true` and reported as a
        # perfect-OCR ceiling -- when a model that read nothing but pixels has
        # produced the project's first genuine end-to-end number. Mislabelling
        # that understates the result and corrupts every table it reaches.
        args.ocr_source = "gold" if payload.get("ocr_hint") else "none"
        args.vision_only = not payload.get("ocr_hint")

    # Wire up OCR only if it is actually needed.
    ocr_engine = None
    preprocess = None
    if args.ocr_source == "engine" and not args.replay:
        from invoice_extract.ocr.base import OcrEngineUnavailableError, get_engine
        from invoice_extract.preprocess.pipeline import make_pipeline

        try:
            ocr_engine = get_engine(args.ocr_engine)
        except OcrEngineUnavailableError as exc:
            print(f"\nOCR engine unavailable:\n{exc}\n", file=sys.stderr)
            return 3
        preprocess = make_pipeline(args.preprocess_config)

    extractor_kwargs: dict = {}
    if args.extractor.startswith("vlm"):
        extractor_kwargs = {
            "use_ocr_hint": not args.vision_only and args.ocr_source != "none",
            "ocr_engine": ocr_engine,
            "preprocess": preprocess,
        }
    elif args.extractor == "rules" and args.ocr_source == "engine":
        extractor_kwargs = {"ocr_engine": ocr_engine, "preprocess": preprocess}

    extractor = None
    if not args.replay:
        try:
            extractor = get_extractor(args.extractor, **extractor_kwargs)
        except Exception as exc:
            print(f"Could not construct extractor {args.extractor!r}: {exc}", file=sys.stderr)
            return 3

    schema = load_schema()
    args.out.mkdir(parents=True, exist_ok=True)
    gold_lines: list[str] = []
    pred_lines: list[str] = []
    stats = {
        "n": 0,
        "parse_failures": 0,
        "schema_failures": 0,
        "arithmetic_flags": 0,
        "total_ms": 0.0,
    }
    started = time.time()

    for i, row in enumerate(rows, start=1):
        doc_id = row["doc_id"]
        gold_doc = json.loads(
            (args.corpus / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        image_path = args.corpus / row["image_path"]
        page = {
            "page_index": 0,
            "image_path": row["image_path"],
            "width": row["width"],
            "height": row["height"],
        }
        # The schema types `dpi` as an integer, so an absent dpi has to be an
        # absent key, not a null. FATURA2's manifest carries no dpi (its pages
        # are vector renders, not scans), and emitting `"dpi": null` made all
        # 50 predictions fail schema validation -- which reads in a report as
        # the extractor producing malformed documents rather than as a corpus
        # that does not record dpi.
        if row.get("dpi") is not None:
            page["dpi"] = row["dpi"]
        pages = [page]

        t0 = time.perf_counter()
        try:
            if args.replay:
                entry = replayed.get(doc_id) or {}
                # A document the model never answered, an empty response and an
                # unparseable response are all the same outcome: no prediction.
                # `extract_json_object` is re-run here rather than trusting the
                # notebook's `parsed`, so parsing is the repo's current parser.
                raw = extract_json_object(entry.get("raw_response") or "") or {}
                stats["total_ms"] += float(entry.get("elapsed_ms") or 0.0)
            elif args.ocr_source == "gold" and hasattr(extractor, "predict_from_text"):
                raw = extractor.predict_from_text(gold_text_for(args.corpus, row))
            elif args.extractor.startswith("vlm") and args.ocr_source == "gold":
                raw = (
                    extractor.generate(
                        [image_path], ocr_text=gold_text_for(args.corpus, row)
                    ).parsed
                    or {}
                )
            else:
                raw = extractor.predict([image_path])
        except Exception as exc:  # a model failure is a data point, not a crash
            print(f"  {doc_id}: extractor raised {type(exc).__name__}: {exc}", file=sys.stderr)
            raw = {}
        if not args.replay:
            stats["total_ms"] += (time.perf_counter() - t0) * 1000

        if not raw:
            stats["parse_failures"] += 1

        pred_doc, consistency = finalize(
            raw,
            doc_id=doc_id,
            source=gold_doc.get(
                "source", {"dataset": "other", "original_id": doc_id, "split": "test"}
            ),
            pages=pages,
            day_first=args.day_first,
        )
        if consistency.problems:
            stats["arithmetic_flags"] += 1
        if validate_jsonschema(pred_doc, schema):
            stats["schema_failures"] += 1

        gold_lines.append(json.dumps(gold_doc))
        pred_lines.append(json.dumps(pred_doc))
        stats["n"] += 1

        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} documents", file=sys.stderr)

    (args.out / "gold.jsonl").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    (args.out / "pred.jsonl").write_text("\n".join(pred_lines) + "\n", encoding="utf-8")

    run = {
        "extractor": args.extractor,
        "replayed_from": str(args.replay) if args.replay else None,
        "replay_model": replay_model,
        "doc_ids_from": str(args.doc_ids) if args.doc_ids else None,
        "profile": args.profile,
        "ocr_source": args.ocr_source,
        "ocr_engine": args.ocr_engine if args.ocr_source == "engine" else None,
        "vision_only": args.vision_only,
        "is_ocr_ceiling": args.ocr_source == "gold",
        "n_docs": stats["n"],
        "parse_failures": stats["parse_failures"],
        "schema_failures": stats["schema_failures"],
        "documents_with_arithmetic_flags": stats["arithmetic_flags"],
        "mean_ms_per_doc": round(stats["total_ms"] / max(stats["n"], 1), 1),
        "elapsed_s": round(time.time() - started, 1),
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}/pred.jsonl and gold.jsonl ({stats['n']} documents)")
    if run["is_ocr_ceiling"]:
        print(
            "NOTE: --ocr-source gold uses the renderer's ground-truth text. These are "
            "CEILING numbers for the extractor given perfect OCR, not end-to-end results.",
            file=sys.stderr,
        )
    if stats["parse_failures"]:
        print(
            f"WARNING: {stats['parse_failures']} document(s) produced no parseable output",
            file=sys.stderr,
        )
    if stats["schema_failures"]:
        print(
            f"WARNING: {stats['schema_failures']} prediction(s) failed schema validation",
            file=sys.stderr,
        )
    print(
        f"\nScore with:\n  python -m invoice_extract.eval.harness "
        f"--gold {args.out}/gold.jsonl --pred {args.out}/pred.jsonl"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
