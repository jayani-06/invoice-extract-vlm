#!/usr/bin/env python
"""Precompute everything the demo shows, so the demo itself is instant.

OCR runs at ~40 s/page on this CPU. A demo that processes on click has a
40-second dead spot on every interaction and a real chance of stalling in front
of an audience, so the app replays cached artifacts instead. Everything cached
here is a genuine output -- real preprocessing, real OCR, real extraction --
just computed in advance rather than during the review.

Two things this deliberately does NOT do:

- **It never invents a result.** If a stage was not run, the cache records
  `null` and the app renders "not run" rather than a placeholder number.
- **It does not hide failures.** `--include-failures` (on by default) makes sure
  the picked documents include ones the extractor gets wrong. A demo that only
  shows successes invites exactly the question you least want asked cold.

Layout written:

    demo_cache/
      index.json                     documents, what was run, when
      docs/<doc_id>.json             gold, OCR, extractions, per-field scoring
      images/<doc_id>/<name>.png     originals, degraded, preprocessed
      reports/*.json                 copies of the ablation results

Usage:
    python scripts/build_demo_cache.py --limit 4 --fatura2-limit 4
    python scripts/build_demo_cache.py --vlm vlm_outputs.json   # merge Colab run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.eval.metrics import compare_documents  # noqa: E402
from invoice_extract.models.postprocess import (  # noqa: E402
    check_arithmetic,
    finalize,
    validate_gstin,
)
from invoice_extract.models.registry import get_extractor  # noqa: E402
from invoice_extract.preprocess.pipeline import Pipeline, get_config  # noqa: E402
from invoice_extract.reading_order import reading_order_text  # noqa: E402

#: Shown on the pipeline screen as toggleable stages, cheapest first.
DEMO_CONFIGS = [
    "raw",
    "gray+border",
    "gray+border+deskew",
    "full(no-binarize)",
    "full+sauvola",
]

#: Fields the demo scores on screen 3 and 4.
DEMO_FIELDS = [
    "parties.vendor.name",
    "invoice_meta.invoice_number",
    "invoice_meta.invoice_date",
    "totals.grand_total",
]


def gold_text(corpus: Path, row: dict) -> str | None:
    path = row.get("ocr_gold_path")
    if not path or not (corpus / path).exists():
        return None
    gold = json.loads((corpus / path).read_text(encoding="utf-8"))
    return reading_order_text(gold["words"], lambda w: tuple(w["bbox"]), lambda w: w["text"])


def score_fields(gold_doc: dict, pred_doc: dict) -> dict:
    """Per-field correct/wrong, using the same comparison the harness uses."""
    report = compare_documents(gold_doc, pred_doc)
    out = {}
    for path in DEMO_FIELDS:
        cmp = report.field_comparisons.get(path)
        gold_flat = gold_doc
        pred_flat = pred_doc
        for part in path.split("."):
            gold_flat = (gold_flat or {}).get(part) if isinstance(gold_flat, dict) else None
            pred_flat = (pred_flat or {}).get(part) if isinstance(pred_flat, dict) else None
        out[path] = {
            "gold": gold_flat,
            "pred": pred_flat,
            "correct": bool(cmp.correct) if cmp else (gold_flat is None and pred_flat is None),
            "score": float(cmp.score) if cmp else None,
        }
    return out


def validation_panel(doc: dict) -> dict:
    """Arithmetic + GSTIN checks -- the 'flag, never silently fix' behaviour.

    Uses `postprocess.check_arithmetic`, not the Phase 0
    `check_totals_consistency`. The latter only compares line totals against the
    subtotal and the tax lines against the tax total; it never verifies
    `grand_total`, so an invoice whose headline figure is wrong would show a
    clean bill of health on screen 3 -- the single worst thing this panel could
    do.
    """
    consistency = check_arithmetic(doc)
    gstin = (doc.get("parties", {}).get("vendor") or {}).get("gstin")
    return {
        "arithmetic": consistency.problems,
        "arithmetic_ok": consistency.ok,
        "gstin": gstin,
        "gstin_problems": validate_gstin(gstin) if gstin else [],
    }


def build_corpus(
    corpus: Path,
    corpus_name: str,
    out: Path,
    *,
    limit: int,
    profiles: list[str],
    ocr_profiles: list[str],
    ocr_configs: list[str],
    engine,
    include_failures: bool,
) -> list[dict]:
    manifest_path = corpus / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  skipping {corpus_name}: no manifest at {manifest_path}", file=sys.stderr)
        return []

    manifest = [json.loads(ln) for ln in manifest_path.read_text().splitlines() if ln.strip()]
    by_doc: dict[str, dict[str, dict]] = {}
    for row in manifest:
        by_doc.setdefault(row["doc_id"], {})[row["profile"]] = row

    rules = get_extractor("rules")

    # Rank documents so the demo set includes visible failures, not just wins.
    ranked: list[tuple[float, str]] = []
    for doc_id, rows in by_doc.items():
        gold_doc = json.loads((corpus / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8"))
        row = rows.get("clean") or next(iter(rows.values()))
        text = gold_text(corpus, row)
        if text is None:
            ranked.append((0.5, doc_id))
            continue
        pred, _ = finalize(
            rules.predict_from_text(text), doc_id, gold_doc["source"], gold_doc["pages"]
        )
        scored = score_fields(gold_doc, pred)
        ranked.append((sum(f["correct"] for f in scored.values()) / len(scored), doc_id))
    ranked.sort()

    if include_failures and len(ranked) > limit:
        # Half from the worst-scoring documents, half from the best.
        n_bad = max(1, limit // 3)
        picked = [d for _, d in ranked[:n_bad]] + [d for _, d in ranked[-(limit - n_bad) :]]
    else:
        picked = [d for _, d in ranked[:limit]]

    entries: list[dict] = []
    for i, doc_id in enumerate(picked, start=1):
        rows = by_doc[doc_id]
        gold_doc = json.loads((corpus / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8"))
        img_dir = out / "images" / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        available = [p for p in profiles if p in rows]
        images: dict[str, str] = {}
        ocr: dict[str, dict] = {}

        for profile in available:
            row = rows[profile]
            source = np.array(Image.open(corpus / row["image_path"]).convert("L"))
            name = f"{profile}.png"
            Image.fromarray(source).save(img_dir / name)
            images[profile] = f"images/{doc_id}/{name}"

            for config in ocr_configs if profile in ocr_profiles else []:
                result = Pipeline(get_config(config))(source)
                key = f"{profile}__{config}"
                pre_name = f"pre__{key}.png".replace("(", "_").replace(")", "_").replace("+", "-")
                Image.fromarray(result.image).save(img_dir / pre_name)
                images[key] = f"images/{doc_id}/{pre_name}"

                if engine is not None:
                    t0 = time.perf_counter()
                    recognised = engine.recognize(result.image)
                    ocr[key] = {
                        "words": [
                            {
                                "text": w.text,
                                "bbox": [round(v, 1) for v in w.bbox],
                                "confidence": w.confidence,
                            }
                            for w in recognised.words
                        ],
                        "text": recognised.text,
                        "ms": round((time.perf_counter() - t0) * 1000),
                        "engine": recognised.engine,
                    }
                    print(
                        f"    {doc_id} {key}: {len(recognised.words)} words "
                        f"in {ocr[key]['ms'] / 1000:.0f}s",
                        file=sys.stderr,
                    )

            # Preprocessed previews for the pipeline screen (no OCR, so cheap).
            for config in DEMO_CONFIGS:
                key = f"{profile}__{config}"
                if key in images:
                    continue
                result = Pipeline(get_config(config))(source)
                pre_name = f"pre__{key}.png".replace("(", "_").replace(")", "_").replace("+", "-")
                Image.fromarray(result.image).save(img_dir / pre_name)
                images[key] = f"images/{doc_id}/{pre_name}"

        # Extraction from ground-truth text: the perfect-OCR ceiling, and the
        # only path FATURA2 supports (it ships tokens, not a full transcript).
        extractions: dict[str, dict] = {}
        base_row = rows.get("clean") or rows[available[0]]
        text = gold_text(corpus, base_row)
        if text is not None:
            raw = rules.predict_from_text(text)
            pred, consistency = finalize(raw, doc_id, gold_doc["source"], gold_doc["pages"])
            extractions["rules"] = {
                "source": "gold-text (perfect-OCR ceiling)",
                "document": pred,
                "per_field": score_fields(gold_doc, pred),
                "validation": validation_panel(pred),
                "arithmetic_flags": consistency.problems,
            }

        # Extraction from real OCR, when we ran it.
        for key, result in ocr.items():
            raw = rules.predict_from_text(result["text"])
            pred, consistency = finalize(raw, doc_id, gold_doc["source"], gold_doc["pages"])
            extractions[f"rules@{key}"] = {
                "source": f"real OCR ({result['engine']}, {key})",
                "document": pred,
                "per_field": score_fields(gold_doc, pred),
                "validation": validation_panel(pred),
                "arithmetic_flags": consistency.problems,
            }

        entry = {
            "doc_id": doc_id,
            "corpus": corpus_name,
            "profiles": available,
            "images": images,
            "ocr": ocr,
            "gold": gold_doc,
            "gold_text": text,
            "extractions": extractions,
            "layout": base_row.get("layout_style"),
        }
        (out / "docs").mkdir(parents=True, exist_ok=True)
        (out / "docs" / f"{doc_id}.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")

        entries.append(
            {
                "doc_id": doc_id,
                "corpus": corpus_name,
                "profiles": available,
                "has_ocr": bool(ocr),
                "rules_score": (
                    sum(f["correct"] for f in extractions["rules"]["per_field"].values())
                    / len(DEMO_FIELDS)
                    if "rules" in extractions
                    else None
                ),
            }
        )
        print(f"  [{i}/{len(picked)}] {doc_id}", file=sys.stderr)

    return entries


def merge_vlm(out: Path, vlm_path: Path) -> int:
    """Fold Colab-produced VLM outputs into the cache.

    The notebook writes {doc_id: {raw_response, parsed, elapsed_ms, model}}.
    Everything else -- coercion, scoring, validation -- happens here, using the
    exact same code path as the rules baseline, so a difference on screen is a
    difference in the model rather than in how each was post-processed.
    """
    payload = json.loads(vlm_path.read_text(encoding="utf-8"))
    outputs = payload.get("outputs", payload)
    model = payload.get("model", "vlm")
    merged = 0

    for doc_id, result in outputs.items():
        doc_path = out / "docs" / f"{doc_id}.json"
        if not doc_path.exists():
            print(f"  {doc_id}: not in cache, skipped", file=sys.stderr)
            continue
        entry = json.loads(doc_path.read_text(encoding="utf-8"))
        gold_doc = entry["gold"]

        parsed = result.get("parsed")
        pred, consistency = finalize(parsed or {}, doc_id, gold_doc["source"], gold_doc["pages"])
        entry["extractions"][f"vlm:{model}"] = {
            "source": f"VLM ({model}), run on Colab",
            "document": pred,
            "per_field": score_fields(gold_doc, pred),
            "validation": validation_panel(pred),
            "arithmetic_flags": consistency.problems,
            "parse_failed": parsed is None,
            "raw_response": (result.get("raw_response") or "")[:4000],
            "elapsed_ms": result.get("elapsed_ms"),
        }
        doc_path.write_text(json.dumps(entry, indent=1), encoding="utf-8")
        merged += 1

    index_path = out / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["vlm_model"] = model
    index["has_vlm"] = merged > 0
    index["vlm_merged"] = merged
    index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"\nMerged VLM outputs for {merged} document(s); model = {model}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("demo_cache"))
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/gst_in_synthetic"))
    ap.add_argument("--fatura2", type=Path, default=Path("data/processed/fatura2"))
    ap.add_argument("--limit", type=int, default=4, help="Documents from our synthetic corpus")
    ap.add_argument("--fatura2-limit", type=int, default=4, help="Documents from FATURA2")
    ap.add_argument("--profiles", nargs="+", default=["clean", "medium", "heavy"])
    ap.add_argument(
        "--ocr-profiles",
        nargs="+",
        default=["medium"],
        help="Profiles to actually run OCR on. Each costs ~40 s/page/config.",
    )
    ap.add_argument("--ocr-configs", nargs="+", default=["full(no-binarize)"])
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--fast", action="store_true", default=True, help="Use the mobile OCR models")
    ap.add_argument("--no-ocr", action="store_true", help="Skip OCR entirely (fast rebuild)")
    ap.add_argument("--no-failures", dest="include_failures", action="store_false", default=True)
    ap.add_argument("--vlm", type=Path, default=None, help="Merge a Colab VLM output JSON and exit")
    args = ap.parse_args(argv)

    if args.vlm:
        if not (args.out / "index.json").exists():
            print(f"No cache at {args.out}; build it first.", file=sys.stderr)
            return 1
        return merge_vlm(args.out, args.vlm)

    engine = None
    if not args.no_ocr:
        from invoice_extract.ocr.base import OcrEngineUnavailableError, get_engine

        try:
            engine = get_engine(args.engine, fast=args.fast)
        except OcrEngineUnavailableError as exc:
            print(f"OCR unavailable, continuing without it:\n{exc}\n", file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "docs").mkdir(exist_ok=True)
    (args.out / "images").mkdir(exist_ok=True)
    (args.out / "reports").mkdir(exist_ok=True)

    started = time.time()
    entries: list[dict] = []

    print("Our synthetic GST corpus:", file=sys.stderr)
    entries += build_corpus(
        args.corpus,
        "gst_in_synthetic",
        args.out,
        limit=args.limit,
        profiles=args.profiles,
        ocr_profiles=args.ocr_profiles,
        ocr_configs=args.ocr_configs,
        engine=engine,
        include_failures=args.include_failures,
    )

    print("FATURA2 (third-party):", file=sys.stderr)
    entries += build_corpus(
        args.fatura2,
        "fatura2",
        args.out,
        limit=args.fatura2_limit,
        profiles=["clean"],
        ocr_profiles=[],
        ocr_configs=[],
        engine=None,
        include_failures=args.include_failures,
    )

    for name, path in (
        ("phase1", Path("reports/phase1_ablation/results.json")),
        ("phase2", Path("reports/phase2_ablation/results.json")),
        ("fatura2", Path("reports/phase2_fatura2/results.json")),
    ):
        if path.exists():
            shutil.copy(path, args.out / "reports" / f"{name}.json")

    index = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - started, 1),
        "documents": entries,
        "ocr_engine": args.engine if engine else None,
        "ocr_profiles": args.ocr_profiles if engine else [],
        "ocr_configs": args.ocr_configs if engine else [],
        "demo_configs": DEMO_CONFIGS,
        "demo_fields": DEMO_FIELDS,
        "has_vlm": False,
        "vlm_model": None,
    }
    (args.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")

    print(f"\nCached {len(entries)} documents to {args.out} in {index['elapsed_s']:.0f}s")
    print(f"  with real OCR: {sum(1 for e in entries if e['has_ocr'])}")
    print("\nRun the demo:  streamlit run demo/app.py")
    if not index["has_vlm"]:
        print(
            "\nNo VLM outputs yet. Run notebooks/vlm_demo_outputs.ipynb on Colab, then:\n"
            "  python scripts/build_demo_cache.py --vlm vlm_outputs.json"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
