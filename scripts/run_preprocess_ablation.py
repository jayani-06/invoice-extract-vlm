#!/usr/bin/env python
"""Phase 1 headline experiment: does preprocessing actually improve OCR, and
which stage earns its place?

Runs the cross product of {degradation profile} x {pipeline config} over the
image corpus, scoring each cell with CER / WER / detection F1, and emits a
markdown table plus a JSON record.

The ablation ladder is cumulative: each config adds exactly one stage to the
one above it, so the row-to-row delta is that stage's contribution. Variant
configs then answer "which method" within a stage, held against a fixed
control. Reading a single end-to-end number instead would tell you the
pipeline works without telling you which parts do.

Usage:
    python scripts/run_preprocess_ablation.py \
        --corpus data/processed/gst_in_synthetic \
        --engine tesseract \
        --profiles light medium heavy \
        --out reports/phase1_ablation

    # No OCR engine installed yet? Verify the plumbing end to end:
    python scripts/run_preprocess_ablation.py --corpus ... --engine oracle --allow-oracle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.eval.ocr_metrics import (  # noqa: E402
    aggregate_ocr,
    score_document,
)
from invoice_extract.ocr.base import (  # noqa: E402
    OcrEngineUnavailableError,
    OcrWord,
    get_engine,
)
from invoice_extract.preprocess.pipeline import (  # noqa: E402
    ABLATION_CONFIGS,
    ADAPTIVE,
    VARIANT_CONFIGS,
    get_config,
    make_pipeline,
)
from invoice_extract.reading_order import reading_order_text  # noqa: E402


def load_manifest(corpus: Path) -> list[dict]:
    path = corpus / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(
            f"No manifest at {path}. Build the image corpus first:\n"
            f"  python scripts/build_image_corpus.py --canonical {corpus}/canonical --out {corpus}"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gold_reading_order_text(words: list[dict]) -> str:
    """Reference text for CER/WER.

    Grouped by geometry, not by the renderer's per-draw-call `line_id`: a
    right-aligned value drawn separately from its label shares a visual line
    with it, and any OCR engine reads it that way. See
    `invoice_extract.reading_order`.
    """
    return reading_order_text(words, lambda w: tuple(w["bbox"]), lambda w: w["text"])


def fmt(value: float, places: int = 4) -> str:
    return "n/a" if value != value else f"{value:.{places}f}"


def markdown_table(
    rows: list[dict], profiles: list[str], metric: str, config_order: list[str]
) -> str:
    """One table per metric: configs down, degradation severity across."""
    lookup = {(r["config"], r["profile"]): r for r in rows}
    header = f"| Pipeline | {' | '.join(profiles)} |"
    sep = "|" + "---|" * (len(profiles) + 1)
    lines = [header, sep]
    for cfg in config_order:
        cells = []
        for p in profiles:
            r = lookup.get((cfg, p))
            cells.append(fmt(r[metric], 4) if r else "-")
        lines.append(f"| `{cfg}` | {' | '.join(cells)} |")
    return "\n".join(lines)


def delta_table(rows: list[dict], profiles: list[str], metric: str, ladder: list[str]) -> str:
    """Per-stage contribution: each row's change from the row above it.

    Negative is an improvement for error-rate metrics, which is why the sign is
    kept rather than reporting an absolute magnitude.
    """
    lookup = {(r["config"], r["profile"]): r for r in rows}
    lines = [f"| Stage added | {' | '.join(profiles)} |", "|" + "---|" * (len(profiles) + 1)]
    for prev, cur in zip(ladder, ladder[1:], strict=False):
        cells = []
        for p in profiles:
            a, b = lookup.get((prev, p)), lookup.get((cur, p))
            if not a or not b or a[metric] != a[metric] or b[metric] != b[metric]:
                cells.append("-")
            else:
                d = b[metric] - a[metric]
                cells.append(f"{d:+.4f}")
        label = cur.replace(prev + "+", "") if cur.startswith(prev + "+") else cur
        lines.append(f"| +{label} | {' | '.join(cells)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra engine kwargs, e.g. --engine-arg psm=11",
    )
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=None,
        help="Degradation profiles to evaluate (default: all in the manifest)",
    )
    ap.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Pipeline config names (default: the full ablation ladder + variants)",
    )
    ap.add_argument("--no-variants", action="store_true", help="Ladder only, skip variant configs")
    ap.add_argument("--limit", type=int, default=None, help="Documents per cell")
    ap.add_argument("--out", type=Path, default=Path("reports/phase1_ablation"))
    ap.add_argument("--iou", type=float, default=0.5, help="Box-match IoU threshold")
    ap.add_argument(
        "--allow-oracle",
        action="store_true",
        help="Permit the ground-truth 'oracle' engine. Its numbers are a ceiling "
        "reference, never an OCR baseline -- reports are watermarked as such.",
    )
    ap.add_argument("--oracle-error-rate", type=float, default=0.0)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Recompute cells already present in the output. Without it, an "
        "interrupted sweep resumes where it stopped.",
    )
    args = ap.parse_args(argv)

    if args.engine == "oracle" and not args.allow_oracle:
        print(
            "Refusing to run the 'oracle' engine without --allow-oracle.\n"
            "It returns ground truth, not recognition: any CER it reports is fictional.\n"
            "Install a real engine (see docs/phase1_preprocessing.md) or pass --allow-oracle "
            "to verify the harness plumbing.",
            file=sys.stderr,
        )
        return 2

    manifest = load_manifest(args.corpus)
    profiles = args.profiles or sorted({r["profile"] for r in manifest})

    if args.configs:
        # Validate every name up front rather than failing an hour into a
        # sweep. `adaptive` is not in CONFIGS_BY_NAME because it has no fixed
        # stage list -- it measures each page and chooses.
        for name in args.configs:
            if name != ADAPTIVE:
                get_config(name)
        configs = list(args.configs)
    else:
        configs = [c.name for c in ABLATION_CONFIGS]
        if not args.no_variants:
            configs += [c.name for c in VARIANT_CONFIGS]

    engine_kwargs = {}
    for pair in args.engine_arg:
        k, _, v = pair.partition("=")
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        engine_kwargs[k] = v

    engine = None
    if args.engine != "oracle":
        try:
            engine = get_engine(args.engine, **engine_kwargs)
        except OcrEngineUnavailableError as exc:
            print(f"\nOCR engine unavailable:\n{exc}\n", file=sys.stderr)
            return 3

    by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in manifest:
        by_profile[row["profile"]].append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"

    # Resume. A full sweep is 12 cells of real OCR and takes an hour and a
    # half; two of them were killed by session teardown and wrote nothing at
    # all, because results.json was only written at the very end. Cells are now
    # flushed as they complete and reloaded on the next run.
    results: list[dict] = []
    elapsed_before = 0.0
    if results_path.exists() and not args.force:
        try:
            previous = json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            # A sweep killed mid-write leaves truncated JSON. Starting over is
            # correct; silently continuing against a half-parsed file is not.
            print(f"{results_path} is unreadable; starting over", file=sys.stderr)
            previous = {}
        else:
            # Only reuse cells measured the same way. An engine or IoU change
            # makes old cells incomparable, and mixing them in one table is
            # worse than recomputing -- the table would silently describe two
            # different experiments.
            same_setup = (
                previous.get("engine") == args.engine
                and previous.get("engine_kwargs") == engine_kwargs
                and previous.get("iou_threshold") == args.iou
                and previous.get("corpus") == str(args.corpus)
            )
            if same_setup:
                results = list(previous.get("results") or [])
                # Carry the earlier wall clock forward, or a resumed sweep
                # reports only the time the final attempt took and the cost of
                # the experiment looks far lower than it was.
                elapsed_before = float(previous.get("elapsed_s") or 0.0)
            elif previous.get("results"):
                print(
                    "Existing results used a different engine, corpus or IoU; "
                    "recomputing all cells.",
                    file=sys.stderr,
                )

    started = time.time()
    total_cells = len(configs) * len(profiles)
    cell = 0
    done = {(r["config"], r["profile"]) for r in results}
    if done:
        print(f"Resuming: {len(done)} of {total_cells} cells already measured", file=sys.stderr)

    def flush() -> None:
        """Write every completed cell. Called after each one, so a kill costs
        the cell in flight rather than the whole sweep."""
        results_path.write_text(
            json.dumps(
                {
                    "engine": args.engine,
                    "engine_kwargs": engine_kwargs,
                    "corpus": str(args.corpus),
                    "iou_threshold": args.iou,
                    "elapsed_s": round(elapsed_before + time.time() - started, 1),
                    "is_oracle": args.engine == "oracle",
                    "complete": len(results) == total_cells,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for name in configs:
        pipeline = make_pipeline(name)
        for profile in profiles:
            cell += 1
            if (name, profile) in done:
                print(
                    f"[{cell}/{total_cells}] {name:<34} {profile:<7} cached",
                    file=sys.stderr,
                )
                continue
            rows = by_profile.get(profile, [])
            if args.limit:
                rows = rows[: args.limit]
            if not rows:
                continue

            reports = []
            for row in rows:
                img_path = args.corpus / row["image_path"]
                gold = json.loads((args.corpus / row["ocr_gold_path"]).read_text())
                image = np.array(Image.open(img_path).convert("L"))

                pre = pipeline(image)
                gold_boxes_src = [tuple(w["bbox"]) for w in gold["words"]]
                gold_boxes = pre.map_boxes(gold_boxes_src)
                gold_pairs = [
                    (w["text"], b) for w, b in zip(gold["words"], gold_boxes, strict=True)
                ]

                if args.engine == "oracle":
                    engine = get_engine(
                        "oracle",
                        words=[
                            OcrWord(w["text"], b, None, w["line_id"])
                            for w, b in zip(gold["words"], gold_boxes, strict=True)
                        ],
                        error_rate=args.oracle_error_rate,
                        quality_aware=True,
                        seed=abs(hash(row["doc_id"])) % (2**31),
                    )

                ocr = engine.recognize(pre.image)
                reports.append(
                    score_document(
                        doc_id=row["doc_id"],
                        gold_text=gold_reading_order_text(gold["words"]),
                        pred_text=ocr.text,
                        gold_boxes=gold_pairs,
                        pred_boxes=[(w.text, w.bbox) for w in ocr.words],
                        iou_threshold=args.iou,
                        elapsed_ms=ocr.elapsed_ms,
                        preprocess_ms=pre.total_ms,
                    )
                )

            agg = aggregate_ocr(reports)
            record = {"config": name, "profile": profile, "n_docs": agg.n_docs, **agg.as_dict()}
            results.append(record)
            flush()
            print(
                f"[{cell}/{total_cells}] {name:<34} {profile:<7} "
                f"CER={fmt(agg.cer_norm, 4)} WER={fmt(agg.wer_norm, 4)} "
                f"detF1={fmt(agg.det_f1, 3)} pre={agg.mean_preprocess_ms:.0f}ms "
                f"ocr={agg.mean_ocr_ms:.0f}ms",
                file=sys.stderr,
            )

    flush()

    # Tables are regenerated from whatever is on disk, so an interrupted sweep
    # still produces a readable report for the cells it did finish.
    results.sort(
        key=lambda r: (
            configs.index(r["config"]) if r["config"] in configs else 99,
            profiles.index(r["profile"]) if r["profile"] in profiles else 99,
        )
    )

    ladder = [c.name for c in ABLATION_CONFIGS if any(r["config"] == c.name for r in results)]
    # `adaptive` is not a rung: it is not the row above it plus one stage, so a
    # per-stage delta against it would not mean anything. Tabulated with the
    # variants instead.
    variants = [c.name for c in VARIANT_CONFIGS if any(r["config"] == c.name for r in results)]
    if any(r["config"] == ADAPTIVE for r in results):
        variants.append(ADAPTIVE)

    md = [f"# Phase 1 preprocessing ablation ({args.engine})", ""]
    if args.engine == "oracle":
        md += [
            "> **These numbers are not an OCR result.** The `oracle` engine returns the "
            "renderer's ground truth with synthetic noise. This table verifies that the "
            "harness, the homography composition and the metrics work end to end. Re-run "
            "with a real engine before citing anything.",
            ">",
            "> In particular, **the direction of the deltas here is meaningless**: the "
            "oracle injects errors from a crude local-contrast heuristic, which reads a "
            "denoised or binarised page as *less* legible and so makes preprocessing look "
            "harmful. Only a real engine can say whether a stage helps.",
            "",
        ]
    md += [
        f"- Corpus: `{args.corpus}` | documents per cell: {results[0]['n_docs'] if results else 0}",
        f"- Box-match IoU threshold: {args.iou}",
        f"- Wall clock: {round(time.time() - started, 1)}s",
        "",
        "## Normalised CER (lower is better)",
        "",
        markdown_table(results, profiles, "cer_norm", ladder + variants),
        "",
        "## Per-stage contribution to CER (negative = improvement)",
        "",
        delta_table(results, profiles, "cer_norm", ladder),
        "",
        "## Normalised WER (lower is better)",
        "",
        markdown_table(results, profiles, "wer_norm", ladder + variants),
        "",
        "## Word detection F1 (higher is better)",
        "",
        markdown_table(results, profiles, "det_f1", ladder + variants),
        "",
        "## Word accuracy - located *and* read correctly (higher is better)",
        "",
        markdown_table(results, profiles, "word_accuracy", ladder + variants),
        "",
        "## Mean preprocessing cost per page (ms)",
        "",
        markdown_table(results, profiles, "mean_preprocess_ms", ladder + variants),
        "",
    ]
    (args.out / "ablation.md").write_text("\n".join(md))

    print(f"\nWrote {args.out / 'ablation.md'} and {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
