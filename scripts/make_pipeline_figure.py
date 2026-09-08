#!/usr/bin/env python
"""Render a before/after contact sheet of the Phase 1 pipeline.

Two uses. Day to day it is visual QA: CER is a single number and will happily
hide a stage that is quietly destroying the page, whereas a human glance at
the strip catches that instantly. For the report it is the figure that shows
the preprocessing working, which no table communicates as directly.

Produces one row per degradation profile: the degraded input, then the output
of each named pipeline configuration.

Usage:
    python scripts/make_pipeline_figure.py --corpus data/processed/gst_in_synthetic \
        --doc-id gst_synth_0000 --out reports/figures/pipeline_stages.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.preprocess.pipeline import Pipeline, get_config  # noqa: E402
from invoice_extract.render import fonts  # noqa: E402

DEFAULT_CONFIGS = [
    "gray+border",
    "gray+border+deskew",
    "gray+border+deskew+illum",
    "full(no-binarize)",
    "full+sauvola",
]


def label_strip(width: int, height: int, text: str, size: int = 20) -> Image.Image:
    strip = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(strip)
    try:
        font = fonts.load("sans", "bold", size)
    except fonts.FontUnavailableError:  # pragma: no cover - no fonts anywhere
        font = None
    draw.text((6, max(0, (height - size) // 2)), text, fill=0, font=font)
    return strip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--doc-id", default=None, help="Defaults to the first document in the manifest")
    ap.add_argument("--profiles", nargs="+", default=["light", "medium", "heavy"])
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--tile-height", type=int, default=520)
    ap.add_argument(
        "--crop-top",
        type=float,
        default=0.42,
        help="Show only the top fraction of the page, where the text density is",
    )
    ap.add_argument("--out", type=Path, default=Path("reports/figures/pipeline_stages.png"))
    args = ap.parse_args(argv)

    manifest_path = args.corpus / "manifest.jsonl"
    if not manifest_path.exists():
        print(
            f"No manifest at {manifest_path}; run scripts/build_image_corpus.py first",
            file=sys.stderr,
        )
        return 1
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]

    doc_id = args.doc_id or manifest[0]["doc_id"]
    rows_meta = {r["profile"]: r for r in manifest if r["doc_id"] == doc_id}
    missing = [p for p in args.profiles if p not in rows_meta]
    if missing:
        print(f"Document {doc_id} has no image for profile(s) {missing}", file=sys.stderr)
        return 1

    pipelines = [(name, Pipeline(get_config(name))) for name in args.configs]
    header_h = 30
    rows: list[list[Image.Image]] = []

    for profile in args.profiles:
        source = np.array(Image.open(args.corpus / rows_meta[profile]["image_path"]).convert("L"))
        tiles = [("degraded input", source)]
        for name, pipeline in pipelines:
            tiles.append((name, pipeline(source).image))

        row: list[Image.Image] = []
        for name, img in tiles:
            crop = img[: int(img.shape[0] * args.crop_top), :]
            pil = Image.fromarray(crop)
            scale = args.tile_height / pil.height
            pil = pil.resize((max(1, int(pil.width * scale)), args.tile_height), Image.LANCZOS)

            tile = Image.new("L", (pil.width, args.tile_height + header_h), 255)
            tile.paste(label_strip(pil.width, header_h, f"{profile} | {name}"), (0, 0))
            tile.paste(pil, (0, header_h))
            ImageDraw.Draw(tile).rectangle([0, 0, tile.width - 1, tile.height - 1], outline=180)
            row.append(tile)
        rows.append(row)

    gap = 8
    width = max(sum(t.width for t in row) + gap * (len(row) - 1) for row in rows)
    height = sum(row[0].height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("L", (width, height), 245)

    y = 0
    for row in rows:
        x = 0
        for tile in row:
            sheet.paste(tile, (x, y))
            x += tile.width + gap
        y += row[0].height + gap

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"Wrote {args.out} ({sheet.width}x{sheet.height}) for doc {doc_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
