#!/usr/bin/env python
"""Convert downloaded CORD ground_truth JSON into canonical schema JSON.

Usage:
    python scripts/convert_cord.py --raw data/raw/cord --out data/processed/cord
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from invoice_extract.data.converters import cord  # noqa: E402
from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    schema = load_schema()

    for split_dir in sorted(p for p in args.raw.iterdir() if p.is_dir()):
        split = split_dir.name  # download_cord.py writes train/validation/test dirs
        canonical_dir = args.out / "canonical" / split
        canonical_dir.mkdir(parents=True, exist_ok=True)

        meta_files = sorted(split_dir.glob("*.json"))
        n_ok = 0
        for meta_path in meta_files:
            raw = json.loads(meta_path.read_text())
            ground_truth = (
                json.loads(raw["ground_truth"])
                if isinstance(raw.get("ground_truth"), str)
                else raw.get("ground_truth", raw)
            )
            doc_id = f"cord_{split}_{meta_path.stem}"
            image_path = split_dir / "images" / f"{meta_path.stem}.png"
            try:
                width, height = Image.open(image_path).size
            except FileNotFoundError:
                width = height = 0

            doc = cord.convert(
                ground_truth,
                doc_id=doc_id,
                split="val" if split.startswith("valid") else split,
                image_path=str(image_path),
                width=width,
                height=height,
            )
            doc_dict = doc.model_dump()
            errors = validate_jsonschema(doc_dict, schema)
            if errors:
                print(f"SKIP {doc_id}: {errors}", file=sys.stderr)
                continue
            (canonical_dir / f"{doc_id}.json").write_text(json.dumps(doc_dict, indent=2))
            n_ok += 1

        print(f"Converted {n_ok}/{len(meta_files)} CORD [{split}] documents to {canonical_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
