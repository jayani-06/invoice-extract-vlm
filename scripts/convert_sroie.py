#!/usr/bin/env python
"""Convert downloaded SROIE key-value files into canonical schema JSON.

Usage:
    python scripts/convert_sroie.py --raw data/raw/sroie --out data/processed/sroie
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from invoice_extract.data.converters import sroie  # noqa: E402
from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    canonical_dir = args.out / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema()

    # SROIE's public mirrors typically use an `entities/*.txt` (JSON-ish key:value)
    # directory alongside `img/*.jpg`. Adjust the glob if your mirror differs.
    kv_files = sorted(args.raw.glob("**/entities/*.txt")) or sorted(args.raw.glob("**/*.txt"))
    if not kv_files:
        print(f"No key-value files found under {args.raw}.", file=sys.stderr)
        return 1

    n_ok = 0
    for kv_path in kv_files:
        try:
            kv = json.loads(kv_path.read_text())
        except json.JSONDecodeError:
            print(
                f"SKIP {kv_path}: not valid JSON, check the mirror's file format", file=sys.stderr
            )
            continue

        doc_id = f"sroie_{kv_path.stem}"
        image_path = kv_path.parent.parent / "img" / f"{kv_path.stem}.jpg"
        try:
            width, height = Image.open(image_path).size
        except FileNotFoundError:
            width = height = 0

        doc = sroie.convert(
            kv,
            doc_id=doc_id,
            split=args.split,
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

    print(f"Converted {n_ok}/{len(kv_files)} SROIE documents to {canonical_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
