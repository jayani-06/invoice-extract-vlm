#!/usr/bin/env python
"""Convert downloaded FATURA annotations into canonical schema JSON.

This is a template driver: it assumes `--raw` contains one JSON annotation
file + one image per document (adjust the glob/loading logic once you've
inspected the real FATURA directory layout after downloading it).

Usage:
    python scripts/convert_fatura.py --raw data/raw/fatura --out data/processed/fatura
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.converters import fatura  # noqa: E402
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

    ann_files = sorted(args.raw.glob("**/*.json"))
    if not ann_files:
        print(
            f"No annotation JSON files found under {args.raw}. "
            "Confirm the download completed and the layout matches what this script expects.",
            file=sys.stderr,
        )
        return 1

    n_ok = 0
    for ann_path in ann_files:
        raw = json.loads(ann_path.read_text())
        doc_id = f"fatura_{ann_path.stem}"
        image_path = str(ann_path.with_suffix(".png"))
        doc = fatura.convert(
            raw,
            doc_id=doc_id,
            split=args.split,
            image_path=image_path,
            width=raw.get("width", 0),
            height=raw.get("height", 0),
        )
        doc_dict = doc.model_dump()
        errors = validate_jsonschema(doc_dict, schema)
        if errors:
            print(f"SKIP {doc_id}: {errors}", file=sys.stderr)
            continue
        (canonical_dir / f"{doc_id}.json").write_text(json.dumps(doc_dict, indent=2))
        n_ok += 1

    print(f"Converted {n_ok}/{len(ann_files)} FATURA documents to {canonical_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
