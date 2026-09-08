#!/usr/bin/env python
"""Convert real (Label-Studio-annotated) GST invoices into canonical schema JSON.

Usage:
    python scripts/convert_gst_in.py --mode real --raw data/raw/gst_in_real --out data/processed/gst_in_real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from invoice_extract.data.converters import gst_in  # noqa: E402
from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["real"],
        default="real",
        help="Synthetic docs are generated directly; see generate_synthetic_gst_invoices.py",
    )
    parser.add_argument(
        "--raw", type=Path, required=True, help="Label Studio JSON export file or directory"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    canonical_dir = args.out / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    schema = load_schema()

    export_path = args.raw if args.raw.is_file() else next(args.raw.glob("*.json"), None)
    if export_path is None:
        print(f"No Label Studio export JSON found at/under {args.raw}", file=sys.stderr)
        return 1

    tasks = json.loads(export_path.read_text())
    n_ok = 0
    for task in tasks:
        doc_id = f"gst_real_{task['id']}"
        image_path = task.get("data", {}).get("image", "")
        try:
            width, height = Image.open(image_path).size
        except (FileNotFoundError, OSError):
            width = height = 0

        doc = gst_in.convert(
            task, doc_id=doc_id, split=args.split, image_path=image_path, width=width, height=height
        )
        doc_dict = doc.model_dump()
        errors = validate_jsonschema(doc_dict, schema)
        if errors:
            print(f"SKIP {doc_id}: {errors}", file=sys.stderr)
            continue
        (canonical_dir / f"{doc_id}.json").write_text(json.dumps(doc_dict, indent=2))
        n_ok += 1

    print(f"Converted {n_ok}/{len(tasks)} real GST invoices to {canonical_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
