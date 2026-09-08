#!/usr/bin/env python
"""Download CORD (Consolidated Receipt Dataset) from HuggingFace Hub.

CORD is public on the Hub as `naver-clova-ix/cord-v2`. Requires the
`datasets` library (`pip install datasets`); HUGGINGFACE_TOKEN in .env is
only needed if you point this at a gated mirror instead.

Usage:
    python scripts/download_cord.py --out data/raw/cord
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CORD_HF_DATASET = "naver-clova-ix/cord-v2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/cord"))
    parser.add_argument("--dataset", type=str, default=CORD_HF_DATASET)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install the `datasets` package first: pip install datasets")
        return 1

    token = os.environ.get("HUGGINGFACE_TOKEN") or None
    ds = load_dataset(args.dataset, token=token)

    args.out.mkdir(parents=True, exist_ok=True)
    for split_name, split in ds.items():
        split_dir = args.out / split_name
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        for i, example in enumerate(split):
            example["image"].save(split_dir / "images" / f"{i:05d}.png")
            meta = {k: v for k, v in example.items() if k != "image"}
            (split_dir / f"{i:05d}.json").write_text(json.dumps(meta, default=str))

    print(
        f"Downloaded CORD into {args.out}. Next: python scripts/convert_cord.py --raw {args.out} --out data/processed/cord"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
