#!/usr/bin/env python
"""Download SROIE (ICDAR 2019 receipt OCR/key-info-extraction task) via Kaggle.

Requires a free Kaggle account and API token: create one at
kaggle.com/settings -> API -> "Create New Token", then set
KAGGLE_USERNAME / KAGGLE_KEY in .env (see .env.example).

Usage:
    python scripts/download_sroie.py --out data/raw/sroie
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

# Community Kaggle mirror of SROIE (task 1+3: OCR + key info extraction).
# Verify this is still the best-maintained mirror before running at scale.
SROIE_KAGGLE_SLUG = "urbikn/sroie-datasetv2"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/sroie"))
    parser.add_argument("--kaggle-slug", type=str, default=SROIE_KAGGLE_SLUG)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", args.kaggle_slug, "-p", str(args.out), "--unzip"],
        check=True,
    )
    print(
        f"Downloaded SROIE into {args.out}. Next: python scripts/convert_sroie.py --raw {args.out} --out data/processed/sroie"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
