#!/usr/bin/env python
"""Download the FATURA dataset (Limam et al., ~10k invoices / ~50 layouts).

FATURA is a research dataset whose canonical hosting location has moved in
the past (project page / Zenodo / community Kaggle mirrors) — confirm the
current URL before relying on this script, then fill in FATURA_SOURCE_URL
(direct zip/tar) or FATURA_KAGGLE_SLUG (if using a Kaggle mirror) below.

This script is intentionally a thin, documented template rather than a
best-effort scraper: pointing it at the wrong mirror silently gives you the
wrong data, which is worse than a script that makes you confirm the source
once.

Usage:
    python scripts/download_fatura.py --out data/raw/fatura
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# --- fill these in after confirming the current dataset location ---
FATURA_SOURCE_URL: str | None = None  # e.g. "https://zenodo.org/records/<id>/files/FATURA.zip"
FATURA_KAGGLE_SLUG: str | None = None  # e.g. "someuser/fatura-invoice-dataset"
# ---------------------------------------------------------------------


def download_via_url(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "fatura_download" / Path(url).name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(archive_path), url], check=True)
    subprocess.run(
        [
            "python",
            "-m",
            "zipfile" if url.endswith(".zip") else "tarfile",
            "-e" if url.endswith(".zip") else "-x",
            str(archive_path),
            str(out_dir),
        ],
        check=False,
    )
    return out_dir


def download_via_kaggle(slug: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Requires `pip install kaggle` and KAGGLE_USERNAME/KAGGLE_KEY in the environment (.env).
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"], check=True
    )
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/fatura"))
    parser.add_argument(
        "--url", type=str, default=FATURA_SOURCE_URL, help="Override FATURA_SOURCE_URL"
    )
    parser.add_argument(
        "--kaggle-slug", type=str, default=FATURA_KAGGLE_SLUG, help="Override FATURA_KAGGLE_SLUG"
    )
    args = parser.parse_args()

    if args.url:
        download_via_url(args.url, args.out)
    elif args.kaggle_slug:
        download_via_kaggle(args.kaggle_slug, args.out)
    else:
        print(
            "No source configured yet.\n"
            "1. Confirm FATURA's current hosting location (project page / Zenodo / Kaggle mirror).\n"
            "2. Set FATURA_SOURCE_URL or FATURA_KAGGLE_SLUG at the top of this script "
            "(or pass --url / --kaggle-slug), then re-run.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Downloaded FATURA into {args.out}. Next: python scripts/convert_fatura.py --raw {args.out} --out data/processed/fatura"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
