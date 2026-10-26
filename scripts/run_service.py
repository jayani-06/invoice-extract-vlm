#!/usr/bin/env python
"""Run the Phase 5 service: the API plus the reviewer page.

    python scripts/run_service.py                 # http://127.0.0.1:8000
    python scripts/run_service.py --port 9000

Then open http://127.0.0.1:8000/review for the review queue, or
http://127.0.0.1:8000/docs for the interactive API.

Configuration is by environment variable (all optional):

    IE_SERVICE_DIR   where the review database and uploaded images live
                     (default data/service)
    IE_CORPUS        corpus whose documents seed the history
                     (default data/processed/gst_in_synthetic)
    IE_LEDGER        Phase 3 ledger directory (default data/processed/ledger)
    IE_OCR_ENGINE    paddle | tesseract (default paddle)
    IE_OCR_FAST      1 = PP-OCRv5 mobile models, ~40 s/page on CPU (default 1)

Binds to 127.0.0.1 by default. There is no authentication (see
docs/limitations.md), so exposing it on a network is a deliberate choice that
needs `--host 0.0.0.0` and an auth layer in front of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            'The service extra is not installed: pip install -e ".[ocr,paddle,service]"',
            file=sys.stderr,
        )
        return 1

    from invoice_extract.service.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
