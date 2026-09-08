#!/usr/bin/env python
"""Thin CLI wrapper: validate every JSON file under schema/examples/.

Usage:
    python scripts/validate_examples.py                # validate schema/examples/*.json
    python scripts/validate_examples.py path/to/doc.json ...   # validate specific files
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.validate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
