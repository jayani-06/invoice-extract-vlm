#!/usr/bin/env python
"""Time the Phase 5 service end to end with a real OCR engine.

Exit criterion 1 of docs/phase5_service.md: a single POST returns a complete
result in under 10 s on CPU with fast models, or the job endpoint returns a
handle immediately. This drives the real app (create_app + FastAPI's
TestClient, so no network stack is involved) with real PaddleOCR on held-out
corpus pages, and prints per-stage timings and routing for each.

    python scripts/time_service.py --pages 6
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/gst_in_synthetic"))
    ap.add_argument("--ledger", type=Path, default=Path("data/processed/ledger"))
    ap.add_argument("--pages", type=int, default=6)
    ap.add_argument("--profiles", nargs="+", default=["clean", "medium"])
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    from fastapi.testclient import TestClient

    from invoice_extract.ocr.base import get_engine
    from invoice_extract.service.api import create_app
    from invoice_extract.service.knowledge import Knowledge
    from invoice_extract.service.review import ReviewService
    from invoice_extract.service.store import ReviewStore

    tmp = Path(tempfile.mkdtemp(prefix="ie_svc_"))
    knowledge = Knowledge.bootstrap(args.corpus, args.ledger)
    print(
        f"knowledge: history {len(knowledge.history)}, ledger {len(knowledge.entries)}, "
        f"vendors {len(knowledge.master)}",
        flush=True,
    )
    t = time.perf_counter()
    engine = get_engine("paddle", fast=True)
    print(f"engine load {time.perf_counter() - t:.1f}s", flush=True)

    service = ReviewService(
        ReviewStore(tmp / "review.sqlite3"), knowledge, lambda: engine, image_dir=tmp / "images"
    )
    client = TestClient(create_app(service))

    manifest = args.corpus / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    seen = {d["doc_id"] for d in knowledge.history}
    picks = [r for r in rows if r["doc_id"] not in seen and r["profile"] in args.profiles]
    picks = picks[: args.pages]

    for row in picks:
        data = (args.corpus / row["image_path"]).read_bytes()
        t = time.perf_counter()
        resp = client.post(
            "/documents", files={"file": (Path(row["image_path"]).name, data, "image/png")}
        )
        elapsed = time.perf_counter() - t
        body = resp.json()
        if resp.status_code != 200:
            print(f"{row['doc_id']} {row['profile']} HTTP {resp.status_code}: {body}")
            continue
        print(
            f"SYNC {row['doc_id']} {row['profile']:6s} {elapsed:6.1f}s "
            f"verdict={body['overall_verdict']} status={body['status']} "
            f"timings={body['timings_ms']} words={body['ocr']['n_words']} "
            f"reasons={body['review_reasons'][:2]}",
            flush=True,
        )

    if picks:
        data = (args.corpus / picks[0]["image_path"]).read_bytes()
        t = time.perf_counter()
        resp = client.post("/jobs", files={"file": ("job.png", data, "image/png")})
        print(f"JOB handle HTTP {resp.status_code} in {time.perf_counter() - t:.3f}s", flush=True)
        url = resp.json()["status_url"]
        while True:
            status = client.get(url).json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(1)
        print(f"JOB {status['status']} after {time.perf_counter() - t:.1f}s {status.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
