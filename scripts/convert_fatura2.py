#!/usr/bin/env python
"""Download and convert FATURA2 into the canonical schema + an image corpus.

FATURA2 (`mathieu1256/FATURA2-invoices`, CC-BY-4.0) is the third-party
benchmark this project has been missing: 10,000 synthetic invoices across 50
layout templates, from Limam et al. (arXiv:2311.11856). Every result before
this ran on invoices we rendered ourselves, so the layout-generalisation claim
had nothing independent behind it.

The original Zenodo release (zenodo.org/records/8261508) was returning 504s, so
this pulls the Hugging Face parquet mirror instead -- same images, same
annotations, one file, no login.

Writes the same layout as the synthetic corpus, so every downstream tool works
unchanged:
    canonical/<doc_id>.json
    images/clean/<doc_id>.png
    manifest.jsonl

Usage:
    python scripts/convert_fatura2.py --out data/processed/fatura2 --limit 200
    python scripts/convert_fatura2.py --audit-tags     # re-verify the tag map
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.converters.fatura2 import TAG_EVIDENCE, convert_record  # noqa: E402
from invoice_extract.data.validate import load_schema, validate_jsonschema  # noqa: E402

BASE = "https://huggingface.co/datasets/mathieu1256/FATURA2-invoices/resolve/main/data"
SPLIT_FILES = {
    "test": "test-00000-of-00001.parquet",
    "train": "train-00000-of-00001.parquet",
}


def download(split: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{split}.parquet"
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"  using cached {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
        return path
    url = f"{BASE}/{SPLIT_FILES[split]}"
    print(f"  downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, path)  # noqa: S310 - fixed https URL
    print(f"  saved {path} ({path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return path


def audit_tags(parquet_path: Path) -> int:
    """Re-derive the tag vocabulary from the data and check it against the map.

    The parquet ships bare integer `ner_tags` with no label names, so the
    mapping in `converters/fatura2.py` was inferred by inspection. This makes
    that inference re-runnable: if a future release renumbers the tags, the
    evidence tokens stop appearing and this fails loudly instead of the
    converter silently writing wrong fields.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    samples: dict[int, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    for group in range(pf.num_row_groups):
        table = pf.read_row_group(group, columns=["ner_tags", "tokens"])
        for tags, tokens in zip(
            table.column("ner_tags").to_pylist(), table.column("tokens").to_pylist(), strict=True
        ):
            for tag, token in zip(tags, tokens, strict=True):
                counts[tag] += 1
                if len(samples[tag]) < 12:
                    samples[tag].append(str(token))

    print(f"{'tag':>4} {'tokens':>8}  sample tokens")
    for tag in sorted(samples):
        print(f"{tag:>4} {counts[tag]:>8}  {samples[tag][:8]}")

    problems = []
    for tag, evidence in TAG_EVIDENCE.items():
        if not evidence:
            continue
        if tag not in samples:
            problems.append(f"tag {tag} is absent from the data entirely")
            continue
        seen = {s.lower() for s in samples[tag]}
        if not any(e.lower() in seen for e in evidence):
            problems.append(f"tag {tag}: expected one of {evidence}, saw {samples[tag][:6]}")

    print()
    if problems:
        print("TAG MAP AUDIT FAILED - the vocabulary has shifted:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1
    print("Tag map audit passed: every mapped tag still carries its expected label token.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", default="test", choices=sorted(SPLIT_FILES))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/fatura2"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/fatura2"))
    ap.add_argument("--limit", type=int, default=None, help="Convert only the first N documents")
    ap.add_argument("--audit-tags", action="store_true", help="Verify the tag map and exit")
    args = ap.parse_args(argv)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: pip install pyarrow", file=sys.stderr)
        return 2

    parquet_path = download(args.split, args.raw)
    if args.audit_tags:
        return audit_tags(parquet_path)

    from PIL import Image

    (args.out / "canonical").mkdir(parents=True, exist_ok=True)
    (args.out / "images" / "clean").mkdir(parents=True, exist_ok=True)
    (args.out / "ocr_gold").mkdir(parents=True, exist_ok=True)

    schema = load_schema()
    pf = pq.ParquetFile(parquet_path)
    rows: list[dict] = []
    n = 0
    invalid = 0
    missing: Counter = Counter()

    for group in range(pf.num_row_groups):
        table = pf.read_row_group(group, columns=["image", "ner_tags", "tokens", "bboxes", "id"])
        batch = table.to_pylist()
        for record in batch:
            if args.limit is not None and n >= args.limit:
                break
            doc_id = f"fatura2_{args.split}_{n:05d}"

            image_bytes = (record.get("image") or {}).get("bytes")
            if not image_bytes:
                continue
            image = Image.open(io.BytesIO(image_bytes)).convert("L")
            rel = f"images/clean/{doc_id}.png"
            image.save(args.out / rel)

            doc = convert_record(
                record,
                doc_id=doc_id,
                image_path=rel,
                width=image.width,
                height=image.height,
                split=args.split,
            )
            errors = validate_jsonschema(doc, schema)
            if errors:
                invalid += 1
                if invalid <= 3:
                    print(f"  {doc_id}: {errors[:2]}", file=sys.stderr)
            (args.out / "canonical" / f"{doc_id}.json").write_text(
                json.dumps(doc, indent=1), encoding="utf-8"
            )

            for field, value in (
                ("vendor.name", doc["parties"]["vendor"].get("name")),
                ("invoice_number", doc["invoice_meta"]["invoice_number"]),
                ("invoice_date", doc["invoice_meta"]["invoice_date"]),
                ("grand_total", doc["totals"]["grand_total"]),
            ):
                if value is None:
                    missing[field] += 1

            # FATURA2's own tokens + boxes double as OCR ground truth, giving a
            # perfect-recognition ceiling on a third-party benchmark without
            # running an engine. It is a PARTIAL page transcript, not a full
            # one: the line-item table is a single "table" placeholder, so the
            # table's text is simply absent. Fine for header-field extraction,
            # useless for CER -- hence the explicit `coverage` marker.
            ocr_rel = f"ocr_gold/{doc_id}.clean.json"
            (args.out / ocr_rel).write_text(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "profile": "clean",
                        "coverage": "annotated_regions_only",
                        "image_width": image.width,
                        "image_height": image.height,
                        "words": [
                            {
                                "text": str(tok),
                                "bbox": [float(v) for v in box],
                                "field_path": None,
                                "line_id": 0,
                            }
                            for tok, box in zip(record["tokens"], record["bboxes"], strict=True)
                            if str(tok) not in ("table", "logo")
                        ],
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )

            rows.append(
                {
                    "doc_id": doc_id,
                    "profile": "clean",
                    "image_path": rel,
                    "ocr_gold_path": ocr_rel,
                    "width": image.width,
                    "height": image.height,
                    "n_words": len(record["tokens"]),
                    "source": "fatura2",
                }
            )
            n += 1
        if args.limit is not None and n >= args.limit:
            break

    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nConverted {n} FATURA2 documents to {args.out}")
    print(f"  schema-invalid: {invalid}")
    print("  fields absent from the annotation (not extraction failures):")
    for field, count in missing.most_common():
        print(f"    {field:<16} {count}/{n} ({count / max(n, 1):.0%})")
    print(
        "\nNOTE: FATURA2 does not tokenise line items (tag 10 is one 'table' "
        "placeholder), so every document has line_items=[]. Score header fields only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
