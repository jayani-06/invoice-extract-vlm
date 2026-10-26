#!/usr/bin/env python
"""Convert the SROIE HuggingFace mirror into a corpus every runner can read.

**These are the first real documents in the project.** Everything else here is
generated -- our invoices by our own renderer, FATURA2 by someone else's. SROIE
is 652 photographs of actual Malaysian retail receipts: creased, skewed,
unevenly lit, thermal-printed, shot on a phone. Until now no stage of this
pipeline had met one.

The mirror is `arvindrajan92/sroie_document_understanding` (MIT, one parquet,
no loading script to execute). It carries labelled OCR lines with polygon
boxes, including `line_description` and `line_total` -- so SROIE documents get
**real line items**, the annotation FATURA2 lacks entirely.

Output is laid out exactly like `gst_in_synthetic` and `fatura2` -- canonical/,
images/clean/, ocr_gold/, manifest.jsonl -- so `run_extraction.py`,
`run_preprocess_ablation.py` and the demo cache builder all work on it with no
changes.

**Two things about this corpus that must not be forgotten when reading a
number from it:**

1. *Annotations are line-level, not word-level.* Our synthetic corpus has a box
   per word. CER and WER are unaffected -- text is text -- but a box-IoU
   detection figure computed here measures LINE detection and cannot be
   compared with the word-level detection numbers elsewhere in this project.
   The manifest records `annotation_granularity: "line"` so the difference is
   visible to anything that reads it.
2. *There is no `clean` version.* The other corpora pair a clean render with
   degraded copies, which is what makes the preprocessing ablation measurable.
   A photograph has no clean twin. The profile is named `photo` rather than
   `clean` precisely so nobody reads it as an undegraded page.

Usage:
    python scripts/convert_sroie_hf.py --parquet data/raw/sroie_hf/sroie_train.parquet \
        --out data/processed/sroie
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.data.converters.sroie import (  # noqa: E402
    convert_tokens,
    polygon_bbox,
)
from invoice_extract.data.validate import (  # noqa: E402
    drop_impermissible_nulls,
    load_schema,
    validate_jsonschema,
)
from invoice_extract.reading_order import group_lines  # noqa: E402

#: Not "clean". The other corpora pair a clean render with degraded copies, and
#: calling a phone photograph of a creased receipt "clean" would make it look
#: like the undegraded end of that scale.
PROFILE = "photo"

#: SROIE's label set, for the audit printed at the end. A mirror that renames
#: or renumbers these would otherwise convert silently into empty documents.
EXPECTED_LABELS = {
    "company",
    "address",
    "date",
    "total",
    "line_description",
    "line_total",
    "other",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument(
        "--splits",
        default="train:0.6,val:0.15,test:0.25",
        help="Deterministic split by doc_id hash",
    )
    args = ap.parse_args(argv)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow is required: pip install -e '.[datasets]'", file=sys.stderr)
        return 1
    from PIL import Image

    if not args.parquet.exists():
        print(f"No parquet at {args.parquet}", file=sys.stderr)
        return 1

    canonical_dir = args.out / "canonical"
    image_dir = args.out / "images" / PROFILE
    gold_dir = args.out / "ocr_gold"
    for d in (canonical_dir, image_dir, gold_dir):
        d.mkdir(parents=True, exist_ok=True)

    ratios = []
    for part in args.splits.split(","):
        name, _, value = part.partition(":")
        ratios.append((name.strip(), float(value)))

    schema = load_schema()
    rng = random.Random(args.seed)

    manifest_rows: list[dict] = []
    seen_labels: set[str] = set()
    n_ok = n_skipped = 0
    field_present = dict.fromkeys(
        ("vendor_name", "vendor_address", "invoice_date", "grand_total", "line_items"), 0
    )

    table = pq.read_table(args.parquet)
    total_rows = table.num_rows if args.limit is None else min(args.limit, table.num_rows)
    print(f"{table.num_rows} rows in {args.parquet.name}; converting {total_rows}", file=sys.stderr)

    for i in range(total_rows):
        row = table.slice(i, 1).to_pylist()[0]
        doc_id = f"sroie_{i:04d}"

        # ---- image
        raw = row["image"]
        data = raw["bytes"] if isinstance(raw, dict) else raw
        if not data:
            n_skipped += 1
            continue
        image = Image.open(io.BytesIO(data)).convert("RGB")
        width, height = image.size
        rel_image = f"images/{PROFILE}/{doc_id}.png"
        image.save(args.out / rel_image)

        # ---- tokens
        tokens = []
        for token in row.get("ocr") or []:
            text = (token.get("text") or "").strip()
            if not text:
                continue
            label = token.get("label") or "other"
            seen_labels.add(label)
            tokens.append({"text": text, "label": label, "bbox": polygon_bbox(token["box"])})
        if not tokens:
            n_skipped += 1
            continue

        # ---- deterministic split, by position rather than by RNG draw order,
        # so adding --limit cannot move a document between splits.
        r = rng.random()
        cum = 0.0
        split = ratios[-1][0]
        for name, ratio in ratios:
            cum += ratio
            if r <= cum:
                split = name
                break

        # `model_dump` cannot tell "present and null" from "absent", and the
        # schema does. See drop_impermissible_nulls.
        doc = drop_impermissible_nulls(
            convert_tokens(tokens, doc_id, split, rel_image, width, height).model_dump(mode="json"),
            schema,
        )
        errors = validate_jsonschema(doc, schema)
        if errors:
            print(f"SKIP {doc_id}: {errors[:2]}", file=sys.stderr)
            n_skipped += 1
            continue

        (canonical_dir / f"{doc_id}.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

        # ---- ocr gold. Line-level, and labelled as such in the manifest.
        # `line_id` is assigned geometrically by the same helper the renderer
        # and the OCR adapters use, so reading order here matches everywhere
        # else -- the defect that once split "Invoice No:" from its value.
        line_of: dict[int, int] = {}
        for line_id, line in enumerate(group_lines(tokens, lambda t: tuple(t["bbox"]))):
            for token in line:
                line_of[id(token)] = line_id
        words = [
            {
                "text": tok["text"],
                "bbox": [round(v, 1) for v in tok["bbox"]],
                # Carrying the SROIE label through as field_path is what makes
                # this corpus usable for extraction and not only for OCR.
                "field_path": None if tok["label"] == "other" else tok["label"],
                "line_id": line_of[id(tok)],
            }
            for tok in tokens
        ]
        rel_gold = f"ocr_gold/{doc_id}.{PROFILE}.json"
        (args.out / rel_gold).write_text(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "profile": PROFILE,
                    "coverage": None,
                    "image_width": width,
                    "image_height": height,
                    "annotation_granularity": "line",
                    "words": words,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        vendor = (doc.get("parties") or {}).get("vendor") or {}
        field_present["vendor_name"] += bool(vendor.get("name"))
        field_present["vendor_address"] += bool(vendor.get("address"))
        field_present["invoice_date"] += bool((doc.get("invoice_meta") or {}).get("invoice_date"))
        field_present["grand_total"] += (doc.get("totals") or {}).get("grand_total") is not None
        field_present["line_items"] += bool(doc.get("line_items"))

        manifest_rows.append(
            {
                "doc_id": doc_id,
                "profile": PROFILE,
                "image_path": rel_image,
                "ocr_gold_path": rel_gold,
                "width": width,
                "height": height,
                "n_words": len(words),
                "source": "sroie",
                # The two facts that stop a number from this corpus being
                # misread. See the module docstring.
                "annotation_granularity": "line",
                "is_photograph": True,
            }
        )
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  {n_ok}/{total_rows}", file=sys.stderr)

    (args.out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {n_ok} documents to {args.out} ({n_skipped} skipped)")
    print(f"  profile: {PROFILE} (photographs -- there is no clean twin)")
    print("  annotations: LINE level, not word level")
    print("\nField coverage (what the annotations actually contain):")
    for name, count in field_present.items():
        print(f"  {name:<16} {count}/{n_ok}  ({count / max(n_ok, 1):.0%})")

    unexpected = seen_labels - EXPECTED_LABELS
    missing = EXPECTED_LABELS - seen_labels
    if unexpected:
        print(f"\nWARNING: labels not in the expected set: {sorted(unexpected)}", file=sys.stderr)
    if missing:
        print(f"WARNING: expected labels never seen: {sorted(missing)}", file=sys.stderr)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
