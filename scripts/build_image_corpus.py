#!/usr/bin/env python
"""Turn canonical-schema JSON documents into a paired image corpus.

For every input document this writes:

  images/clean/<doc_id>.png        the pristine render
  images/<profile>/<doc_id>.png    one degraded scan per severity profile
  ocr_gold/<doc_id>.<profile>.json word-level ground truth in *that image's*
                                   coordinate space
  canonical/<doc_id>.json          the input doc, updated in place with real
                                   page dimensions, image_path and grounding
  manifest.jsonl                   one row per (doc, profile) with the layout
                                   style and the degradations actually applied

The paired clean/degraded structure is the point. A real scan corpus can only
tell you the OCR was wrong; a paired one tells you *which* degradation cost
you the characters, because the same document exists at every severity with
identical ground truth.

Usage:
    python scripts/build_image_corpus.py \
        --canonical data/processed/gst_in_synthetic/canonical \
        --out data/processed/gst_in_synthetic \
        --dpi 200 --profiles clean light medium heavy --limit 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.degrade.transforms import PROFILES, degrade  # noqa: E402
from invoice_extract.render.invoice_renderer import InvoiceRenderer  # noqa: E402


def write_ocr_gold(path: Path, doc_id: str, profile: str, words, size, applied) -> None:
    path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "profile": profile,
                "image_width": size[0],
                "image_height": size[1],
                "degradations_applied": applied,
                "words": [
                    {
                        "text": w["text"],
                        "bbox": [round(v, 2) for v in w["bbox"]],
                        "field_path": w["field_path"],
                        "line_id": w["line_id"],
                    }
                    for w in words
                ],
            },
            indent=1,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--canonical",
        type=Path,
        required=True,
        help="Directory of canonical-schema *.json documents",
    )
    ap.add_argument("--out", type=Path, required=True, help="Corpus root to write into")
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Render resolution. 200 balances realism against corpus size; "
        "the plan's ingestion target is 300",
    )
    ap.add_argument(
        "--profiles",
        nargs="+",
        default=["clean", "light", "medium", "heavy"],
        choices=sorted(PROFILES),
    )
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N documents")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--quality", type=int, default=92, help="PNG is lossless; used for JPEG output only"
    )
    ap.add_argument("--format", default="png", choices=["png", "jpg"])
    args = ap.parse_args(argv)

    docs = sorted(args.canonical.glob("*.json"))
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        print(f"No canonical documents found in {args.canonical}", file=sys.stderr)
        return 1

    renderer = InvoiceRenderer(dpi=args.dpi)
    out_root: Path = args.out
    (out_root / "ocr_gold").mkdir(parents=True, exist_ok=True)
    for profile in args.profiles:
        (out_root / "images" / profile).mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    rows: list[dict] = []
    pruned_fields = 0

    for i, doc_path in enumerate(docs):
        doc = json.loads(doc_path.read_text())
        doc_id = doc["doc_id"]
        # Seed per document, not globally, so adding documents later does not
        # change the rendering of existing ones.
        seed = args.seed + hash(doc_id) % 10_000_000

        render = renderer.render(doc, seed=seed)
        clean = np.array(render.image)
        gold_words = [
            {"text": w.text, "bbox": w.bbox, "field_path": w.field_path, "line_id": w.line_id}
            for w in render.words
        ]

        for profile_name in args.profiles:
            profile = PROFILES[profile_name]
            rng = random.Random(seed + hash(profile_name) % 100_000)
            result = degrade(clean, profile, rng)

            mapped = result.map_boxes([w["bbox"] for w in gold_words])
            words = [{**w, "bbox": b} for w, b in zip(gold_words, mapped, strict=True)]

            ext = "png" if args.format == "png" else "jpg"
            img_path = out_root / "images" / profile_name / f"{doc_id}.{ext}"
            pil = Image.fromarray(result.image)
            if ext == "jpg":
                pil.save(img_path, quality=args.quality)
            else:
                pil.save(img_path)

            write_ocr_gold(
                out_root / "ocr_gold" / f"{doc_id}.{profile_name}.json",
                doc_id,
                profile_name,
                words,
                result.image.shape[::-1],
                result.applied,
            )

            rows.append(
                {
                    "doc_id": doc_id,
                    "profile": profile_name,
                    "image_path": str(img_path.relative_to(out_root)).replace("\\", "/"),
                    "ocr_gold_path": f"ocr_gold/{doc_id}.{profile_name}.json",
                    "width": int(result.image.shape[1]),
                    "height": int(result.image.shape[0]),
                    "dpi": args.dpi,
                    "n_words": len(words),
                    "degradations_applied": result.applied,
                    "layout_style": render.style.as_dict(),
                }
            )

        # Gold must describe the image. The generator emits party fields the
        # renderer's compact buyer block never draws (buyer phone/email, for
        # instance), and an annotation containing values that appear nowhere on
        # the page is unextractable by construction -- it would charge every
        # model, forever, for not hallucinating. Anything the renderer did not
        # ground is therefore dropped from the annotation.
        grounded = {g["field_path"] for g in render.grounding}
        for role, party in list((doc.get("parties") or {}).items()):
            if not isinstance(party, dict):
                continue
            for key in list(party):
                if party[key] is not None and f"parties.{role}.{key}" not in grounded:
                    party[key] = None
                    pruned_fields += 1

        # Point the canonical document at its clean render and attach the
        # renderer's grounding, so Phase 2 has real boxes to train/evaluate on.
        clean_rel = f"images/clean/{doc_id}.{'png' if args.format == 'png' else 'jpg'}"
        doc["pages"] = [
            {
                "page_index": 0,
                "image_path": clean_rel,
                "width": int(clean.shape[1]),
                "height": int(clean.shape[0]),
                "dpi": args.dpi,
            }
        ]
        doc["grounding"] = render.grounding
        doc_path.write_text(json.dumps(doc, indent=1))

        if (i + 1) % 25 == 0 or i + 1 == len(docs):
            print(f"  {i + 1}/{len(docs)} documents rendered", file=sys.stderr)

    with manifest_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nWrote {len(rows)} images across {len(args.profiles)} profile(s) to {out_root}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
