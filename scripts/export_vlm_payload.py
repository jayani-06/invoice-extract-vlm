#!/usr/bin/env python
"""Package the demo documents for a Colab VLM run.

Writes `vlm_payload.zip` containing one image per cached demo document plus the
exact prompt each one should get. You upload that to
`notebooks/vlm_demo_outputs.ipynb`, run it on a T4, and bring back
`vlm_outputs.json`.

The prompts are built here, not in the notebook, so the model sees byte-for-byte
what `models/prompting.py` produces. If the notebook rebuilt them, the demo
comparison would be against a prompt that no other part of the project uses.

Usage:
    python scripts/export_vlm_payload.py
    python scripts/export_vlm_payload.py --with-ocr-hint   # hybrid mode
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.models.prompting import SYSTEM_PROMPT, build_prompt  # noqa: E402
from invoice_extract.models.vlm import DEFAULT_PRESET, MODEL_PRESETS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, default=Path("demo_cache"))
    ap.add_argument("--out", type=Path, default=Path("vlm_payload.zip"))
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(MODEL_PRESETS))
    ap.add_argument(
        "--with-ocr-hint",
        action="store_true",
        help="Hybrid mode: include the OCR transcript in the prompt. Off by default "
        "so the demo shows the model reading pixels rather than re-reading OCR.",
    )
    ap.add_argument(
        "--profile",
        default="clean",
        help="Which page image to send. 'clean' shows the model at its best; "
        "'medium' or 'heavy' shows it against scan damage.",
    )
    args = ap.parse_args(argv)

    index_path = args.cache / "index.json"
    if not index_path.exists():
        print(f"No demo cache at {args.cache}. Run `make demo-cache` first.", file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text(encoding="utf-8"))

    documents = []
    files: list[tuple[Path, str]] = []

    for entry in index["documents"]:
        doc = json.loads(
            (args.cache / "docs" / f"{entry['doc_id']}.json").read_text(encoding="utf-8")
        )
        rel = doc["images"].get(args.profile) or doc["images"].get(doc["profiles"][0])
        if not rel:
            continue
        source = args.cache / rel
        if not source.exists():
            continue

        arc = f"images/{entry['doc_id']}.png"
        files.append((source, arc))
        documents.append(
            {
                "doc_id": entry["doc_id"],
                "corpus": entry["corpus"],
                "image": arc,
                "prompt": build_prompt(doc.get("gold_text") if args.with_ocr_hint else None),
            }
        )

    if not documents:
        print("No cached images to export.", file=sys.stderr)
        return 1

    manifest = {
        "model_id": MODEL_PRESETS[args.preset].model_id,
        "preset": args.preset,
        "system_prompt": SYSTEM_PROMPT,
        "ocr_hint": args.with_ocr_hint,
        "profile": args.profile,
        "documents": documents,
    }

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for source, arc in files:
            z.write(source, arc)

    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out} ({size_mb:.1f} MB) with {len(documents)} documents")
    print(f"  model : {manifest['model_id']}")
    print(f"  mode  : {'hybrid (image + OCR text)' if args.with_ocr_hint else 'vision only'}")
    print(f"  image : {args.profile}")
    print("\nNext: upload it to notebooks/vlm_demo_outputs.ipynb on a Colab T4 GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
