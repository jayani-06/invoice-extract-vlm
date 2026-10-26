#!/usr/bin/env python
"""Package documents for a Colab VLM run.

Two modes, one payload format:

- `--cache demo_cache` (default) packages the handful of documents the demo
  screens show. Small, fast, for `build_demo_cache.py --vlm`.
- `--corpus data/processed/<name>` packages an evaluation slice. Repeatable, so
  a single upload can carry both corpora and one Colab session produces every
  number Phase 2 needs.

Either way the zip holds one image per document plus the exact prompt that
document should get. You upload it to `notebooks/vlm_eval.ipynb`, run it on a
T4, and bring back `vlm_outputs.json`.

The prompts are built here, not in the notebook, so the model sees byte-for-byte
what `models/prompting.py` produces. If the notebook rebuilt them, the
comparison would be against a prompt that no other part of the project uses.

In corpus mode a selection record is written next to the zip. Scoring the VLM
against the rules baseline is only meaningful on identical documents, and
`run_extraction.py --doc-ids` consumes that record to guarantee it.

Usage:
    python scripts/export_vlm_payload.py                      # demo payload
    python scripts/export_vlm_payload.py --with-ocr-hint      # hybrid mode

    python scripts/export_vlm_payload.py \
        --corpus data/processed/gst_in_synthetic \
        --corpus data/processed/fatura2 \
        --limit 50 --out vlm_eval_payload.zip
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
from invoice_extract.reading_order import reading_order_text  # noqa: E402

#: Split preference for corpus mode. Held-out documents are taken first so the
#: run always contains every test-split document the corpus has, and a
#: test-only subset can be scored separately afterwards. `gst_in_synthetic` has
#: only 17 test documents, so any run larger than that necessarily includes
#: train-split ones -- which is harmless here because neither the rules
#: baseline nor a zero-shot VLM is fitted on this corpus, but it has to be
#: reported rather than assumed.
SPLIT_ORDER = {"test": 0, "val": 1, "train": 2}


def load_manifest(corpus: Path, profile: str) -> list[dict]:
    path = corpus / "manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"No manifest at {path}. Build the image corpus first.")
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    matching = [r for r in rows if r["profile"] == profile]
    if not matching:
        have = sorted({r["profile"] for r in rows})
        raise SystemExit(f"No rows for profile {profile!r} in {corpus.name}; have {have}")
    return matching


def ocr_hint_for(corpus: Path, row: dict) -> str:
    gold = json.loads((corpus / row["ocr_gold_path"]).read_text(encoding="utf-8"))
    return reading_order_text(gold["words"], lambda w: tuple(w["bbox"]), lambda w: w["text"])


def documents_from_corpus(
    corpus: Path, profile: str, limit: int | None, with_ocr_hint: bool
) -> tuple[list[dict], list[tuple[Path, str]], dict]:
    rows = load_manifest(corpus, profile)

    splits: dict[str, str | None] = {}
    for row in rows:
        canonical = corpus / "canonical" / f"{row['doc_id']}.json"
        if canonical.exists():
            doc = json.loads(canonical.read_text(encoding="utf-8"))
            splits[row["doc_id"]] = (doc.get("source") or {}).get("split")

    # Stable: split preference first, manifest order within a split. No RNG, so
    # the same slice comes back on every machine and every rerun.
    rows.sort(key=lambda r: SPLIT_ORDER.get(splits.get(r["doc_id"]) or "", 3))
    if limit:
        rows = rows[:limit]

    documents: list[dict] = []
    files: list[tuple[Path, str]] = []
    for row in rows:
        source = corpus / row["image_path"]
        if not source.exists():
            continue
        arc = f"images/{corpus.name}/{row['doc_id']}.png"
        files.append((source, arc))
        documents.append(
            {
                "doc_id": row["doc_id"],
                "corpus": corpus.name,
                "split": splits.get(row["doc_id"]),
                "image": arc,
                "prompt": build_prompt(ocr_hint_for(corpus, row) if with_ocr_hint else None),
            }
        )

    composition: dict[str, int] = {}
    for d in documents:
        key = d["split"] or "unknown"
        composition[key] = composition.get(key, 0) + 1
    selection = {
        "corpus": corpus.name,
        "corpus_path": str(corpus),
        "profile": profile,
        "n": len(documents),
        "split_composition": composition,
        "doc_ids": [d["doc_id"] for d in documents],
    }
    return documents, files, selection


def documents_from_cache(
    cache: Path, profile: str, with_ocr_hint: bool
) -> tuple[list[dict], list[tuple[Path, str]]]:
    index_path = cache / "index.json"
    if not index_path.exists():
        raise SystemExit(f"No demo cache at {cache}. Run `make demo-cache` first.")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    documents: list[dict] = []
    files: list[tuple[Path, str]] = []
    for entry in index["documents"]:
        doc = json.loads((cache / "docs" / f"{entry['doc_id']}.json").read_text(encoding="utf-8"))
        rel = doc["images"].get(profile) or doc["images"].get(doc["profiles"][0])
        if not rel:
            continue
        source = cache / rel
        if not source.exists():
            continue

        arc = f"images/{entry['doc_id']}.png"
        files.append((source, arc))
        documents.append(
            {
                "doc_id": entry["doc_id"],
                "corpus": entry["corpus"],
                "image": arc,
                "prompt": build_prompt(doc.get("gold_text") if with_ocr_hint else None),
            }
        )
    return documents, files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, default=Path("demo_cache"))
    ap.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=[],
        help="Evaluation mode: package a slice of this corpus. Repeatable.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Documents per corpus")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(MODEL_PRESETS))
    ap.add_argument(
        "--with-ocr-hint",
        action="store_true",
        help="Hybrid mode: include the OCR transcript in the prompt. Off by default "
        "so the run shows the model reading pixels rather than re-reading OCR.",
    )
    ap.add_argument(
        "--profile",
        default="clean",
        help="Which page image to send. 'clean' shows the model at its best; "
        "'medium' or 'heavy' shows it against scan damage.",
    )
    args = ap.parse_args(argv)

    out = args.out or Path("vlm_eval_payload.zip" if args.corpus else "vlm_payload.zip")

    documents: list[dict] = []
    files: list[tuple[Path, str]] = []
    selections: list[dict] = []

    if args.corpus:
        for corpus in args.corpus:
            docs, fs, selection = documents_from_corpus(
                corpus, args.profile, args.limit, args.with_ocr_hint
            )
            documents += docs
            files += fs
            selections.append(selection)
    else:
        documents, files = documents_from_cache(args.cache, args.profile, args.with_ocr_hint)

    if not documents:
        print("No images to export.", file=sys.stderr)
        return 1

    manifest = {
        "model_id": MODEL_PRESETS[args.preset].model_id,
        "preset": args.preset,
        "max_new_tokens": MODEL_PRESETS[args.preset].max_new_tokens,
        "four_bit": MODEL_PRESETS[args.preset].four_bit,
        "system_prompt": SYSTEM_PROMPT,
        "ocr_hint": args.with_ocr_hint,
        "profile": args.profile,
        "documents": documents,
    }

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for source, arc in files:
            z.write(source, arc)

    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out} ({size_mb:.1f} MB) with {len(documents)} documents")
    print(f"  model : {manifest['model_id']}")
    print(f"  mode  : {'hybrid (image + OCR text)' if args.with_ocr_hint else 'vision only'}")
    print(f"  image : {args.profile}")

    if selections:
        record = out.with_suffix(".selection.json")
        record.write_text(json.dumps({"selections": selections}, indent=2), encoding="utf-8")
        print(f"  split : {record}")
        for s in selections:
            print(f"    {s['corpus']}: {s['n']} docs, {s['split_composition']}")
        print(
            "\nScore the baseline on the SAME documents, or the comparison is "
            f"between two different test sets:\n"
            f"  python scripts/run_extraction.py --corpus <path> --extractor rules "
            f"--ocr-source gold --doc-ids {record} --out reports/phase2/<name>"
        )

    print(f"\nNext: upload {out} to notebooks/vlm_eval.ipynb on a Colab T4 GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
