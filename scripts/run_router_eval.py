#!/usr/bin/env python
"""Fit the per-field router on a training split, score it on a held-out one.

The split is the whole point. A router fitted on the documents it is scored
against will always beat both extractors, because it is choosing with the
answers in hand -- that is an oracle, not a system. So the policy is learned
on one set of documents and reported on another it has never seen.

Reads the `pred.jsonl` / `gold.jsonl` that `run_extraction.py` already writes,
so no model is re-run: the inputs are the same end-to-end predictions quoted
everywhere else.

Usage:
    python scripts/run_router_eval.py \
        --corpus gst_in_synthetic \
        --extractor rules=reports/phase2/rules_e2e_gst_in_synthetic \
        --extractor vlm=reports/phase2/vlm_gst_in_synthetic \
        --default rules --out reports/phase2/router_gst
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.eval.metrics import aggregate, compare_documents  # noqa: E402
from invoice_extract.models.router import fit, route  # noqa: E402

HEADER_FIELDS = [
    "parties.vendor.name",
    "parties.vendor.gstin",
    "invoice_meta.invoice_number",
    "invoice_meta.invoice_date",
    "totals.grand_total",
    "totals.subtotal",
]


def read_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            out[doc["doc_id"]] = doc
    return out


def fmt(v: float, places: int = 3) -> str:
    return "n/a" if v != v else f"{v:.{places}f}"


def score(gold: dict[str, dict], preds: dict[str, dict]) -> dict:
    empty = {"doc_id": "", "line_items": [], "tax_lines": []}
    agg = aggregate([compare_documents(gold[d], preds.get(d, empty)) for d in gold])
    header = [
        agg.per_field[f].accuracy
        for f in HEADER_FIELDS
        if f in agg.per_field and agg.per_field[f].accuracy == agg.per_field[f].accuracy
    ]
    return {
        "n": agg.n_docs,
        "header_accuracy": sum(header) / len(header) if header else float("nan"),
        "critical_fields": agg.doc_critical_accuracy,
        "macro_field_accuracy": agg.macro_field_accuracy,
        "per_field": {f: r.accuracy for f, r in agg.per_field.items()},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", required=True, help="Corpus name, for the report")
    ap.add_argument(
        "--extractor",
        action="append",
        required=True,
        metavar="NAME=RUNDIR",
        help="A run directory written by run_extraction.py. Repeatable.",
    )
    ap.add_argument("--default", required=True, help="Extractor supplying unrouted fields")
    ap.add_argument(
        "--fit-split",
        default="train,val",
        help="source.split values used to FIT. Everything else is held out.",
    )
    ap.add_argument("--min-margin", type=float, default=0.05)
    ap.add_argument(
        "--fit-fraction",
        type=float,
        default=0.66,
        help="Used only when the corpus has no usable source.split.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    runs: dict[str, Path] = {}
    for spec in args.extractor:
        name, _, path = spec.partition("=")
        runs[name] = Path(path)
    if args.default not in runs:
        print(f"--default {args.default!r} is not among {sorted(runs)}", file=sys.stderr)
        return 2

    gold = read_jsonl(next(iter(runs.values())) / "gold.jsonl")
    by_extractor = {name: read_jsonl(p / "pred.jsonl") for name, p in runs.items()}

    fit_splits = {s.strip() for s in args.fit_split.split(",")}
    fit_ids = {d for d, g in gold.items() if (g.get("source") or {}).get("split") in fit_splits}
    test_ids = set(gold) - fit_ids

    if not fit_ids or not test_ids:
        # FATURA2 labels every document `test`, so there is no split to fit on.
        # Fall back to a deterministic positional division by doc_id. It is an
        # arbitrary boundary, but it is still a genuine hold-out: the policy
        # never sees the documents it is scored on, which is the only property
        # that matters here.
        present = sorted({(g.get("source") or {}).get("split") for g in gold.values()})
        ordered = sorted(gold)
        cut = int(len(ordered) * args.fit_fraction)
        fit_ids, test_ids = set(ordered[:cut]), set(ordered[cut:])
        print(
            f"No usable split (corpus has {present}); falling back to the first "
            f"{len(fit_ids)} doc_ids to fit and the remaining {len(test_ids)} to score.",
            file=sys.stderr,
        )
        if not fit_ids or not test_ids:
            print("Not enough documents to split.", file=sys.stderr)
            return 1

    policy = fit(
        {d: gold[d] for d in fit_ids},
        {n: {d: p[d] for d in fit_ids if d in p} for n, p in by_extractor.items()},
        default=args.default,
        min_margin=args.min_margin,
    )

    # Scored only on documents the policy has never seen.
    held_gold = {d: gold[d] for d in test_ids}
    routed = {d: route(policy, by_extractor, d) for d in test_ids}

    results = {name: score(held_gold, preds) for name, preds in by_extractor.items()}
    results["router"] = score(held_gold, routed)

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": args.corpus,
        "fit_documents": len(fit_ids),
        "held_out_documents": len(test_ids),
        "fit_splits": sorted(fit_splits),
        "min_margin": args.min_margin,
        "policy": policy.as_dict(),
        "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    policy.save(args.out / "policy.json")

    names = [*runs, "router"]
    md = [
        f"# Per-field router — {args.corpus}",
        "",
        f"Policy fitted on **{len(fit_ids)}** documents ({', '.join(sorted(fit_splits))}), "
        f"scored on **{len(test_ids)}** held out.",
        "",
        "| Extractor | Header acc | Critical-fields | Macro field acc |",
        "|---|---|---|---|",
    ]
    for name in names:
        r = results[name]
        bold = "**" if name == "router" else ""
        md.append(
            f"| {bold}`{name}`{bold} | {bold}{fmt(r['header_accuracy'])}{bold} "
            f"| {fmt(r['critical_fields'])} | {fmt(r['macro_field_accuracy'])} |"
        )

    md += ["", "## What the policy learned", ""]
    if policy.choices:
        md += ["| Field | Taken from | Margin on the fit split | n |", "|---|---|---|---|"]
        for path, c in sorted(policy.choices.items()):
            md.append(f"| `{path}` | `{c.winner}` | {c.margin:+.3f} | {c.n} |")
    else:
        md.append(
            f"Nothing cleared the {args.min_margin:.0%} margin, so every field "
            f"comes from `{args.default}`. The router is then exactly that "
            "extractor, which is the honest outcome when the two do not differ."
        )
    (args.out / "router.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\n=== {args.corpus}: fitted on {len(fit_ids)}, scored on {len(test_ids)} held out ===")
    print(f"{'extractor':<12}{'header':>10}{'critical':>10}{'macro':>10}")
    for name in names:
        r = results[name]
        print(
            f"{name:<12}{fmt(r['header_accuracy']):>10}"
            f"{fmt(r['critical_fields']):>10}{fmt(r['macro_field_accuracy']):>10}"
        )
    print(f"\nrouted fields: {len(policy.choices)}")
    for path, c in sorted(policy.choices.items()):
        print(f"  {path:<34} -> {c.winner:<8} (+{c.margin:.3f} on fit)")
    print(f"\nWrote {args.out}/router.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
