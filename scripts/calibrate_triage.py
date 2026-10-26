#!/usr/bin/env python
"""Fit the triage thresholds on the train split, and report how well they work.

The paired corpus is what makes this possible: every page exists at four
degradation levels and the manifest records which, so the true answer to "does
this page need preprocessing" is known for calibration even though it is
unknowable at inference time.

**Only the train split is used.** Fitting on the evaluation pages and then
reporting accuracy on them would make the gates look better than they are, in
exactly the way the rest of this project is careful to avoid.

For each signal the threshold is the midpoint between the clean mean and the
degraded mean, which is the error-minimising boundary when the two spreads are
similar. The report prints each signal's separation (the gap between the two
means in units of their pooled standard deviation) so a gate that is really a
coin flip is visible as one rather than hiding behind a plausible number.

Usage:
    python scripts/calibrate_triage.py --corpus data/processed/gst_in_synthetic
    python scripts/calibrate_triage.py --apply    # rewrite DEFAULT_THRESHOLDS
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invoice_extract.preprocess.triage import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    assess,
    triage,
)

#: Which direction means "degraded" for each signal. `laplacian_variance` and
#: `text_height_px` go the other way -- a blurry or downscaled page has LOWER
#: values -- and getting that backwards would invert the gate while still
#: producing a threshold that looks reasonable.
HIGHER_IS_WORSE = {
    "skew_deg": True,
    "border_darkness": True,
    "illumination_spread": True,
    "noise_sigma": True,
    "laplacian_variance": False,
    "text_height_px": False,
}

#: The stage each signal gates, for the report.
SIGNAL_STAGE = {
    "skew_deg": "deskew",
    "border_darkness": "remove_border",
    "illumination_spread": "normalize_illumination",
    "noise_sigma": "denoise",
    "laplacian_variance": "upscale",
    "text_height_px": "upscale",
}

#: Which recorded degradations each signal is supposed to detect.
#:
#: This is what the signals are scored against, rather than the profile label.
#: Degradations are applied probabilistically, so a `heavy` page may carry no
#: rotation at all -- and such a page genuinely does not need deskewing.
#: Scoring every signal against "was this page degraded in any way" therefore
#: punishes a gate for correctly declining, and the first version of this
#: script did exactly that: it reported the skew gate at 0.583 accuracy when
#: most of its apparent errors were pages that were never rotated.
#:
#: The manifest records `degradations_applied` per page, so the honest question
#: is available: given that THIS degradation was applied, does THIS signal see
#: it?
#: A gate below this separation is dropped rather than wired up weakly.
#: 0.8 SD is roughly where two normal distributions stop being separable
#: enough for a single threshold to beat guessing by a useful margin.
MIN_SEPARATION_SD = 0.8

SIGNAL_TARGETS = {
    "skew_deg": ("rotate", "perspective"),
    "border_darkness": ("scan_border",),
    "illumination_spread": ("illumination", "shadow"),
    "noise_sigma": ("gaussian_noise", "salt_pepper", "paper_texture"),
    "laplacian_variance": ("blur", "motion_blur"),
    "text_height_px": ("rescale",),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/processed/gst_in_synthetic"))
    ap.add_argument("--limit", type=int, default=40, help="Documents per profile")
    ap.add_argument("--out", type=Path, default=Path("reports/phase1_triage"))
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite DEFAULT_THRESHOLDS in preprocess/triage.py with the fitted values",
    )
    args = ap.parse_args(argv)

    manifest_path = args.corpus / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}. Build the image corpus first.", file=sys.stderr)
        return 1
    rows = [json.loads(ln) for ln in manifest_path.read_text().splitlines() if ln.strip()]

    splits: dict[str, str] = {}
    for path in (args.corpus / "canonical").glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        splits[doc["doc_id"]] = (doc.get("source") or {}).get("split") or "unknown"

    train = [r for r in rows if splits.get(r["doc_id"]) == "train"]
    if not train:
        print("No train-split pages in the manifest; nothing to calibrate on.", file=sys.stderr)
        return 1

    by_profile: dict[str, list[dict]] = defaultdict(list)
    for row in train:
        if len(by_profile[row["profile"]]) < args.limit:
            by_profile[row["profile"]].append(row)

    print(
        "Calibrating on the TRAIN split only: "
        + ", ".join(f"{p}={len(v)}" for p, v in sorted(by_profile.items())),
        file=sys.stderr,
    )

    pages: list[dict] = []
    for profile, profile_rows in sorted(by_profile.items()):
        for i, row in enumerate(profile_rows, 1):
            image = np.array(Image.open(args.corpus / row["image_path"]).convert("L"))
            pages.append(
                {
                    "profile": profile,
                    "applied": set(row.get("degradations_applied") or []),
                    **assess(image).as_dict(),
                }
            )
            if i % 20 == 0 or i == len(profile_rows):
                print(f"  {profile}: {i}/{len(profile_rows)}", file=sys.stderr)

    fitted: dict[str, float] = {}
    report_rows = []
    dropped: list[str] = []
    for signal in DEFAULT_THRESHOLDS:
        targets = set(SIGNAL_TARGETS[signal])
        with_it = [p[signal] for p in pages if p["applied"] & targets and p[signal] == p[signal]]
        without = [
            p[signal] for p in pages if not (p["applied"] & targets) and p[signal] == p[signal]
        ]
        if not with_it or not without:
            continue
        a, b = np.array(without), np.array(with_it)
        threshold = float((a.mean() + b.mean()) / 2)
        pooled = float(np.sqrt((a.var() + b.var()) / 2)) or 1e-9
        separation = abs(float(b.mean() - a.mean())) / pooled
        higher_worse = HIGHER_IS_WORSE[signal]
        correct = (
            ((b > threshold).sum() + (a <= threshold).sum())
            if higher_worse
            else ((b < threshold).sum() + (a >= threshold).sum())
        )
        accuracy = float(correct) / (len(a) + len(b))
        # A signal that cannot see the degradation it exists to detect is not a
        # weak gate, it is noise, and wiring it up makes the policy worse than
        # having no gate at all. `text_height_px` was exactly this: 0.17 SD of
        # separation, 0.375 accuracy, and it was single-handedly firing
        # `upscale` on 19 of 30 clean pages. Dropped rather than kept with a
        # caveat.
        if separation < MIN_SEPARATION_SD:
            dropped.append(signal)
        else:
            fitted[signal] = round(threshold, 4 if threshold < 10 else 1)
        report_rows.append(
            {
                "signal": signal,
                "stage": SIGNAL_STAGE[signal],
                "targets": sorted(targets),
                "without_mean": round(float(a.mean()), 4),
                "with_mean": round(float(b.mean()), 4),
                "n_with": len(b),
                "n_without": len(a),
                "threshold": round(threshold, 4 if threshold < 10 else 1),
                "separation_sd": round(separation, 2),
                "gate_accuracy": round(accuracy, 3),
                "direction": "higher is worse" if higher_worse else "lower is worse",
                "used": signal not in dropped,
            }
        )

    if dropped:
        print(
            f"\nDROPPED (separation < {MIN_SEPARATION_SD} SD, carries no signal): "
            f"{', '.join(dropped)}",
            file=sys.stderr,
        )

    # How the whole policy behaves, using the fitted thresholds: does a clean
    # page get left alone, and does a degraded page get worked on?
    behaviour = {}
    for profile, profile_rows in sorted(by_profile.items()):
        touched = 0
        stage_counts: dict[str, int] = defaultdict(int)
        for row in profile_rows:
            image = np.array(Image.open(args.corpus / row["image_path"]).convert("L"))
            decision = triage(image, fitted)
            touched += decision.degraded
            for stage in decision.stages:
                stage_counts[stage] += 1
        behaviour[profile] = {
            "n": len(profile_rows),
            "preprocessed": touched,
            "left_alone": len(profile_rows) - touched,
            "stage_fire_rate": {
                s: round(c / len(profile_rows), 3) for s, c in sorted(stage_counts.items())
            },
        }

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": str(args.corpus),
        "split": "train",
        "documents_per_profile": args.limit,
        "signals": report_rows,
        "dropped_signals": dropped,
        "fitted_thresholds": fitted,
        "policy_behaviour": behaviour,
    }
    (args.out / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Triage calibration",
        "",
        f"Fitted on the **train split** of `{args.corpus.name}`, "
        f"{args.limit} pages per profile. Never on the evaluation pages.",
        "",
        "## Per-signal separation",
        "",
        "| Signal | Gates | Detects | Absent | Present | Threshold | Sep (SD) | Acc | Used |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report_rows:
        md.append(
            f"| `{r['signal']}` | `{r['stage']}` | {', '.join(r['targets'])} "
            f"| {r['without_mean']} | {r['with_mean']} | {r['threshold']} "
            f"| {r['separation_sd']} | {r['gate_accuracy']} "
            f"| {'yes' if r['used'] else '**no**'} |"
        )
    md += [
        "",
        "Each signal is scored against whether *the degradation it detects* was "
        "applied to that page, not against the page's profile label. "
        "Degradations are applied probabilistically, so a `heavy` page may carry "
        "no rotation at all, and such a page genuinely does not need deskewing "
        "-- scoring against the profile punished the skew gate for correctly "
        "declining and reported it at 0.583 when it is in fact far better.",
        "",
        f"Separation is the gap between the two means in pooled standard "
        f"deviations. A signal below {MIN_SEPARATION_SD} SD is **dropped**, not "
        "kept with a caveat: a gate that cannot see what it exists to detect "
        "makes the policy worse than having no gate.",
        "",
        "## What the policy actually does",
        "",
        "| Profile | n | Preprocessed | Left alone |",
        "|---|---|---|---|",
    ]
    for profile, b in behaviour.items():
        md.append(f"| `{profile}` | {b['n']} | {b['preprocessed']} | {b['left_alone']} |")
    md += [
        "",
        "The target is the top-left and bottom-right of that table: clean pages "
        "left alone, degraded pages worked on. A clean page that gets "
        "preprocessed pays the 8.9-point word-accuracy penalty the ablation "
        "measured, and a degraded page left alone forgoes up to 14.1 points.",
        "",
        "## Stage fire rate by profile",
        "",
        "| Profile | "
        + " | ".join(
            f"`{s}`" for s in sorted({s for b in behaviour.values() for s in b["stage_fire_rate"]})
        )
        + " |",
    ]
    all_stages = sorted({s for b in behaviour.values() for s in b["stage_fire_rate"]})
    md.append("|" + "---|" * (len(all_stages) + 1))
    for profile, b in behaviour.items():
        cells = [f"{b['stage_fire_rate'].get(s, 0.0):.2f}" for s in all_stages]
        md.append(f"| `{profile}` | {' | '.join(cells)} |")
    (args.out / "calibration.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print(f"{'signal':<22} {'absent':>9} {'present':>9} {'thresh':>9} {'sep':>5} {'acc':>5}  used")
    for r in report_rows:
        print(
            f"{r['signal']:<22} {r['without_mean']:>9} {r['with_mean']:>9} "
            f"{r['threshold']:>9} {r['separation_sd']:>5} {r['gate_accuracy']:>5}  "
            f"{'yes' if r['used'] else 'NO'}"
        )
    print()
    for profile, b in behaviour.items():
        print(f"  {profile:<7} {b['preprocessed']:>3}/{b['n']:<3} preprocessed")
    print(f"\nWrote {args.out}/calibration.json and calibration.md")

    if args.apply:
        target = Path(__file__).resolve().parents[1] / "src/invoice_extract/preprocess/triage.py"
        text = target.read_text(encoding="utf-8")
        start = text.index("DEFAULT_THRESHOLDS: dict[str, float] = {")
        end = text.index("}", start) + 1
        block = (
            "DEFAULT_THRESHOLDS: dict[str, float] = {\n"
            + "".join(f'    "{k}": {v},\n' for k, v in fitted.items())
            + "}"
        )
        target.write_text(text[:start] + block + text[end:], encoding="utf-8")
        print(f"Rewrote DEFAULT_THRESHOLDS in {target.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
