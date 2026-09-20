# Demo

A five-screen walkthrough of the pipeline, built for a live review.

## Run it

```powershell
.\run.ps1 demo-cache    # once, ~4 min (precomputes OCR + extraction)
.\run.ps1 demo          # opens http://localhost:8501
```

On macOS/Linux the equivalents are `make demo-cache` and `make demo`. Either way
the underlying commands are just:

```bash
python scripts/build_demo_cache.py
python -m streamlit run demo/app.py
```

If you only changed the UI, `.\run.ps1 demo-cache-fast` rebuilds in about a minute
by skipping OCR — screen 2 will show "not run" until you do a full build.

## Why it replays a cache instead of processing live

OCR costs about 40 seconds per page on this CPU. Processing on click would give
the demo a dead spot on every interaction and a real chance of stalling in
front of an audience. Everything on screen is a genuine output — real
preprocessing, real OCR, real extraction — just computed in advance.

The app never fabricates. A stage that was not run renders as **"not run"**,
never as a dash or a placeholder number that could be mistaken for a result.

## Adding the model comparison

Screen 4 shows rules-vs-model once you have run the VLM. There is no GPU on the
dev machine, so this takes one Colab session:

```powershell
.\run.ps1 demo-vlm-payload     # writes vlm_payload.zip (~0.5 MB)
```

Then open `notebooks/vlm_demo_outputs.ipynb` in Colab, set **Runtime > Change
runtime type > T4 GPU**, upload the zip, run all cells, and download
`vlm_outputs.json`. Back here:

```powershell
.\run.ps1 demo-vlm-merge
```

Scoring, coercion and validation all happen locally through the same code the
rules baseline uses, so a difference on screen is a difference in the model
rather than in how each was post-processed.

## The five screens

| Screen | Shows |
|---|---|
| 1. Pipeline | Damage slider and cleanup stages, side by side |
| 2. Read (OCR) | Word boxes over the page, plus the transcript |
| 3. Extract | Structured JSON, per-field correct/wrong, automatic checks |
| 4. Does it generalise? | Our invoice vs FATURA2 — **the point of the demo** |
| 5. Findings | The two ablation tables, and the honest limits |

## Suggested five minutes

| Time | Screen | Say |
|---|---|---|
| 0:00 | 1 | "Real scans arrive skewed, shadowed, bordered. Here's the same invoice at four damage levels, and what our cleanup does." *(drag the slider)* |
| 1:00 | 2 | "PaddleOCR reads it. Boxes are what it found." |
| 2:00 | 3 | "Structured JSON out. Green matched ground truth. Note it *flags* that the totals don't add up — it never silently corrects them, because that's the signal the anomaly-detection stage needs." |
| 3:00 | 4 | "Now the interesting part. Same code, an invoice from a public dataset we didn't design. Vendor name: zero. 98% on our data, 56% on theirs. That's the case for a vision-language model." |
| 4:00 | 5 | "And a negative result we're reporting rather than hiding: classical preprocessing didn't improve character accuracy on a modern OCR engine. It improved *localisation* by 21 points — which turned out to be what matters." |

The arc — it works, here's where it breaks, here's what that tells us, here's
what we're doing about it — is a stronger review than a demo that only succeeds.
The document picker on each screen is pre-loaded with failing cases as well as
passing ones for exactly that reason.

## Notes

- `.streamlit/config.toml` pins the light theme. Without it, Streamlit follows
  the OS setting and a dark-mode laptop renders white-on-dark widgets that no
  amount of CSS reliably overrides.
- Nothing in `demo/` is imported by `src/` — the research pipeline stays clean.
- `demo_cache/` is gitignored; rebuild it with `.\run.ps1 demo-cache`.
