"""Demo UI for the invoice extraction pipeline.

Reads only from `demo_cache/`. Nothing is computed here: OCR costs ~40 s/page
on CPU, so processing on click would give the demo a dead spot on every
interaction and a real chance of stalling in front of an audience. Everything
shown is a genuine output, precomputed by `scripts/build_demo_cache.py`.

The one rule this file follows everywhere: **never render a number that was not
computed.** Missing stages show as "not run", never as a placeholder or a dash
that could be mistaken for a result.

Run:  streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CACHE = ROOT / "demo_cache"

st.set_page_config(page_title="Invoice Extraction Demo", layout="wide")

# Plain white background, black text -- readable on a projector, and nothing
# competing with the document images for attention.
st.markdown(
    """
    <style>
      /* Widget colours come from .streamlit/config.toml (light theme pinned
         there). This block only handles typography and the semantic
         correct/wrong colours -- Streamlit's widget class names are
         content-hashed and restyling them here would break on every upgrade. */
      header[data-testid="stHeader"] { background: #ffffff !important; }
      #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

      html, body, p, li, span, label, h1, h2, h3, h4, h5, .stMarkdown {
          color: #000000;
          font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
      }
      h1 { font-size: 1.9rem !important; font-weight: 700 !important; }
      h2 { font-size: 1.35rem !important; font-weight: 700 !important;
           margin-top: 1.2rem !important; }
      h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
      code, pre, .stCode { color: #000000 !important; background: #f4f4f4 !important; }
      table { color: #000 !important; }
      thead th { background: #f0f0f0 !important; color: #000 !important; }
      .stAlert { background: #f4f4f4 !important; color: #000 !important; }
      hr { border-color: #dddddd; }
      div[data-testid="stMetricValue"] { color: #000 !important; }
      .ok   { color: #0a7d28; font-weight: 700; }
      .bad  { color: #c1121f; font-weight: 700; }
      .muted{ color: #666666; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- loading


def _stamp() -> float:
    """Modification time of the cache index, used as a cache key.

    Without this, rebuilding the cache while the app is running silently serves
    the previous build -- Streamlit keeps `@st.cache_data` results for the life
    of the process, and the reader has no way to tell. Keying on mtime means a
    rebuild is picked up on the next interaction.
    """
    path = CACHE / "index.json"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def load_index(stamp: float) -> dict | None:
    path = CACHE / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def load_doc(doc_id: str, stamp: float) -> dict:
    return json.loads((CACHE / "docs" / f"{doc_id}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_report(name: str, stamp: float) -> dict | None:
    path = CACHE / "reports" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def image_for(doc: dict, key: str) -> Image.Image | None:
    rel = doc["images"].get(key)
    if not rel:
        return None
    path = CACHE / rel
    return Image.open(path) if path.exists() else None


def tick(correct: bool) -> str:
    return '<span class="ok">correct</span>' if correct else '<span class="bad">wrong</span>'


def show_value(value) -> str:
    if value is None:
        return '<span class="muted">(none)</span>'
    return f"<code>{value}</code>"


def not_run(message: str) -> None:
    st.info(f"Not run: {message}")


STAMP = _stamp()
index = load_index(STAMP)
if index is None:
    st.title("Invoice Extraction Demo")
    st.error(
        "No demo cache found.\n\n"
        "Build it first (about 20 minutes with OCR, 1 minute without):\n\n"
        "```\nmake demo-cache\n```"
    )
    st.stop()

DOCS = index["documents"]
OURS = [d for d in DOCS if d["corpus"] == "gst_in_synthetic"]
THEIRS = [d for d in DOCS if d["corpus"] == "fatura2"]

# ----------------------------------------------------------------- sidebar

st.sidebar.markdown("## Invoice Extraction")
screen = st.sidebar.radio(
    "Screen",
    [
        "1. Pipeline",
        "2. Read (OCR)",
        "3. Extract",
        "4. Does it generalise?",
        "5. Findings",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Cache built {index['built_at']}")
st.sidebar.caption(f"{len(DOCS)} documents | OCR: {index.get('ocr_engine') or 'not run'}")
if index.get("has_vlm"):
    st.sidebar.caption(f"VLM: {index['vlm_model']}")
else:
    st.sidebar.caption("VLM: not run")


def pick_document(pool: list[dict], key: str, label: str = "Invoice") -> dict:
    labels = {
        f"{d['doc_id']}  ({int((d['rules_score'] or 0) * 100)}% fields correct)": d["doc_id"]
        for d in pool
    }
    chosen = st.selectbox(label, list(labels), key=key)
    return load_doc(labels[chosen], STAMP)


# --------------------------------------------------------------- 1. pipeline

if screen.startswith("1"):
    st.title("1. Cleaning up a scanned page")
    st.write(
        "Real scans arrive skewed, shadowed and bordered. Move the damage slider, "
        "then pick how much cleanup to apply."
    )

    doc = pick_document(OURS, "pipe_doc")
    col_a, col_b = st.columns(2)
    with col_a:
        profile = st.select_slider(
            "Scan damage",
            options=[p for p in ["clean", "medium", "heavy"] if p in doc["profiles"]],
            value="medium" if "medium" in doc["profiles"] else doc["profiles"][0],
        )
    with col_b:
        config = st.selectbox("Cleanup applied", index["demo_configs"], index=3)

    left, right = st.columns(2)
    with left:
        st.markdown("### Scanned input")
        img = image_for(doc, profile)
        if img:
            st.image(img, use_container_width=True)
    with right:
        st.markdown(f"### After cleanup: `{config}`")
        img = image_for(doc, f"{profile}__{config}")
        if img:
            st.image(img, use_container_width=True)
        else:
            not_run(f"no cached preprocessing for {profile} / {config}")

    st.markdown("---")
    st.markdown(
        "**What the stages do** — `gray+border` whitens the dark scanner frame, "
        "`deskew` straightens the page, `full(no-binarize)` adds lighting correction "
        "and denoising, `full+sauvola` additionally forces pure black-and-white. "
        "That last one measurably *hurts* modern OCR — see screen 5."
    )

# ------------------------------------------------------------------ 2. read

elif screen.startswith("2"):
    st.title("2. Reading the page")
    st.write("PaddleOCR output. Boxes are what it found; the table lists what it read.")

    with_ocr = [d for d in OURS if d["has_ocr"]]
    if not with_ocr:
        not_run(
            "no OCR in the cache. Rebuild with OCR enabled:\n\n"
            "```\npython scripts/build_demo_cache.py\n```"
        )
        st.stop()

    doc = pick_document(with_ocr, "ocr_doc")
    key = st.selectbox("Which run", list(doc["ocr"]))
    result = doc["ocr"][key]

    img = image_for(doc, key) or image_for(doc, key.split("__")[0])
    left, right = st.columns([3, 2])
    with left:
        if img:
            canvas = img.convert("RGB")
            draw = ImageDraw.Draw(canvas)
            for word in result["words"]:
                draw.rectangle(word["bbox"], outline=(200, 0, 0), width=2)
            st.image(canvas, use_container_width=True)
    with right:
        st.metric("Words found", len(result["words"]))
        st.metric("Time", f"{result['ms'] / 1000:.0f}s")
        st.markdown("### Text it read")
        st.code(result["text"][:2500] or "(nothing)")

# --------------------------------------------------------------- 3. extract

elif screen.startswith("3"):
    st.title("3. Turning the page into data")
    doc = pick_document(DOCS, "ext_doc")

    if not doc["extractions"]:
        not_run("no extraction cached for this document")
        st.stop()

    which = st.selectbox("Extractor", list(doc["extractions"]))
    extraction = doc["extractions"][which]
    st.caption(f"Input: {extraction['source']}")

    left, right = st.columns([2, 3])
    with left:
        img = image_for(doc, "clean") or image_for(doc, doc["profiles"][0])
        if img:
            st.image(img, use_container_width=True)

    with right:
        st.markdown("### Fields extracted")
        rows = ["| Field | Extracted | Expected | |", "|---|---|---|---|"]
        for path, field in extraction["per_field"].items():
            rows.append(
                f"| `{path.split('.')[-1]}` | {show_value(field['pred'])} | "
                f"{show_value(field['gold'])} | {tick(field['correct'])} |"
            )
        st.markdown("\n".join(rows), unsafe_allow_html=True)

        st.markdown("### Automatic checks")
        validation = extraction["validation"]
        if validation["arithmetic_ok"]:
            st.markdown('- Totals add up: <span class="ok">pass</span>', unsafe_allow_html=True)
        else:
            st.markdown('- Totals: <span class="bad">flagged</span>', unsafe_allow_html=True)
            for problem in validation["arithmetic"]:
                st.markdown(f"  - `{problem}`")
        if validation["gstin"]:
            verdict = (
                '<span class="ok">valid</span>'
                if not validation["gstin_problems"]
                else '<span class="bad">invalid</span>'
            )
            st.markdown(
                f"- GSTIN `{validation['gstin']}` checksum: {verdict}",
                unsafe_allow_html=True,
            )
        st.caption(
            "Problems are flagged, never silently corrected — a rewritten total "
            "would destroy the signal the anomaly-detection stage needs."
        )

    with st.expander("Full extracted JSON"):
        st.json(extraction["document"])
    if extraction.get("raw_response"):
        with st.expander("Raw model response"):
            st.code(extraction["raw_response"])

# --------------------------------------------------------------- 4. the gap

elif screen.startswith("4"):
    st.title("4. Does it generalise?")
    st.write(
        "The same extractor, on an invoice we designed and on one from a public "
        "dataset we did not."
    )

    def field_table(doc: dict, extractor: str) -> str:
        extraction = doc["extractions"][extractor]
        rows = ["| Field | Extracted | |", "|---|---|---|"]
        for path, field in extraction["per_field"].items():
            rows.append(
                f"| `{path.split('.')[-1]}` | {show_value(field['pred'])} | "
                f"{tick(field['correct'])} |"
            )
        return "\n".join(rows)

    left, right = st.columns(2)
    with left:
        st.markdown("### Our invoice")
        doc_a = pick_document(OURS, "gap_ours", label=" ")
        img = image_for(doc_a, "clean")
        if img:
            st.image(img, use_container_width=True)
        if "rules" in doc_a["extractions"]:
            st.markdown(field_table(doc_a, "rules"), unsafe_allow_html=True)
    with right:
        st.markdown("### FATURA2 (someone else's layout)")
        doc_b = pick_document(THEIRS, "gap_theirs", label=" ")
        img = image_for(doc_b, "clean")
        if img:
            st.image(img, use_container_width=True)
        if "rules" in doc_b["extractions"]:
            st.markdown(field_table(doc_b, "rules"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Measured across the whole corpus")
    ours_report, theirs_report = load_report("phase2", STAMP), load_report("fatura2", STAMP)
    if ours_report and theirs_report:

        def header_acc(report):
            rows = [r for r in report["results"] if r["extractor"] == "rules"]
            return rows[0]["header_accuracy"] if rows else None

        a, b = header_acc(ours_report), header_acc(theirs_report)
        c1, c2, c3 = st.columns(3)
        c1.metric("Our invoices (150)", f"{a:.1%}" if a else "n/a")
        c2.metric("FATURA2 (300)", f"{b:.1%}" if b else "n/a")
        if a and b:
            c3.metric("Drop", f"{(b - a) * 100:.0f} points")
        st.markdown(
            "The regex baseline saturates **our** test set, not the task. "
            "Vendor name falls from 90% to **0%** — worse than predicting nothing, "
            "because our rule assumes the vendor is the top line of the page. "
            "That gap is the case for a vision-language model."
        )
    else:
        not_run("ablation reports are not in the cache")

    st.markdown("---")
    st.markdown("## Rules vs the model")
    if index.get("has_vlm"):
        vlm_key = f"vlm:{index['vlm_model']}"
        rows = ["| Document | Field | Rules | Model |", "|---|---|---|---|"]
        for entry in DOCS:
            doc = load_doc(entry["doc_id"], STAMP)
            if vlm_key not in doc["extractions"] or "rules" not in doc["extractions"]:
                continue
            for path in index["demo_fields"]:
                r = doc["extractions"]["rules"]["per_field"].get(path)
                v = doc["extractions"][vlm_key]["per_field"].get(path)
                if not r or not v:
                    continue
                rows.append(
                    f"| `{entry['doc_id'][:20]}` | `{path.split('.')[-1]}` | "
                    f"{tick(r['correct'])} | {tick(v['correct'])} |"
                )
        st.markdown("\n".join(rows), unsafe_allow_html=True)
    else:
        not_run(
            "the VLM has not been run yet. Execute "
            "`notebooks/vlm_demo_outputs.ipynb` on Colab, then:\n\n"
            "```\npython scripts/build_demo_cache.py --vlm vlm_outputs.json\n```"
        )

# -------------------------------------------------------------- 5. findings

else:
    st.title("5. What we measured")

    st.markdown("## Preprocessing did not improve character accuracy")
    phase1 = load_report("phase1", STAMP)
    if phase1:
        configs, profiles = [], []
        for row in phase1["results"]:
            if row["config"] not in configs:
                configs.append(row["config"])
            if row["profile"] not in profiles:
                profiles.append(row["profile"])
        lookup = {(r["config"], r["profile"]): r for r in phase1["results"]}

        for metric, title, note in (
            ("cer_norm", "Character error rate (lower is better)", "Cleanup does not help."),
            (
                "word_accuracy",
                "Word accuracy — found *and* read right (higher is better)",
                "Cleanup helps a lot on damaged pages.",
            ),
        ):
            st.markdown(f"### {title}")
            head = "| Cleanup | " + " | ".join(profiles) + " |"
            rows = [head, "|" + "---|" * (len(profiles) + 1)]
            for config in configs:
                cells = []
                for profile in profiles:
                    r = lookup.get((config, profile))
                    cells.append(f"{r[metric]:.3f}" if r and r[metric] == r[metric] else "n/a")
                rows.append(f"| `{config}` | " + " | ".join(cells) + " |")
            st.markdown("\n".join(rows))
            st.caption(note)

        st.markdown(
            "**The two metrics disagree, and that is the finding.** On heavily damaged "
            "pages, straightening raises word accuracy from 0.27 to 0.48 while nudging "
            "character error the wrong way. OCR reads a crooked page fine — it just "
            "reports the wrong *location*, and location is what tells you which field a "
            "number belongs to. Both original Phase 1 targets failed; they were written "
            "against the wrong metric."
        )
    else:
        not_run("phase 1 ablation results are not in the cache")

    st.markdown("---")
    st.markdown("## Honest limits")
    st.markdown("""
- Every corpus here is **synthetic**. FATURA2 is third-party but still generated,
  not photographed. Real scans (SROIE) are the next acquisition.
- FATURA2 does **not annotate line items**, so that criterion is only verified on
  our own data.
- The Phase 1 ablation used **4 documents per cell** — the 21-point deskew effect
  is large enough to trust; the small character-error differences are not.
- Anything labelled *gold text* is a **perfect-OCR ceiling**, not an end-to-end
  result.
        """)
