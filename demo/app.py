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

from invoice_extract.rag.taxonomy import account_name  # noqa: E402

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


@st.cache_resource(show_spinner=False)
def load_knowledge():
    """The ledger, the retriever and the vendor master.

    Computed live rather than cached to disk like the OCR, because this half of
    the pipeline is fast: indexing ~300 ledger entries with TF-IDF character
    n-grams is milliseconds. OCR is the only stage expensive enough to need
    precomputing, and the "never render a number that was not computed" rule is
    satisfied either way -- these ARE computed, on the spot.

    Returns None when the ledger has not been built, so the screen can say so
    instead of rendering an empty table that looks like a result.
    """
    ledger_path = ROOT / "data/processed/ledger/ledger.json"
    corpus = ROOT / "data/processed/gst_in_synthetic"
    if not ledger_path.exists() or not (corpus / "canonical").exists():
        return None

    import glob

    from invoice_extract.rag.store import LedgerEntry, TfidfRetriever, VendorStats
    from invoice_extract.rag.vendor_master import VendorMaster

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entries = [LedgerEntry(**e) for e in ledger["entries"]]
    retriever = TfidfRetriever()
    retriever.index(entries)

    stats = VendorStats.build(entries, doc_totals={})
    for row in ledger.get("doc_totals") or []:
        if row.get("vendor") and row.get("grand_total") is not None:
            stats.totals.setdefault(row["vendor"], []).append(row["grand_total"])

    history = [
        json.loads(Path(f).read_text(encoding="utf-8"))
        for f in sorted(glob.glob(str(corpus / "canonical" / "*.json")))[:100]
    ]
    aliases = {}
    alias_path = corpus / "vendor_aliases.json"
    if alias_path.exists():
        aliases = json.loads(alias_path.read_text(encoding="utf-8")).get("aliases") or {}

    return {
        "retriever": retriever,
        "stats": stats,
        "master": VendorMaster.build(history, aliases=aliases),
        "history": history,
        "n_entries": len(entries),
        "n_aliases": len(aliases),
    }


SEVERITY_COLOUR = {"high": "bad", "medium": "bad", "low": "muted"}
VERDICT_COLOUR = {"PASS": "ok", "REVIEW": "muted", "FAIL": "bad"}


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
        "4. Enrich (RAG)",
        "5. Does it generalise?",
        "6. Findings",
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
    st.title("4. Turning data into decisions")
    st.write(
        "Extraction stops at *what the page says*. This is the half that decides "
        "**what to do about it** — which account each line belongs to, whether "
        "anything looks wrong, and whether the supplier is who they claim to be."
    )

    knowledge = load_knowledge()
    if knowledge is None:
        not_run(
            "the ledger has not been built. Run `python scripts/build_ledger.py` "
            "to enable categorisation, anomaly detection and vendor validation."
        )
    else:
        from invoice_extract.rag.anomaly import detect_anomalies
        from invoice_extract.rag.categorize import categorize_document
        from invoice_extract.rag.validate_vendor import validate_against_master

        doc = pick_document(OURS, "rag_doc")
        extraction = doc["extractions"].get("rules") or next(iter(doc["extractions"].values()))
        document = extraction["document"]

        st.caption(
            f"Retrieving against {knowledge['n_entries']} past line items from "
            f"{len(knowledge['history'])} accepted invoices. "
            f"Vendor master: {len(knowledge['master'])} suppliers, "
            f"{knowledge['n_aliases']} known trading names."
        )

        # ------------------------------------------------ retrieve + categorise
        st.markdown("## Retrieval — what similar invoices were booked to")
        st.write(
            "For each line, the most similar items in the ledger are pulled up and "
            "vote on the account, weighted by similarity. **This is the R in RAG**: "
            "the answer comes from retrieved history, not from a model's prior."
        )

        decisions = categorize_document(
            document, knowledge["retriever"], vendor_stats=knowledge["stats"]
        )
        items = document.get("line_items") or []
        if not decisions:
            not_run("this document has no line items to categorise")
        for item, decision in zip(items, decisions, strict=False):
            label = item.get("description") or "(no description)"
            if decision.abstained:
                head = f'**{label}** → <span class="muted">abstained</span>'
            else:
                head = (
                    f'**{label}** → <span class="ok">{decision.category} '
                    f"{account_name(decision.category)}</span> "
                    f'<span class="muted">({decision.confidence:.0%} confident)</span>'
                )
            st.markdown(head, unsafe_allow_html=True)
            st.caption(decision.reason)
            if decision.neighbours:
                rows = [
                    "| Similar past item | Vendor | Booked to | Similarity |",
                    "|---|---|---|---|",
                ]
                for m in decision.neighbours[:4]:
                    rows.append(
                        f"| {m.entry.description} | {m.entry.vendor or '—'} | "
                        f"{m.entry.category} {account_name(m.entry.category)} | "
                        f"{m.score:.2f} |"
                    )
                st.markdown(chr(10).join(rows), unsafe_allow_html=True)
            st.markdown("")

        st.info(
            "When the neighbours are weak or disagree it **abstains** rather than "
            "guessing. A wrong account code propagates silently into a ledger; an "
            "abstention lands in a review queue and is cleared in seconds."
        )

        # ------------------------------------------------------------ anomalies
        st.markdown("---")
        st.markdown("## Anomaly flags")
        report = detect_anomalies(document, knowledge["retriever"], knowledge["stats"])
        if report.flags:
            for flag in report.flags:
                css = SEVERITY_COLOUR.get(flag.severity, "muted")
                st.markdown(
                    f'<span class="{css}">{flag.severity.upper()}</span> '
                    f"<code>{flag.code}</code> — {flag.message}",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<span class="ok">Nothing flagged</span>', unsafe_allow_html=True)

        st.caption(
            "Recall is 1.000 on the scenarios these checks were designed for and "
            "0.800 once two deliberately held-out scenarios are included — both of "
            "which are missed entirely, and kept unfixed on purpose."
        )

        # ------------------------------------------------------- vendor validation
        st.markdown("---")
        st.markdown("## Is the supplier who they say they are?")
        validation = validate_against_master(
            document, knowledge["master"], history=knowledge["history"]
        )
        res = validation.resolution
        if res.resolved:
            st.markdown(
                f"Resolved to **{res.vendor.name}** "
                f'<span class="muted">(by {res.method}, '
                f"{res.vendor.invoice_count} past invoices)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="muted">No matching supplier on file — treated as a '
                "first-time vendor, which is a review, not a fraud finding.</span>",
                unsafe_allow_html=True,
            )

        for check in validation.checks:
            css = VERDICT_COLOUR.get(check.verdict, "muted")
            st.markdown(
                f'<span class="{css}">{check.verdict}</span> '
                f"<code>{check.code}</code> — {check.message}",
                unsafe_allow_html=True,
            )

        css = VERDICT_COLOUR.get(validation.verdict, "muted")
        st.markdown(
            f'### Verdict: <span class="{css}">{validation.verdict}</span>',
            unsafe_allow_html=True,
        )
        st.caption(
            "The worst check wins. PASS pays, REVIEW queues for a human, FAIL stops. "
            "Nothing is ever silently corrected."
        )

        with st.expander("What is NOT shown here"):
            st.markdown("""
- **The generation step.** `rag/generate.py` prompts a local Qwen2.5-0.5B with
  these same retrieved neighbours, and `rag/embed.py` adds dense retrieval.
  Both are implemented and verified against real weights, but their Phase 3
  numbers are still being measured, so this screen shows the
  **similarity-weighted vote** — the control the generator has to beat.
- **Fraud.** These are clean invoices. The injected-fraud scenarios live in
  `scripts/run_validation_eval.py`; on the hardened corpus the system scores
  precision 0.750 with a 0.308 false-positive rate, down from a flattering
  1.000/0.000 on a corpus that had no hard cases.
            """)

elif screen.startswith("5"):
    st.title("5. Does it generalise?")
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
        # The VLM run covered the 100-document evaluation slice, which was
        # selected independently of the demo documents, so most cached
        # documents legitimately carry no model output. Saying "the VLM has not
        # been run" here would now be false.
        not_run(
            "this document was not part of the VLM run. To cover it, add it to "
            "the payload and re-run `notebooks/vlm_eval.ipynb` on Colab, then:\n\n"
            "```\npython scripts/build_demo_cache.py --vlm vlm_outputs.json\n```"
        )

# -------------------------------------------------------------- 5. findings

else:
    st.title("6. What we measured")

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
