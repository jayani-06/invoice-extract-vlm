"""One invoice image, all the way through every stage, to a final verdict.

Every other test file exercises a single phase in isolation. That leaves the
handoffs untested, and the handoffs are where this project has actually broken:

- the renderer emitted text grouped by draw call, so OCR gold split labels from
  their values (found in Phase 2, by eye, not by a test)
- duplicate detection was keyed normalised and looked up raw, so the signal was
  dead while every unit test passed
- coercion let a postal address through as a GSTIN, producing a schema-invalid
  document that no per-module test caught

None of those were module bugs. They were contract bugs between modules, and
only a test that runs the whole chain can see them.

The chain:

    image -> preprocess -> OCR -> reading order -> extract -> coerce+validate
          -> categorise (Phase 3) -> anomaly flags (Phase 3)
          -> vendor resolution + record validation (Phase 4)

Marked `slow` because real OCR is ~40 s/page on CPU. Run the fast subset with
`pytest -m "not slow"`; this file's OCR test is the only thing that costs time.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from invoice_extract.data.validate import validate_jsonschema
from invoice_extract.models.postprocess import finalize
from invoice_extract.models.registry import get_extractor
from invoice_extract.preprocess.pipeline import Pipeline, get_config
from invoice_extract.rag.anomaly import detect_anomalies
from invoice_extract.rag.categorize import categorize_document
from invoice_extract.rag.store import LedgerEntry, TfidfRetriever, VendorStats
from invoice_extract.rag.taxonomy import category_of
from invoice_extract.rag.validate_vendor import PASS, REVIEW, validate_against_master
from invoice_extract.rag.vendor_master import VendorMaster
from invoice_extract.reading_order import reading_order_text

CORPUS = Path("data/processed/gst_in_synthetic")
pytestmark = pytest.mark.skipif(
    not (CORPUS / "manifest.jsonl").exists(), reason="image corpus not built"
)


def manifest_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (CORPUS / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def corpus_docs() -> list[dict]:
    return [
        json.loads(Path(p).read_text(encoding="utf-8"))
        for p in sorted(glob.glob(str(CORPUS / "canonical" / "*.json")))
    ]


@pytest.fixture(scope="module")
def history(corpus_docs) -> list[dict]:
    """The first 100 documents stand in as accepted history."""
    return corpus_docs[:100]


@pytest.fixture(scope="module")
def incoming(corpus_docs, history) -> dict:
    """A held-out document whose vendor the system HAS seen before.

    Picked by that criterion rather than by index. An arbitrary held-out
    document often belongs to a first-time supplier, and then every Phase 4
    check correctly short-circuits to "unknown vendor" -- which exercises none
    of the identity logic this file is meant to cover. A separate test below
    handles the genuinely-new-vendor path.
    """
    known = {((doc.get("parties") or {}).get("vendor") or {}).get("name") for doc in history}
    for doc in corpus_docs[100:]:
        vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        if vendor in known and doc.get("line_items"):
            return doc
    pytest.skip("no held-out document shares a vendor with the history split")


@pytest.fixture(scope="module")
def knowledge(history):
    """Everything the RAG stages retrieve against, built from history alone."""
    entries = [
        LedgerEntry(
            description=item.get("description") or "",
            category=category_of(item.get("description")),
            vendor=((doc.get("parties") or {}).get("vendor") or {}).get("name"),
            unit_price=item.get("unit_price"),
            line_total=item.get("line_total"),
            hsn_sac_code=item.get("hsn_sac_code"),
            doc_id=doc.get("doc_id"),
        )
        for doc in history
        for item in doc.get("line_items") or []
    ]
    retriever = TfidfRetriever()
    retriever.index(entries)

    stats = VendorStats.build(entries)
    for doc in history:
        vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
        total = (doc.get("totals") or {}).get("grand_total")
        if vendor and total is not None:
            stats.totals.setdefault(vendor, []).append(total)

    return retriever, stats, VendorMaster.build(history)


class TestChainOnGoldText:
    """The full chain with OCR replaced by the renderer's own text.

    Fast, so it runs on every commit. It cannot catch OCR-specific breakage,
    which is what the slow test below is for -- but it does catch every
    contract bug between extraction, enrichment and validation.
    """

    @pytest.fixture(scope="class")
    def result(self, incoming, knowledge):
        retriever, stats, master = knowledge
        row = next(
            r
            for r in manifest_rows()
            if r["doc_id"] == incoming["doc_id"] and r["profile"] == "clean"
        )
        gold = json.loads((CORPUS / row["ocr_gold_path"]).read_text(encoding="utf-8"))
        text = reading_order_text(gold["words"], lambda w: tuple(w["bbox"]), lambda w: w["text"])

        raw = get_extractor("rules").predict_from_text(text)
        pred, consistency = finalize(raw, incoming["doc_id"], incoming["source"], incoming["pages"])
        return {
            "text": text,
            "pred": pred,
            "consistency": consistency,
            "categories": categorize_document(pred, retriever, vendor_stats=stats),
            "anomalies": detect_anomalies(pred, retriever, stats),
            "validation": validate_against_master(pred, master, history=[]),
        }

    def test_ocr_text_reaches_the_extractor(self, result, incoming):
        """The handoff that broke once: text has to carry label and value on
        the same line, or every label-anchored rule fails."""
        assert result["text"], "no text produced"
        number = incoming["invoice_meta"]["invoice_number"]
        assert number in result["text"]
        # Label and value on one line -- the reading-order defect split these.
        line = next(ln for ln in result["text"].splitlines() if number in ln)
        assert "invoice" in line.lower()

    def test_extraction_produces_a_schema_valid_document(self, result):
        assert validate_jsonschema(result["pred"]) == []

    def test_critical_fields_survive_the_chain(self, result, incoming):
        pred, gold = result["pred"], incoming
        assert pred["invoice_meta"]["invoice_number"] == gold["invoice_meta"]["invoice_number"]
        assert pred["invoice_meta"]["invoice_date"] == gold["invoice_meta"]["invoice_date"]
        assert pred["totals"]["grand_total"] == pytest.approx(
            gold["totals"]["grand_total"], rel=0.01
        )

    def test_a_clean_invoice_reconciles(self, result):
        """Arithmetic flags on an untouched corpus document would mean the
        extraction or the generator is wrong."""
        assert result["consistency"].ok, result["consistency"].problems

    def test_categorisation_runs_on_the_extracted_document(self, result):
        decisions = result["categories"]
        assert len(decisions) == len(result["pred"]["line_items"])
        assert any(not d.abstained for d in decisions), "every line abstained"
        for d in decisions:
            assert d.neighbours or d.abstained

    def test_anomaly_detection_runs_and_stays_quiet(self, result):
        report = result["anomalies"]
        actionable = [f for f in report.flags if f.severity in ("high", "medium")]
        assert actionable == [], [f.code for f in actionable]

    def test_vendor_validation_resolves_and_passes(self, result):
        report = result["validation"]
        assert report.resolution.resolved, "known vendor failed to resolve"
        assert report.verdict in (PASS, REVIEW)
        assert report.explain()

    def test_the_whole_result_is_json_serialisable(self, result):
        """Phase 5 will serve this over HTTP; anything that cannot round-trip
        through JSON is a problem now, not later."""
        json.dumps(
            {
                "document": result["pred"],
                "categories": [d.as_dict() for d in result["categories"]],
                "anomalies": result["anomalies"].as_dict(),
                "validation": result["validation"].as_dict(),
            }
        )


class TestTamperedInvoiceIsCaught:
    """The same chain, on a document corrupted the way fraud would corrupt it.

    A pipeline that passes everything is indistinguishable from one that checks
    nothing, so the end-to-end test has to include a case that must fail.
    """

    def test_redirected_bank_account_reaches_a_fail_verdict(self, incoming, knowledge):
        _, _, master = knowledge
        tampered = json.loads(json.dumps(incoming))
        tampered.setdefault("payment_info", {})["account_number"] = "00000000000000"

        report = validate_against_master(tampered, master, history=[])
        if report.resolution.resolved and report.resolution.vendor.invoice_count >= 2:
            assert "bank_account_changed" in {c.code for c in report.checks}
        else:
            pytest.skip("this document's vendor has too little history to check identity")

    def test_broken_arithmetic_is_flagged_end_to_end(self, incoming, knowledge):
        retriever, stats, _ = knowledge
        tampered = json.loads(json.dumps(incoming))
        tampered["totals"]["grand_total"] = 999_999.0

        report = detect_anomalies(tampered, retriever, stats)
        assert "arithmetic_mismatch" in {f.code for f in report.flags}
        assert report.max_severity == "high"

    def test_a_first_time_supplier_is_reviewed_not_failed(self, corpus_docs, history, knowledge):
        """The other half of resolution: a vendor with no record must reach
        REVIEW, not FAIL. Paying nobody is as broken as paying everybody."""
        _, _, master = knowledge
        known = {((doc.get("parties") or {}).get("vendor") or {}).get("name") for doc in history}
        unseen = next(
            (
                doc
                for doc in corpus_docs[100:]
                if ((doc.get("parties") or {}).get("vendor") or {}).get("name") not in known
            ),
            None,
        )
        if unseen is None:
            pytest.skip("every held-out vendor appears in the history split")

        report = validate_against_master(unseen, master, history=[])
        assert not report.resolution.resolved
        assert report.verdict == REVIEW
        assert {c.code for c in report.checks} == {"unknown_vendor"}


@pytest.mark.slow
class TestChainWithRealOcr:
    """The same chain again, but reading actual pixels.

    This is the only test that can catch OCR-specific regressions -- a
    preprocessing change that destroys the page, an engine upgrade that alters
    the output shape, a reading-order change that scrambles lines.
    """

    def test_image_to_verdict(self, incoming, knowledge):
        ocr = pytest.importorskip("paddleocr", reason="PaddleOCR not installed")
        del ocr
        from invoice_extract.ocr.base import OcrEngineUnavailableError, get_engine

        retriever, stats, master = knowledge
        row = next(
            r
            for r in manifest_rows()
            if r["doc_id"] == incoming["doc_id"] and r["profile"] == "medium"
        )

        try:
            engine = get_engine("paddle", fast=True)
        except OcrEngineUnavailableError as exc:
            pytest.skip(str(exc))

        image = np.array(Image.open(CORPUS / row["image_path"]).convert("L"))
        preprocessed = Pipeline(get_config("full(no-binarize)"))(image)
        recognised = engine.recognize(preprocessed.image)

        assert recognised.words, "OCR returned nothing"
        assert recognised.text.strip(), "OCR produced no text"

        raw = get_extractor("rules").predict_from_text(recognised.text)
        pred, _ = finalize(raw, incoming["doc_id"], incoming["source"], incoming["pages"])

        # Schema validity is the hard guarantee: whatever OCR returns, the
        # document that comes out the other side must be well-formed.
        assert validate_jsonschema(pred) == []

        # Every downstream stage must accept it without raising.
        categorize_document(pred, retriever, vendor_stats=stats)
        detect_anomalies(pred, retriever, stats)
        report = validate_against_master(pred, master, history=[])
        assert report.verdict in (PASS, REVIEW, "FAIL")
