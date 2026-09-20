"""Tests for the demo cache contract.

The demo is what gets shown to an audience, so the failure mode that matters is
not a crash -- it is the app confidently displaying something that was never
computed. These tests pin the two properties that prevent that:

1. Every field the UI reads exists in the cache with the shape it expects.
2. A stage that did not run is recorded as absent, never as a zero or a blank
   that renders like a real measurement.

The Streamlit app itself is not exercised here; it is a thin renderer over this
contract, and testing it would mean testing Streamlit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from invoice_extract.models.postprocess import finalize

# The cache builder is a script, not part of the installed package, so make it
# importable rather than skipping these tests -- they cover the contract the
# demo actually depends on.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_demo_cache  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "demo_cache"


def _doc(**totals):
    return {
        "doc_id": "d1",
        "source": {"dataset": "other", "original_id": "d1", "split": "test"},
        "pages": [{"page_index": 0, "image_path": "x.png", "width": 10, "height": 10}],
        "parties": {"vendor": {"name": "Acme", "gstin": "29AAAAA0000A1ZY"}},
        "invoice_meta": {"invoice_number": "INV-1", "invoice_date": "2026-01-12"},
        "line_items": [{"quantity": 2, "unit_price": 50.0, "line_total": 100.0}],
        "tax_lines": [{"type": "IGST", "rate": 18, "amount": 18.0}],
        "totals": {"subtotal": 100.0, "tax_total": 18.0, "grand_total": 118.0, **totals},
    }


class TestScoreFields:
    def test_identical_documents_score_all_correct(self):
        gold = _doc()
        scored = build_demo_cache.score_fields(gold, gold)
        assert set(scored) == set(build_demo_cache.DEMO_FIELDS)
        assert all(f["correct"] for f in scored.values())

    def test_a_wrong_value_is_marked_wrong_and_both_sides_shown(self):
        gold = _doc()
        pred = json.loads(json.dumps(gold))
        pred["invoice_meta"]["invoice_number"] = "WRONG-9"
        scored = build_demo_cache.score_fields(gold, pred)
        field = scored["invoice_meta.invoice_number"]
        assert field["correct"] is False
        assert field["gold"] == "INV-1"
        assert field["pred"] == "WRONG-9"

    def test_both_sides_missing_counts_as_correct(self):
        """Predicting nothing for a field the document does not have is right,
        not wrong -- otherwise the demo would show red for correct behaviour."""
        gold, pred = _doc(), _doc()
        for doc in (gold, pred):
            doc["parties"]["vendor"]["name"] = None
        scored = build_demo_cache.score_fields(gold, pred)
        assert scored["parties.vendor.name"]["correct"] is True

    def test_every_scored_field_is_json_serialisable(self):
        json.dumps(build_demo_cache.score_fields(_doc(), _doc()))


class TestValidationPanel:
    def test_a_consistent_invoice_passes_both_checks(self):
        panel = build_demo_cache.validation_panel(_doc())
        assert panel["arithmetic_ok"] is True
        assert panel["gstin_problems"] == []

    def test_broken_arithmetic_is_flagged_not_corrected(self):
        doc = _doc(grand_total=9999.0)
        panel = build_demo_cache.validation_panel(doc)
        assert panel["arithmetic_ok"] is False
        assert panel["arithmetic"]
        # The document itself must be untouched: the demo's claim is that
        # problems are surfaced, never silently repaired.
        assert doc["totals"]["grand_total"] == 9999.0

    def test_bad_gstin_checksum_is_reported(self):
        doc = _doc()
        doc["parties"]["vendor"]["gstin"] = "29AAAAA0000A1Z5"  # wrong check digit
        assert build_demo_cache.validation_panel(doc)["gstin_problems"]

    def test_absent_gstin_produces_no_problems(self):
        doc = _doc()
        doc["parties"]["vendor"].pop("gstin")
        panel = build_demo_cache.validation_panel(doc)
        assert panel["gstin"] is None
        assert panel["gstin_problems"] == []


@pytest.mark.skipif(not (CACHE / "index.json").exists(), reason="demo cache not built")
class TestBuiltCache:
    @pytest.fixture(scope="class")
    def index(self) -> dict:
        return json.loads((CACHE / "index.json").read_text(encoding="utf-8"))

    def test_index_declares_what_was_and_was_not_run(self, index):
        """`has_vlm` / `ocr_engine` are how the UI decides between showing a
        result and showing 'not run'. They must always be present."""
        assert "has_vlm" in index and "ocr_engine" in index
        assert isinstance(index["documents"], list) and index["documents"]

    def test_every_indexed_document_has_a_detail_file(self, index):
        for entry in index["documents"]:
            assert (CACHE / "docs" / f"{entry['doc_id']}.json").exists(), entry["doc_id"]

    def test_every_referenced_image_exists(self, index):
        """A missing image renders as a blank panel mid-demo."""
        for entry in index["documents"]:
            doc = json.loads(
                (CACHE / "docs" / f"{entry['doc_id']}.json").read_text(encoding="utf-8")
            )
            for key, rel in doc["images"].items():
                assert (CACHE / rel).exists(), f"{entry['doc_id']}: {key} -> {rel}"

    def test_the_demo_set_includes_a_failure(self, index):
        """A demo that only shows successes invites the question you least want
        asked cold, so the picker is seeded with imperfect documents too."""
        scores = [e["rules_score"] for e in index["documents"] if e["rules_score"] is not None]
        assert scores, "no scored documents in the cache"
        assert min(scores) < 1.0, "every cached document extracts perfectly"

    def test_both_corpora_are_represented(self, index):
        """Screen 4 compares our layouts against FATURA2; it needs both."""
        corpora = {e["corpus"] for e in index["documents"]}
        assert "gst_in_synthetic" in corpora
        assert "fatura2" in corpora

    def test_extractions_carry_everything_the_ui_renders(self, index):
        for entry in index["documents"]:
            doc = json.loads(
                (CACHE / "docs" / f"{entry['doc_id']}.json").read_text(encoding="utf-8")
            )
            for name, extraction in doc["extractions"].items():
                for key in ("source", "document", "per_field", "validation"):
                    assert key in extraction, f"{entry['doc_id']}/{name} missing {key}"
                for path, field in extraction["per_field"].items():
                    assert set(field) >= {"gold", "pred", "correct"}, path

    def test_ocr_entries_record_real_timings(self, index):
        """`has_ocr` must mean OCR actually ran, not that a key exists."""
        for entry in index["documents"]:
            doc = json.loads(
                (CACHE / "docs" / f"{entry['doc_id']}.json").read_text(encoding="utf-8")
            )
            assert entry["has_ocr"] == bool(doc["ocr"])
            for key, result in doc["ocr"].items():
                assert result["ms"] > 0, f"{entry['doc_id']}/{key} has no elapsed time"
                assert "engine" in result

    def test_cached_documents_still_validate_against_the_schema(self, index):
        from invoice_extract.data.validate import validate_jsonschema

        for entry in index["documents"]:
            doc = json.loads(
                (CACHE / "docs" / f"{entry['doc_id']}.json").read_text(encoding="utf-8")
            )
            assert validate_jsonschema(doc["gold"]) == [], entry["doc_id"]
            for name, extraction in doc["extractions"].items():
                assert (
                    validate_jsonschema(extraction["document"]) == []
                ), f"{entry['doc_id']}/{name}"


class TestVlmMerge:
    def test_a_parse_failure_is_recorded_rather_than_dropped(self, tmp_path):
        """An unparseable model response is a result -- it scores as a missed
        document. Silently skipping it would flatter the model."""
        gold = _doc()
        pred, _ = finalize({}, "d1", gold["source"], gold["pages"])
        scored = build_demo_cache.score_fields(gold, pred)
        assert not any(f["correct"] for f in scored.values())
