"""Tests for the FATURA2 converter.

FATURA2 is the project's only third-party benchmark, and the tag vocabulary it
is decoded with was *inferred* rather than shipped -- the parquet stores bare
integer `ner_tags` with no label names. That makes these tests unusually
load-bearing: a silent tag shift would not crash anything, it would just
produce confidently wrong gold, and every model would then be scored against
it forever.

So the tests pin the inferred semantics against hand-written records whose
correct output is obvious by inspection, and separately assert the properties
that must hold across the real converted corpus.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from invoice_extract.data.converters.fatura2 import (
    TAG_BILL_TO,
    TAG_DUE_DATE,
    TAG_INVOICE_DATE,
    TAG_INVOICE_NUMBER,
    TAG_TABLE,
    TAG_TOTAL,
    TAG_VENDOR,
    convert_record,
)
from invoice_extract.data.validate import validate_jsonschema

CORPUS = Path("data/processed/fatura2")


def record(pairs: list[tuple[int, str]], doc_id: str = "1") -> dict:
    """Build a FATURA2-shaped record from (tag, token) pairs."""
    return {
        "ner_tags": [t for t, _ in pairs],
        "tokens": [w for _, w in pairs],
        "bboxes": [[i * 10, 0, i * 10 + 9, 12] for i in range(len(pairs))],
        "id": doc_id,
    }


def convert(pairs: list[tuple[int, str]]) -> dict:
    return convert_record(
        record(pairs), doc_id="d1", image_path="images/clean/d1.png", width=595, height=841
    )


class TestFieldDecoding:
    def test_total_separates_amount_from_label_and_currency(self):
        doc = convert([(TAG_TOTAL, "TOTAL"), (TAG_TOTAL, "441.14"), (TAG_TOTAL, "EUR")])
        assert doc["totals"]["grand_total"] == pytest.approx(441.14)
        assert doc["currency"] == "EUR"

    def test_thousands_separator_in_total(self):
        doc = convert([(TAG_TOTAL, "TOTAL"), (TAG_TOTAL, "12,204.22"), (TAG_TOTAL, "USD")])
        assert doc["totals"]["grand_total"] == pytest.approx(12204.22)
        assert doc["currency"] == "USD"

    def test_invoice_number_drops_its_label_tokens(self):
        doc = convert(
            [
                (TAG_INVOICE_NUMBER, "INVOICE"),
                (TAG_INVOICE_NUMBER, "#"),
                (TAG_INVOICE_NUMBER, "INV/30-14/832"),
            ]
        )
        assert doc["invoice_meta"]["invoice_number"] == "INV/30-14/832"

    def test_invoice_number_with_the_other_label_wording(self):
        doc = convert(
            [
                (TAG_INVOICE_NUMBER, "Invoice"),
                (TAG_INVOICE_NUMBER, "number"),
                (TAG_INVOICE_NUMBER, "9Y2M7d-647"),
            ]
        )
        assert doc["invoice_meta"]["invoice_number"] == "9Y2M7d-647"

    def test_dates_are_normalised_to_iso(self):
        doc = convert(
            [
                (TAG_INVOICE_DATE, "Date"),
                (TAG_INVOICE_DATE, "03-Jan-1994"),
                (TAG_DUE_DATE, "Due"),
                (TAG_DUE_DATE, "Date"),
                (TAG_DUE_DATE, "06-Feb-2007"),
            ]
        )
        assert doc["invoice_meta"]["invoice_date"] == "1994-01-03"
        assert doc["invoice_meta"]["due_date"] == "2007-02-06"

    def test_vendor_name_is_taken_verbatim(self):
        doc = convert([(TAG_VENDOR, "Mclean-Cochran")])
        assert doc["parties"]["vendor"]["name"] == "Mclean-Cochran"

    def test_buyer_splits_name_from_address_at_the_street_number(self):
        doc = convert(
            [
                (TAG_BILL_TO, "BILL_TO"),
                (TAG_BILL_TO, "Sarah"),
                (TAG_BILL_TO, "Ross"),
                (TAG_BILL_TO, "073"),
                (TAG_BILL_TO, "Ayers"),
                (TAG_BILL_TO, "Street"),
            ]
        )
        assert doc["parties"]["buyer"]["name"] == "Sarah Ross"
        assert doc["parties"]["buyer"]["address"] == "073 Ayers Street"

    def test_line_items_are_always_empty(self):
        """Tag 10 is a single "table" placeholder -- FATURA2 does not tokenise
        the line-item table, so there is genuinely nothing to extract."""
        doc = convert([(TAG_TABLE, "table"), (TAG_TOTAL, "TOTAL"), (TAG_TOTAL, "10.00")])
        assert doc["line_items"] == []

    def test_missing_fields_become_null_not_guesses(self):
        doc = convert([(TAG_TOTAL, "TOTAL"), (TAG_TOTAL, "10.00")])
        assert doc["invoice_meta"]["invoice_number"] is None
        assert doc["invoice_meta"]["invoice_date"] is None
        assert doc["parties"]["vendor"].get("name") is None

    def test_grounding_boxes_are_produced_for_present_fields(self):
        doc = convert([(TAG_TOTAL, "TOTAL"), (TAG_TOTAL, "441.14"), (TAG_VENDOR, "Acme")])
        paths = {g["field_path"] for g in doc["grounding"]}
        assert "totals.grand_total" in paths
        assert "parties.vendor.name" in paths
        for g in doc["grounding"]:
            x0, y0, x1, y1 = g["bbox"]
            assert x0 < x1 and y0 < y1

    def test_an_empty_record_still_validates(self):
        doc = convert([])
        assert validate_jsonschema(doc) == []

    def test_source_records_provenance_and_licence(self):
        doc = convert([(TAG_TOTAL, "TOTAL"), (TAG_TOTAL, "1.00")])
        assert doc["source"]["dataset"] == "fatura"
        assert doc["source"]["license"] == "CC-BY-4.0"


@pytest.mark.skipif(
    not glob.glob(str(CORPUS / "canonical" / "*.json")), reason="FATURA2 corpus not converted"
)
class TestConvertedCorpus:
    def _docs(self, n: int = 60) -> list[dict]:
        return [
            json.loads(Path(p).read_text(encoding="utf-8"))
            for p in sorted(glob.glob(str(CORPUS / "canonical" / "*.json")))[:n]
        ]

    def test_every_document_is_schema_valid(self):
        for doc in self._docs():
            assert validate_jsonschema(doc) == [], doc["doc_id"]

    def test_every_document_references_an_image_that_exists(self):
        for doc in self._docs():
            assert (CORPUS / doc["pages"][0]["image_path"]).exists(), doc["doc_id"]

    def test_grand_total_is_almost_always_recovered(self):
        """`TOTAL: <amount> <currency>` is the one pattern that holds across all
        50 layouts; if this regresses, the tag map has shifted."""
        docs = self._docs()
        got = sum(1 for d in docs if d["totals"]["grand_total"] is not None)
        assert got / len(docs) > 0.95

    def test_dates_that_were_extracted_are_iso_formatted(self):
        import re

        for doc in self._docs():
            for key in ("invoice_date", "due_date"):
                value = doc["invoice_meta"][key]
                if value is not None:
                    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), f"{doc['doc_id']}: {value}"

    def test_line_items_are_empty_across_the_corpus(self):
        assert all(d["line_items"] == [] for d in self._docs())

    def test_ocr_gold_is_marked_as_partial_coverage(self):
        """The transcript omits the line-item table, so anything computing CER
        from it would be measuring against a page with a hole in it."""
        paths = sorted(glob.glob(str(CORPUS / "ocr_gold" / "*.json")))
        if not paths:
            pytest.skip("no ocr_gold written")
        gold = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
        assert gold["coverage"] == "annotated_regions_only"
        assert all(w["text"] not in ("table", "logo") for w in gold["words"])
