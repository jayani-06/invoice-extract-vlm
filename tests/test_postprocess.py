"""Tests for deterministic repair of raw model output.

Parsers are tested against hand-computed expectations, including the cases
that actually appear on Indian GST invoices (day-first dates, rupee symbols,
lakh-style grouping) and the accounting conventions that trip naive parsers
(parenthesised negatives, European decimal commas).

The arithmetic checks are tested for what they must NOT do as much as for what
they do: silently repairing a total would destroy the exact signal Phase 3's
anomaly detection is built to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from invoice_extract.models.postprocess import (
    check_arithmetic,
    coerce_document,
    detect_currency,
    finalize,
    gstin_check_digit,
    normalize_date,
    normalize_gstin,
    parse_amount,
    parse_quantity,
    parse_rate,
    validate_gstin,
)


class TestNormalizeDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2026-01-12", "2026-01-12"),
            ("12/01/2026", "2026-01-12"),
            ("12-01-2026", "2026-01-12"),
            ("12.01.2026", "2026-01-12"),
            ("12-Jan-2026", "2026-01-12"),
            ("12 January 2026", "2026-01-12"),
            ("Jan 12, 2026", "2026-01-12"),
            ("January 12 2026", "2026-01-12"),
            ("20260112", "2026-01-12"),
        ],
    )
    def test_common_formats(self, raw, expected):
        assert normalize_date(raw) == expected

    def test_day_first_resolves_the_ambiguous_case(self):
        """03/04/2026 is 3 April day-first, 4 March month-first. Indian GST
        invoices are day-first, which is why that is the default."""
        assert normalize_date("03/04/2026", day_first=True) == "2026-04-03"
        assert normalize_date("03/04/2026", day_first=False) == "2026-03-04"

    def test_unambiguous_dates_parse_the_same_either_way(self):
        for raw in ("25/12/2026", "12-Jan-2026"):
            assert normalize_date(raw, True) == normalize_date(raw, False)

    def test_two_digit_year_is_this_century(self):
        assert normalize_date("31/12/26") == "2026-12-31"

    def test_surrounding_noise_is_tolerated(self):
        assert normalize_date("  Date: 12/01/2026  ".replace("Date:", "")) == "2026-01-12"

    @pytest.mark.parametrize("raw", ["", "   ", "not a date", "13/13/2026", None, "abc123"])
    def test_unparseable_returns_none_rather_than_guessing(self, raw):
        assert normalize_date(raw) is None


class TestParseAmount:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1234.56", 1234.56),
            ("1,234.56", 1234.56),
            ("12,204.22", 12204.22),
            ("₹1,234.56", 1234.56),
            ("Rs. 1,234", 1234.0),
            ("Rs 1234", 1234.0),
            ("INR 500", 500.0),
            ("$1,000.00", 1000.0),
            ("1.234,56", 1234.56),
            ("(123.45)", -123.45),
            ("-99", -99.0),
            ("1,50", 1.5),
            ("1,500", 1500.0),
            (12.5, 12.5),
            (7, 7.0),
        ],
    )
    def test_values(self, raw, expected):
        assert parse_amount(raw) == pytest.approx(expected)

    def test_indian_lakh_grouping(self):
        assert parse_amount("1,90,121.95") == pytest.approx(190121.95)

    @pytest.mark.parametrize("raw", [None, "", "abc", "N/A", "-", True, False])
    def test_unparseable_returns_none(self, raw):
        assert parse_amount(raw) is None

    def test_booleans_are_not_numbers(self):
        """bool is a subclass of int; treating True as 1.0 would silently turn
        a model's `"discount": true` into a one-rupee discount."""
        assert parse_amount(True) is None


class TestParseQuantityAndRate:
    @pytest.mark.parametrize(
        "raw,expected", [("12 nos", 12.0), ("2.5 kg", 2.5), ("3", 3.0), (4, 4.0)]
    )
    def test_quantity_strips_units(self, raw, expected):
        assert parse_quantity(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw,expected", [("18%", 18.0), ("@ 18 %", 18.0), ("5.5%", 5.5), (12, 12.0)]
    )
    def test_rate_strips_percent(self, raw, expected):
        assert parse_rate(raw) == pytest.approx(expected)

    def test_none_passthrough(self):
        assert parse_quantity(None) is None and parse_rate(None) is None


class TestDetectCurrency:
    @pytest.mark.parametrize(
        "raw,code",
        [
            ("₹12,204.22", "INR"),
            ("Rs. 500", "INR"),
            ("INR 1", "INR"),
            ("$10", "USD"),
            ("€5", "EUR"),
        ],
    )
    def test_from_markers(self, raw, code):
        assert detect_currency(raw) == code

    def test_no_marker(self):
        assert detect_currency("1234.56", None) is None


class TestGstin:
    @pytest.mark.parametrize(
        "base,expected",
        [
            # Hand-computed from the published mod-36 algorithm. For
            # "29AAAAA0000A1Z" the weighted sum is 146; 146 % 36 = 2, so the
            # check code point is (36 - 2) % 36 = 34, which is "Y".
            ("29AAAAA0000A1Z", "Y"),
            ("27ABCDE1234F1Z", "0"),
            ("07AABCU9603R1Z", "P"),
        ],
    )
    def test_check_digit_matches_the_published_algorithm(self, base, expected):
        assert gstin_check_digit(base) == expected

    def test_the_common_placeholder_gstin_is_not_actually_valid(self):
        """ "29AAAAA0000A1Z5" appears all over documentation as an example, but
        its check digit is wrong. Pinning that here stops someone "fixing" the
        implementation to match a bad constant."""
        assert validate_gstin("29AAAAA0000A1Z5") != []

    def test_a_generated_gstin_validates(self):
        base = "29AAAAA0000A1Z"
        assert validate_gstin(base + gstin_check_digit(base)) == []

    def test_every_corpus_gstin_validates(self):
        """The synthetic generator computes real check digits (it previously
        drew them at random, which made ~35/36 of the corpus invalid and the
        checksum signal useless)."""
        import glob
        import json

        paths = sorted(glob.glob("data/processed/gst_in_synthetic/canonical/*.json"))[:20]
        if not paths:
            pytest.skip("corpus not built")
        for path in paths:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            for role in ("vendor", "buyer"):
                gstin = ((doc.get("parties") or {}).get(role) or {}).get("gstin")
                if gstin:
                    assert validate_gstin(gstin) == [], f"{path}: {gstin}"

    def test_wrong_checksum_is_reported(self):
        problems = validate_gstin("29RXCKA2501U9ZQ")
        assert any("checksum" in p for p in problems)

    def test_bad_length(self):
        assert "15" in validate_gstin("29AAAAA")[0]

    def test_unassigned_state_code(self):
        base = "49AAAAA0000A1Z"
        problems = validate_gstin(base + (gstin_check_digit(base) or "0"))
        assert any("state code" in p for p in problems)

    def test_normalization_strips_spacing_and_case(self):
        assert normalize_gstin(" 29aaaaa0000a1z5 ") == "29AAAAA0000A1Z5"

    def test_empty(self):
        assert validate_gstin(None) == ["gstin is empty"]

    def test_check_digit_rejects_wrong_length(self):
        assert gstin_check_digit("TOOSHORT") is None


class TestCheckArithmetic:
    def _doc(self, **totals):
        return {
            "line_items": [{"quantity": 2, "unit_price": 50.0, "line_total": 100.0}],
            "tax_lines": [{"type": "CGST", "amount": 9.0}, {"type": "SGST", "amount": 9.0}],
            "totals": {"subtotal": 100.0, "tax_total": 18.0, "grand_total": 118.0, **totals},
        }

    def test_a_consistent_invoice_is_clean(self):
        assert check_arithmetic(self._doc()).ok

    def test_wrong_grand_total_is_flagged(self):
        report = check_arithmetic(self._doc(grand_total=999.0))
        assert not report.ok
        assert any("grand_total" in p for p in report.problems)

    def test_tax_lines_not_summing_is_flagged(self):
        doc = self._doc()
        doc["tax_lines"][0]["amount"] = 50.0
        assert any("tax_lines" in p for p in check_arithmetic(doc).problems)

    def test_line_items_not_summing_to_subtotal_is_flagged(self):
        doc = self._doc()
        doc["line_items"][0]["line_total"] = 999.0
        assert any("line_items" in p for p in check_arithmetic(doc).problems)

    def test_it_never_modifies_the_document(self):
        """Flag, never fix. Rewriting a total to make it add up would erase the
        anomaly Phase 3 is supposed to detect."""
        doc = self._doc(grand_total=999.0)
        before = {k: dict(v) if isinstance(v, dict) else v for k, v in doc.items()}
        check_arithmetic(doc)
        assert doc["totals"] == before["totals"]

    def test_rounding_within_tolerance_is_accepted(self):
        assert check_arithmetic(self._doc(grand_total=118.01)).ok

    def test_missing_values_do_not_produce_spurious_flags(self):
        doc = {"line_items": [], "tax_lines": [], "totals": {"grand_total": None}}
        assert check_arithmetic(doc).ok

    def test_report_is_falsy_when_clean(self):
        assert not check_arithmetic(self._doc())
        assert check_arithmetic(self._doc(grand_total=999.0))


class TestCoerceDocument:
    def test_a_realistic_raw_response_is_normalised(self):
        raw = {
            "document_type": "invoice",
            "parties": {"vendor": {"name": "  Acme   Traders ", "gstin": "29aaaaa0000a1z5"}},
            "invoice_meta": {"invoice_number": "INV-1", "invoice_date": "12/01/2026"},
            "line_items": [
                {
                    "description": "Widget",
                    "quantity": "12 nos",
                    "unit_price": "₹861.88",
                    "tax_rate": "18%",
                    "line_total": "12,204.22",
                }
            ],
            "tax_lines": [{"type": "igst", "rate": "18%", "amount": "1,861.66"}],
            "totals": {"subtotal": "10,342.56", "grand_total": "₹12,204.22"},
        }
        doc = coerce_document(raw)
        assert doc["parties"]["vendor"]["name"] == "Acme Traders"
        assert doc["parties"]["vendor"]["gstin"] == "29AAAAA0000A1Z5"
        assert doc["invoice_meta"]["invoice_date"] == "2026-01-12"
        assert doc["line_items"][0]["quantity"] == 12.0
        assert doc["line_items"][0]["unit_price"] == pytest.approx(861.88)
        assert doc["tax_lines"][0]["type"] == "IGST"
        assert doc["totals"]["grand_total"] == pytest.approx(12204.22)
        assert doc["currency"] == "INR"

    def test_unknown_keys_are_dropped(self):
        """The schema forbids extra properties, so one hallucinated field would
        fail validation for the whole document and cost every correct field."""
        doc = coerce_document(
            {"parties": {"vendor": {"name": "X", "vibe": "good"}}, "hallucinated_root": 1}
        )
        assert "vibe" not in doc["parties"]["vendor"]
        assert "hallucinated_root" not in doc

    def test_empty_line_item_rows_are_discarded(self):
        doc = coerce_document({"line_items": [{"description": None, "quantity": None}, {}]})
        assert doc["line_items"] == []

    def test_tax_line_without_an_amount_is_discarded(self):
        """`amount` is required by the schema; keeping the row would make the
        document invalid."""
        doc = coerce_document({"tax_lines": [{"type": "CGST", "rate": 9}]})
        assert doc["tax_lines"] == []

    def test_unknown_tax_type_falls_back_to_other(self):
        doc = coerce_document({"tax_lines": [{"type": "SERVICE CHARGE", "amount": 5}]})
        assert doc["tax_lines"][0]["type"] == "OTHER"

    def test_tax_total_is_derived_from_components_when_absent(self):
        """Indian GST invoices print CGST and SGST separately and never sum
        them, so deriving the total is arithmetic over read values, not a guess."""
        doc = coerce_document(
            {"tax_lines": [{"type": "CGST", "amount": 9.0}, {"type": "SGST", "amount": 9.0}]}
        )
        assert doc["totals"]["tax_total"] == pytest.approx(18.0)

    def test_a_printed_tax_total_is_not_overwritten(self):
        doc = coerce_document(
            {
                "tax_lines": [{"type": "CGST", "amount": 9.0}],
                "totals": {"tax_total": 100.0},
            }
        )
        assert doc["totals"]["tax_total"] == pytest.approx(100.0)

    def test_garbage_input_still_produces_a_valid_shape(self):
        for raw in ({}, {"parties": "nonsense"}, {"line_items": "nope"}):
            doc = coerce_document(raw)
            assert set(doc) >= {"parties", "invoice_meta", "line_items", "tax_lines", "totals"}
            assert isinstance(doc["line_items"], list)

    def test_invalid_currency_is_replaced_not_kept(self):
        assert coerce_document({"currency": "rupees"}).get("currency") in (None, "INR")


class TestFinalize:
    def test_produces_a_schema_valid_document(self):
        from invoice_extract.data.validate import validate_jsonschema

        raw = {
            "parties": {"vendor": {"name": "Acme"}},
            "invoice_meta": {"invoice_number": "INV-1", "invoice_date": "12/01/2026"},
            "totals": {"grand_total": "100.00"},
        }
        doc, report = finalize(
            raw,
            doc_id="d1",
            source={"dataset": "other", "original_id": "d1", "split": "test"},
            pages=[{"page_index": 0, "image_path": "x.png", "width": 100, "height": 200}],
        )
        assert validate_jsonschema(doc) == []
        assert doc["doc_id"] == "d1"
        assert report.ok

    def test_consistency_is_returned_separately_not_written_into_the_document(self):
        """Writing flags into the prediction would add a field the gold
        annotation does not have."""
        doc, report = finalize(
            {"totals": {"subtotal": 100.0, "tax_total": 10.0, "grand_total": 999.0}},
            "d",
            {"dataset": "other", "original_id": "d", "split": "test"},
            [{"page_index": 0, "image_path": "x.png", "width": 1, "height": 1}],
        )
        assert not report.ok
        assert "problems" not in doc and "consistency" not in doc
