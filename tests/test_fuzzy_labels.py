"""Label matching that survives OCR damage.

The regex baseline scores 1.000 against the renderer's own text and 0.653
through a real engine. The mechanism is mechanical rather than subtle: OCR
returns `lnvoice N0:` and `invoice\\s*(?:no|number)` does not match, so the
field is not misread — it is **lost**, and comes out null.

The danger in fixing this is obvious: a matcher loose enough to accept
`lnvoice N0` may also accept `Invoice Total` as `Invoice Date`. Measured, plain
similarity cannot separate them — real damage scores 71–86 and that confusable
pair scores 72, so the ranges overlap and no threshold exists. Folding
OCR-confusable glyphs first is what makes a threshold possible at all, because
`t` and `d` are not confusable and survive the fold.

These tests pin both halves: damaged labels are found, and labels that merely
resemble each other stay apart.
"""

from __future__ import annotations

import pytest

from invoice_extract.models.rules import RuleBasedExtractor, _fold, _fuzzy_label_match


class TestDamagedLabelsAreRecovered:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("lnvoice N0: INV-2026-004001", "invoice_number"),
            ("1nvoice Date: 2026-01-18", "invoice_date"),
            ("Inv0ice Numbor: INV-9", "invoice_number"),
            ("Bi11 No: B-77", "invoice_number"),
            ("P0 Number: PO-2113", "po_number"),
            ("Due Date: 2026-03-04", "due_date"),
            # A bare DATE: is what receipts print, and what cost this
            # extractor 157 of 157 SROIE documents before it was added.
            ("DATE: 25/12/2018", "invoice_date"),
        ],
    )
    def test_it_finds_the_field(self, line, expected):
        match = _fuzzy_label_match(line)
        assert match is not None, f"{line!r} matched nothing"
        assert match[0] == expected

    def test_the_value_survives_intact(self):
        assert _fuzzy_label_match("lnvoice N0: INV-2026-004001")[1] == "INV-2026-004001"


class TestLabelsThatMerelyResembleEachOtherStayApart:
    @pytest.mark.parametrize(
        "line",
        [
            "Invoice Total: 36,694.41",  # total vs date -- the dangerous pair
            "Grand Total: 9.00",
            "Thank you for your business: come again",
            "Authorised Signatory: ____",
            "Bank: Axis Bank",
        ],
    )
    def test_it_does_not_match(self, line):
        assert _fuzzy_label_match(line) is None, f"{line!r} was wrongly claimed"

    def test_a_long_line_is_never_a_label(self):
        """A label is short. A sentence containing a colon is not one, and
        comparing it would only manufacture matches."""
        line = "Please note that all goods sold are final and non returnable: see terms"
        assert _fuzzy_label_match(line) is None

    def test_a_line_with_no_separator_is_not_a_label(self):
        assert _fuzzy_label_match("INVOICE") is None

    def test_a_label_with_no_value_is_not_a_match(self):
        assert _fuzzy_label_match("Invoice No:") is None


class TestGlyphFolding:
    def test_confusable_glyphs_collapse(self):
        assert _fold("lnv0ice n0") == _fold("invoice no")
        assert _fold("bi11") == _fold("bill")

    def test_non_confusable_glyphs_do_not(self):
        """`t` and `d` are not confusable, which is exactly why folding makes
        a threshold possible: it closes the gap for real damage without
        closing it for `total` versus `date`."""
        assert _fold("invoice total") != _fold("invoice date")


class TestExactMatchingStillWins:
    def test_a_clean_line_is_unaffected(self):
        """The fuzzy pass only sees lines every exact pattern already failed
        on, so a correctly-matched line can never be re-routed by a near
        miss."""
        text = "\n".join(
            [
                "Acme Supplies",
                "Invoice No: INV-2026-004001",
                "Invoice Date: 2026-01-18",
                "Grand Total: 36694.41",
            ]
        )
        out = RuleBasedExtractor().predict_from_text(text)
        meta = out["invoice_meta"]
        assert meta["invoice_number"] == "INV-2026-004001"
        assert meta["invoice_date"] == "2026-01-18"

    def test_a_damaged_line_is_recovered_end_to_end(self):
        text = "\n".join(
            [
                "Acme Supplies",
                "lnvoice N0: INV-2026-004001",
                "1nvoice Date: 2026-01-18",
                "Grand Total: 36694.41",
            ]
        )
        out = RuleBasedExtractor().predict_from_text(text)
        meta = out["invoice_meta"]
        assert meta["invoice_number"] == "INV-2026-004001"
        assert meta["invoice_date"] == "2026-01-18"

    def test_a_damaged_total_is_not_read_as_a_date(self):
        """The regression this guards: `Invoice Total` must not populate
        `invoice_date` just because the line was otherwise unclaimed."""
        out = RuleBasedExtractor().predict_from_text("Acme\nInvoice Total: 36694.41")
        assert out["invoice_meta"]["invoice_date"] is None
