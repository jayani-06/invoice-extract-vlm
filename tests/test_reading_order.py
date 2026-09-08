"""Tests for visual-line grouping.

This module exists because of a real defect: the renderer originally assigned
a line id per draw call, so a right-aligned value drawn separately from its
left-aligned label landed on its own line. The gold text read

    Invoice No:
    INV-2026-4733

for a row every OCR engine reads as one line, which would have inflated CER
and WER with pure line-break noise. These tests pin the geometric behaviour
that replaced it.
"""

from __future__ import annotations

from invoice_extract.reading_order import group_lines, reading_order_text

BBOX = lambda item: item[1]  # noqa: E731
TEXT = lambda item: item[0]  # noqa: E731


def word(text: str, x0: float, y0: float, width: float = 40, height: float = 12):
    return (text, (x0, y0, x0 + width, y0 + height))


class TestGroupLines:
    def test_label_and_far_right_value_share_a_line(self):
        """The exact defect this module was written for: two words on the same
        visual row but drawn far apart horizontally."""
        items = [word("Invoice", 100, 500), word("INV-2026-4733", 900, 500, width=180)]
        lines = group_lines(items, BBOX)
        assert len(lines) == 1
        assert [TEXT(i) for i in lines[0]] == ["Invoice", "INV-2026-4733"]

    def test_rows_separated_vertically_become_separate_lines(self):
        items = [word("first", 100, 100), word("second", 100, 140)]
        assert len(group_lines(items, BBOX)) == 2

    def test_slight_vertical_offset_still_counts_as_one_line(self):
        """Baseline jitter from mixed font sizes must not split a row."""
        items = [word("Total", 100, 500), word("1234.00", 400, 503)]
        assert len(group_lines(items, BBOX)) == 1

    def test_words_are_ordered_left_to_right_within_a_line(self):
        items = [word("third", 300, 100), word("first", 100, 100), word("second", 200, 100)]
        assert [TEXT(i) for i in group_lines(items, BBOX)[0]] == ["first", "second", "third"]

    def test_lines_are_ordered_top_to_bottom(self):
        items = [word("bottom", 100, 900), word("top", 100, 100), word("middle", 100, 500)]
        lines = group_lines(items, BBOX)
        assert [TEXT(ln[0]) for ln in lines] == ["top", "middle", "bottom"]

    def test_tolerance_scales_with_text_size(self):
        """A 4px offset splits small text but not large text -- the rule is
        relative to the median height, so it behaves the same at any dpi."""
        small = [word("a", 0, 0, height=6), word("b", 100, 8, height=6)]
        large = [word("a", 0, 0, height=40), word("b", 100, 8, height=40)]
        assert len(group_lines(small, BBOX)) == 2
        assert len(group_lines(large, BBOX)) == 1

    def test_empty_input(self):
        assert group_lines([], BBOX) == []

    def test_single_item(self):
        assert len(group_lines([word("solo", 0, 0)], BBOX)) == 1

    def test_every_item_appears_exactly_once(self):
        items = [word(f"w{i}", (i % 5) * 100, (i // 5) * 40) for i in range(20)]
        grouped = [i for line in group_lines(items, BBOX) for i in line]
        assert sorted(TEXT(i) for i in grouped) == sorted(TEXT(i) for i in items)

    def test_a_drifting_row_does_not_absorb_the_next_one(self):
        """Words stepping down a few pixels at a time must not chain into one
        runaway line -- the running centre is a mean, not the latest value."""
        items = [word(f"w{i}", i * 60, i * 5) for i in range(12)]
        lines = group_lines(items, BBOX)
        assert len(lines) > 1


class TestReadingOrderText:
    def test_renders_rows_as_lines(self):
        items = [
            word("Invoice", 100, 500),
            word("No:", 200, 500),
            word("INV-1", 900, 500),
            word("Date:", 100, 540),
            word("2026-01-12", 900, 540),
        ]
        assert reading_order_text(items, BBOX, TEXT) == "Invoice No: INV-1\nDate: 2026-01-12"

    def test_empty(self):
        assert reading_order_text([], BBOX, TEXT) == ""
