"""Tests for the synthetic invoice renderer.

The renderer is the source of Phase 1's ground truth, so these tests are less
about "does it draw" and more about "is the ground truth it emits actually
true" -- a word box that does not sit on its glyphs would silently poison
every CER and detection number downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from invoice_extract.render import fonts
from invoice_extract.render.invoice_renderer import InvoiceRenderer, LayoutStyle
from invoice_extract.render.layout import Canvas, merge_boxes

SCHEMA_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "schema" / "examples" / "example_invoice.json"
)


@pytest.fixture(scope="module")
def doc() -> dict:
    return json.loads(SCHEMA_EXAMPLE.read_text())


@pytest.fixture(scope="module")
def rendered(doc):
    return InvoiceRenderer(dpi=150).render(doc, seed=42)


def ink_fraction(image: np.ndarray, bbox, threshold: int = 160) -> float:
    """Share of dark pixels inside a box."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = image[y0:y1, x0:x1]
    return float((patch < threshold).mean()) if patch.size else 0.0


class TestFonts:
    def test_resolves_a_usable_font(self):
        assert fonts.resolve("sans", "regular").is_file()

    def test_falls_back_across_families(self):
        # Every declared family must resolve to *something*, even on a machine
        # that has only one typeface installed.
        for family in fonts.FAMILIES:
            assert fonts.resolve(family, "bold").is_file()

    def test_available_families_are_deduplicated(self):
        families = fonts.available_families()
        assert families
        paths = {str(fonts.resolve(f, "regular")) for f in families}
        assert len(paths) == len(families), "families collapsing to the same file should be merged"


class TestCanvas:
    def test_word_boxes_sit_on_their_glyphs(self):
        canvas = Canvas(600, 120)
        font = fonts.load("sans", "regular", 32)
        boxes = canvas.text((20, 30), "Invoice Total 1234", font)
        image = np.array(canvas.image)

        assert len(boxes) == 3
        for box in boxes:
            assert ink_fraction(image, box.bbox) > 0.05, f"{box.text!r} box contains no ink"

    def test_word_boxes_do_not_overlap(self):
        canvas = Canvas(600, 120)
        boxes = canvas.text((20, 30), "alpha beta gamma", fonts.load("sans", "regular", 28))
        for a, b in zip(boxes, boxes[1:], strict=False):
            assert a.bbox[2] <= b.bbox[0] + 1e-6

    @pytest.mark.parametrize("align,anchor", [("left", 100.0), ("right", 500.0), ("center", 300.0)])
    def test_alignment_anchors(self, align, anchor):
        canvas = Canvas(600, 120)
        boxes = canvas.text(
            (anchor, 30), "hello world", fonts.load("sans", "regular", 24), align=align
        )
        merged = merge_boxes(boxes)
        if align == "left":
            assert merged[0] == pytest.approx(anchor, abs=1.0)
        elif align == "right":
            assert merged[2] == pytest.approx(anchor, abs=1.0)
        else:
            assert (merged[0] + merged[2]) / 2 == pytest.approx(anchor, abs=1.5)

    def test_wrapped_text_respects_max_width(self):
        canvas = Canvas(600, 400)
        font = fonts.load("sans", "regular", 20)
        boxes, _ = canvas.wrapped_text(
            (20, 20), "one two three four five six seven eight nine ten", font, 200
        )
        assert len({b.line_id for b in boxes}) > 1
        for box in boxes:
            assert box.bbox[2] <= 20 + 200 + 2

    def test_empty_text_draws_nothing(self):
        canvas = Canvas(100, 50)
        assert canvas.text((10, 10), "   ", fonts.load("sans", "regular", 12)) == []
        assert canvas.words == []

    def test_ellipsize_keeps_text_inside_max_width(self):
        canvas = Canvas(400, 60)
        font = fonts.load("sans", "regular", 20)
        boxes = canvas.text(
            (10, 10), "a very long description that will not fit", font, max_width=100
        )
        assert merge_boxes(boxes)[2] <= 10 + 100 + 2


class TestRenderer:
    def test_produces_a_page_with_ground_truth(self, rendered):
        assert rendered.size[0] > 500 and rendered.size[1] > 700
        assert len(rendered.words) > 20
        assert rendered.grounding

    def test_every_word_box_contains_ink(self, rendered):
        image = np.array(rendered.image)
        empty = [w for w in rendered.words if ink_fraction(image, w.bbox) < 0.02]
        assert (
            not empty
        ), f"{len(empty)} word boxes contain no ink, e.g. {[w.text for w in empty[:5]]}"

    def test_critical_fields_are_grounded(self, rendered):
        paths = {g["field_path"] for g in rendered.grounding}
        for required in (
            "parties.vendor.name",
            "invoice_meta.invoice_number",
            "invoice_meta.invoice_date",
            "totals.grand_total",
        ):
            assert required in paths, f"{required} was never drawn with a field_path"

    def test_grounding_boxes_lie_inside_the_page(self, rendered):
        w, h = rendered.size
        for g in rendered.grounding:
            x0, y0, x1, y1 = g["bbox"]
            assert 0 <= x0 < x1 <= w + 1
            assert 0 <= y0 < y1 <= h + 1

    def test_line_item_values_reach_the_page(self, doc, rendered):
        """A dropped table column would still leave a plausible-looking invoice,
        so assert the actual values are present rather than trusting the layout."""
        text = rendered.text
        for item in doc["line_items"]:
            if item.get("description"):
                assert item["description"].split()[0] in text

    def test_seed_is_deterministic(self, doc):
        a = InvoiceRenderer(dpi=100).render(doc, seed=7)
        b = InvoiceRenderer(dpi=100).render(doc, seed=7)
        assert a.style == b.style
        assert np.array_equal(np.array(a.image), np.array(b.image))
        assert [w.text for w in a.words] == [w.text for w in b.words]

    def test_different_seeds_give_different_layouts(self, doc):
        styles = {InvoiceRenderer(dpi=100).render(doc, seed=s).style for s in range(25)}
        assert len(styles) > 5, "layout sampling is not exploring the space"

    @pytest.mark.parametrize("header_style", ["left", "right", "centered"])
    @pytest.mark.parametrize("rule_style", ["grid", "horizontal", "none"])
    def test_layout_variants_all_render(self, doc, header_style, rule_style):
        style = LayoutStyle(header_style=header_style, rule_style=rule_style)
        result = InvoiceRenderer(dpi=100).render(doc, style=style)
        assert len(result.words) > 15
        image = np.array(result.image)
        assert (image < 160).any(), "page rendered blank"

    def test_reading_order_text_starts_at_the_top(self, rendered):
        first_line = rendered.text.splitlines()[0]
        top_words = sorted(rendered.words, key=lambda w: w.bbox[1])[:3]
        assert any(w.text in first_line for w in top_words)

    def test_handles_a_document_with_no_optional_blocks(self):
        minimal = {
            "doc_id": "minimal",
            "document_type": "invoice",
            "parties": {"vendor": {"name": "Solo Vendor"}},
            "invoice_meta": {"invoice_number": "X-1", "invoice_date": "2026-01-01"},
            "line_items": [],
            "tax_lines": [],
            "totals": {"grand_total": 100.0},
        }
        result = InvoiceRenderer(dpi=100).render(minimal, seed=3)
        assert "Solo" in result.text
        assert any(g["field_path"] == "totals.grand_total" for g in result.grounding)
