"""Render a canonical-schema invoice document to a page image + word-level
ground truth.

Why render instead of annotate: the research gap this project targets sits
*downstream* of extraction (RAG enrichment + vendor validation), so spending
weeks on manual OCR annotation to measure Phase 1 would be misallocated
effort. Rendering gives pixel-exact word boxes for free and, combined with
`invoice_extract.degrade`, a paired (clean, degraded) corpus -- which is
precisely what the preprocessing ablation needs and what real scan corpora
cannot provide, since you never have the clean original of a noisy scan.

Layout diversity is sampled, not templated. `LayoutStyle.sample(rng)` draws
from independent axes (typeface, rule style, block placement, alignment,
density, shading), so the corpus covers a combinatorial layout space rather
than N fixed templates. The FATURA finding is that template-tuned extraction
collapses on unseen layouts; a fixed-template synthetic set would quietly
reproduce that same weakness inside our own evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from invoice_extract.reading_order import reading_order_text
from invoice_extract.render import fonts
from invoice_extract.render.layout import Canvas, WordBox, merge_boxes

RuleStyle = Literal["grid", "horizontal", "none"]
HeaderStyle = Literal["left", "right", "centered"]

# A4. 150 dpi is the low end of flatbed-scanner output and keeps corpus
# generation fast; 300 dpi matches the Phase 1 ingestion target and is what
# the ablation runs at.
A4_INCHES = (8.27, 11.69)


def page_size(dpi: int) -> tuple[int, int]:
    return (round(A4_INCHES[0] * dpi), round(A4_INCHES[1] * dpi))


@dataclass(frozen=True)
class LayoutStyle:
    """One point in the layout space.

    Sampled per document and recorded in the corpus manifest, so any rendering
    is exactly reproducible and per-layout accuracy can be broken out during
    evaluation.
    """

    font_family: str = "sans"
    base_pt: int = 10
    header_style: HeaderStyle = "left"
    rule_style: RuleStyle = "horizontal"
    shade_table_header: bool = True
    totals_on_right: bool = True
    show_hsn_column: bool = True
    show_buyer_block: bool = True
    margin_ratio: float = 0.075
    line_spacing: float = 1.35
    label_colon: bool = True
    uppercase_labels: bool = False

    @classmethod
    def sample(cls, rng: random.Random, families: list[str] | None = None) -> LayoutStyle:
        families = families or fonts.available_families()
        return cls(
            font_family=rng.choice(families),
            base_pt=rng.choice([9, 10, 10, 11, 12]),
            header_style=rng.choice(["left", "right", "centered"]),
            rule_style=rng.choice(["grid", "horizontal", "horizontal", "none"]),
            shade_table_header=rng.random() < 0.7,
            totals_on_right=rng.random() < 0.8,
            show_hsn_column=rng.random() < 0.75,
            show_buyer_block=rng.random() < 0.85,
            margin_ratio=rng.choice([0.055, 0.065, 0.075, 0.09]),
            line_spacing=rng.choice([1.2, 1.35, 1.5]),
            label_colon=rng.random() < 0.7,
            uppercase_labels=rng.random() < 0.35,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "font_family": self.font_family,
            "base_pt": self.base_pt,
            "header_style": self.header_style,
            "rule_style": self.rule_style,
            "shade_table_header": self.shade_table_header,
            "totals_on_right": self.totals_on_right,
            "show_hsn_column": self.show_hsn_column,
            "show_buyer_block": self.show_buyer_block,
            "margin_ratio": self.margin_ratio,
            "line_spacing": self.line_spacing,
            "label_colon": self.label_colon,
            "uppercase_labels": self.uppercase_labels,
        }


@dataclass
class RenderResult:
    image: Image.Image
    words: list[WordBox]
    style: LayoutStyle
    dpi: int
    grounding: list[dict[str, Any]]

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def text(self) -> str:
        """Reading-order plain text, used as the reference string for CER/WER.

        Lines come from geometry, not from `WordBox.line_id`. A line id is per
        draw call, so a right-aligned value drawn separately from its label
        would land on its own line -- producing a reference no OCR engine could
        ever reproduce, and inflating CER with pure line-break noise. See
        `invoice_extract.reading_order`.
        """
        return reading_order_text(self.words, lambda w: w.bbox, lambda w: w.text)


def _fmt_amount(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_qty(v: Any) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_rate(v: Any) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):g}%"
    except (TypeError, ValueError):
        return str(v)


def _fmt_str(v: Any) -> str:
    return "" if v is None else str(v)


LINE_ITEM_FORMATTERS = {
    "line_no": _fmt_str,
    "description": _fmt_str,
    "hsn_sac_code": _fmt_str,
    "quantity": _fmt_qty,
    "unit_price": _fmt_amount,
    "tax_rate": _fmt_rate,
    "line_total": _fmt_amount,
}


class InvoiceRenderer:
    """Renders one canonical document.

    Stateless apart from the dpi, so a single instance is safe to reuse across
    a whole corpus.
    """

    def __init__(self, dpi: int = 300) -> None:
        self.dpi = dpi

    # ------------------------------------------------------------- public

    def render(
        self, doc: dict[str, Any], style: LayoutStyle | None = None, seed: int | None = None
    ) -> RenderResult:
        rng = random.Random(seed if seed is not None else doc.get("doc_id", ""))
        style = style or LayoutStyle.sample(rng)

        w, h = page_size(self.dpi)
        canvas = Canvas(w, h)
        margin = w * style.margin_ratio
        scale = self.dpi / 72.0  # points -> pixels

        def f(weight: str = "regular", pt: float | None = None):
            return fonts.load(
                style.font_family, weight, max(6, round((pt or style.base_pt) * scale))
            )

        y = margin
        y = self._draw_header(canvas, doc, style, margin, y, w, f)
        y = self._draw_parties(canvas, doc, style, margin, y, w, f)
        y = self._draw_meta(canvas, doc, style, margin, y, w, f)
        y = self._draw_items_table(canvas, doc, style, margin, y, w, f)
        y = self._draw_totals(canvas, doc, style, margin, y, w, f)
        self._draw_footer(canvas, doc, style, margin, y, w, h, f)

        return RenderResult(
            image=canvas.image,
            words=canvas.words,
            style=style,
            dpi=self.dpi,
            grounding=self._grounding(canvas.words),
        )

    # -------------------------------------------------------------- parts

    def _label(self, text: str, style: LayoutStyle) -> str:
        t = text.upper() if style.uppercase_labels else text
        return t + ":" if style.label_colon else t

    def _draw_header(self, c, doc, style, margin, y, w, f) -> float:
        vendor = (doc.get("parties") or {}).get("vendor") or {}
        title = "TAX INVOICE" if doc.get("document_type") == "invoice" else "RECEIPT"
        title_font = f("bold", style.base_pt * 1.9)
        name_font = f("bold", style.base_pt * 1.35)
        body = f("regular")
        right_edge = w - margin

        if style.header_style == "centered":
            c.text((w / 2, y), title, title_font, align="center")
            y += title_font.size * 1.5
            c.text(
                (w / 2, y),
                vendor.get("name") or "",
                name_font,
                field_path="parties.vendor.name",
                align="center",
            )
            y += name_font.size * 1.35
            block_w = (w - 2 * margin) / 2
            _, y = c.wrapped_text(
                (w / 2 - block_w / 2, y),
                vendor.get("address") or "",
                body,
                block_w,
                field_path="parties.vendor.address",
                line_spacing=style.line_spacing,
            )
            info_x, info_align = margin, "left"
        else:
            left_header = style.header_style == "left"
            name_x, name_align = (margin, "left") if left_header else (right_edge, "right")
            title_x, title_align = (right_edge, "right") if left_header else (margin, "left")

            c.text((title_x, y), title, title_font, align=title_align)
            c.text(
                (name_x, y),
                vendor.get("name") or "",
                name_font,
                field_path="parties.vendor.name",
                align=name_align,
            )
            y += name_font.size * 1.4
            block_w = (w - 2 * margin) * 0.45
            addr_x = margin if left_header else right_edge - block_w
            _, y = c.wrapped_text(
                (addr_x, y),
                vendor.get("address") or "",
                body,
                block_w,
                field_path="parties.vendor.address",
                line_spacing=style.line_spacing,
            )
            info_x = margin if left_header else right_edge - block_w
            info_align = "left"

        for key, label, path in (
            ("gstin", "GSTIN", "parties.vendor.gstin"),
            ("tax_id", "Tax ID", "parties.vendor.tax_id"),
            ("phone", "Phone", "parties.vendor.phone"),
            ("email", "Email", "parties.vendor.email"),
        ):
            if vendor.get(key):
                c.text(
                    (info_x, y),
                    f"{self._label(label, style)} {vendor[key]}",
                    body,
                    field_path=path,
                    align=info_align,
                )
                y += body.size * style.line_spacing

        y += body.size * 0.8
        if style.rule_style != "none":
            c.hline(margin, w - margin, y, width=max(1, round(self.dpi / 150)))
            y += body.size * 0.9
        return y

    def _draw_parties(self, c, doc, style, margin, y, w, f) -> float:
        if not style.show_buyer_block:
            return y
        buyer = (doc.get("parties") or {}).get("buyer")
        if not buyer or not any(buyer.values()):
            return y

        body, bold = f(), f("bold")
        block_w = (w - 2 * margin) * 0.5
        c.text((margin, y), self._label("Bill To", style), bold)
        y += bold.size * style.line_spacing

        if buyer.get("name"):
            c.text((margin, y), buyer["name"], body, field_path="parties.buyer.name")
            y += body.size * style.line_spacing
        if buyer.get("address"):
            _, y = c.wrapped_text(
                (margin, y),
                buyer["address"],
                body,
                block_w,
                field_path="parties.buyer.address",
                line_spacing=style.line_spacing,
            )
        if buyer.get("gstin"):
            c.text(
                (margin, y),
                f"{self._label('GSTIN', style)} {buyer['gstin']}",
                body,
                field_path="parties.buyer.gstin",
            )
            y += body.size * style.line_spacing
        return y + body.size * 0.8

    def _draw_meta(self, c, doc, style, margin, y, w, f) -> float:
        meta = doc.get("invoice_meta") or {}
        body, bold = f(), f("bold")
        rows = [
            ("Invoice No", meta.get("invoice_number"), "invoice_meta.invoice_number"),
            ("Invoice Date", meta.get("invoice_date"), "invoice_meta.invoice_date"),
            ("Due Date", meta.get("due_date"), "invoice_meta.due_date"),
            ("PO Number", meta.get("po_number"), "invoice_meta.po_number"),
            ("Payment Terms", meta.get("payment_terms"), "invoice_meta.payment_terms"),
        ]
        rows = [r for r in rows if r[1]]
        if not rows:
            return y

        # The meta block sits opposite the totals block, so across the corpus
        # the two do not always land on the same side of the page.
        block_w = (w - 2 * margin) * 0.34
        x_label = margin if not style.totals_on_right else w - margin - block_w
        x_value = x_label + block_w
        for label, value, path in rows:
            c.text((x_label, y), self._label(label, style), bold)
            c.text((x_value, y), str(value), body, field_path=path, align="right")
            y += body.size * style.line_spacing
        return y + body.size * 0.9

    def _columns(self, style: LayoutStyle) -> list[tuple[str, float, str, str]]:
        """(header, relative width, align, line-item key) for the items table."""
        cols: list[tuple[str, float, str, str]] = [
            ("#", 0.05, "left", "line_no"),
            ("Description", 0.36 if style.show_hsn_column else 0.46, "left", "description"),
        ]
        if style.show_hsn_column:
            cols.append(("HSN/SAC", 0.10, "left", "hsn_sac_code"))
        cols += [
            ("Qty", 0.08, "right", "quantity"),
            ("Rate", 0.13, "right", "unit_price"),
            ("Tax %", 0.08, "right", "tax_rate"),
            ("Amount", 0.16, "right", "line_total"),
        ]
        return cols

    def _draw_items_table(self, c, doc, style, margin, y, w, f) -> float:
        items = doc.get("line_items") or []
        if not items:
            return y

        body, bold = f(), f("bold")
        table_w = w - 2 * margin
        cols = self._columns(style)

        edges, x = [], margin
        for _, rel, _, _ in cols:
            edges.append((x, x + rel * table_w))
            x += rel * table_w

        row_h = body.size * style.line_spacing * 1.5
        pad = body.size * 0.35
        head_top = y

        if style.shade_table_header:
            c.rect((margin, head_top, margin + table_w, head_top + row_h), outline=None, fill=225)
        for (header, _, align, _), (x0, x1) in zip(cols, edges, strict=True):
            hx = x0 + pad if align == "left" else x1 - pad
            c.text((hx, head_top + pad), header, bold, align=align)
        y = head_top + row_h
        if style.rule_style != "none":
            c.hline(margin, margin + table_w, head_top, width=1)
            c.hline(margin, margin + table_w, y, width=1)

        for idx, item in enumerate(items):
            for (_, _, align, key), (x0, x1) in zip(cols, edges, strict=True):
                raw = item.get(key)
                if key == "line_no" and raw is None:
                    raw = idx + 1
                text = LINE_ITEM_FORMATTERS[key](raw)
                if not text:
                    continue
                tx = x0 + pad if align == "left" else x1 - pad
                c.text(
                    (tx, y + pad),
                    text,
                    body,
                    field_path=f"line_items[{idx}].{key}",
                    align=align,
                    max_width=(x1 - x0) - 2 * pad,
                )
            y += row_h
            if style.rule_style == "grid":
                c.hline(margin, margin + table_w, y, width=1)

        if style.rule_style == "grid":
            for x0, _ in edges:
                c.vline(x0, head_top, y, width=1)
            c.vline(margin + table_w, head_top, y, width=1)
        elif style.rule_style == "horizontal":
            c.hline(margin, margin + table_w, y, width=1)
        return y + body.size * 0.9

    def _draw_totals(self, c, doc, style, margin, y, w, f) -> float:
        totals = doc.get("totals") or {}
        tax_lines = doc.get("tax_lines") or []
        body, bold = f(), f("bold")
        block_w = (w - 2 * margin) * 0.42
        x_right = (w - margin) if style.totals_on_right else (margin + block_w)
        x_label = x_right - block_w

        rows: list[tuple[str, str, str, bool]] = []
        if totals.get("subtotal") is not None:
            rows.append(("Subtotal", _fmt_amount(totals["subtotal"]), "totals.subtotal", False))
        if totals.get("discount_total"):
            rows.append(
                ("Discount", _fmt_amount(totals["discount_total"]), "totals.discount_total", False)
            )
        for i, tl in enumerate(tax_lines):
            rate = f" @ {float(tl['rate']):g}%" if tl.get("rate") is not None else ""
            rows.append(
                (
                    f"{tl['type']}{rate}",
                    _fmt_amount(tl.get("amount")),
                    f"tax_lines[{i}].amount",
                    False,
                )
            )
        if totals.get("shipping"):
            rows.append(("Shipping", _fmt_amount(totals["shipping"]), "totals.shipping", False))
        if totals.get("round_off"):
            rows.append(("Round Off", _fmt_amount(totals["round_off"]), "totals.round_off", False))
        if totals.get("grand_total") is not None:
            rows.append(
                ("Grand Total", _fmt_amount(totals["grand_total"]), "totals.grand_total", True)
            )

        for label, value, path, emphasise in rows:
            font_l = bold if emphasise else body
            if emphasise and style.rule_style != "none":
                c.hline(x_label, x_right, y - body.size * 0.25, width=1)
            c.text((x_label, y), self._label(label, style), font_l)
            c.text((x_right, y), value, font_l, field_path=path, align="right")
            y += body.size * style.line_spacing * (1.3 if emphasise else 1.0)

        if totals.get("amount_in_words"):
            y += body.size * 0.5
            _, y = c.wrapped_text(
                (margin, y),
                f"{self._label('Amount in words', style)} {totals['amount_in_words']}",
                body,
                w - 2 * margin,
                field_path="totals.amount_in_words",
                line_spacing=style.line_spacing,
            )
        return y + body.size

    def _draw_footer(self, c, doc, style, margin, y, w, h, f) -> None:
        pay = doc.get("payment_info") or {}
        body, bold = f(), f("bold")
        y = max(y, h - margin - body.size * 10)

        if any(pay.get(k) for k in ("bank_name", "account_number", "ifsc_or_swift", "upi_id")):
            c.text((margin, y), self._label("Payment Details", style), bold)
            y += bold.size * style.line_spacing
            for key, label, path in (
                ("bank_name", "Bank", "payment_info.bank_name"),
                ("account_number", "A/C No", "payment_info.account_number"),
                ("ifsc_or_swift", "IFSC", "payment_info.ifsc_or_swift"),
                ("upi_id", "UPI", "payment_info.upi_id"),
            ):
                if pay.get(key):
                    c.text(
                        (margin, y),
                        f"{self._label(label, style)} {pay[key]}",
                        body,
                        field_path=path,
                    )
                    y += body.size * style.line_spacing
        c.text((w - margin, h - margin - body.size), "Authorised Signatory", body, align="right")

    # --------------------------------------------------------- ground truth

    @staticmethod
    def _grounding(words: list[WordBox]) -> list[dict[str, Any]]:
        """Collapse word boxes into one grounding entry per canonical field,
        matching the schema's `grounding[]` shape."""
        by_field: dict[str, list[WordBox]] = {}
        for wb in words:
            if wb.field_path:
                by_field.setdefault(wb.field_path, []).append(wb)

        out = []
        for path, boxes in by_field.items():
            merged = merge_boxes(boxes)
            if merged:
                out.append(
                    {
                        "field_path": path,
                        "page_index": 0,
                        "bbox": [round(v, 2) for v in merged],
                        "confidence": 1.0,
                    }
                )
        return sorted(out, key=lambda g: g["field_path"])
