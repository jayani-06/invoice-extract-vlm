"""Drawing primitives that record word-level ground truth as they draw.

The whole point of rendering synthetic invoices rather than hand-annotating
scans is that we know, exactly, which pixels hold which word of which
canonical field. Every text-drawing call here returns the `WordBox`es it
produced, so an image and its OCR ground truth are generated in lockstep and
cannot drift apart.

Coordinates are absolute pixels in the rendered page, origin top-left,
`bbox = (x0, y0, x1, y1)` — the same convention as the canonical schema's
`grounding[].bbox`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

Align = Literal["left", "right", "center"]


@dataclass(frozen=True)
class WordBox:
    """One whitespace-delimited token and where it landed on the page."""

    text: str
    bbox: tuple[float, float, float, float]
    field_path: str | None = None  # canonical dotted path, e.g. "totals.grand_total"
    line_id: int = 0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def scaled(self, sx: float, sy: float) -> WordBox:
        x0, y0, x1, y1 = self.bbox
        return WordBox(
            self.text, (x0 * sx, y0 * sy, x1 * sx, y1 * sy), self.field_path, self.line_id
        )


@dataclass
class Canvas:
    """A page being drawn, plus the ground truth accumulated so far."""

    width: int
    height: int
    background: int = 255
    words: list[WordBox] = field(default_factory=list)
    _line_counter: int = 0

    def __post_init__(self) -> None:
        self.image = Image.new("L", (self.width, self.height), color=self.background)
        self.draw = ImageDraw.Draw(self.image)

    # ---------------------------------------------------------------- text

    def text(
        self,
        xy: tuple[float, float],
        s: str,
        font: ImageFont.FreeTypeFont,
        *,
        field_path: str | None = None,
        align: Align = "left",
        fill: int = 0,
        max_width: float | None = None,
    ) -> list[WordBox]:
        """Draw one line of text and record a box per whitespace-delimited word.

        `xy` is the anchor: the left edge for align="left", the right edge for
        "right", the midpoint for "center". Returns the boxes produced (also
        appended to `self.words`).
        """
        s = " ".join(str(s).split())
        if not s:
            return []
        if max_width is not None:
            s = self._ellipsize(s, font, max_width)

        x, y = xy
        total = font.getlength(s)
        if align == "right":
            x -= total
        elif align == "center":
            x -= total / 2

        self.draw.text((x, y), s, font=font, fill=fill)

        ascent, descent = font.getmetrics()
        line_id = self._line_counter
        self._line_counter += 1

        boxes: list[WordBox] = []
        cursor = 0
        for word in s.split(" "):
            prefix_w = font.getlength(s[:cursor])
            word_w = font.getlength(word)
            # Trim the box to the glyphs' real ink extent vertically; horizontal
            # advance width is what OCR engines report, so keep that as-is.
            top = y + ascent - self._ink_ascent(word, font)
            bottom = y + ascent + self._ink_descent(word, font)
            boxes.append(
                WordBox(
                    text=word,
                    bbox=(x + prefix_w, top, x + prefix_w + word_w, bottom),
                    field_path=field_path,
                    line_id=line_id,
                )
            )
            cursor += len(word) + 1

        self.words.extend(boxes)
        return boxes

    def wrapped_text(
        self,
        xy: tuple[float, float],
        s: str,
        font: ImageFont.FreeTypeFont,
        max_width: float,
        *,
        field_path: str | None = None,
        line_spacing: float = 1.25,
        fill: int = 0,
        max_lines: int = 6,
    ) -> tuple[list[WordBox], float]:
        """Draw text wrapped to `max_width`. Returns (boxes, y after the block)."""
        s = " ".join(str(s).split())
        if not s:
            return [], xy[1]

        lines: list[str] = []
        current = ""
        for word in s.split(" "):
            trial = f"{current} {word}".strip()
            if font.getlength(trial) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)

        step = (font.getmetrics()[0] + font.getmetrics()[1]) * line_spacing
        boxes: list[WordBox] = []
        x, y = xy
        for line in lines:
            boxes.extend(self.text((x, y), line, font, field_path=field_path, fill=fill))
            y += step
        return boxes, y

    # --------------------------------------------------------------- rules

    def hline(self, x0: float, x1: float, y: float, *, width: int = 1, fill: int = 0) -> None:
        self.draw.line([(x0, y), (x1, y)], fill=fill, width=width)

    def vline(self, x: float, y0: float, y1: float, *, width: int = 1, fill: int = 0) -> None:
        self.draw.line([(x, y0), (x, y1)], fill=fill, width=width)

    def rect(
        self,
        box: tuple[float, float, float, float],
        *,
        outline: int | None = 0,
        fill: int | None = None,
        width: int = 1,
    ) -> None:
        self.draw.rectangle(box, outline=outline, fill=fill, width=width)

    # ------------------------------------------------------------ helpers

    def _ellipsize(self, s: str, font: ImageFont.FreeTypeFont, max_width: float) -> str:
        if font.getlength(s) <= max_width:
            return s
        while s and font.getlength(s + "…") > max_width:
            s = s[:-1]
        return (s + "…") if s else s

    @staticmethod
    def _ink_ascent(word: str, font: ImageFont.FreeTypeFont) -> float:
        bbox = font.getbbox(word)  # (x0, y0, x1, y1) relative to the text origin
        ascent = font.getmetrics()[0]
        return max(0.0, ascent - bbox[1])

    @staticmethod
    def _ink_descent(word: str, font: ImageFont.FreeTypeFont) -> float:
        bbox = font.getbbox(word)
        ascent = font.getmetrics()[0]
        return max(0.0, bbox[3] - ascent)


def merge_boxes(boxes: list[WordBox]) -> tuple[float, float, float, float] | None:
    """Union bbox over a set of word boxes — used to give a whole field one
    grounding box even though it was drawn as several words."""
    if not boxes:
        return None
    return (
        min(b.bbox[0] for b in boxes),
        min(b.bbox[1] for b in boxes),
        max(b.bbox[2] for b in boxes),
        max(b.bbox[3] for b in boxes),
    )
