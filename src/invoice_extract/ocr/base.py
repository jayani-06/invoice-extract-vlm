"""Engine-agnostic OCR interface.

Phase 1 has to compare preprocessing configurations, not OCR engines -- but
the two are entangled (a binarisation that helps Tesseract can hurt an engine
that does its own thresholding). Pinning every engine behind one interface
keeps the ablation honest: the same runner, the same metrics, the same
corpus, with the engine as an explicit axis rather than a hidden constant.

Engines are constructed lazily through `get_engine`, so importing this module
never requires any engine to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from invoice_extract.reading_order import group_lines, reading_order_text


@dataclass(frozen=True)
class OcrWord:
    """One recognised token. Mirrors `render.layout.WordBox` so predictions and
    ground truth are directly comparable."""

    text: str
    bbox: tuple[float, float, float, float]
    confidence: float | None = None
    line_id: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass
class OcrResult:
    words: list[OcrWord]
    engine: str
    elapsed_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Reading-order text: group by line, then left-to-right within a line.

        Engines disagree about reading order on multi-column layouts, so we
        impose our own rather than trusting theirs -- otherwise a CER
        difference between engines could be pure ordering, not recognition.
        """
        if not self.words:
            return ""
        return reading_order_text(self.words, lambda w: w.bbox, lambda w: w.text)

    @property
    def mean_confidence(self) -> float | None:
        vals = [w.confidence for w in self.words if w.confidence is not None]
        return sum(vals) / len(vals) if vals else None


class OcrEngine(Protocol):
    name: str

    def recognize(self, image: np.ndarray) -> OcrResult: ...


class OcrEngineUnavailableError(RuntimeError):
    """The engine's Python package or its underlying binary is not installed."""


def group_into_lines(words: list[OcrWord], tolerance_frac: float = 0.6) -> list[OcrWord]:
    """Assign `line_id` by vertical overlap, for engines that report no line
    structure of their own.

    Two words share a line when their vertical centres sit within
    `tolerance_frac` of the median word height -- a scale-free rule, so it
    works the same at 150 and 300 dpi.
    """
    out: list[OcrWord] = []
    for line_id, line in enumerate(group_lines(words, lambda w: w.bbox, tolerance_frac)):
        out.extend(OcrWord(w.text, w.bbox, w.confidence, line_id) for w in line)
    return out


_ENGINE_FACTORIES: dict[str, Any] = {}


def register_engine(name: str, factory) -> None:
    _ENGINE_FACTORIES[name] = factory


def available_engines() -> list[str]:
    return sorted(_ENGINE_FACTORIES)


def get_engine(name: str, **kwargs) -> OcrEngine:
    """Construct an engine by name, importing its module only on demand."""
    if name not in _ENGINE_FACTORIES:
        # Import the adapters lazily so a missing optional dependency does not
        # break anything that only needs the interface.
        from invoice_extract.ocr import oracle, paddle, tesseract  # noqa: F401
    if name not in _ENGINE_FACTORIES:
        raise OcrEngineUnavailableError(
            f"unknown OCR engine {name!r}; registered: {available_engines()}"
        )
    return _ENGINE_FACTORIES[name](**kwargs)
