"""PaddleOCR adapter.

PaddleOCR installs entirely through pip (no system binary), and its detection
model handles rotated and curved text far better than Tesseract -- which
matters for receipts and for pages our deskew stage did not fully level.
It is the intended primary engine per the Phase 1 plan, with Tesseract as
the fallback and as a second data point in the ablation.

PaddleOCR returns quadrilaterals, not axis-aligned boxes. We keep the quad in
`meta` and expose its bounding rectangle as `bbox`, so downstream metrics stay
uniform across engines while the richer geometry remains available.
"""

from __future__ import annotations

import time

import numpy as np

from invoice_extract.ocr.base import (
    OcrEngineUnavailableError,
    OcrResult,
    OcrWord,
    group_into_lines,
    register_engine,
)

INSTALL_HINT = (
    "PaddleOCR is pip-installable with no system binary:\n"
    "  pip install 'invoice-extract-vlm[paddle]'\n"
    "First run downloads ~20 MB of detection/recognition weights."
)


class PaddleEngine:
    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        det_db_box_thresh: float = 0.5,
        min_confidence: float = 0.0,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise OcrEngineUnavailableError(f"paddleocr is not installed.\n{INSTALL_HINT}") from exc

        # PaddleOCR's constructor signature has churned across releases; probe
        # rather than pin, so a version bump does not silently disable the engine.
        kwargs = {
            "lang": lang,
            "use_angle_cls": True,
            "show_log": False,
            "det_db_box_thresh": det_db_box_thresh,
        }
        if use_gpu:
            kwargs["use_gpu"] = True
        try:
            self._ocr = PaddleOCR(**kwargs)
        except (TypeError, ValueError):
            self._ocr = PaddleOCR(lang=lang)

        self.min_confidence = min_confidence

    def recognize(self, image: np.ndarray) -> OcrResult:
        img = image
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # Paddle expects 3 channels

        t0 = time.perf_counter()
        raw = self._ocr.ocr(img, cls=True)
        elapsed = (time.perf_counter() - t0) * 1000

        lines = raw[0] if raw and isinstance(raw[0], list) else (raw or [])
        words: list[OcrWord] = []
        quads: list[list[list[float]]] = []
        for entry in lines or []:
            try:
                quad, (text, conf) = entry[0], entry[1]
            except (TypeError, ValueError, IndexError):
                continue
            text = (text or "").strip()
            if not text or float(conf) < self.min_confidence:
                continue
            xs = [float(p[0]) for p in quad]
            ys = [float(p[1]) for p in quad]
            # Paddle emits whole text lines; split into words and apportion the
            # line box horizontally, so word-level metrics remain comparable to
            # Tesseract's genuinely per-word output.
            tokens = text.split()
            if not tokens:
                continue
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            total_chars = sum(len(t) for t in tokens) + (len(tokens) - 1)
            cursor = 0
            for tok in tokens:
                frac_start = cursor / total_chars
                frac_end = (cursor + len(tok)) / total_chars
                words.append(
                    OcrWord(
                        tok,
                        (x0 + (x1 - x0) * frac_start, y0, x0 + (x1 - x0) * frac_end, y1),
                        float(conf),
                    )
                )
                cursor += len(tok) + 1
            quads.append([[float(p[0]), float(p[1])] for p in quad])

        return OcrResult(
            words=group_into_lines(words),
            engine=self.name,
            elapsed_ms=elapsed,
            meta={"quads": quads, "note": "word boxes apportioned from line boxes"},
        )


register_engine("paddle", PaddleEngine)
