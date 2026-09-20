"""PaddleOCR adapter (supports the 3.x pipeline API, falls back to 2.x).

PaddleOCR installs entirely through pip (no system binary), and its detection
model handles rotated and curved text far better than Tesseract -- which
matters for receipts and for pages our deskew stage did not fully level. It is
the primary engine per the Phase 1 plan.

Two things this adapter has to work around, both discovered by running it:

1. **oneDNN crashes PP-OCRv6 on this CPU.** paddlepaddle 3.3.1 raises
   `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support` from
   the oneDNN executor during text detection. Passing `enable_mkldnn=False`
   avoids it entirely, at some cost in CPU throughput. It is the default here
   because a crash is worse than being slower, and the flag is exposed so a
   machine without the bug can turn it back on.

2. **Word boxes are opt-in.** With `return_word_box=True` the 3.x pipeline
   returns real per-word geometry (`text_word` / `text_word_boxes`) rather than
   only line boxes. That matters because Phase 1's detection metrics are
   word-level: apportioning a line box across its words by character count --
   the only option on 2.x -- fabricates geometry and quietly flatters or
   punishes the IoU numbers. Real boxes are used whenever available, and the
   result's `meta` records which path was taken so a run is never ambiguous
   about it.
"""

from __future__ import annotations

import time
from typing import Any

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
    "  pip install paddlepaddle paddleocr\n"
    "First run downloads the detection/recognition weights (~100 MB)."
)


class PaddleEngine:
    name = "paddle"

    #: Lightweight detection/recognition pair. Measured on this project's
    #: corpus at 200 dpi, CPU, mkldnn disabled: ~40 s/page versus ~105 s/page
    #: for the default PP-OCRv6 "medium" models -- a 2.6x speedup that is the
    #: difference between an ablation finishing in an hour and in a day. Not
    #: the default, because it trades some accuracy for that speed; pass
    #: `fast=True` (or `--engine-arg fast=1`) to opt in, and report which was
    #: used, since the two are not interchangeable in a results table.
    FAST_MODELS = {
        "ocr_version": "PP-OCRv5",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "PP-OCRv5_mobile_rec",
    }

    def __init__(
        self,
        lang: str = "en",
        *,
        enable_mkldnn: bool = False,
        text_det_box_thresh: float | None = None,
        min_confidence: float = 0.0,
        ocr_version: str | None = None,
        return_word_box: bool = True,
        fast: bool = False,
        **extra: Any,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise OcrEngineUnavailableError(f"paddleocr is not installed.\n{INSTALL_HINT}") from exc

        self.min_confidence = min_confidence
        self.return_word_box = return_word_box

        # The 3.x constructor. Document pre-processing (orientation classify,
        # unwarping) is disabled because our own preprocessing pipeline owns
        # that step -- leaving both on would make the Phase 1 ablation
        # meaningless, since Paddle would silently deskew pages we are
        # measuring our deskew stage on.
        kwargs: dict[str, Any] = {
            "lang": lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "return_word_box": return_word_box,
            "enable_mkldnn": enable_mkldnn,
        }
        if text_det_box_thresh is not None:
            kwargs["text_det_box_thresh"] = text_det_box_thresh
        if fast:
            kwargs.update(self.FAST_MODELS)
        if ocr_version:
            kwargs["ocr_version"] = ocr_version
        kwargs.update(extra)
        self.fast = fast

        try:
            self._ocr = PaddleOCR(**kwargs)
            self._api = "3.x"
        except (TypeError, ValueError):
            # 2.x signature: different argument names, and no word boxes.
            self._ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
            self._api = "2.x"
            self.return_word_box = False
            self.fast = False

    # ------------------------------------------------------------ 3.x path

    def _words_from_3x(self, result: dict) -> tuple[list[OcrWord], str]:
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []
        word_texts = result.get("text_word") or []
        word_boxes = result.get("text_word_boxes") or []

        words: list[OcrWord] = []
        if self.return_word_box and len(word_texts) == len(texts) and word_boxes:
            for i, tokens in enumerate(word_texts):
                boxes = word_boxes[i]
                score = float(scores[i]) if i < len(scores) else None
                for token, box in zip(tokens, boxes, strict=False):
                    token = str(token).strip()
                    if not token:  # Paddle emits the spaces between words too
                        continue
                    x0, y0, x1, y1 = (float(v) for v in box)
                    words.append(OcrWord(token, (x0, y0, x1, y1), score))
            if words:
                return words, "word_boxes"

        # Fall back to line boxes, splitting the text but keeping the line's
        # own geometry for every token in it.
        boxes = result.get("rec_boxes")
        if boxes is None:
            boxes = [None] * len(texts)
        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) else None
            if score is not None and score < self.min_confidence:
                continue
            box = boxes[i]
            if box is None:
                continue
            x0, y0, x1, y1 = (float(v) for v in np.asarray(box).ravel()[:4])
            for token in text.split():
                words.append(OcrWord(token, (x0, y0, x1, y1), score))
        return words, "line_boxes"

    # ------------------------------------------------------------ 2.x path

    def _words_from_2x(self, raw) -> list[OcrWord]:
        lines = raw[0] if raw and isinstance(raw[0], list) else (raw or [])
        words: list[OcrWord] = []
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
            box = (min(xs), min(ys), max(xs), max(ys))
            for token in text.split():
                words.append(OcrWord(token, box, float(conf)))
        return words

    # ----------------------------------------------------------- recognize

    def recognize(self, image: np.ndarray) -> OcrResult:
        img = image
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # Paddle expects 3 channels

        t0 = time.perf_counter()
        if self._api == "3.x":
            results = self._ocr.predict(img)
            elapsed = (time.perf_counter() - t0) * 1000
            words, geometry = ([], "none")
            if results:
                words, geometry = self._words_from_3x(results[0])
        else:
            raw = self._ocr.ocr(img, cls=True)
            elapsed = (time.perf_counter() - t0) * 1000
            words, geometry = self._words_from_2x(raw), "line_boxes"

        if self.min_confidence:
            words = [
                w for w in words if w.confidence is None or w.confidence >= self.min_confidence
            ]

        return OcrResult(
            words=group_into_lines(words),
            engine=self.name,
            elapsed_ms=elapsed,
            meta={"api": self._api, "geometry": geometry, "fast_models": self.fast},
        )


register_engine("paddle", PaddleEngine)
