"""Ground-truth "OCR" — a harness instrument, not a result.

This engine returns the renderer's own word boxes, optionally with synthetic
character errors injected. It exists for two legitimate purposes:

1. **Plumbing verification.** It lets the whole Phase 1 chain -- render,
   degrade, preprocess, recognise, score -- be exercised and unit-tested with
   no OCR engine installed, so a broken metric or a mis-composed homography
   surfaces immediately rather than being blamed on the engine.

2. **A ceiling row in the ablation.** `error_rate=0` is a perfect-recognition
   reference: it shows what the downstream field-extraction score would be if
   OCR were solved, separating "OCR lost it" from "extraction lost it".

It is NOT an OCR baseline and must never be reported as one. The runner
labels its rows `oracle` and `scripts/run_preprocess_ablation.py` refuses to
write a headline CER/WER table from it without `--allow-oracle`.

The quality-aware mode is deliberately crude: it samples local contrast under
each gold box and raises the error probability where the box is washed out or
smeared. That makes the fake errors correlate with real degradation, which is
enough to check that the metrics move in the right direction -- and nothing more.
"""

from __future__ import annotations

import random
import string
import time

import numpy as np

from invoice_extract.ocr.base import OcrResult, OcrWord, register_engine

# Substitutions a real engine actually makes, rather than uniform random noise:
# glyph confusions dominate OCR error, and using them keeps the injected CER
# in the same neighbourhood as a genuine engine's.
CONFUSIONS = {
    "0": "O",
    "O": "0",
    "1": "l",
    "l": "1",
    "I": "1",
    "5": "S",
    "S": "5",
    "8": "B",
    "B": "8",
    "2": "Z",
    "Z": "2",
    "6": "b",
    "b": "6",
    "rn": "m",
    "m": "rn",
    "c": "e",
    "e": "c",
    ",": ".",
    ".": ",",
}


class OracleEngine:
    """Returns known ground truth. Must be constructed per document."""

    name = "oracle"

    def __init__(
        self,
        words: list[OcrWord] | None = None,
        error_rate: float = 0.0,
        quality_aware: bool = False,
        seed: int = 0,
    ) -> None:
        self.words = words or []
        self.error_rate = error_rate
        self.quality_aware = quality_aware
        self.rng = random.Random(seed)

    def set_ground_truth(self, words: list[OcrWord]) -> None:
        self.words = words

    # ------------------------------------------------------------ quality

    @staticmethod
    def _local_quality(image: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
        """Rough legibility score in [0, 1] for the patch under `bbox`.

        Standard deviation stands in for ink/paper separation: a crisp word has
        both dark and light pixels, while a blurred or washed-out one collapses
        toward a single value.
        """
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1) + 1), min(h, int(y1) + 1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            return 0.0
        return float(min(1.0, patch.std() / 60.0))

    def _corrupt(self, text: str, p: float) -> str:
        if p <= 0:
            return text
        out = []
        for ch in text:
            if self.rng.random() >= p:
                out.append(ch)
                continue
            roll = self.rng.random()
            if roll < 0.6 and ch in CONFUSIONS:
                out.append(CONFUSIONS[ch])
            elif roll < 0.8:
                pass  # deletion
            else:
                out.append(self.rng.choice(string.ascii_letters + string.digits))
        return "".join(out) or text[:1]

    # ------------------------------------------------------------ recognise

    def recognize(self, image: np.ndarray) -> OcrResult:
        t0 = time.perf_counter()
        out: list[OcrWord] = []
        for w in self.words:
            p = self.error_rate
            if self.quality_aware:
                quality = self._local_quality(image, w.bbox)
                # Scale the base rate by how illegible the patch looks.
                p = min(1.0, self.error_rate + (1.0 - quality) * 0.15)
            text = self._corrupt(w.text, p)
            if not text:
                continue
            out.append(OcrWord(text, w.bbox, confidence=None, line_id=w.line_id))

        return OcrResult(
            words=out,
            engine=self.name,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            meta={
                "synthetic": True,
                "error_rate": self.error_rate,
                "quality_aware": self.quality_aware,
                "warning": "ground-truth instrument, not an OCR baseline",
            },
        )


register_engine("oracle", OracleEngine)
