"""Tesseract 5 adapter (via pytesseract).

Tesseract needs a system binary in addition to the Python wrapper, so this
module fails loudly and with installation instructions rather than silently
producing empty results -- an empty OcrResult would look like a catastrophic
preprocessing failure in the ablation table.
"""

from __future__ import annotations

import shutil
import time

import numpy as np

from invoice_extract.ocr.base import (
    OcrEngineUnavailableError,
    OcrResult,
    OcrWord,
    register_engine,
)

INSTALL_HINT = (
    "Tesseract needs both the Python wrapper and the engine binary:\n"
    "  pip install 'invoice-extract-vlm[ocr]'\n"
    "  Windows : winget install --id UB-Mannheim.TesseractOCR\n"
    "  macOS   : brew install tesseract\n"
    "  Debian  : sudo apt install tesseract-ocr\n"
    "If the binary is installed but not on PATH, set TESSERACT_CMD in .env."
)

# Page segmentation mode 6 ("assume a single uniform block of text") is a poor
# fit for invoices, which are multi-block by nature; 3 (fully automatic page
# segmentation) is the right default and 11/12 (sparse text) are worth trying
# on receipts.
DEFAULT_PSM = 3
DEFAULT_OEM = 3  # LSTM + legacy, whichever the build supports


class TesseractEngine:
    name = "tesseract"

    def __init__(
        self,
        psm: int = DEFAULT_PSM,
        oem: int = DEFAULT_OEM,
        lang: str = "eng",
        min_confidence: float = 0.0,
        cmd: str | None = None,
    ) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise OcrEngineUnavailableError(
                f"pytesseract is not installed.\n{INSTALL_HINT}"
            ) from exc

        self._pytesseract = pytesseract
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        binary = cmd or pytesseract.pytesseract.tesseract_cmd
        if shutil.which(binary) is None:
            raise OcrEngineUnavailableError(
                f"Tesseract binary {binary!r} not found on PATH.\n{INSTALL_HINT}"
            )

        self.psm = psm
        self.oem = oem
        self.lang = lang
        self.min_confidence = min_confidence

    @property
    def config(self) -> str:
        return f"--psm {self.psm} --oem {self.oem}"

    def recognize(self, image: np.ndarray) -> OcrResult:
        t0 = time.perf_counter()
        data = self._pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=self._pytesseract.Output.DICT,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        words: list[OcrWord] = []
        # Tesseract already reports block/paragraph/line structure; reuse it
        # rather than re-deriving lines geometrically.
        line_keys: dict[tuple[int, int, int, int], int] = {}
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            if not text:
                continue
            conf = float(data["conf"][i])
            if conf < 0:  # -1 marks a structural row, not a word
                continue
            conf /= 100.0
            if conf < self.min_confidence:
                continue
            key = (
                data["page_num"][i],
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )
            line_id = line_keys.setdefault(key, len(line_keys))
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append(
                OcrWord(text, (float(x), float(y), float(x + w), float(y + h)), conf, line_id)
            )

        return OcrResult(
            words=words,
            engine=self.name,
            elapsed_ms=elapsed,
            meta={"psm": self.psm, "oem": self.oem, "lang": self.lang},
        )


register_engine("tesseract", TesseractEngine)
