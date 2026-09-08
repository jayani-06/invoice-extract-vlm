"""OCR-stage metrics: CER, WER, and word-box detection quality.

Phase 1's exit criteria are stated in CER, so the definitions need to be
pinned down rather than left to whichever library is installed:

- **CER / WER** are edit distance over the *whole page's* reading-order text,
  normalised by the reference length. Page-level rather than per-word, because
  OCR errors include insertions and deletions that no word-to-word alignment
  can represent -- a dropped word has to cost something.

- **Normalisation matters and is reported both ways.** Raw CER punishes an
  engine for a stray space or a lowercase letter; normalised CER (collapsed
  whitespace, case-folded, punctuation stripped) reflects what actually
  survives into field extraction. Reporting only one of the two would let a
  preprocessing choice look better than it is.

- **Detection P/R/F1** scores whether words were *found*, independent of
  whether they were *read*. This is what separates a preprocessing failure
  (text destroyed, boxes missing) from a recognition failure (boxes right,
  characters wrong) -- the distinction the ablation exists to make.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import Levenshtein

Box = tuple[float, float, float, float]

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(s: str, *, case_fold: bool = True, strip_punct: bool = True) -> str:
    """Collapse the differences that do not survive into field extraction."""
    s = unicodedata.normalize("NFKC", s)
    if case_fold:
        s = s.lower()
    if strip_punct:
        s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate. Unbounded above (insertions can exceed the reference)."""
    ref = _WS.sub(" ", reference).strip()
    hyp = _WS.sub(" ", hypothesis).strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, as token-level edit distance."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class DetectionReport:
    n_gold: int
    n_pred: int
    n_matched: int
    n_matched_correct_text: int
    mean_iou: float

    @property
    def precision(self) -> float:
        return self.n_matched / self.n_pred if self.n_pred else float("nan")

    @property
    def recall(self) -> float:
        return self.n_matched / self.n_gold if self.n_gold else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p != p or r != r or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def word_accuracy(self) -> float:
        """Share of gold words both located and read correctly -- the metric
        that actually predicts downstream field-extraction success."""
        return self.n_matched_correct_text / self.n_gold if self.n_gold else float("nan")


def match_boxes(
    gold: list[tuple[str, Box]],
    pred: list[tuple[str, Box]],
    iou_threshold: float = 0.5,
) -> DetectionReport:
    """Greedy highest-IoU matching between gold and predicted word boxes.

    Greedy rather than optimal assignment: word boxes rarely contend for the
    same partner (they are small and spatially separated), and greedy is
    O(n log n) with a spatial prefilter instead of O(n^3), which matters when
    the ablation runs thousands of pages.
    """
    if not gold or not pred:
        return DetectionReport(len(gold), len(pred), 0, 0, 0.0)

    pred_arr = np.array([b for _, b in pred], dtype=np.float64)
    used = np.zeros(len(pred), dtype=bool)
    n_matched = 0
    n_text_ok = 0
    ious: list[float] = []

    for g_text, g_box in gold:
        gx0, gy0, gx1, gy1 = g_box
        # Cheap rejection: only consider predictions whose box overlaps at all.
        candidates = np.where(
            (~used)
            & (pred_arr[:, 0] < gx1)
            & (pred_arr[:, 2] > gx0)
            & (pred_arr[:, 1] < gy1)
            & (pred_arr[:, 3] > gy0)
        )[0]
        if candidates.size == 0:
            continue

        best_idx, best_iou = -1, 0.0
        for idx in candidates:
            score = iou(g_box, tuple(pred_arr[idx]))
            if score > best_iou:
                best_idx, best_iou = int(idx), score

        if best_idx >= 0 and best_iou >= iou_threshold:
            used[best_idx] = True
            n_matched += 1
            ious.append(best_iou)
            if normalize_text(pred[best_idx][0]) == normalize_text(g_text):
                n_text_ok += 1

    return DetectionReport(
        n_gold=len(gold),
        n_pred=len(pred),
        n_matched=n_matched,
        n_matched_correct_text=n_text_ok,
        mean_iou=float(np.mean(ious)) if ious else 0.0,
    )


@dataclass
class OcrReport:
    doc_id: str
    cer_raw: float
    wer_raw: float
    cer_norm: float
    wer_norm: float
    detection: DetectionReport
    elapsed_ms: float = 0.0
    preprocess_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "cer_raw": self.cer_raw,
            "wer_raw": self.wer_raw,
            "cer_norm": self.cer_norm,
            "wer_norm": self.wer_norm,
            "det_precision": self.detection.precision,
            "det_recall": self.detection.recall,
            "det_f1": self.detection.f1,
            "det_mean_iou": self.detection.mean_iou,
            "word_accuracy": self.detection.word_accuracy,
            "n_gold_words": self.detection.n_gold,
            "n_pred_words": self.detection.n_pred,
            "ocr_ms": self.elapsed_ms,
            "preprocess_ms": self.preprocess_ms,
        }


def score_document(
    doc_id: str,
    gold_text: str,
    pred_text: str,
    gold_boxes: list[tuple[str, Box]],
    pred_boxes: list[tuple[str, Box]],
    *,
    iou_threshold: float = 0.5,
    elapsed_ms: float = 0.0,
    preprocess_ms: float = 0.0,
) -> OcrReport:
    return OcrReport(
        doc_id=doc_id,
        cer_raw=cer(gold_text, pred_text),
        wer_raw=wer(gold_text, pred_text),
        cer_norm=cer(normalize_text(gold_text), normalize_text(pred_text)),
        wer_norm=wer(normalize_text(gold_text), normalize_text(pred_text)),
        detection=match_boxes(gold_boxes, pred_boxes, iou_threshold),
        elapsed_ms=elapsed_ms,
        preprocess_ms=preprocess_ms,
    )


@dataclass
class AggregateOcrReport:
    n_docs: int
    cer_raw: float
    wer_raw: float
    cer_norm: float
    wer_norm: float
    det_precision: float
    det_recall: float
    det_f1: float
    word_accuracy: float
    mean_iou: float
    median_cer_norm: float
    p90_cer_norm: float
    mean_ocr_ms: float
    mean_preprocess_ms: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_ocr(reports: list[OcrReport]) -> AggregateOcrReport:
    """Aggregate per document, weighting every page equally.

    Micro-averaging over characters would let one dense page dominate the
    score; per-page macro-averaging matches how the system is actually judged
    (invoice by invoice). The median and p90 are reported alongside the mean
    because CER distributions are long-tailed -- a handful of catastrophic
    pages move the mean far more than they move the typical experience.
    """
    if not reports:
        # Name the fields rather than counting positional NaNs: a miscount here
        # is silent until someone evaluates an empty split.
        nan = float("nan")
        return AggregateOcrReport(
            n_docs=0,
            cer_raw=nan,
            wer_raw=nan,
            cer_norm=nan,
            wer_norm=nan,
            det_precision=nan,
            det_recall=nan,
            det_f1=nan,
            word_accuracy=nan,
            mean_iou=nan,
            median_cer_norm=nan,
            p90_cer_norm=nan,
            mean_ocr_ms=nan,
            mean_preprocess_ms=nan,
        )

    cer_norms = sorted(r.cer_norm for r in reports)
    n_gold = sum(r.detection.n_gold for r in reports)
    n_pred = sum(r.detection.n_pred for r in reports)
    n_matched = sum(r.detection.n_matched for r in reports)
    n_text_ok = sum(r.detection.n_matched_correct_text for r in reports)

    precision = n_matched / n_pred if n_pred else float("nan")
    recall = n_matched / n_gold if n_gold else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    return AggregateOcrReport(
        n_docs=len(reports),
        cer_raw=_safe_mean([r.cer_raw for r in reports]),
        wer_raw=_safe_mean([r.wer_raw for r in reports]),
        cer_norm=_safe_mean([r.cer_norm for r in reports]),
        wer_norm=_safe_mean([r.wer_norm for r in reports]),
        det_precision=precision,
        det_recall=recall,
        det_f1=f1,
        word_accuracy=n_text_ok / n_gold if n_gold else float("nan"),
        mean_iou=_safe_mean([r.detection.mean_iou for r in reports if r.detection.n_matched]),
        median_cer_norm=float(np.median(cer_norms)),
        p90_cer_norm=float(np.percentile(cer_norms, 90)),
        mean_ocr_ms=_safe_mean([r.elapsed_ms for r in reports]),
        mean_preprocess_ms=_safe_mean([r.preprocess_ms for r in reports]),
    )
