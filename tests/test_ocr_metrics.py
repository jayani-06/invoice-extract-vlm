"""Tests for the OCR-stage metrics.

These are the numbers Phase 1's exit criteria are stated in, so they are
tested against hand-computed values rather than against themselves. A metric
that is merely self-consistent can still be the wrong metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from invoice_extract.eval.ocr_metrics import (
    OcrReport,
    aggregate_ocr,
    cer,
    iou,
    match_boxes,
    normalize_text,
    score_document,
    wer,
)
from invoice_extract.ocr.base import OcrResult, OcrWord, group_into_lines
from invoice_extract.ocr.oracle import OracleEngine


class TestNormalization:
    def test_case_and_punctuation_are_folded(self):
        assert normalize_text("Invoice, No.: INV-001") == "invoice no inv 001"

    def test_whitespace_is_collapsed(self):
        assert normalize_text("a   b\n\tc") == "a b c"

    def test_options_can_be_disabled(self):
        assert normalize_text("Total: 10", case_fold=False, strip_punct=False) == "Total: 10"

    def test_unicode_is_normalised(self):
        assert normalize_text("ﬁle") == "file"


class TestCer:
    def test_identical_strings_score_zero(self):
        assert cer("hello world", "hello world") == 0.0

    def test_single_substitution(self):
        # "kitten" -> "sitten": 1 edit over 6 characters.
        assert cer("kitten", "sitten") == pytest.approx(1 / 6)

    def test_classic_levenshtein_case(self):
        # kitten -> sitting is 3 edits over 6 characters.
        assert cer("kitten", "sitting") == pytest.approx(3 / 6)

    def test_deletion_is_penalised(self):
        assert cer("abcd", "abc") == pytest.approx(1 / 4)

    def test_insertions_can_exceed_one(self):
        """CER is unbounded above by design: an engine that hallucinates a wall
        of text should not be capped at the same score as one that outputs
        nothing."""
        assert cer("ab", "abcdefghij") > 1.0

    def test_empty_reference(self):
        assert cer("", "") == 0.0
        assert cer("", "spurious") == 1.0

    def test_empty_hypothesis_scores_one(self):
        assert cer("abcdef", "") == 1.0


class TestWer:
    def test_identical(self):
        assert wer("the quick brown fox", "the quick brown fox") == 0.0

    def test_one_word_wrong(self):
        assert wer("the quick brown fox", "the quick brown cat") == pytest.approx(1 / 4)

    def test_dropped_word(self):
        assert wer("a b c d", "a b d") == pytest.approx(1 / 4)

    def test_empty_reference(self):
        assert wer("", "") == 0.0
        assert wer("", "x") == 1.0

    def test_is_insensitive_to_intra_word_errors_that_cer_catches(self):
        """The two metrics answer different questions; a single typo is one
        whole word wrong but only one character wrong."""
        ref, hyp = "invoice total", "invoicc total"
        assert wer(ref, hyp) == pytest.approx(0.5)
        assert cer(ref, hyp) < 0.1


class TestIou:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_touching_boxes_do_not_overlap(self):
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # Intersection 50, union 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_contained_box(self):
        assert iou((0, 0, 10, 10), (2, 2, 4, 4)) == pytest.approx(4 / 100)


class TestMatchBoxes:
    def test_perfect_match(self):
        gold = [("hello", (0, 0, 10, 10)), ("world", (20, 0, 30, 10))]
        report = match_boxes(gold, list(gold))
        assert report.n_matched == 2
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.word_accuracy == 1.0

    def test_located_but_misread_counts_for_detection_not_accuracy(self):
        """The distinction the ablation exists to make: preprocessing that
        preserves layout but loses characters should show high detection F1
        and low word accuracy."""
        gold = [("hello", (0, 0, 10, 10))]
        pred = [("he11o", (0, 0, 10, 10))]
        report = match_boxes(gold, pred)
        assert report.n_matched == 1
        assert report.f1 == 1.0
        assert report.word_accuracy == 0.0

    def test_missed_word_lowers_recall(self):
        gold = [("a", (0, 0, 10, 10)), ("b", (20, 0, 30, 10))]
        report = match_boxes(gold, [gold[0]])
        assert report.recall == pytest.approx(0.5)
        assert report.precision == 1.0

    def test_spurious_word_lowers_precision(self):
        gold = [("a", (0, 0, 10, 10))]
        pred = [("a", (0, 0, 10, 10)), ("ghost", (50, 50, 60, 60))]
        report = match_boxes(gold, pred)
        assert report.recall == 1.0
        assert report.precision == pytest.approx(0.5)

    def test_below_threshold_overlap_is_not_a_match(self):
        gold = [("a", (0, 0, 10, 10))]
        pred = [("a", (8, 0, 18, 10))]  # IoU well under 0.5
        assert match_boxes(gold, pred, iou_threshold=0.5).n_matched == 0

    def test_a_prediction_is_consumed_by_one_gold_word_only(self):
        gold = [("a", (0, 0, 10, 10)), ("b", (0, 0, 10, 10))]
        pred = [("a", (0, 0, 10, 10))]
        report = match_boxes(gold, pred)
        assert report.n_matched == 1
        assert report.precision == 1.0
        assert report.recall == pytest.approx(0.5)

    def test_empty_inputs(self):
        assert match_boxes([], []).n_matched == 0
        assert match_boxes([("a", (0, 0, 1, 1))], []).n_matched == 0
        assert match_boxes([], [("a", (0, 0, 1, 1))]).n_matched == 0

    def test_text_matching_is_normalisation_aware(self):
        gold = [("Total:", (0, 0, 10, 10))]
        pred = [("total", (0, 0, 10, 10))]
        assert match_boxes(gold, pred).word_accuracy == 1.0


class TestScoreAndAggregate:
    def _report(self, cer_norm: float, n_gold: int = 10, n_ok: int = 8) -> OcrReport:
        return score_document(
            doc_id="d",
            gold_text="a b c",
            pred_text="a b c",
            gold_boxes=[(f"w{i}", (i * 10.0, 0.0, i * 10.0 + 8, 10.0)) for i in range(n_gold)],
            pred_boxes=[(f"w{i}", (i * 10.0, 0.0, i * 10.0 + 8, 10.0)) for i in range(n_ok)],
        )

    def test_score_document_populates_both_normalisations(self):
        report = score_document(
            doc_id="doc1",
            gold_text="Invoice No: INV-001",
            pred_text="invoice no inv 001",
            gold_boxes=[("Invoice", (0, 0, 10, 10))],
            pred_boxes=[("invoice", (0, 0, 10, 10))],
        )
        assert report.cer_raw > 0
        assert report.cer_norm == pytest.approx(0.0)
        assert report.detection.word_accuracy == 1.0

    def test_aggregate_of_nothing_is_all_nan(self):
        agg = aggregate_ocr([])
        assert agg.n_docs == 0
        assert agg.cer_norm != agg.cer_norm

    def test_aggregate_reports_tail_as_well_as_mean(self):
        reports = [self._report(0.0) for _ in range(9)]
        agg = aggregate_ocr(reports)
        assert agg.n_docs == 10 - 1
        assert agg.median_cer_norm == pytest.approx(0.0)
        assert agg.p90_cer_norm >= 0.0

    def test_detection_is_micro_averaged_over_words(self):
        reports = [self._report(0.0, n_gold=10, n_ok=8), self._report(0.0, n_gold=10, n_ok=10)]
        agg = aggregate_ocr(reports)
        assert agg.det_recall == pytest.approx(18 / 20)
        assert agg.det_precision == pytest.approx(1.0)

    def test_aggregate_dict_is_serialisable(self):
        import json

        json.dumps(aggregate_ocr([self._report(0.0)]).as_dict())


class TestOcrResult:
    def test_reading_order_is_imposed_not_inherited(self):
        """Words are returned deliberately out of order; the result must still
        read top-to-bottom, left-to-right."""
        words = [
            OcrWord("world", (50, 0, 90, 10), line_id=0),
            OcrWord("second", (0, 40, 40, 50), line_id=1),
            OcrWord("hello", (0, 0, 40, 10), line_id=0),
        ]
        assert OcrResult(words=words, engine="t").text == "hello world\nsecond"

    def test_empty_result(self):
        assert OcrResult(words=[], engine="t").text == ""
        assert OcrResult(words=[], engine="t").mean_confidence is None

    def test_mean_confidence_ignores_missing_values(self):
        words = [OcrWord("a", (0, 0, 1, 1), 0.9), OcrWord("b", (2, 0, 3, 1), None)]
        assert OcrResult(words=words, engine="t").mean_confidence == pytest.approx(0.9)

    def test_group_into_lines_by_vertical_overlap(self):
        words = [
            OcrWord("a", (0, 0, 10, 20)),
            OcrWord("b", (20, 2, 30, 22)),  # same line, slightly offset
            OcrWord("c", (0, 100, 10, 120)),  # clearly a new line
        ]
        grouped = group_into_lines(words)
        by_text = {w.text: w.line_id for w in grouped}
        assert by_text["a"] == by_text["b"]
        assert by_text["c"] != by_text["a"]

    def test_group_into_lines_handles_empty(self):
        assert group_into_lines([]) == []


class TestOracleEngine:
    def test_returns_ground_truth_unchanged_at_zero_error(self):
        words = [OcrWord("Invoice", (0, 0, 50, 20)), OcrWord("12345", (60, 0, 110, 20))]
        engine = OracleEngine(words=words, error_rate=0.0)
        result = engine.recognize(np.full((100, 200), 255, dtype=np.uint8))
        assert [w.text for w in result.words] == ["Invoice", "12345"]

    def test_flags_itself_as_synthetic(self):
        result = OracleEngine(words=[OcrWord("x", (0, 0, 1, 1))]).recognize(
            np.zeros((10, 10), dtype=np.uint8)
        )
        assert result.meta["synthetic"] is True
        assert "not an OCR baseline" in result.meta["warning"]

    def test_error_rate_corrupts_text(self):
        words = [OcrWord("ABCDEFGHIJKLMNOP", (0, 0, 200, 20))]
        result = OracleEngine(words=words, error_rate=0.5, seed=1).recognize(
            np.full((50, 250), 255, dtype=np.uint8)
        )
        assert result.words[0].text != "ABCDEFGHIJKLMNOP"

    def test_is_reproducible_from_a_seed(self):
        words = [OcrWord("ABCDEFGHIJ", (0, 0, 100, 20))]
        image = np.full((50, 150), 255, dtype=np.uint8)
        a = OracleEngine(words=words, error_rate=0.3, seed=5).recognize(image)
        b = OracleEngine(words=words, error_rate=0.3, seed=5).recognize(image)
        assert [w.text for w in a.words] == [w.text for w in b.words]

    def test_quality_aware_mode_punishes_illegible_patches(self):
        """A flat (contrast-free) patch should draw more errors than a crisp one."""
        words = [OcrWord("ABCDEFGHIJKLMNOPQRST", (0, 0, 100, 20))]
        flat = np.full((30, 120), 128, dtype=np.uint8)
        crisp = np.full((30, 120), 255, dtype=np.uint8)
        crisp[5:15, 0:100:2] = 0

        def error_chars(image):
            out = OracleEngine(words=words, error_rate=0.0, quality_aware=True, seed=3).recognize(
                image
            )
            got = out.words[0].text
            return sum(
                1 for a, b in zip("ABCDEFGHIJKLMNOPQRST", got, strict=False) if a != b
            ) + abs(20 - len(got))

        assert error_chars(flat) > error_chars(crisp)
