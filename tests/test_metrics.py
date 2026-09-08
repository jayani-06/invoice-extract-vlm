import json
from pathlib import Path

import pytest

from invoice_extract.eval.metrics import (
    aggregate,
    anls,
    compare_documents,
    compare_field,
    match_line_items,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> dict[str, dict]:
    return {json.loads(line)["doc_id"]: json.loads(line) for line in path.read_text().splitlines()}


@pytest.fixture(scope="module")
def fixtures():
    gold = read_jsonl(REPO_ROOT / "tests" / "fixtures" / "gold.jsonl")
    pred = read_jsonl(REPO_ROOT / "tests" / "fixtures" / "pred.jsonl")
    return gold, pred


def test_anls_identical_strings_is_one():
    assert anls("Sundar Traders", "Sundar Traders") == 1.0


def test_anls_completely_different_strings_is_low():
    assert anls("Sundar Traders", "xyz") < 0.3


def test_compare_field_numeric_within_tolerance_is_correct():
    cmp = compare_field("totals.grand_total", 8110.5, 8111.0)
    assert cmp.correct


def test_compare_field_numeric_outside_tolerance_is_incorrect():
    cmp = compare_field("totals.grand_total", 8110.5, 9000.0)
    assert not cmp.correct


def test_compare_field_date_requires_exact_match():
    assert compare_field("invoice_meta.invoice_date", "2026-08-14", "2026-08-14").correct
    assert not compare_field("invoice_meta.invoice_date", "2026-08-14", "2026-08-15").correct


def test_compare_field_both_none_is_correct():
    cmp = compare_field("invoice_meta.due_date", None, None)
    assert cmp.correct and cmp.score == 1.0


def test_compare_field_one_none_is_incorrect():
    assert not compare_field("invoice_meta.due_date", "2026-09-13", None).correct
    assert not compare_field("invoice_meta.due_date", None, "2026-09-13").correct


def test_match_line_items_perfect_match():
    items = [{"description": "Widget", "line_total": 100.0}]
    result = match_line_items(items, items)
    assert result.n_matched == 1
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_match_line_items_missing_prediction_hurts_recall():
    gold = [
        {"description": "Widget A", "line_total": 100.0},
        {"description": "Widget B", "line_total": 50.0},
    ]
    pred = [{"description": "Widget A", "line_total": 100.0}]
    result = match_line_items(gold, pred)
    assert result.n_matched == 1
    assert result.recall == 0.5
    assert result.precision == 1.0


def test_compare_documents_on_fixture_doc1_detects_known_errors(fixtures):
    gold, pred = fixtures
    report = compare_documents(gold["gst_synth_0001"], pred["gst_synth_0001"])
    # Known injected error: buyer address typo in the pred fixture.
    assert report.field_comparisons["parties.buyer.address"].score < 1.0
    # Known injected error: grand_total off by 0.5, within 1% tolerance -> still correct.
    assert report.field_comparisons["totals.grand_total"].correct
    # Known injected error: line item 2 quantity wrong (4 vs 5) -> line item field mismatch,
    # but the item should still MATCH (same description/amount) so recall stays 1.0.
    assert report.line_items.n_matched == 2
    assert report.line_items.field_reports["quantity"].n_correct == 1  # only item 1's qty matches


def test_compare_documents_on_fixture_doc2_missing_line_item(fixtures):
    gold, pred = fixtures
    report = compare_documents(gold["gst_synth_0002"], pred["gst_synth_0002"])
    assert report.line_items.n_gold == 2
    assert report.line_items.n_pred == 1
    assert report.line_items.recall == 0.5
    # due_date present in gold, null in pred -> incorrect
    assert not report.field_comparisons["invoice_meta.due_date"].correct
    # invoice_number has an injected typo (0 -> O) -> ANLS < 1 but likely still >= threshold
    assert report.field_comparisons["invoice_meta.invoice_number"].score < 1.0


def test_aggregate_smoke_over_fixtures(fixtures):
    gold, pred = fixtures
    reports = [compare_documents(gold[doc_id], pred[doc_id]) for doc_id in gold]
    agg = aggregate(reports)
    assert agg.n_docs == 2
    assert 0.0 <= agg.macro_field_accuracy <= 1.0
    assert 0.0 <= agg.micro_field_accuracy <= 1.0
    assert 0.0 <= agg.doc_critical_accuracy <= 1.0
    assert agg.line_items_micro.n_gold == 4
    assert agg.line_items_micro.n_pred == 3
