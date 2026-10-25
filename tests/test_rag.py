"""Tests for Phase 3: retrieval-based categorisation and anomaly detection.

Two properties carry most of the weight here, because both failed silently
during development and neither would have crashed anything:

1. **Abstention has to actually happen.** A categoriser that always answers
   looks identical in aggregate accuracy to one that abstains well, until you
   separate coverage from accuracy. The threshold tests pin that behaviour.

2. **Flags must fire for the right reason.** The duplicate-invoice check scored
   0/2 on injected duplicates because the history was keyed on normalised
   numbers and looked up with raw ones. Nothing errored; the signal was simply
   dead. Every flag type therefore gets a test that triggers it directly.
"""

from __future__ import annotations

from datetime import date

import pytest

from invoice_extract.rag.anomaly import (
    ESCALATION_PAIRS,
    VALID_SPLIT_RATES,
    detect_anomalies,
    escalate_combinations,
    normalize_invoice_number,
    rule_flags,
    semantic_flags,
    statistical_flags,
)
from invoice_extract.rag.categorize import (
    MIN_NEIGHBOUR_SIMILARITY,
    categorize_document,
    categorize_line_item,
)
from invoice_extract.rag.store import LedgerEntry, TfidfRetriever, VendorStats
from invoice_extract.rag.taxonomy import (
    CANONICAL_ITEM_CATEGORY,
    CHART_OF_ACCOUNTS,
    UNCATEGORISED,
    category_of,
)

TODAY = date(2026, 9, 20)


def ledger() -> list[LedgerEntry]:
    """A small ledger covering three accounts."""
    rows = [
        ("A4 Copier Paper (500 sheets)", "5010", "Sundar Traders"),
        ("A4 Copier Paper (500 sheets)", "5010", "Ramesh Enterprises"),
        ("Ballpoint Pens (box of 50)", "5010", "Sundar Traders"),
        ("Printer Toner Cartridge", "5010", "Sundar Traders"),
        ("Wireless Mouse", "5020", "Krishna Electronics"),
        ("Laptop Bag", "5020", "Krishna Electronics"),
        ("USB-C Cable 1m", "5020", "Krishna Electronics"),
        ("Office Chair", "5030", "Ganesh Furnishings"),
        ("LED Desk Lamp", "5030", "Ganesh Furnishings"),
        ("Steel Filing Cabinet", "5030", "Ganesh Furnishings"),
    ]
    return [
        LedgerEntry(description=d, category=c, vendor=v, line_total=100.0, doc_id=f"d{i}")
        for i, (d, c, v) in enumerate(rows)
    ]


@pytest.fixture(scope="module")
def retriever() -> TfidfRetriever:
    r = TfidfRetriever()
    r.index(ledger())
    return r


@pytest.fixture(scope="module")
def stats() -> VendorStats:
    s = VendorStats.build(ledger())
    # Eight invoices spread around the 1,180 that `invoice()` bills, so the
    # baseline document is genuinely unremarkable. An earlier fixture clustered
    # these at 950-1100, which made the supposedly clean invoice a real 3.9-sigma
    # outlier -- the test was wrong, not the detector.
    s.totals["Sundar Traders"] = [1180.0, 1050.0, 1300.0, 900.0, 1400.0, 1150.0, 1250.0, 1000.0]
    return s


def invoice(**over) -> dict:
    doc = {
        "doc_id": "test-1",
        "parties": {"vendor": {"name": "Sundar Traders", "gstin": "29AAAAA0000A1ZY"}},
        "invoice_meta": {"invoice_number": "INV-2026-0001", "invoice_date": "2026-05-01"},
        "line_items": [
            {
                "description": "A4 Copier Paper (500 sheets)",
                "quantity": 2,
                "unit_price": 500.0,
                "line_total": 1000.0,
            }
        ],
        "tax_lines": [
            {"type": "CGST", "rate": 9.0, "amount": 90.0},
            {"type": "SGST", "rate": 9.0, "amount": 90.0},
        ],
        "totals": {"subtotal": 1000.0, "tax_total": 180.0, "grand_total": 1180.0},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(doc.get(key), dict):
            doc[key].update(value)
        else:
            doc[key] = value
    return doc


# ------------------------------------------------------------------ taxonomy


class TestTaxonomy:
    def test_every_canonical_item_maps_to_a_real_account(self):
        for description, code in CANONICAL_ITEM_CATEGORY.items():
            assert code in CHART_OF_ACCOUNTS, description
            assert code != UNCATEGORISED, description

    def test_unknown_description_is_uncategorised(self):
        """Variants must NOT resolve here, or the evaluation would be comparing
        the categoriser against itself."""
        assert category_of("A4 COPIER PPR 500SH") == UNCATEGORISED
        assert category_of(None) == UNCATEGORISED
        assert category_of("") == UNCATEGORISED


# -------------------------------------------------------------- retrieval


class TestRetriever:
    def test_finds_the_exact_item_first(self, retriever):
        matches = retriever.search("Wireless Mouse", k=3)
        assert matches[0].entry.description == "Wireless Mouse"
        assert matches[0].score > 0.5

    @pytest.mark.parametrize(
        "variant,expected",
        [
            ("A4 COPIER PPR (500 SHEETS)", "5010"),
            ("Mouse Wireless", "5020"),
            ("laptop-bag", "5020"),
            ("0ffice Chair", "5030"),
            ("STL FILE CAB", "5030"),
        ],
    )
    def test_surface_variants_still_reach_the_right_account(self, retriever, variant, expected):
        """The point of character n-grams: vendors write the same item
        differently every time, and word tokens share almost nothing between
        'Steel Filing Cabinet' and 'STL FILE CAB'."""
        matches = retriever.search(variant, k=3)
        assert matches, variant
        assert matches[0].entry.category == expected, f"{variant} -> {matches[0].entry}"

    def test_empty_ledger_returns_nothing(self):
        r = TfidfRetriever()
        r.index([])
        assert r.search("anything") == []

    def test_empty_query_returns_nothing(self, retriever):
        assert retriever.search("") == []
        assert retriever.search("   ") == []

    def test_scores_are_cosine_similarities(self, retriever):
        for match in retriever.search("Office Chair", k=5):
            assert 0.0 < match.score <= 1.0 + 1e-9


# ---------------------------------------------------------- categorisation


class TestCategorize:
    def test_exact_match(self, retriever, stats):
        d = categorize_line_item(
            {"description": "Office Chair"},
            retriever,
            vendor="Ganesh Furnishings",
            vendor_stats=stats,
        )
        assert d.category == "5030"
        assert not d.abstained
        assert d.confidence > 0.5

    def test_variant_match(self, retriever, stats):
        d = categorize_line_item({"description": "OFC CHAIR"}, retriever, vendor_stats=stats)
        assert d.category == "5030"

    def test_abstains_on_a_completely_unrelated_item(self, retriever):
        """Industrial plant has nothing to do with a stationery ledger. The
        honest answer is 'I don't know', not the nearest of three bad options."""
        d = categorize_line_item({"description": "Hydraulic excavator track pin"}, retriever)
        assert d.abstained
        assert d.category == UNCATEGORISED
        assert d.reason

    def test_abstains_when_there_is_no_description(self, retriever):
        d = categorize_line_item({"description": None}, retriever)
        assert d.abstained
        assert "no description" in d.reason

    def test_abstains_against_an_empty_ledger(self):
        empty = TfidfRetriever()
        empty.index([])
        d = categorize_line_item({"description": "Office Chair"}, empty)
        assert d.abstained

    def test_decision_carries_its_evidence(self, retriever):
        """RAG's claimed advantage over a classifier is explainability, which
        is only real if the neighbours survive into the output."""
        d = categorize_line_item({"description": "Wireless Mouse"}, retriever)
        assert d.neighbours
        assert all(m.score >= MIN_NEIGHBOUR_SIMILARITY for m in d.neighbours)
        assert d.explain()
        assert "neighbours" in d.as_dict()

    def test_decision_is_json_serialisable(self, retriever):
        import json

        json.dumps(categorize_line_item({"description": "Laptop Bag"}, retriever).as_dict())

    def test_document_level_categorisation(self, retriever, stats):
        decisions = categorize_document(invoice(), retriever, vendor_stats=stats)
        assert len(decisions) == 1
        assert decisions[0].category == "5010"

    def test_document_with_no_line_items(self, retriever):
        assert categorize_document({"line_items": []}, retriever) == []


# ------------------------------------------------------------ rule signals


class TestRuleFlags:
    def codes(self, doc, **kw):
        return {f.code for f in rule_flags(doc, today=TODAY, **kw)}

    def test_a_clean_invoice_raises_nothing(self):
        assert self.codes(invoice()) == set()

    def test_future_invoice_date(self):
        assert "future_invoice_date" in self.codes(
            invoice(invoice_meta={"invoice_date": "2029-01-01"})
        )

    def test_due_date_before_invoice_date(self):
        assert "due_before_invoice" in self.codes(
            invoice(invoice_meta={"invoice_date": "2026-05-01", "due_date": "2026-04-01"})
        )

    def test_duplicate_invoice_number(self):
        codes = self.codes(invoice(), seen_invoice_numbers={"INV20260001": "earlier-doc"})
        assert "duplicate_invoice_number" in codes

    def test_duplicate_detection_survives_punctuation_differences(self):
        """Vendors reissue INV-2026-0001 as 'inv 2026 0001'. Comparing raw
        strings would miss every one -- this is the bug that made the check
        score 0/2 on injected duplicates."""
        doc = invoice(invoice_meta={"invoice_number": "inv/2026/0001"})
        assert "duplicate_invoice_number" in self.codes(
            doc, seen_invoice_numbers={"INV20260001": "earlier-doc"}
        )

    def test_a_document_does_not_flag_itself(self):
        assert "duplicate_invoice_number" not in self.codes(
            invoice(), seen_invoice_numbers={"INV20260001": "test-1"}
        )

    def test_invalid_gst_rate(self):
        assert "invalid_gst_rate" in self.codes(
            invoice(tax_lines=[{"type": "CGST", "rate": 11.5, "amount": 10.0}])
        )

    @pytest.mark.parametrize("rate", sorted(VALID_SPLIT_RATES))
    def test_statutory_rates_are_accepted(self, rate):
        assert "invalid_gst_rate" not in self.codes(
            invoice(tax_lines=[{"type": "CGST", "rate": rate, "amount": 10.0}])
        )

    def test_invalid_gstin(self):
        assert "invalid_gstin" in self.codes(
            invoice(parties={"vendor": {"name": "X", "gstin": "29AAAAA0000A1Z5"}})
        )

    def test_arithmetic_mismatch(self):
        assert "arithmetic_mismatch" in self.codes(
            invoice(totals={"subtotal": 1000.0, "tax_total": 180.0, "grand_total": 99999.0})
        )

    def test_missing_fields_do_not_raise_spurious_flags(self):
        assert self.codes({"doc_id": "x", "invoice_meta": {}, "totals": {}}) == set()


class TestNormalizeInvoiceNumber:
    @pytest.mark.parametrize(
        "raw", ["INV-2026-0001", "inv/2026/0001", "INV 2026 0001", "inv.2026.0001"]
    )
    def test_equivalent_forms_collapse(self, raw):
        assert normalize_invoice_number(raw) == "INV20260001"

    def test_empty(self):
        assert normalize_invoice_number(None) is None
        assert normalize_invoice_number("") is None
        assert normalize_invoice_number("---") is None


# ----------------------------------------------------- statistical signals


class TestStatisticalFlags:
    def test_amount_far_from_vendor_history_is_flagged(self, stats):
        codes = {
            f.code for f in statistical_flags(invoice(totals={"grand_total": 500_000.0}), stats)
        }
        assert "unusual_amount_for_vendor" in codes

    def test_a_normal_amount_is_not_flagged(self, stats):
        codes = {f.code for f in statistical_flags(invoice(totals={"grand_total": 1005.0}), stats)}
        assert "unusual_amount_for_vendor" not in codes

    def test_thin_history_produces_no_zscore(self):
        """A z-score from two invoices is noise wearing a number's clothes."""
        thin = VendorStats()
        thin.totals["New Vendor"] = [100.0, 120.0]
        doc = invoice(parties={"vendor": {"name": "New Vendor"}}, totals={"grand_total": 99_999.0})
        assert thin.amount_zscore("New Vendor", 99_999.0) is None
        assert not [
            f for f in statistical_flags(doc, thin) if f.code == "unusual_amount_for_vendor"
        ]

    def test_round_total_is_low_severity_only(self, stats):
        flags = statistical_flags(invoice(totals={"grand_total": 50_000.0}), stats)
        round_flags = [f for f in flags if f.code == "suspiciously_round_total"]
        assert round_flags and round_flags[0].severity == "low"


# -------------------------------------------------------- semantic signals


class TestSemanticFlags:
    def test_unknown_vendor_is_low_severity(self, retriever, stats):
        """Businesses onboard suppliers constantly. Flagging every first
        invoice as actionable put 12 flags on clean documents and dropped
        anomaly precision to 0.41."""
        doc = invoice(parties={"vendor": {"name": "Never Seen Ltd"}})
        flags = [f for f in semantic_flags(doc, retriever, stats) if f.code == "unknown_vendor"]
        assert flags and flags[0].severity == "low"

    def test_known_vendor_is_not_flagged(self, retriever, stats):
        codes = {f.code for f in semantic_flags(invoice(), retriever, stats)}
        assert "unknown_vendor" not in codes

    def test_novel_line_item(self, retriever, stats):
        doc = invoice(line_items=[{"description": "Hydraulic excavator track pin"}])
        assert "novel_line_item" in {f.code for f in semantic_flags(doc, retriever, stats)}

    def test_a_familiar_item_is_not_novel(self, retriever, stats):
        assert "novel_line_item" not in {
            f.code for f in semantic_flags(invoice(), retriever, stats)
        }


class TestEscalation:
    def test_two_low_signals_together_escalate(self):
        from invoice_extract.rag.anomaly import Flag

        low = [
            Flag("unknown_vendor", "low", "", "semantic"),
            Flag("suspiciously_round_total", "low", "", "statistical"),
        ]
        escalated = escalate_combinations(low)
        combined = [f for f in escalated if f.code.startswith("combined_")]
        assert combined and combined[0].severity == "medium"

    def test_one_low_signal_alone_does_not_escalate(self):
        from invoice_extract.rag.anomaly import Flag

        only = [Flag("unknown_vendor", "low", "", "semantic")]
        assert escalate_combinations(only) == only

    def test_escalation_pairs_are_all_low_signals(self):
        """Escalating a pair that already contains a medium/high flag would be
        double-counting, not new information."""
        low_codes = {"unknown_vendor", "suspiciously_round_total", "novel_line_item"}
        for a, b in ESCALATION_PAIRS:
            assert a in low_codes or b in low_codes


# ------------------------------------------------------------------ facade


class TestDetectAnomalies:
    def test_clean_invoice_is_clean(self, retriever, stats):
        report = detect_anomalies(invoice(), retriever, stats, today=TODAY)
        actionable = [f for f in report.flags if f.severity in ("high", "medium")]
        assert actionable == []

    def test_report_sorts_most_severe_first(self, retriever, stats):
        doc = invoice(
            invoice_meta={"invoice_date": "2029-01-01"},
            parties={"vendor": {"name": "Never Seen Ltd"}},
        )
        report = detect_anomalies(doc, retriever, stats, today=TODAY)
        severities = [f.severity for f in report.flags]
        assert severities == sorted(
            severities, key=lambda s: -{"high": 3, "medium": 2, "low": 1}[s]
        )
        assert report.max_severity == "high"

    def test_report_groups_by_signal_family(self, retriever, stats):
        doc = invoice(
            invoice_meta={"invoice_date": "2029-01-01"},
            parties={"vendor": {"name": "Never Seen Ltd"}},
        )
        grouped = detect_anomalies(doc, retriever, stats, today=TODAY).by_signal()
        assert "rule" in grouped and "semantic" in grouped

    def test_report_is_json_serialisable(self, retriever, stats):
        import json

        json.dumps(detect_anomalies(invoice(), retriever, stats, today=TODAY).as_dict())

    def test_it_never_mutates_the_document(self, retriever, stats):
        import copy

        doc = invoice(totals={"grand_total": 99999.0})
        before = copy.deepcopy(doc)
        detect_anomalies(doc, retriever, stats, today=TODAY)
        assert doc == before
