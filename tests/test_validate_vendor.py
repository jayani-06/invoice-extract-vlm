"""Tests for Phase 4: vendor resolution and record validation.

Three defects during development shaped these tests, and all three were silent
— nothing crashed, the numbers were just wrong:

1. **rapidfuzz is case-sensitive by default.** "KRISHNA TEXTILES" scored 18.75
   against "Krishna Textiles" and failed to resolve. OCR produces upper-case
   names constantly, so this cost about a third of all resolutions.

2. **Stripping legal suffixes merged distinct companies.** "Arjun & Co" and
   "Arjun Pvt Ltd" both reduce to "arjun", so a clean invoice resolved to the
   wrong supplier and then mismatched on bank account, GSTIN and address at
   once — turning a correct invoice into a confident FAIL.

3. **GSTIN-first resolution hid GSTIN fraud.** An invoice carrying another
   supplier's GSTIN resolved to *that* supplier, whose record then matched
   perfectly. It scored 0/3 on injected swaps.

Each has a test that fails if it comes back.
"""

from __future__ import annotations

import pytest

from invoice_extract.rag.validate_vendor import (
    FAIL,
    MIN_HISTORY_FOR_IDENTITY_CHECK,
    PASS,
    REVIEW,
    name_similarity,
    normalize_company_name,
    resolve_vendor,
    validate_against_master,
)
from invoice_extract.rag.vendor_master import VendorMaster, rate_key


def past_invoice(
    vendor="Krishna Textiles",
    gstin="29AAAAA0000A1ZY",
    account="22966171396568",
    address="58 Anna Salai, Bengaluru, Karnataka 207473",
    total=1000.0,
    price=500.0,
    doc_id="hist-1",
    item="Office Chair",
    date="2026-03-01",
) -> dict:
    return {
        "doc_id": doc_id,
        "parties": {"vendor": {"name": vendor, "gstin": gstin, "address": address}},
        "invoice_meta": {"invoice_number": f"INV-{doc_id}", "invoice_date": date},
        "line_items": [
            {"description": item, "quantity": 2, "unit_price": price, "line_total": price * 2}
        ],
        "tax_lines": [],
        "totals": {"grand_total": total},
        "payment_info": {
            "account_number": account,
            "ifsc_or_swift": "HDFC0001234",
            "bank_name": "HDFC Bank",
        },
    }


@pytest.fixture(scope="module")
def master() -> VendorMaster:
    """Two suppliers whose names differ only by legal suffix -- the case that
    broke resolution -- plus enough history for identity checks to engage."""
    history = [
        past_invoice(doc_id="h1", price=500.0),
        past_invoice(doc_id="h2", price=505.0),
        past_invoice(doc_id="h3", price=495.0),
        past_invoice(
            vendor="Arjun & Co",
            gstin="29BBBBB1111B1Z4",
            account="99887766554433",
            doc_id="h4",
            price=300.0,
            address="7 MG Road, Chennai, Tamil Nadu 600001",
        ),
        past_invoice(
            vendor="Arjun & Co",
            gstin="29BBBBB1111B1Z4",
            account="99887766554433",
            doc_id="h5",
            price=310.0,
            address="7 MG Road, Chennai, Tamil Nadu 600001",
        ),
        past_invoice(
            vendor="Arjun Pvt Ltd",
            gstin="27CCCCC2222C1Z9",
            account="11223344556677",
            doc_id="h6",
            price=800.0,
            address="9 Ring Road, Mumbai, Maharashtra 400001",
        ),
        past_invoice(
            vendor="Arjun Pvt Ltd",
            gstin="27CCCCC2222C1Z9",
            account="11223344556677",
            doc_id="h7",
            price=790.0,
            address="9 Ring Road, Mumbai, Maharashtra 400001",
        ),
    ]
    return VendorMaster.build(history)


# -------------------------------------------------------------- name matching


class TestNameNormalisation:
    def test_case_is_folded(self):
        assert normalize_company_name("KRISHNA TEXTILES") == normalize_company_name(
            "Krishna Textiles"
        )

    def test_ampersand_expands(self):
        assert normalize_company_name("Meera & Co") == normalize_company_name("Meera and Co")

    def test_punctuation_flattens(self):
        assert normalize_company_name("Krishna-Textiles, Ltd.") == "krishna textiles ltd"

    def test_legal_suffixes_are_kept(self):
        """Stripping them merges "Arjun & Co" with "Arjun Pvt Ltd", which are
        different companies. The suffix is often the only distinguishing part."""
        assert normalize_company_name("Arjun & Co") != normalize_company_name("Arjun Pvt Ltd")


class TestNameSimilarity:
    @pytest.mark.parametrize(
        "a,b",
        [
            ("KRISHNA TEXTILES", "Krishna Textiles"),
            ("Meera and Co", "Meera & Co"),
            ("KrishnaTextiles", "Krishna Textiles"),
            ("Textiles Krishna", "Krishna Textiles"),
        ],
    )
    def test_surface_variants_score_high(self, a, b):
        assert name_similarity(a, b) >= 88.0, f"{a} vs {b}"

    def test_different_companies_score_low(self):
        assert name_similarity("Arjun & Co", "Arjun Pvt Ltd") < 70.0

    def test_case_sensitivity_regression(self):
        """rapidfuzz scores this 18.75 without normalisation."""
        assert name_similarity("KRISHNA TEXTILES", "Krishna Textiles") == 100.0

    def test_empty_names(self):
        assert name_similarity("", "Krishna") == 0.0
        assert name_similarity("Krishna", "") == 0.0


# ----------------------------------------------------------------- resolution


class TestResolution:
    def test_exact_name(self, master):
        r = resolve_vendor(past_invoice(gstin=None), master)
        assert r.resolved and r.vendor.name == "Krishna Textiles"
        assert r.method == "exact_name"

    def test_gstin_wins_even_with_a_mangled_name(self, master):
        r = resolve_vendor(past_invoice(vendor="KR1SHNA TXTLS"), master)
        assert r.resolved and r.vendor.name == "Krishna Textiles"
        assert r.method == "gstin"

    def test_uppercase_name_resolves(self, master):
        r = resolve_vendor(past_invoice(vendor="KRISHNA TEXTILES", gstin=None), master)
        assert r.resolved and r.vendor.name == "Krishna Textiles"

    def test_lost_spacing_resolves(self, master):
        r = resolve_vendor(past_invoice(vendor="KrishnaTextiles", gstin=None), master)
        assert r.resolved and r.vendor.name == "Krishna Textiles"

    def test_similar_but_distinct_companies_do_not_merge(self, master):
        r = resolve_vendor(past_invoice(vendor="Arjun & Co", gstin=None), master)
        assert r.resolved and r.vendor.name == "Arjun & Co"

    def test_a_genuinely_unknown_vendor_is_unresolved(self, master):
        r = resolve_vendor(past_invoice(vendor="Zenith Holdings 404", gstin=None), master)
        assert not r.resolved
        assert r.method == "unresolved"

    def test_name_and_gstin_pointing_elsewhere_is_a_conflict(self, master):
        """An invoice naming one supplier but carrying another's GSTIN must not
        quietly resolve to the GSTIN's owner -- that hid the fraud entirely."""
        doc = past_invoice(vendor="Arjun & Co", gstin="29AAAAA0000A1ZY")
        r = resolve_vendor(doc, master)
        assert r.method == "conflict"


# ----------------------------------------------------------------- validation


class TestValidation:
    def codes(self, doc, master, **kw):
        return {c.code for c in validate_against_master(doc, master, **kw).checks}

    def test_a_consistent_invoice_passes(self, master):
        report = validate_against_master(past_invoice(doc_id="new-1"), master)
        assert report.verdict == PASS
        assert report.checks == []

    def test_changed_bank_account_fails(self, master):
        """The single highest-value check: invoice-redirection fraud changes
        nothing on the page except the account number."""
        doc = past_invoice(account="00000000000000", doc_id="new-2")
        report = validate_against_master(doc, master)
        assert "bank_account_changed" in {c.code for c in report.checks}
        assert report.verdict == FAIL

    def test_gstin_belonging_to_another_supplier_fails(self, master):
        doc = past_invoice(vendor="Arjun & Co", gstin="29AAAAA0000A1ZY", doc_id="new-3")
        report = validate_against_master(doc, master)
        assert "gstin_mismatch" in {c.code for c in report.checks}
        assert report.verdict == FAIL

    def test_unknown_vendor_is_review_not_fail(self, master):
        """A first invoice from a new supplier warrants a look, not a refusal --
        it is indistinguishable from a legitimate onboarding."""
        doc = past_invoice(vendor="Zenith Holdings 404", gstin=None, doc_id="new-4")
        report = validate_against_master(doc, master)
        assert report.verdict == REVIEW
        assert {c.code for c in report.checks} == {"unknown_vendor"}

    def test_price_far_above_the_vendors_own_rate_is_caught(self, master):
        doc = past_invoice(price=5000.0, doc_id="new-5")  # usual is ~500
        report = validate_against_master(doc, master)
        assert "price_deviation" in {c.code for c in report.checks}
        assert report.verdict == FAIL

    def test_normal_price_variation_is_not_flagged(self, master):
        assert "price_deviation" not in self.codes(
            past_invoice(price=512.0, doc_id="new-6"), master
        )

    def test_duplicate_payment(self, master):
        history = [past_invoice(doc_id="h1", total=1000.0, date="2026-03-01")]
        doc = past_invoice(doc_id="new-7", total=1000.0, date="2026-03-10")
        assert "possible_duplicate_payment" in self.codes(doc, master, history=history)

    def test_same_amount_long_apart_is_not_a_duplicate(self, master):
        """Recurring charges are the same amount every month; only a tight
        window suggests a double submission."""
        history = [past_invoice(doc_id="h1", total=1000.0, date="2026-01-01")]
        doc = past_invoice(doc_id="new-8", total=1000.0, date="2026-09-01")
        assert "possible_duplicate_payment" not in self.codes(doc, master, history=history)

    def test_thin_history_suppresses_identity_checks(self):
        """One prior invoice does not establish what normal looks like, and
        alerting anyway trains reviewers to dismiss alerts."""
        thin = VendorMaster.build([past_invoice(doc_id="only-one")])
        assert thin.get("Krishna Textiles").invoice_count < MIN_HISTORY_FOR_IDENTITY_CHECK
        doc = past_invoice(account="00000000000000", doc_id="new-9")
        assert "bank_account_changed" not in self.codes(doc, thin)

    def test_the_worst_check_decides_the_verdict(self, master):
        doc = past_invoice(account="00000000000000", price=5000.0, doc_id="new-10")
        report = validate_against_master(doc, master)
        assert report.verdict == FAIL
        assert len(report.checks) >= 2

    def test_report_is_json_serialisable(self, master):
        import json

        json.dumps(validate_against_master(past_invoice(doc_id="new-11"), master).as_dict())

    def test_report_explains_itself(self, master):
        doc = past_invoice(account="00000000000000", doc_id="new-12")
        assert "FAIL" in validate_against_master(doc, master).explain()

    def test_validation_never_mutates_the_invoice(self, master):
        import copy

        doc = past_invoice(account="00000000000000", doc_id="new-13")
        before = copy.deepcopy(doc)
        validate_against_master(doc, master)
        assert doc == before


# -------------------------------------------------------------- vendor master


class TestVendorMaster:
    def test_it_accretes_identity_across_invoices(self, master):
        record = master.get("Krishna Textiles")
        assert record.invoice_count == 3
        assert record.gstins == {"29AAAAA0000A1ZY"}
        assert record.bank_accounts == {"22966171396568"}

    def test_contract_rate_is_the_median_of_observations(self, master):
        rate = master.get("Krishna Textiles").rates[rate_key("Office Chair")]
        assert rate.observations == 3
        assert rate.median_unit_price == pytest.approx(500.0)

    def test_market_rate_pools_across_vendors(self, master):
        """Most (vendor, item) pairs appear once in a real ledger, so a
        per-vendor rate alone leaves nearly every line unchecked."""
        market = master.market_rates[rate_key("Office Chair")]
        assert market.observations == 7

    def test_rate_for_prefers_the_vendor_then_falls_back(self, master):
        rate, source = master.rate_for(master.get("Krishna Textiles"), rate_key("Office Chair"))
        assert source == "vendor" and rate.observations == 3

    def test_lookup_by_gstin(self, master):
        assert master.by_gstin("27CCCCC2222C1Z9").name == "Arjun Pvt Ltd"
        assert master.by_gstin("99ZZZZZ9999Z9Z9") is None

    def test_round_trips_through_json(self, master):
        import json

        restored = VendorMaster.from_dict(json.loads(json.dumps(master.as_dict())))
        assert len(restored) == len(master)
        assert restored.get("Krishna Textiles").bank_accounts == {"22966171396568"}
        assert restored.market_rates.keys() == master.market_rates.keys()

    def test_rate_key_normalises_surface_differences(self):
        assert rate_key("A4 Copier Paper (500 sheets)") == rate_key("a4 copier paper 500 sheets")

    def test_building_from_nothing(self):
        assert len(VendorMaster.build([])) == 0
