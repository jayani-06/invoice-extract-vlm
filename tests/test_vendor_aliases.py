"""Trading names, branch addresses, and legitimate bank changes.

Three Phase 4 limitations came from the generator giving each supplier exactly
one address, one name and one bank account forever:

- `address_mismatch` never executed, so its 75% threshold was never validated.
- Resolution matched on the legal name alone, which is easier than reality.
- Every bank-account change in the corpus was fraud, so `bank_account_changed`
  was scored against a population containing no honest counter-example.

Fixing the generator is only half of it. A company billing as both "Anand
Industries" and "Anand Ind." produces **two records with half the history
each** unless the master knows they are one entity, and a thin record is worse
than no record: identity checks switch on at two invoices, so halving the
history moves suppliers into the band where they are checked but the evidence
is not there yet.

The alias map is explicit, never inferred. Fuzzy-merging names would also merge
`Arjun & Co` with `Arjun Pvt Ltd`, which this project deliberately keeps apart.
"""

from __future__ import annotations

from invoice_extract.rag.validate_vendor import resolve_vendor
from invoice_extract.rag.vendor_master import VendorMaster

LEGAL = "Anand Industries"
TRADING = "Anand Ind."
ALIASES = {TRADING: LEGAL}


def invoice(
    name: str,
    *,
    gstin: str = "29AAAAA0000A1ZY",
    address: str = "1 MG Road, Bengaluru",
    account: str = "49151157779884",
    doc_id: str = "d",
) -> dict:
    return {
        "doc_id": doc_id,
        "parties": {"vendor": {"name": name, "gstin": gstin, "address": address}},
        "payment_info": {"account_number": account},
        "invoice_meta": {"invoice_number": f"INV-{doc_id}", "invoice_date": "2026-03-01"},
        "totals": {"grand_total": 1000.0},
        "line_items": [],
    }


class TestAliasFolding:
    def test_two_billing_names_become_one_record(self):
        master = VendorMaster.build(
            [invoice(LEGAL, doc_id="1"), invoice(TRADING, doc_id="2"), invoice(LEGAL, doc_id="3")],
            aliases=ALIASES,
        )
        assert len(master.records) == 1
        record = master.get(LEGAL)
        assert record.invoice_count == 3, "history split across the two names"

    def test_without_the_alias_map_the_history_splits(self):
        """The failure this exists to prevent, pinned so the fix cannot quietly
        regress. Two records of 2 and 1 invoices instead of one of 3 puts the
        supplier in the band where identity checks run on evidence too thin to
        support them."""
        master = VendorMaster.build(
            [invoice(LEGAL, doc_id="1"), invoice(TRADING, doc_id="2"), invoice(LEGAL, doc_id="3")]
        )
        assert len(master.records) == 2
        assert master.get(LEGAL).invoice_count == 2

    def test_a_trading_name_resolves_to_the_legal_entity(self):
        master = VendorMaster.build([invoice(LEGAL), invoice(LEGAL, doc_id="2")], aliases=ALIASES)
        resolution = resolve_vendor(invoice(TRADING), master)
        assert resolution.resolved
        assert resolution.vendor.name == LEGAL

    def test_an_aliased_invoice_is_not_read_as_a_gstin_conflict(self):
        """`resolve_vendor` reports a conflict when the GSTIN and the name point
        at different suppliers -- the fix for GSTIN fraud scoring 0/3. Comparing
        raw billing names there would make every aliased invoice a conflict, so
        it compares canonical entities."""
        master = VendorMaster.build([invoice(LEGAL), invoice(LEGAL, doc_id="2")], aliases=ALIASES)
        resolution = resolve_vendor(invoice(TRADING), master)
        assert resolution.method != "conflict"

    def test_aliases_are_searchable_by_similarity(self):
        """An invoice arriving as a *misspelled* trading name has to match the
        alias. The legal name may share few characters with it, so excluding
        aliases from the search list would leave those unresolvable."""
        master = VendorMaster.build([invoice(LEGAL), invoice(LEGAL, doc_id="2")], aliases=ALIASES)
        assert TRADING in master.names()

    def test_an_unknown_name_is_still_unknown(self):
        """Aliasing must not become a way to resolve anything to anything."""
        master = VendorMaster.build([invoice(LEGAL)], aliases=ALIASES)
        assert master.get("Completely Different Pvt Ltd") is None

    def test_canonical_name_is_identity_for_a_legal_name(self):
        master = VendorMaster.build([invoice(LEGAL)], aliases=ALIASES)
        assert master.canonical_name(LEGAL) == LEGAL
        assert master.canonical_name(TRADING) == LEGAL
        assert master.canonical_name("  ") == ""


class TestBranchAddresses:
    def test_a_known_branch_does_not_flag(self):
        """Once the master has seen a site, billing from it is unremarkable.
        `address_mismatch` scores the invoice against *every* known address and
        takes the best, so a second site must not look like a changed one."""
        branch = "14 Industrial Estate, Chennai"
        master = VendorMaster.build(
            [invoice(LEGAL, doc_id="1"), invoice(LEGAL, address=branch, doc_id="2")],
            aliases=ALIASES,
        )
        record = master.get(LEGAL)
        assert len(record.addresses) == 2

        from invoice_extract.rag.validate_vendor import validate_against_master

        report = validate_against_master(invoice(LEGAL, address=branch), master, history=[])
        assert "address_mismatch" not in {c.code for c in report.checks}

    def test_an_unseen_address_reaches_review_not_fail(self):
        """A first invoice from a new branch genuinely is worth a glance, so
        flagging it is correct. It must be REVIEW -- an address the master has
        not seen is not evidence of fraud, and FAIL would block a legitimate
        payment."""
        from invoice_extract.rag.validate_vendor import FAIL, validate_against_master

        master = VendorMaster.build(
            [invoice(LEGAL, doc_id="1"), invoice(LEGAL, doc_id="2")], aliases=ALIASES
        )
        report = validate_against_master(
            invoice(LEGAL, address="99 Nowhere Lane, Kolkata 700001"), master, history=[]
        )
        codes = {c.code for c in report.checks}
        assert "address_mismatch" in codes, "the check still never fires"
        assert report.verdict != FAIL
