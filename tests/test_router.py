"""The per-field router, and the one way it could lie.

Combining two extractors is easy to do and easy to fake. A router fitted on
the documents it is then scored against will beat both of them every time,
because it is choosing with the answers in hand. That is an oracle, and it
would produce exactly the kind of impressive, meaningless number this project
exists to avoid.

So these tests pin the hold-out discipline, the margin that stops the policy
chasing noise, and the merge actually taking fields from where it says.
"""

from __future__ import annotations

import json

from invoice_extract.models.router import MIN_MARGIN, RoutingPolicy, fit, route


def doc(doc_id: str, *, vendor: str | None, total: float | None) -> dict:
    return {
        "doc_id": doc_id,
        "source": {"dataset": "other", "original_id": doc_id, "split": "train"},
        "parties": {"vendor": {"name": vendor}},
        "invoice_meta": {"invoice_number": "INV-1", "invoice_date": "2026-01-01"},
        "line_items": [],
        "tax_lines": [],
        "totals": {"grand_total": total},
    }


def world(n: int = 10):
    """`a` always reads the vendor, never the total. `b` the reverse."""
    gold, a, b = {}, {}, {}
    for i in range(n):
        d = f"d{i}"
        gold[d] = doc(d, vendor=f"Acme {i}", total=100.0 + i)
        a[d] = doc(d, vendor=f"Acme {i}", total=None)
        b[d] = doc(d, vendor=None, total=100.0 + i)
    return gold, {"a": a, "b": b}


class TestItLearnsTheRightSplit:
    def test_each_field_goes_to_whoever_reads_it(self):
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        assert policy.winner_for("totals.grand_total") == "b"
        assert policy.winner_for("parties.vendor.name") == "a"

    def test_unrouted_fields_fall_back_to_the_default(self):
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        assert policy.winner_for("invoice_meta.invoice_date") == "a"
        assert "invoice_meta.invoice_date" not in policy.choices

    def test_a_field_both_read_equally_is_not_routed(self):
        """Switching on a tie makes the policy depend on which documents
        happened to land in the fit, not on either extractor being better."""
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        assert "invoice_meta.invoice_number" not in policy.choices

    def test_a_margin_below_the_floor_is_ignored(self):
        """One document in ten is noise at this sample size."""
        gold, preds = world(n=10)
        preds["b"]["d0"]["parties"]["vendor"]["name"] = "Acme 0"  # b now wins 1/10
        policy = fit(gold, preds, default="a", min_margin=MIN_MARGIN)
        assert "parties.vendor.name" not in policy.choices

    def test_a_large_margin_is_taken(self):
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        choice = policy.choices["totals.grand_total"]
        assert choice.winner == "b"
        assert choice.margin > 0.5


class TestTheMerge:
    def test_fields_come_from_the_chosen_extractor(self):
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        merged = route(policy, preds, "d3")
        assert merged["parties"]["vendor"]["name"] == "Acme 3"  # from a
        assert merged["totals"]["grand_total"] == 103.0  # from b

    def test_the_base_document_is_not_mutated(self):
        """`route` is called per document in a loop; mutating the source would
        corrupt every later call and the corruption would look like a model
        result."""
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        before = json.dumps(preds["a"]["d3"], sort_keys=True)
        route(policy, preds, "d3")
        assert json.dumps(preds["a"]["d3"], sort_keys=True) == before

    def test_a_document_missing_from_the_donor_keeps_the_default(self):
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        del preds["b"]["d3"]
        merged = route(policy, preds, "d3")
        assert merged["parties"]["vendor"]["name"] == "Acme 3"

    def test_a_document_missing_from_the_default_yields_nothing(self):
        """Not a half-built document: an empty result scores as a miss, which
        is the truth, where a partial merge would score as a partial success."""
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        del preds["a"]["d3"]
        assert route(policy, preds, "d3") == {}


class TestPolicyRoundTrip:
    def test_a_saved_policy_reloads_identically(self, tmp_path):
        """The policy is the artefact that has to be auditable: it is what
        makes the router a reported method rather than a number."""
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        path = tmp_path / "policy.json"
        policy.save(path)
        reloaded = RoutingPolicy.load(path)
        assert reloaded.default == policy.default
        assert set(reloaded.choices) == set(policy.choices)
        for key, choice in policy.choices.items():
            assert reloaded.choices[key].winner == choice.winner

    def test_the_policy_is_human_readable(self, tmp_path):
        gold, preds = world()
        path = tmp_path / "policy.json"
        fit(gold, preds, default="a").save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["default"] == "a"
        entry = payload["fields"]["totals.grand_total"]
        assert entry["winner"] == "b"
        assert "scores" in entry and "margin" in entry and "n" in entry


class TestItCannotQuietlyBecomeAnOracle:
    def test_fitting_and_scoring_on_the_same_documents_is_the_callers_problem(self):
        """`fit` cannot detect a leak, and pretending otherwise would be worse
        than documenting it. This pins the fact so the runner's split is
        understood as the only thing preventing it -- and the runner has its
        own guard against an empty hold-out."""
        gold, preds = world()
        policy = fit(gold, preds, default="a")
        # Scoring on the fit set would be trivially perfect for both fields.
        merged = route(policy, preds, "d0")
        assert merged["parties"]["vendor"]["name"] == gold["d0"]["parties"]["vendor"]["name"]
        assert merged["totals"]["grand_total"] == gold["d0"]["totals"]["grand_total"]
