"""The retrieve-augment-generate path, and the guarantees it must not lose.

Phase 3 shipped retrieval plus a similarity-weighted vote. Adding a generator
is the easy part; keeping the two properties that made the vote path safe is
not, and each of these failures produces a *better-looking* number rather than
an error:

- **Abstention.** The vote abstains when neighbours are weak or disagree, which
  is why it is safe in front of a ledger. A generator that always answers shows
  higher coverage and is strictly worse, because a wrong account code
  propagates silently while an abstention lands in a review queue.
- **Retrieval-grounding.** If the model is prompted when retrieval found
  nothing, RAG has quietly become zero-shot classification and the number is no
  longer a measurement of retrieval at all.
- **Constrained output.** "Probably office supplies" is not an account code. A
  parser that guesses from prose converts a disobedient answer into a confident
  one.

`ScriptedGenerator` is used throughout here. It exists to exercise prompt
construction, parsing and the abstention paths — it is never the source of a
reported number. `models/vlm.py` was unit-tested through a fake for weeks and
the fake could not tell us the generation call was wrong.
"""

from __future__ import annotations

import json

import pytest

from invoice_extract.rag.generate import (
    ABSTAIN,
    SYSTEM_PROMPT,
    GeneratedDecision,
    ScriptedGenerator,
    _parse_choice,
    build_prompt,
    categorize_document_rag,
    categorize_line_item_rag,
    get_generator,
)
from invoice_extract.rag.store import LedgerEntry, Match, TfidfRetriever
from invoice_extract.rag.taxonomy import CHART_OF_ACCOUNTS, UNCATEGORISED

PAPER = "A4 Copier Paper (500 sheets)"
MOUSE = "Wireless Mouse"


@pytest.fixture
def ledger() -> list[LedgerEntry]:
    return [
        LedgerEntry(description=PAPER, category="5010", vendor="Acme Stationers"),
        LedgerEntry(description="Ballpoint Pens (box of 50)", category="5010", vendor="Acme"),
        LedgerEntry(description="Printer Toner Cartridge", category="5010", vendor="Acme"),
        LedgerEntry(description=MOUSE, category="5020", vendor="Bolt Computers"),
        LedgerEntry(description="USB-C Cable 1m", category="5020", vendor="Bolt Computers"),
        LedgerEntry(description="Office Chair", category="5030", vendor="Deskworks"),
    ]


@pytest.fixture
def retriever(ledger) -> TfidfRetriever:
    r = TfidfRetriever()
    r.index(ledger)
    return r


class TestConstrainedOutput:
    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("5010", "5010"),
            ("  5030  ", "5030"),
            ("5010.", "5010"),
            ("5010 Office Supplies", "5010"),
            # A single code buried in prose is a formatting failure, not an
            # uncertain decision, so it is honoured.
            ("probably Office Supplies (5010)", "5010"),
            ("Account code: 5060", "5060"),
            # Everything below must abstain.
            (ABSTAIN, None),
            ("abstain", None),
            ("", None),
            ("   ", None),
            ("I think it is office supplies", None),
            ("1234", None),  # not in the chart
            (UNCATEGORISED, None),  # cannot be *chosen*
        ],
    )
    def test_only_a_valid_code_is_accepted(self, response, expected):
        assert _parse_choice(response) == expected

    @pytest.mark.parametrize("response", ["5010 or 5020", "maybe 5020, or 5030?", "5010/5020"])
    def test_two_codes_is_an_abstention(self, response):
        """The model telling us it cannot choose must not be read as a choice.

        `"5010 or 5020"` begins with a valid code, so checking the first token
        before checking for ambiguity returned 5010 and silently discarded the
        uncertainty. The ambiguity check therefore runs first.
        """
        assert _parse_choice(response) is None


class TestTheAbstentionGuaranteeSurvives:
    def test_an_abstaining_model_produces_an_abstention(self, retriever):
        decision = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=[ABSTAIN])
        )
        assert decision.abstained
        assert decision.category == UNCATEGORISED
        assert decision.generated is True, "the model was asked; that must be recorded"

    def test_an_unusable_answer_produces_an_abstention_not_a_guess(self, retriever):
        decision = categorize_line_item_rag(
            {"description": PAPER},
            retriever,
            ScriptedGenerator(replies=["It looks like stationery to me!"]),
        )
        assert decision.abstained
        assert "unusably" in decision.reason or "abstained" in decision.reason

    def test_an_out_of_vocabulary_code_produces_an_abstention(self, retriever):
        """A model inventing 7777 has not chosen an account; it has hallucinated
        one, and booking it would be worse than booking nothing."""
        decision = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=["7777"])
        )
        assert decision.abstained

    def test_a_line_item_with_no_description_is_never_sent_to_the_model(self, retriever):
        generator = ScriptedGenerator(replies=["5010"])
        decision = categorize_line_item_rag({"description": "  "}, retriever, generator)
        assert decision.abstained
        assert generator.prompts == [], "nothing to categorise, so nothing to ask"
        assert decision.generated is False


class TestRetrievalGrounding:
    def test_no_retrieved_context_means_the_model_is_not_asked(self, ledger):
        """The property that keeps this RAG rather than zero-shot.

        With an empty ledger there is nothing to generate *from*. Prompting
        anyway would still produce an answer, and that answer would be the
        model's prior knowledge rather than a retrieval result -- so the number
        would no longer measure what the phase claims to measure.
        """
        empty = TfidfRetriever()
        empty.index([])
        generator = ScriptedGenerator(replies=["5010"])

        decision = categorize_line_item_rag({"description": PAPER}, empty, generator)

        assert decision.abstained
        assert generator.prompts == [], "prompted with no context: this is zero-shot, not RAG"
        assert decision.generated is False
        assert "not sent to the model" in decision.reason

    def test_the_prompt_carries_the_retrieved_neighbours(self, retriever):
        generator = ScriptedGenerator(replies=["5010"])
        categorize_line_item_rag({"description": PAPER}, retriever, generator)

        prompt = generator.prompts[0]
        assert PAPER in prompt
        assert "5010" in prompt
        # The human-assigned category of a neighbour is the evidence; without
        # it the model is being asked to classify from descriptions alone.
        assert "Office Supplies" in prompt

    def test_the_prompt_offers_abstain_and_lists_the_allowed_codes(self, retriever):
        generator = ScriptedGenerator(replies=["5010"])
        categorize_line_item_rag({"description": PAPER}, retriever, generator)

        prompt = generator.prompts[0]
        assert ABSTAIN in prompt, "a model never offered the option cannot take it"
        for code in CHART_OF_ACCOUNTS:
            if code != UNCATEGORISED:
                assert code in prompt
        assert UNCATEGORISED not in prompt.split("Allowed account codes:")[1].split("Rules:")[0]

    def test_confidence_comes_from_retrieval_not_the_model(self, retriever):
        """A 0.5B model's self-reported certainty is worth nothing, and token
        log-probabilities over a 4-digit code are not calibrated either. The
        available honest signal is how strongly the retrieved context supports
        the answer, so a model answering *against* its context must score lower
        than one answering with it."""
        with_context = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=["5010"])
        )
        against_context = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=["5030"])
        )
        assert with_context.confidence > against_context.confidence


class TestOutputContract:
    def test_a_generated_decision_is_still_a_category_decision(self, retriever):
        """Every existing consumer -- the demo, the eval harness, the
        end-to-end test -- takes a CategoryDecision. Breaking that contract
        would mean touching all of them."""
        decision = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=["5010"])
        )
        from invoice_extract.rag.categorize import CategoryDecision

        assert isinstance(decision, CategoryDecision)
        assert decision.explain()

    def test_the_raw_response_is_retained(self, retriever):
        """Needed to audit a decision after the fact, and to detect the model
        drifting into a new failure mode."""
        decision = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=["5010 Office Supplies"])
        )
        assert decision.raw_response == "5010 Office Supplies"

    def test_the_decision_round_trips_through_json(self, retriever):
        """Phase 5 serves this over HTTP."""
        decision = categorize_line_item_rag(
            {"description": PAPER}, retriever, ScriptedGenerator(replies=["5010"])
        )
        payload = json.loads(json.dumps(decision.as_dict()))
        assert payload["generated"] is True
        assert payload["category"] == "5010"

    def test_a_whole_document_is_categorised(self, retriever):
        doc = {
            "parties": {"vendor": {"name": "Acme Stationers"}},
            "line_items": [{"description": PAPER}, {"description": MOUSE}],
        }
        generator = ScriptedGenerator(replies=["5010", "5020"])
        decisions = categorize_document_rag(doc, retriever, generator)

        assert [d.category for d in decisions] == ["5010", "5020"]
        assert len(generator.prompts) == 2


class TestGeneratorRegistry:
    def test_scripted_is_reachable_by_name(self):
        assert isinstance(get_generator("scripted"), ScriptedGenerator)

    def test_an_unknown_generator_raises(self):
        with pytest.raises(KeyError):
            get_generator("gpt-9")

    def test_local_is_constructible_without_loading_weights(self):
        """Constructing must not download a gigabyte. Loading is deferred to
        the first call so a runner can validate its arguments cheaply."""
        from invoice_extract.rag.generate import LocalGenerator

        g = get_generator("local")
        assert isinstance(g, LocalGenerator)
        assert g._model is None


class TestPromptConstruction:
    def test_vendor_history_is_included_when_known(self):
        matches = [
            Match(entry=LedgerEntry(description=PAPER, category="5010", vendor="Acme"), score=0.9)
        ]
        prompt = build_prompt(
            PAPER, matches, vendor="Acme", hsn="4802", usual={"5010": 7, "5020": 1}
        )
        assert "Acme" in prompt
        assert "4802" in prompt
        assert "7x" in prompt

    def test_the_system_prompt_demands_a_bare_code(self):
        assert "one account code" in SYSTEM_PROMPT

    def test_neighbours_are_presented_closest_first(self):
        """A small model reads position as importance, so the ordering is part
        of the evidence rather than incidental."""
        matches = [
            Match(entry=LedgerEntry(description="near", category="5010"), score=0.9),
            Match(entry=LedgerEntry(description="far", category="5020"), score=0.3),
        ]
        prompt = build_prompt("x", matches)
        assert prompt.index("'near'") < prompt.index("'far'")


class TestGeneratedDecisionDefaults:
    def test_it_defaults_to_not_generated(self):
        """`generated` separates "the model said this" from "we never asked".
        Defaulting to True would let a short-circuited decision be reported as
        a model result."""
        assert GeneratedDecision(UNCATEGORISED, 0.0).generated is False
