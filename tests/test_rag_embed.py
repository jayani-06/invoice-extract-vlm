"""The dense retriever, against the same `Retriever` contract as TF-IDF.

Both backends have to be interchangeable, because `categorize` and `anomaly`
only ever see the Protocol and their similarity thresholds are tuned to the
score scale. A retriever that returned, say, raw dot products on unnormalised
vectors would still "work" — it would just silently shift every threshold in
Phase 3, and the result would look like a retrieval-quality change rather than
a units bug.

Marked `slow`: the encoder is ~90 MB and loading it costs seconds. Run with
`pytest -m "not slow"` to skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from invoice_extract.rag.embed import (
    DEFAULT_MODEL,
    DenseRetriever,
    EmbeddingModelUnavailableError,
    get_retriever,
)
from invoice_extract.rag.store import LedgerEntry, TfidfRetriever

LEDGER = [
    LedgerEntry(description="A4 Copier Paper (500 sheets)", category="5010", vendor="Acme"),
    LedgerEntry(description="Ballpoint Pens (box of 50)", category="5010", vendor="Acme"),
    LedgerEntry(description="Wireless Mouse", category="5020", vendor="Bolt"),
    LedgerEntry(description="USB-C Cable 1m", category="5020", vendor="Bolt"),
    LedgerEntry(description="Office Chair", category="5030", vendor="Deskworks"),
]


class TestRegistry:
    def test_both_backends_are_reachable_by_name(self):
        assert isinstance(get_retriever("tfidf"), TfidfRetriever)
        assert isinstance(get_retriever("dense"), DenseRetriever)

    def test_an_unknown_backend_raises(self):
        with pytest.raises(KeyError):
            get_retriever("qdrant")

    def test_constructing_dense_does_not_load_the_model(self):
        """A runner must be able to validate its arguments without paying for
        90 MB of weights."""
        assert DenseRetriever()._model is None

    def test_a_missing_model_raises_rather_than_falling_back(self):
        """Never silently downgrade to TF-IDF. A report saying "dense" while
        TF-IDF produced the numbers is the most expensive class of mistake in
        this project, and it has made the equivalent one before (a demo cache
        two corpus regenerations stale, displayed as current)."""
        r = DenseRetriever(model_name="definitely/not-a-real-model-xyz")
        with pytest.raises(EmbeddingModelUnavailableError):
            r.index(LEDGER)


class TestEmptyIndex:
    def test_searching_an_unindexed_retriever_returns_nothing(self):
        assert DenseRetriever().search("paper") == []

    def test_an_empty_ledger_indexes_without_loading_the_model(self):
        """Phase 3's no-context path indexes an empty ledger, and it must not
        pay for the encoder to find out there is nothing to encode."""
        r = DenseRetriever()
        r.index([])
        assert r.search("paper") == []
        assert r._model is None


@pytest.mark.slow
class TestAgainstTheRealEncoder:
    @pytest.fixture(scope="class")
    def dense(self) -> DenseRetriever:
        r = DenseRetriever()
        try:
            r.index(LEDGER)
        except EmbeddingModelUnavailableError as exc:
            pytest.skip(f"{DEFAULT_MODEL} not available: {exc}")
        return r

    def test_scores_are_cosine_similarities_in_range(self, dense):
        """`Match.score` promises [0, 1] and the categoriser's thresholds
        (MIN_NEIGHBOUR_SIMILARITY 0.25, MIN_TOP_SIMILARITY 0.35) assume it.
        Unnormalised vectors would put scores on an arbitrary scale and shift
        every threshold without any error."""
        for m in dense.search("copier paper", k=5):
            assert 0.0 <= m.score <= 1.0

    def test_an_exact_match_scores_near_one(self, dense):
        top = dense.search("A4 Copier Paper (500 sheets) | Acme", k=1)[0]
        assert top.score > 0.9

    def test_it_retrieves_the_right_category(self, dense):
        top = dense.search("A4 photocopier paper, ream of 500", k=1)[0]
        assert top.entry.category == "5010"

    def test_k_is_respected(self, dense):
        assert len(dense.search("paper", k=2)) == 2
        assert len(dense.search("paper", k=99)) == len(LEDGER)

    def test_embeddings_do_not_depend_on_batch_composition(self, dense):
        """Mean pooling over padding would make a short description's vector
        depend on the longest text batched with it, so the same input would
        embed differently run to run depending on batching."""
        alone = dense.encode(["Wireless Mouse"])
        batched = dense.encode(
            ["Wireless Mouse", "a considerably longer description than the first one"]
        )
        assert np.allclose(alone[0], batched[0], atol=1e-5)

    def test_vectors_are_unit_length(self, dense):
        norms = np.linalg.norm(dense.encode(["paper", "mouse", "chair"]), axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_it_finds_a_synonym_tfidf_would_miss(self, dense):
        """The case that justifies dense retrieval at all. Character n-grams
        capture surface variation ("A4 PAPER 500SH"); they cannot connect two
        phrasings that share almost no characters. If dense retrieval does not
        win here it is unlikely to win anywhere on this corpus."""
        tfidf = TfidfRetriever()
        tfidf.index(LEDGER)
        query = "cordless pointing device for a computer"

        dense_top = dense.search(query, k=1)[0]
        tfidf_hits = tfidf.search(query, k=1)

        assert dense_top.entry.description == "Wireless Mouse"
        if tfidf_hits and tfidf_hits[0].entry.description == "Wireless Mouse":
            # Not a failure, but worth knowing: it would mean the corpus does
            # not actually contain the case dense retrieval is for.
            assert dense_top.score > tfidf_hits[0].score
