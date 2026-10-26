"""Dense retrieval: the `Retriever` backend the Phase 3 plan actually called for.

`store.TfidfRetriever` is a character-n-gram floor that needs nothing but
scikit-learn, and it is what every Phase 3 number so far was measured with.
This is the dense half, behind the same `Retriever` Protocol, so `categorize`
and `anomaly` are untouched by the swap.

Implemented against `transformers` directly rather than `sentence-transformers`,
with explicit mean pooling and L2 normalisation. That is the whole of what the
sentence-transformers wrapper does for a MiniLM checkpoint, and writing the
three lines here avoids a dependency whose version resolution has repeatedly
fought with the rest of this project's pins.

**Why dense retrieval might not win here.** Character n-grams are unusually
well suited to this task: the variation retrieval has to survive is *surface*
("A4 Copier Paper (500 sheets)" vs "A4 PAPER 500SH"), which is exactly what
3-5 grams capture and exactly what a semantic embedding throws away. The
interesting cases for embeddings are near-synonyms with no shared characters
("courier" vs "freight forwarding"), and whether those dominate is an empirical
question this module exists to answer rather than assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from invoice_extract.rag.store import LedgerEntry, Match

#: Small, fast, CPU-friendly, and the standard baseline sentence encoder.
#: 384-dimensional, ~90 MB, ~22M parameters.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the encoder cannot be loaded.

    Separate from a generic ImportError so callers can degrade to TF-IDF and
    *say so in the report*, rather than silently producing numbers from a
    different retriever than the one they asked for.
    """


@dataclass
class DenseRetriever:
    """Sentence-embedding retrieval with exact cosine search.

    Exact, not approximate: at a few hundred to a few hundred thousand entries
    a dense matrix product is milliseconds, and an approximate index would add
    a recall number that then has to be defended. `store.TfidfRetriever`
    carries the same note for the same reason.
    """

    model_name: str = DEFAULT_MODEL
    batch_size: int = 32
    max_length: int = 128
    entries: list[LedgerEntry] = field(default_factory=list)
    _matrix: Any = None
    _model: Any = None
    _tokenizer: Any = None

    # ------------------------------------------------------------- loading

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise EmbeddingModelUnavailableError(
                f"transformers and torch are required for dense retrieval: {exc}\n"
                "  pip install -e '.[rag]'"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
        except Exception as exc:  # pragma: no cover - network / cache dependent
            raise EmbeddingModelUnavailableError(
                f"could not load {self.model_name!r}: {exc}"
            ) from exc
        self._model.eval()

    # ------------------------------------------------------------ encoding

    def encode(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised embeddings, one row per text."""
        import torch

        self._ensure_loaded()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state

            # Mean pooling over real tokens only. Including padding would make
            # a short description's vector depend on the longest text in its
            # batch, so identical inputs would embed differently depending on
            # what they were batched with.
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

            # L2 normalise so a dot product IS cosine similarity, which keeps
            # the score on the same [0, 1]-ish scale `Match` promises and the
            # categoriser's thresholds assume.
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)

    # ----------------------------------------------------------- Retriever

    def index(self, entries: list[LedgerEntry]) -> None:
        self.entries = list(entries)
        if not self.entries:
            self._matrix = None
            return
        self._matrix = self.encode([e.text() for e in self.entries])

    def search(self, query: str, k: int = 5) -> list[Match]:
        if self._matrix is None or not self.entries or not (query or "").strip():
            return []
        vector = self.encode([query])[0]
        scores = self._matrix @ vector
        # Cosine on normalised vectors is in [-1, 1]; the categoriser's
        # similarity thresholds and weighted vote both assume a non-negative
        # score, so a negative similarity is "no match", not a vote against.
        scores = np.clip(scores, 0.0, 1.0)
        top = np.argsort(-scores)[:k]
        return [Match(entry=self.entries[i], score=float(scores[i])) for i in top]


def get_retriever(kind: str = "tfidf", **kwargs):
    """Build a retriever by name, so runners can sweep both backends.

    Raises rather than falling back. A silent fallback to TF-IDF would report
    dense-retrieval numbers that were never produced by a dense retriever,
    which is the single most expensive class of mistake in this project.
    """
    if kind == "tfidf":
        from invoice_extract.rag.store import TfidfRetriever

        return TfidfRetriever(**kwargs)
    if kind == "dense":
        return DenseRetriever(**kwargs)
    raise KeyError(f"unknown retriever {kind!r}; have 'tfidf' and 'dense'")
