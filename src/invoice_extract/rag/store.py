"""Retrieval over a historical ledger of line items.

The Phase 3 plan calls for a dense vector store (Qdrant/Chroma + a sentence
embedding model). This module ships the interface plus a **TF-IDF character
n-gram backend** that needs nothing beyond scikit-learn, for the same reason
Phase 2 shipped a regex baseline before the VLM: a cheap method that runs today
establishes the floor a model has to clear. Swapping in dense embeddings is a
new `Retriever` implementation, not a rewrite -- `categorize` and `anomaly`
only ever see this interface.

Character n-grams rather than word tokens, because the thing retrieval has to
survive here is *surface* variation in how vendors write the same item:
"A4 Copier Paper (500 sheets)" against "A4 PAPER 500SH". Word tokens share
almost nothing between those two; 3-to-5 character grams share a great deal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class LedgerEntry:
    """One historical line item, already categorised by a human."""

    description: str
    category: str
    vendor: str | None = None
    unit_price: float | None = None
    quantity: float | None = None
    line_total: float | None = None
    hsn_sac_code: str | None = None
    doc_id: str | None = None

    def text(self) -> str:
        """What gets indexed.

        Vendor is folded in because the same words mean different accounts
        depending on who sent the invoice -- "cartridge" from a stationer is
        Office Supplies, from a plant-hire firm it is Facilities.
        """
        parts = [self.description or ""]
        if self.vendor:
            parts.append(self.vendor)
        if self.hsn_sac_code:
            parts.append(f"hsn{self.hsn_sac_code}")
        return " | ".join(parts)


@dataclass
class Match:
    entry: LedgerEntry
    score: float  # cosine similarity in [0, 1]


class Retriever(Protocol):
    """Anything that can find similar historical line items."""

    def index(self, entries: list[LedgerEntry]) -> None: ...

    def search(self, query: str, k: int = 5) -> list[Match]: ...


@dataclass
class TfidfRetriever:
    """Character n-gram TF-IDF with cosine similarity.

    Exact nearest-neighbour search by dense matrix product. At ledger sizes of
    a few hundred thousand rows that is still milliseconds, and it avoids an
    approximate index whose recall would be one more thing to defend.
    """

    ngram_range: tuple[int, int] = (3, 5)
    min_df: int = 1
    sublinear_tf: bool = True
    entries: list[LedgerEntry] = field(default_factory=list)
    _vectorizer: Any = None
    _matrix: Any = None

    def index(self, entries: list[LedgerEntry]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.entries = list(entries)
        if not self.entries:
            self._vectorizer = None
            self._matrix = None
            return

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
        )
        self._matrix = self._vectorizer.fit_transform(e.text() for e in self.entries)

    def search(self, query: str, k: int = 5) -> list[Match]:
        if not self.entries or self._vectorizer is None or not (query or "").strip():
            return []

        vector = self._vectorizer.transform([query])
        # Both sides are L2-normalised by TfidfVectorizer, so the dot product
        # is already the cosine similarity.
        scores = (self._matrix @ vector.T).toarray().ravel()
        if not scores.size:
            return []

        top = np.argsort(-scores)[:k]
        return [Match(self.entries[i], float(scores[i])) for i in top if scores[i] > 0]

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class VendorStats:
    """Per-vendor amount distribution, for the statistical anomaly signal.

    Kept alongside the retriever because both are views of the same ledger and
    both are rebuilt together; splitting them invites the two drifting out of
    sync, which would make an anomaly score reference a vendor history the
    retriever no longer has.
    """

    counts: dict[str, int] = field(default_factory=dict)
    totals: dict[str, list[float]] = field(default_factory=dict)
    categories: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def build(
        cls, entries: list[LedgerEntry], doc_totals: dict[str, float] | None = None
    ) -> VendorStats:
        stats = cls()
        for entry in entries:
            vendor = entry.vendor
            if not vendor:
                continue
            stats.counts[vendor] = stats.counts.get(vendor, 0) + 1
            by_cat = stats.categories.setdefault(vendor, {})
            by_cat[entry.category] = by_cat.get(entry.category, 0) + 1
        for vendor, amount in (doc_totals or {}).items():
            stats.totals.setdefault(vendor, []).append(amount)
        return stats

    def amount_zscore(self, vendor: str | None, amount: float | None) -> float | None:
        """How unusual this amount is for this vendor, in standard deviations.

        Returns None when there is too little history to say anything. A
        z-score computed from two invoices is noise wearing a number's
        clothes, and acting on it would bury real anomalies in false alarms.
        """
        if vendor is None or amount is None:
            return None
        history = self.totals.get(vendor, [])
        if len(history) < 4:
            return None
        mean = float(np.mean(history))
        std = float(np.std(history))
        if std < 1e-6:
            return 0.0 if abs(amount - mean) < 1e-6 else float("inf")
        return (amount - mean) / std

    def known_vendor(self, vendor: str | None) -> bool:
        return bool(vendor) and vendor in self.counts

    def usual_categories(self, vendor: str | None) -> dict[str, int]:
        return dict(self.categories.get(vendor or "", {}))
