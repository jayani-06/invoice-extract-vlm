"""Assign an accounting category to a line item, by retrieval.

The method: embed the line item, pull the k most similar entries out of the
historical ledger, and let them vote on the category -- weighted by similarity,
so a near-exact match outweighs three loose ones.

Two properties this is built for, both of which matter more than raw accuracy:

**It abstains.** When the retrieved neighbours are weak or disagree, the result
is `UNCATEGORISED` with the evidence attached, not a confident guess. A wrong
account code propagates silently into a ledger; an abstention lands in a review
queue. Phase 5's reviewer can clear an abstention in seconds and would never
catch a plausible-looking wrong code.

**It explains.** Every decision carries the neighbours that produced it and
their scores, so the answer is "Office Supplies, because four of the five
closest past items from this vendor were Office Supplies" rather than a bare
label. That is the stated advantage of RAG over a fine-tuned classifier in
ref [8], and it is only real if the evidence survives into the output.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from invoice_extract.rag.store import LedgerEntry, Match, Retriever, VendorStats
from invoice_extract.rag.taxonomy import UNCATEGORISED, account_name

#: Below this cosine similarity a neighbour is treated as uninformative. Set
#: from the character-n-gram scale, where an unrelated item pair typically
#: lands near 0.1-0.2 and a genuine surface variant near 0.4+.
MIN_NEIGHBOUR_SIMILARITY = 0.25

#: The winning category must hold this share of the weighted vote. Below it the
#: neighbours disagree enough that abstaining is the honest answer.
MIN_VOTE_SHARE = 0.5

#: And the top neighbour must be at least this similar, regardless of vote
#: share -- five weak neighbours agreeing is still five weak neighbours.
MIN_TOP_SIMILARITY = 0.35


@dataclass
class CategoryDecision:
    category: str
    confidence: float
    votes: dict[str, float] = field(default_factory=dict)
    neighbours: list[Match] = field(default_factory=list)
    reason: str = ""

    @property
    def abstained(self) -> bool:
        return self.category == UNCATEGORISED

    def explain(self) -> str:
        """One line a human reviewer can act on."""
        if self.abstained:
            return f"No confident category: {self.reason}"
        top = ", ".join(f"{m.entry.description!r} ({m.score:.2f})" for m in self.neighbours[:3])
        return (
            f"{account_name(self.category)} [{self.category}] at {self.confidence:.0%} "
            f"— nearest past items: {top}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "category_name": account_name(self.category),
            "confidence": round(self.confidence, 4),
            "abstained": self.abstained,
            "reason": self.reason,
            "votes": {k: round(v, 4) for k, v in self.votes.items()},
            "neighbours": [
                {
                    "description": m.entry.description,
                    "category": m.entry.category,
                    "vendor": m.entry.vendor,
                    "score": round(m.score, 4),
                }
                for m in self.neighbours
            ],
        }


def _query_for(item: dict[str, Any], vendor: str | None) -> str:
    """Build the retrieval query the same way the ledger was indexed."""
    return LedgerEntry(
        description=item.get("description") or "",
        category="",
        vendor=vendor,
        hsn_sac_code=item.get("hsn_sac_code"),
    ).text()


def categorize_line_item(
    item: dict[str, Any],
    retriever: Retriever,
    *,
    vendor: str | None = None,
    k: int = 5,
    vendor_stats: VendorStats | None = None,
) -> CategoryDecision:
    """Categorise one line item against the historical ledger."""
    description = (item.get("description") or "").strip()
    if not description:
        return CategoryDecision(UNCATEGORISED, 0.0, reason="line item has no description")

    matches = [
        m
        for m in retriever.search(_query_for(item, vendor), k=k)
        if m.score >= MIN_NEIGHBOUR_SIMILARITY
    ]
    if not matches:
        return CategoryDecision(
            UNCATEGORISED, 0.0, reason="no similar item in the ledger", neighbours=[]
        )

    votes: dict[str, float] = defaultdict(float)
    for match in matches:
        weight = match.score
        # A vendor who has bought this category before is weak corroboration,
        # so it nudges rather than decides.
        if vendor_stats and vendor:
            seen = vendor_stats.usual_categories(vendor).get(match.entry.category, 0)
            if seen:
                weight *= 1.15
        votes[match.entry.category] += weight

    total = sum(votes.values()) or 1.0
    best, best_weight = max(votes.items(), key=lambda kv: kv[1])
    share = best_weight / total
    top_similarity = matches[0].score

    if share < MIN_VOTE_SHARE:
        return CategoryDecision(
            UNCATEGORISED,
            share,
            votes=dict(votes),
            neighbours=matches,
            reason=f"neighbours disagree (top category only {share:.0%} of the vote)",
        )
    if top_similarity < MIN_TOP_SIMILARITY:
        return CategoryDecision(
            UNCATEGORISED,
            share,
            votes=dict(votes),
            neighbours=matches,
            reason=f"closest match too weak ({top_similarity:.2f})",
        )

    # Confidence blends agreement with closeness: unanimous-but-distant and
    # near-but-split should both read as less certain than near-and-unanimous.
    confidence = share * min(1.0, top_similarity / 0.8)
    return CategoryDecision(
        category=best,
        confidence=confidence,
        votes=dict(votes),
        neighbours=matches,
        reason=f"{share:.0%} of weighted vote, closest match {top_similarity:.2f}",
    )


def categorize_document(
    doc: dict[str, Any],
    retriever: Retriever,
    *,
    k: int = 5,
    vendor_stats: VendorStats | None = None,
) -> list[CategoryDecision]:
    """Categorise every line item on a document."""
    vendor = ((doc.get("parties") or {}).get("vendor") or {}).get("name")
    return [
        categorize_line_item(item, retriever, vendor=vendor, k=k, vendor_stats=vendor_stats)
        for item in doc.get("line_items") or []
    ]
