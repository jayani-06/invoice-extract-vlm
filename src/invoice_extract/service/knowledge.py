"""What the service retrieves against, and how it grows.

Phases 3 and 4 each build their reference data from a fixed history inside an
evaluation script. A running service needs the same objects -- retriever,
vendor statistics, vendor master, the invoice numbers already seen -- built once
at start-up and then *extended* as reviewers accept documents. That accretion is
the design `rag/vendor_master.py` describes ("nobody types in a vendor master by
hand, it accretes"), and this module is where it actually happens.

Everything is rebuilt from the full accepted history on every accretion rather
than patched in place. The ledgers here are hundreds of rows, a rebuild is
milliseconds, and an incremental update to medians and TF-IDF weights is a
second implementation of the same maths that would have to be kept in sync.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from invoice_extract.rag.anomaly import normalize_invoice_number
from invoice_extract.rag.store import LedgerEntry, TfidfRetriever, VendorStats
from invoice_extract.rag.taxonomy import UNCATEGORISED
from invoice_extract.rag.vendor_master import VendorMaster


def _vendor_name(doc: dict[str, Any]) -> str | None:
    return ((doc.get("parties") or {}).get("vendor") or {}).get("name")


@dataclass
class Knowledge:
    """Accepted history and every index derived from it."""

    history: list[dict[str, Any]] = field(default_factory=list)
    entries: list[LedgerEntry] = field(default_factory=list)
    retriever: TfidfRetriever = field(default_factory=TfidfRetriever)
    stats: VendorStats = field(default_factory=VendorStats)
    master: VendorMaster = field(default_factory=VendorMaster)
    seen_invoice_numbers: dict[str, str] = field(default_factory=dict)

    # ---------------------------------------------------------------- build

    @classmethod
    def build(
        cls, history: list[dict[str, Any]], entries: list[LedgerEntry] | None = None
    ) -> Knowledge:
        """Derive every index from accepted documents and categorised lines.

        `entries` are the categorised ledger lines. They are passed separately
        because a category is a *human* judgement (or a confident machine one
        a human accepted) and cannot be recomputed from the document alone.
        """
        knowledge = cls(history=list(history), entries=list(entries or []))
        knowledge._rebuild()
        return knowledge

    def _rebuild(self) -> None:
        self.retriever = TfidfRetriever()
        self.retriever.index(self.entries)

        self.stats = VendorStats.build(self.entries)
        for doc in self.history:
            vendor = _vendor_name(doc)
            total = (doc.get("totals") or {}).get("grand_total")
            if vendor and total is not None:
                self.stats.totals.setdefault(vendor, []).append(float(total))

        self.master = VendorMaster.build(self.history)

        self.seen_invoice_numbers = {}
        for doc in self.history:
            number = normalize_invoice_number((doc.get("invoice_meta") or {}).get("invoice_number"))
            if number and doc.get("doc_id"):
                self.seen_invoice_numbers.setdefault(number, doc["doc_id"])

    @classmethod
    def bootstrap(cls, corpus_dir: Path | None, ledger_dir: Path | None) -> Knowledge:
        """Start-up history: the Phase 3 ledger and the documents behind it.

        Uses the same split `scripts/build_ledger.py` wrote, so the service
        starts from exactly the history the Phase 3 and 4 numbers were measured
        against. With neither present the service still runs -- every vendor is
        then unknown and every document goes to review, which is the correct
        behaviour for an empty supplier file.
        """
        entries: list[LedgerEntry] = []
        history: list[dict[str, Any]] = []
        ledger_path = (ledger_dir / "ledger.json") if ledger_dir else None
        if ledger_path and ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            entries = [LedgerEntry(**e) for e in ledger.get("entries", [])]
            doc_ids = {e.doc_id for e in entries if e.doc_id}
            if corpus_dir:
                for doc_id in sorted(doc_ids):
                    path = corpus_dir / "canonical" / f"{doc_id}.json"
                    if path.exists():
                        history.append(json.loads(path.read_text(encoding="utf-8")))
        return cls.build(history, entries)

    # --------------------------------------------------------------- accrete

    def accrete(self, doc: dict[str, Any], categories: list[str | None]) -> int:
        """Add one accepted document. Returns how many ledger lines it added.

        `categories[i]` is the final category for line `i`: the reviewer's
        correction if there was one, otherwise the machine's accepted decision.
        A line still `UNCATEGORISED` (or `None`) is **not** added to the ledger:
        an abstention nobody resolved is not a label, and indexing it would
        teach retrieval to predict "uncategorised".
        """
        vendor = _vendor_name(doc)
        added = 0
        for item, category in zip(doc.get("line_items") or [], categories, strict=False):
            if not category or category == UNCATEGORISED:
                continue
            self.entries.append(
                LedgerEntry(
                    description=item.get("description") or "",
                    category=category,
                    vendor=vendor,
                    unit_price=item.get("unit_price"),
                    quantity=item.get("quantity"),
                    line_total=item.get("line_total"),
                    hsn_sac_code=item.get("hsn_sac_code"),
                    doc_id=doc.get("doc_id"),
                )
            )
            added += 1
        self.history.append(doc)
        self._rebuild()
        return added
