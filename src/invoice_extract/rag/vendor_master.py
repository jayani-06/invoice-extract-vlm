"""The vendor master: what we believe to be true about each supplier.

Phase 4 validates an incoming invoice against this record. Everything here is
derived from *past accepted invoices*, which is how a real accounts system
builds a supplier file — nobody types in a vendor master by hand, it accretes.

The important consequence is that the master is only as trustworthy as the
history behind it. A vendor with one prior invoice has a "known" bank account
in the same sense that one coin flip establishes a bias, so every record
carries `invoice_count` and the validators refuse to raise a confident
mismatch below a minimum of history. Getting that wrong produces exactly the
failure that kills these systems in practice: an alert on every new supplier,
which trains the reviewer to click through alerts.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContractRate:
    """What this vendor has historically charged for one item."""

    description: str
    median_unit_price: float
    min_unit_price: float
    max_unit_price: float
    observations: int

    def deviation(self, unit_price: float) -> float:
        """Relative deviation from the historical median."""
        if self.median_unit_price <= 0:
            return 0.0
        return (unit_price - self.median_unit_price) / self.median_unit_price


@dataclass
class VendorRecord:
    name: str
    gstins: set[str] = field(default_factory=set)
    addresses: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    bank_accounts: set[str] = field(default_factory=set)
    ifsc_codes: set[str] = field(default_factory=set)
    invoice_count: int = 0
    invoice_totals: list[float] = field(default_factory=list)
    rates: dict[str, ContractRate] = field(default_factory=dict)
    doc_ids: list[str] = field(default_factory=list)

    @property
    def primary_gstin(self) -> str | None:
        return next(iter(sorted(self.gstins)), None)

    @property
    def primary_account(self) -> str | None:
        return next(iter(sorted(self.bank_accounts)), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gstins": sorted(self.gstins),
            "addresses": sorted(self.addresses),
            "phones": sorted(self.phones),
            "emails": sorted(self.emails),
            "bank_accounts": sorted(self.bank_accounts),
            "ifsc_codes": sorted(self.ifsc_codes),
            "invoice_count": self.invoice_count,
            "invoice_totals": self.invoice_totals,
            "rates": {
                k: {
                    "description": r.description,
                    "median_unit_price": r.median_unit_price,
                    "min_unit_price": r.min_unit_price,
                    "max_unit_price": r.max_unit_price,
                    "observations": r.observations,
                }
                for k, r in self.rates.items()
            },
            "doc_ids": self.doc_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VendorRecord:
        record = cls(
            name=data["name"],
            gstins=set(data.get("gstins") or []),
            addresses=set(data.get("addresses") or []),
            phones=set(data.get("phones") or []),
            emails=set(data.get("emails") or []),
            bank_accounts=set(data.get("bank_accounts") or []),
            ifsc_codes=set(data.get("ifsc_codes") or []),
            invoice_count=data.get("invoice_count", 0),
            invoice_totals=list(data.get("invoice_totals") or []),
            doc_ids=list(data.get("doc_ids") or []),
        )
        for key, r in (data.get("rates") or {}).items():
            record.rates[key] = ContractRate(
                description=r["description"],
                median_unit_price=r["median_unit_price"],
                min_unit_price=r["min_unit_price"],
                max_unit_price=r["max_unit_price"],
                observations=r["observations"],
            )
        return record


def rate_key(description: str | None) -> str:
    """Normalised key for matching an item to a contract rate.

    Case and punctuation are stripped because the same item arrives spelled
    differently on every invoice -- the same surface-variation problem Phase 3
    solves with character n-grams, handled here with a cheaper exact-after-
    normalisation match since we are looking up one known vendor's own catalogue
    rather than searching a whole ledger.
    """
    return "".join(ch for ch in (description or "").lower() if ch.isalnum())


class VendorMaster:
    """The whole supplier file."""

    def __init__(
        self,
        records: dict[str, VendorRecord] | None = None,
        market_rates: dict[str, ContractRate] | None = None,
    ) -> None:
        self.records: dict[str, VendorRecord] = records or {}
        #: What *every* supplier charges for an item, pooled. Used when a
        #: specific vendor has too little history of their own: most
        #: (vendor, item) pairs appear once in a real ledger, so a per-vendor
        #: rate alone leaves almost every line unchecked.
        self.market_rates: dict[str, ContractRate] = market_rates or {}

    def __len__(self) -> int:
        return len(self.records)

    def __contains__(self, name: str) -> bool:
        return name in self.records

    def get(self, name: str) -> VendorRecord | None:
        return self.records.get(name)

    def names(self) -> list[str]:
        return list(self.records)

    def by_gstin(self, gstin: str) -> VendorRecord | None:
        """GSTIN is the one globally unique vendor identifier, so an exact hit
        here outranks any amount of name similarity."""
        for record in self.records.values():
            if gstin in record.gstins:
                return record
        return None

    # ------------------------------------------------------------- building

    @classmethod
    def build(cls, documents: list[dict[str, Any]]) -> VendorMaster:
        """Accrete a supplier file from past accepted invoices."""
        prices: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        master = cls()

        for doc in documents:
            vendor = (doc.get("parties") or {}).get("vendor") or {}
            name = (vendor.get("name") or "").strip()
            if not name:
                continue

            record = master.records.setdefault(name, VendorRecord(name=name))
            record.invoice_count += 1
            if doc.get("doc_id"):
                record.doc_ids.append(doc["doc_id"])

            for value, target in (
                (vendor.get("gstin"), record.gstins),
                (vendor.get("address"), record.addresses),
                (vendor.get("phone"), record.phones),
                (vendor.get("email"), record.emails),
            ):
                if value:
                    target.add(str(value).strip())

            payment = doc.get("payment_info") or {}
            if payment.get("account_number"):
                record.bank_accounts.add(str(payment["account_number"]).strip())
            if payment.get("ifsc_or_swift"):
                record.ifsc_codes.add(str(payment["ifsc_or_swift"]).strip())

            total = (doc.get("totals") or {}).get("grand_total")
            if total is not None:
                record.invoice_totals.append(float(total))

            for item in doc.get("line_items") or []:
                key = rate_key(item.get("description"))
                price = item.get("unit_price")
                if key and price is not None:
                    prices[name][key].append(float(price))

        pooled: dict[str, list[float]] = defaultdict(list)
        for name, by_item in prices.items():
            record = master.records[name]
            for key, observed in by_item.items():
                pooled[key].extend(observed)
                record.rates[key] = ContractRate(
                    description=key,
                    median_unit_price=statistics.median(observed),
                    min_unit_price=min(observed),
                    max_unit_price=max(observed),
                    observations=len(observed),
                )

        for key, observed in pooled.items():
            master.market_rates[key] = ContractRate(
                description=key,
                median_unit_price=statistics.median(observed),
                min_unit_price=min(observed),
                max_unit_price=max(observed),
                observations=len(observed),
            )

        return master

    # ------------------------------------------------------- serialisation

    def rate_for(self, vendor: VendorRecord, key: str) -> tuple[ContractRate | None, str]:
        """Best available price reference for (vendor, item), and its source.

        Prefers the vendor's own history; falls back to the pooled market rate.
        The caller needs the source because a cross-vendor comparison tolerates
        far more spread than a vendor's own contract price.
        """
        own = vendor.rates.get(key)
        if own and own.observations >= 2:
            return own, "vendor"
        market = self.market_rates.get(key)
        if market and market.observations >= 3:
            return market, "market"
        if own:
            return own, "vendor_thin"
        return None, "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "records": {k: v.as_dict() for k, v in self.records.items()},
            "market_rates": {
                k: {
                    "description": r.description,
                    "median_unit_price": r.median_unit_price,
                    "min_unit_price": r.min_unit_price,
                    "max_unit_price": r.max_unit_price,
                    "observations": r.observations,
                }
                for k, r in self.market_rates.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VendorMaster:
        market = {
            k: ContractRate(
                description=r["description"],
                median_unit_price=r["median_unit_price"],
                min_unit_price=r["min_unit_price"],
                max_unit_price=r["max_unit_price"],
                observations=r["observations"],
            )
            for k, r in (data.get("market_rates") or {}).items()
        }
        return cls(
            {k: VendorRecord.from_dict(v) for k, v in (data.get("records") or {}).items()},
            market,
        )
