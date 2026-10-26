"""Routing, the review queue's rules, and feedback into the knowledge base.

**Routing.** Every processed document gets one overall verdict:

- `FAIL`   Phase 4 returned FAIL. Status `blocked`: not payable until a named
           reviewer records an *override* with a reason.
- `REVIEW` Phase 4 returned REVIEW, *or* a line's category abstained, *or*
           Phase 3 raised a high/medium anomaly flag. Status `in_review`.
- `PASS`   none of the above. Status `auto_accepted`.

Anomaly flags never escalate a document to FAIL on their own: Phase 3 rates
how surprising a document is, Phase 4 rates whether it is payable, and mixing
the two scales here would invent a policy neither phase measured.

**The decision rules** (enforced here, pinned by `tests/test_service.py`):

1. A FAIL can never be accepted by `accept`. The only way to make it payable is
   `override`, which requires a non-empty reason, and the reviewer's name is
   stored with it. There is no automatic path from FAIL to payable.
2. A document cannot be accepted while any abstained line is still
   uncategorised: an unresolved abstention is an unresolved question.
3. Corrections are recorded, not applied silently. The machine's extraction is
   stored untouched; the accepted copy is a separate document, and each
   difference between the two is a decision with old value, new value and
   reviewer.
4. Decided documents are final. A second decision on the same document is a
   conflict, not an edit -- the log is append-only.

**Feedback.** An accepted document (auto or human) is added to the history the
vendor master is built from, and its categorised lines to the ledger the
categoriser retrieves against -- with the reviewer's corrections, which is the
point: a corrected vendor name becomes a supplier record the next invoice from
that supplier resolves against.
"""

from __future__ import annotations

import copy
import io
import re
import threading
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from invoice_extract.ocr.base import OcrEngine
from invoice_extract.rag.taxonomy import UNCATEGORISED, is_real_category
from invoice_extract.rag.validate_vendor import FAIL, PASS, REVIEW
from invoice_extract.service.knowledge import Knowledge
from invoice_extract.service.pipeline import DEFAULT_PREPROCESS, ChainResult, run_chain_on_image
from invoice_extract.service.schemas import PAYABLE_STATUSES, ProcessResult, QueueItem
from invoice_extract.service.store import ReviewStore

ACTIONABLE_SEVERITIES = ("high", "medium")
OPEN_STATUSES = ("in_review", "blocked")
_LINE_FIELD = re.compile(r"^line_items\[(\d+)\]\.category$")


class ReviewError(Exception):
    """A request the review rules refuse. `status` is the HTTP code to use."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# ------------------------------------------------------------------ routing


def route(chain: ChainResult) -> tuple[str, str, list[str], list[tuple[str, int | None]]]:
    """Overall verdict, status, human-readable reasons, and the queue items."""
    validation = chain.validation
    reasons: list[str] = []
    if validation.verdict == FAIL:
        reasons += [f"FAIL: {c.message}" for c in validation.failures]
    elif validation.verdict == REVIEW:
        reasons += [f"REVIEW: {c.message}" for c in validation.reviews]

    abstained = [i for i, d in enumerate(chain.categories) if d.abstained]
    reasons += [
        f"line {i + 1}: category abstained ({chain.categories[i].reason})" for i in abstained
    ]

    flags = [f for f in chain.anomalies.flags if f.severity in ACTIONABLE_SEVERITIES]
    reasons += [f"anomaly ({f.severity}): {f.message}" for f in flags]

    if validation.verdict == FAIL:
        overall, status = FAIL, "blocked"
    elif validation.verdict == REVIEW or abstained or flags:
        overall, status = REVIEW, "in_review"
    else:
        overall, status = PASS, "auto_accepted"

    items: list[tuple[str, int | None]] = []
    if overall != PASS:
        items.append(("document", None))
        items += [("line_category", i) for i in abstained]
    return overall, status, reasons, items


def _machine_categories(result: dict[str, Any]) -> list[str]:
    return [row["category"] for row in result["categories"]]


def decode_image(data: bytes) -> tuple[np.ndarray, Image.Image]:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # PIL raises several unrelated types
        raise ReviewError(415, f"not a readable image: {exc}") from exc
    return np.array(image.convert("L")), image


# ------------------------------------------------------------------ service


class ReviewService:
    def __init__(
        self,
        store: ReviewStore,
        knowledge: Knowledge,
        engine_factory: Callable[[], OcrEngine],
        *,
        image_dir: Path,
        preprocess: str = DEFAULT_PREPROCESS,
        extractor: str = "rules",
        today: Callable[[], date] = date.today,
    ) -> None:
        self.store = store
        self.knowledge = knowledge
        self._engine_factory = engine_factory
        self._engine: OcrEngine | None = None
        # One OCR call at a time: PaddleOCR is not thread-safe, and on CPU two
        # concurrent pages are slower than two sequential ones anyway.
        self._ocr_lock = threading.Lock()
        self._knowledge_lock = threading.RLock()
        self.image_dir = Path(image_dir)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess = preprocess
        self.extractor = extractor
        self.today = today

        # Re-apply everything accepted in earlier runs, so a restart does not
        # forget what reviewers taught it.
        for record in store.accepted_documents():
            self.knowledge.accrete(record["document"], record["categories"])

    # ------------------------------------------------------------ process

    def engine(self) -> OcrEngine:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    def process_image(self, data: bytes, filename: str | None = None) -> dict[str, Any]:
        array, image = decode_image(data)
        doc_id = f"svc_{uuid.uuid4().hex[:12]}"
        suffix = Path(filename or "").suffix.lower() or ".png"
        image_path = self.image_dir / f"{doc_id}{suffix}"
        image_path.write_bytes(data)

        pages = [
            {
                "page_index": 0,
                "image_path": image_path.name,
                "width": image.width,
                "height": image.height,
            }
        ]
        source = {"dataset": "other", "original_id": filename or doc_id, "split": "test"}

        with self._ocr_lock, self._knowledge_lock:
            chain = run_chain_on_image(
                array,
                self.engine(),
                self.knowledge,
                doc_id=doc_id,
                source=source,
                pages=pages,
                preprocess=self.preprocess,
                extractor=self.extractor,
                today=self.today(),
            )
        return self._register(chain, filename, str(image_path))

    def _register(self, chain: ChainResult, filename: str | None, image_path: str) -> dict:
        overall, status, reasons, items = route(chain)
        result = chain.as_dict()
        result.update(
            overall_verdict=overall,
            status=status,
            payable=status in PAYABLE_STATUSES,
            review_reasons=reasons,
        )
        # Validate the outgoing contract here, not only at the HTTP layer, so a
        # drift between a stage's as_dict() and the response model fails loudly
        # at the first document rather than when someone opens the API.
        ProcessResult.model_validate(result)

        final = None
        if status == "auto_accepted":
            final = {"document": chain.document, "categories": _machine_categories(result)}
        self.store.add_document(
            doc_id=chain.doc_id,
            filename=filename,
            image_path=image_path,
            overall_verdict=overall,
            status=status,
            result=result,
            open_items=items,
            final_document=final,
        )
        if final is not None:
            with self._knowledge_lock:
                self.knowledge.accrete(final["document"], final["categories"])
        return result

    # ------------------------------------------------------------- reading

    def _require(self, doc_id: str) -> dict[str, Any]:
        record = self.store.get_document(doc_id)
        if record is None:
            raise ReviewError(404, f"no document {doc_id}")
        return record

    def document_record(self, doc_id: str) -> dict[str, Any]:
        record = self._require(doc_id)
        result = dict(record["result"])
        result["status"] = record["status"]
        result["payable"] = record["status"] in PAYABLE_STATUSES
        return {
            "doc_id": doc_id,
            "filename": record["filename"],
            "created_at": record["created_at"],
            "status": record["status"],
            "payable": record["status"] in PAYABLE_STATUSES,
            "overall_verdict": record["overall_verdict"],
            "result": result,
            "final_document": (record["final_document"] or {}).get("document"),
            "decisions": self.store.decisions(doc_id),
        }

    def image_path(self, doc_id: str) -> Path:
        record = self._require(doc_id)
        return Path(record["image_path"])

    def queue(self, state: str | None = "open") -> list[dict[str, Any]]:
        out = []
        cache: dict[str, dict] = {}
        for item in self.store.items(state=state):
            record = cache.setdefault(item["doc_id"], self._require(item["doc_id"]))
            out.append(self._queue_item(item, record))
        return out

    def _queue_item(self, item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        result = record["result"]
        doc = result["document"]
        base = {
            "item_id": item["item_id"],
            "doc_id": item["doc_id"],
            "kind": item["kind"],
            "line_index": item["line_index"],
            "state": item["state"],
            "overall_verdict": record["overall_verdict"],
            "image_url": f"/documents/{item['doc_id']}/image",
        }
        if item["kind"] == "line_category":
            i = item["line_index"]
            category = result["categories"][i]
            base.update(
                extracted=doc["line_items"][i],
                explanation=category["explanation"],
                neighbours=category["neighbours"],
            )
        else:
            flags = result["anomalies"]["flags"]
            base.update(
                extracted={
                    "vendor": (doc.get("parties") or {}).get("vendor"),
                    "invoice_meta": doc.get("invoice_meta"),
                    "totals": doc.get("totals"),
                    "payment_info": doc.get("payment_info"),
                },
                explanation=" | ".join(
                    [result["validation"]["explanation"]]
                    + [f"{f['severity']}: {f['message']}" for f in flags]
                ),
                checks=result["validation"]["checks"],
                flags=flags,
            )
        return QueueItem.model_validate(base).model_dump()

    # ----------------------------------------------------------- decisions

    def decide_line(
        self,
        item_id: int,
        *,
        reviewer: str,
        action: str,
        category: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        reviewer = (reviewer or "").strip()
        if not reviewer:
            raise ReviewError(422, "a decision needs a reviewer name")
        item = self.store.get_item(item_id)
        if item is None or item["kind"] != "line_category":
            raise ReviewError(404, f"no line-category item {item_id}")
        if item["state"] != "open":
            raise ReviewError(409, f"item {item_id} is already resolved")
        record = self._require(item["doc_id"])
        if record["status"] not in OPEN_STATUSES:
            raise ReviewError(409, f"document {item['doc_id']} is already {record['status']}")

        i = item["line_index"]
        machine = record["result"]["categories"][i]
        if action == "accept":
            if machine["abstained"]:
                raise ReviewError(
                    409,
                    "this line abstained, so there is no category to accept; "
                    "use action 'correct' with a category",
                )
            new = machine["category"]
        elif action == "correct":
            if not category or not is_real_category(category):
                raise ReviewError(422, f"{category!r} is not a category in the chart of accounts")
            new = category
        else:
            raise ReviewError(422, f"unknown action {action!r}")

        with self.store.transaction() as conn:
            self.store.record(
                conn,
                doc_id=item["doc_id"],
                item_id=item_id,
                reviewer=reviewer,
                action=action,
                field=f"line_items[{i}].category",
                old_value=machine["category"],
                new_value=new,
                reason=note,
            )
            self.store.resolve_item(conn, item_id)
        return self.document_record(item["doc_id"])

    def _reviewed_categories(self, doc_id: str, n_lines: int) -> dict[int, str]:
        out: dict[int, str] = {}
        for decision in self.store.decisions(doc_id):
            match = _LINE_FIELD.match(decision["field"] or "")
            if match and int(match.group(1)) < n_lines:
                out[int(match.group(1))] = decision["new_value"]
        return out

    def decide_document(
        self,
        doc_id: str,
        *,
        reviewer: str,
        action: str,
        reason: str | None = None,
        vendor_name: str | None = None,
        categories: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        reviewer = (reviewer or "").strip()
        if not reviewer:
            raise ReviewError(422, "a decision needs a reviewer name")
        record = self._require(doc_id)
        status = record["status"]
        if status not in OPEN_STATUSES:
            raise ReviewError(409, f"document {doc_id} is already {status}; decisions are final")

        if action == "accept" and status == "blocked":
            raise ReviewError(
                409,
                "this document reached FAIL and cannot be accepted; a reviewer may "
                "'override' it with a reason, which is recorded against their name",
            )
        if action == "override":
            if status != "blocked":
                raise ReviewError(409, "only a FAIL document needs an override; use 'accept'")
            if not (reason or "").strip():
                raise ReviewError(422, "an override must state a reason")
        if action not in ("accept", "override", "reject"):
            raise ReviewError(422, f"unknown action {action!r}")

        result = record["result"]
        original = result["document"]
        n_lines = len(original.get("line_items") or [])
        categories = {int(k): v for k, v in (categories or {}).items()}
        for i, code in categories.items():
            if not 0 <= i < n_lines:
                raise ReviewError(422, f"no line {i} on this document")
            if not is_real_category(code):
                raise ReviewError(422, f"{code!r} is not a category in the chart of accounts")

        if action == "reject":
            with self.store.transaction() as conn:
                self.store.record(
                    conn, doc_id=doc_id, reviewer=reviewer, action="reject", reason=reason
                )
                for item in self.store.items(state="open", doc_id=doc_id):
                    self.store.resolve_item(conn, item["item_id"])
                self.store.set_status(conn, doc_id, "rejected", None)
            return self.document_record(doc_id)

        # --- accept / override: build the human-approved copy -------------
        final_categories = [
            UNCATEGORISED if row["abstained"] else row["category"] for row in result["categories"]
        ]
        for i, code in self._reviewed_categories(doc_id, n_lines).items():
            final_categories[i] = code
        for i, code in categories.items():
            final_categories[i] = code

        open_lines = [
            item
            for item in self.store.items(state="open", doc_id=doc_id)
            if item["kind"] == "line_category"
            and final_categories[item["line_index"]] == UNCATEGORISED
        ]
        if open_lines:
            lines = ", ".join(str(item["line_index"]) for item in open_lines)
            raise ReviewError(
                409, f"categorise the abstained line(s) {lines} before accepting the document"
            )

        final_doc = copy.deepcopy(original)
        old_vendor = ((original.get("parties") or {}).get("vendor") or {}).get("name")
        corrected_vendor = (vendor_name or "").strip() or None
        if corrected_vendor and corrected_vendor != old_vendor:
            final_doc["parties"]["vendor"]["name"] = corrected_vendor

        new_status = "accepted_with_override" if action == "override" else "accepted"
        with self.store.transaction() as conn:
            if corrected_vendor and corrected_vendor != old_vendor:
                self.store.record(
                    conn,
                    doc_id=doc_id,
                    reviewer=reviewer,
                    action="correct",
                    field="parties.vendor.name",
                    old_value=old_vendor,
                    new_value=corrected_vendor,
                )
            machine = _machine_categories(result)
            items_by_line = {
                item["line_index"]: item["item_id"]
                for item in self.store.items(state="open", doc_id=doc_id)
                if item["kind"] == "line_category"
            }
            for i, code in categories.items():
                self.store.record(
                    conn,
                    doc_id=doc_id,
                    item_id=items_by_line.get(i),
                    reviewer=reviewer,
                    action="correct",
                    field=f"line_items[{i}].category",
                    old_value=machine[i],
                    new_value=code,
                )
            self.store.record(
                conn,
                doc_id=doc_id,
                reviewer=reviewer,
                action=action,
                field="status",
                old_value=status,
                new_value=new_status,
                reason=reason,
            )
            for item in self.store.items(state="open", doc_id=doc_id):
                self.store.resolve_item(conn, item["item_id"])
            self.store.set_status(
                conn, doc_id, new_status, {"document": final_doc, "categories": final_categories}
            )

        with self._knowledge_lock:
            self.knowledge.accrete(final_doc, final_categories)
        return self.document_record(doc_id)
