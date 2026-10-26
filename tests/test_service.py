"""Phase 5: the API contract, the review-queue rules, and the feedback loop.

Self-contained: documents come from the synthetic generator and are rendered
in-process, so these tests need neither the image corpus nor an OCR engine.
OCR is replaced by `RenderedText`, which returns the renderer's own word boxes
for the image just uploaded. That makes the OCR step a known quantity, which
is the point -- these tests are about what the service *does with* a result,
and real OCR is covered by `test_pipeline_end_to_end.py::TestChainWithRealOcr`.

The three Phase 5 exit criteria that are about behaviour each have a test
class here:

- every response field validates against the canonical schema -> TestContract
- a FAIL is not payable without an attributable override   -> TestFailIsNeverAutoAccepted
- reviewer corrections change the vendor master            -> TestCorrectionsReachTheVendorMaster
"""

from __future__ import annotations

import copy
import io
import json
import random
import sys
import time
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="install the service extra: pip install -e '.[service]'")
pytest.importorskip("httpx", reason="install the service extra: pip install -e '.[service]'")

from fastapi.testclient import TestClient  # noqa: E402

from invoice_extract.data.validate import validate_jsonschema  # noqa: E402
from invoice_extract.ocr.base import OcrResult, OcrWord  # noqa: E402
from invoice_extract.rag.store import LedgerEntry  # noqa: E402
from invoice_extract.rag.taxonomy import UNCATEGORISED, category_of  # noqa: E402
from invoice_extract.render.invoice_renderer import InvoiceRenderer  # noqa: E402
from invoice_extract.service.api import create_app  # noqa: E402
from invoice_extract.service.knowledge import Knowledge  # noqa: E402
from invoice_extract.service.review import ReviewService, route  # noqa: E402
from invoice_extract.service.schemas import DocumentRecord, ProcessResult, QueueItem  # noqa: E402
from invoice_extract.service.store import ReviewStore  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import generate_synthetic_gst_invoices as generator  # noqa: E402

#: Past every generated invoice date, so no document trips the future-date rule.
TODAY = date(2030, 1, 1)
HISTORY_SIZE = 60


def vendor_of(doc: dict) -> str:
    return doc["parties"]["vendor"]["name"]


# ------------------------------------------------------------------ world


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    """A fresh synthetic corpus, generated the same way as the real one."""
    rng = random.Random(5050)
    generator.reset_party_registry()
    generator.reset_price_registry()
    generator.reset_invoice_sequence()
    return [generator.gen_document(rng, f"svc_test_{i:03d}", "test") for i in range(110)]


@pytest.fixture(scope="module")
def history(corpus) -> list[dict]:
    return corpus[:HISTORY_SIZE]


@pytest.fixture(scope="module")
def fresh(corpus) -> list[dict]:
    """Documents the service has never seen."""
    return corpus[HISTORY_SIZE:]


def build_knowledge(history: list[dict]) -> Knowledge:
    entries = [
        LedgerEntry(
            description=item["description"],
            category=category_of(item["description"]),
            vendor=vendor_of(doc),
            unit_price=item.get("unit_price"),
            line_total=item.get("line_total"),
            hsn_sac_code=item.get("hsn_sac_code"),
            doc_id=doc["doc_id"],
        )
        for doc in history
        for item in doc["line_items"]
    ]
    return Knowledge.build(copy.deepcopy(history), entries)


class RenderedText:
    """Stands in for OCR: returns the rendered page's own words.

    The service decodes the upload and preprocesses it before calling the
    engine, so the engine cannot see which document it was given. Tests queue
    the words for the next upload instead.
    """

    name = "rendered-text-fixture"

    def __init__(self, delay: float = 0.0) -> None:
        self.pending: list[list[OcrWord]] = []
        self.delay = delay
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return OcrResult(self.pending.pop(0), self.name, meta={"fixture": True})


def render(doc: dict, *, rename: dict[str, str] | None = None) -> tuple[bytes, list[OcrWord]]:
    """PNG bytes plus the words an OCR engine would ideally read from them.

    `rename` swaps tokens in the *read* text only, which is how an OCR error on
    the vendor name reaches the extractor while the page itself is genuine.
    """
    # The corpus dpi. At 100 dpi wide prices are ellipsised ("1,824.…") and the
    # extractor cannot read them, which would test the renderer, not the service.
    result = InvoiceRenderer(dpi=200).render(doc, seed=1)
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    words = [
        OcrWord((rename or {}).get(w.text, w.text), tuple(w.bbox), 1.0, w.line_id)
        for w in result.words
    ]
    return buf.getvalue(), words


@pytest.fixture
def world(history, tmp_path):
    engine = RenderedText()
    service = ReviewService(
        ReviewStore(tmp_path / "review.sqlite3"),
        build_knowledge(history),
        lambda: engine,
        image_dir=tmp_path / "images",
        today=lambda: TODAY,
    )
    client = TestClient(create_app(service))

    def post(doc: dict, endpoint: str = "/documents", **render_kwargs):
        data, words = render(doc, **render_kwargs)
        engine.pending.append(words)
        return client.post(endpoint, files={"file": (f"{doc['doc_id']}.png", data, "image/png")})

    return {"client": client, "service": service, "engine": engine, "post": post, "tmp": tmp_path}


def pick(docs: list[dict], predicate) -> dict:
    for doc in docs:
        if predicate(doc):
            return copy.deepcopy(doc)
    pytest.skip("generated corpus has no document matching this scenario")


def known_vendor(history: list[dict], minimum: int = 1):
    counts: dict[str, int] = {}
    for doc in history:
        counts[vendor_of(doc)] = counts.get(vendor_of(doc), 0) + 1
    return lambda doc: counts.get(vendor_of(doc), 0) >= minimum


def unknown_vendor(history: list[dict]):
    names = {vendor_of(d) for d in history}
    return lambda doc: vendor_of(doc) not in names


# --------------------------------------------------------------- contract


class TestContract:
    def test_a_known_vendor_invoice_returns_a_complete_result(self, world, history, fresh):
        doc = pick(fresh, known_vendor(history))
        resp = world["post"](doc)
        assert resp.status_code == 200, resp.text
        body = resp.json()

        ProcessResult.model_validate(body)
        for key in (
            "document",
            "categories",
            "anomalies",
            "validation",
            "consistency_problems",
            "overall_verdict",
            "status",
            "payable",
            "timings_ms",
        ):
            assert key in body
        assert len(body["categories"]) == len(body["document"]["line_items"])
        assert body["validation"]["resolution"]["resolved"]
        assert (
            body["document"]["invoice_meta"]["invoice_number"]
            == doc["invoice_meta"]["invoice_number"]
        )

    def test_every_returned_document_validates_against_the_canonical_schema(
        self, world, history, fresh
    ):
        """Exit criterion 2. Checked with the JSON Schema itself, not only the
        pydantic mirror, so a drift between the two would show here."""
        for doc in fresh[:12]:
            body = world["post"](doc).json()
            assert validate_jsonschema(body["document"]) == [], doc["doc_id"]
            record = world["client"].get(f"/documents/{body['doc_id']}").json()
            DocumentRecord.model_validate(record)
            if record["final_document"] is not None:
                assert validate_jsonschema(record["final_document"]) == []

    def test_the_service_does_not_repair_the_document(self, world, history, fresh):
        """The extracted document comes back as extracted; arithmetic problems
        are reported beside it. Same contract test_demo.py pins for the demo."""
        doc = pick(fresh, known_vendor(history))
        doc["totals"]["grand_total"] = round(doc["totals"]["grand_total"] + 5000.0, 2)
        body = world["post"](doc).json()
        assert body["document"]["totals"]["grand_total"] == pytest.approx(
            doc["totals"]["grand_total"]
        )
        assert body["consistency_problems"]
        assert "arithmetic_mismatch" in {f["code"] for f in body["anomalies"]["flags"]}
        assert body["status"] != "auto_accepted"

    def test_the_uploaded_page_is_served_back(self, world, fresh):
        body = world["post"](fresh[0]).json()
        image = world["client"].get(f"/documents/{body['doc_id']}/image")
        assert image.status_code == 200
        assert image.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_a_non_image_upload_is_refused(self, world):
        resp = world["client"].post(
            "/documents", files={"file": ("x.png", b"not an image", "image/png")}
        )
        assert resp.status_code == 415

    def test_a_schema_invalid_extraction_is_reported_not_returned(self, world, fresh, monkeypatch):
        import invoice_extract.service.pipeline as pipeline

        monkeypatch.setattr(pipeline, "validate_jsonschema", lambda doc: ["totals: broken"])
        resp = world["post"](fresh[0])
        assert resp.status_code == 422
        assert resp.json()["problems"] == ["totals: broken"]
        assert world["client"].get("/review/queue").json() == []

    def test_the_job_endpoint_returns_a_handle_immediately(self, world, fresh):
        """Exit criterion 1, second branch: with a slow engine the handle comes
        back at once and the result arrives by polling."""
        world["engine"].delay = 1.5
        t = time.perf_counter()
        resp = world["post"](fresh[1], endpoint="/jobs")
        elapsed = time.perf_counter() - t
        assert resp.status_code == 202
        assert elapsed < 1.0, f"handle took {elapsed:.2f}s"
        handle = resp.json()

        for _ in range(100):
            status = world["client"].get(handle["status_url"]).json()
            if status["status"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert status["status"] == "done", status
        ProcessResult.model_validate(status["result"])

    def test_health_reports_the_loaded_knowledge(self, world, history):
        health = world["client"].get("/health").json()
        assert health["history_documents"] == len(history)
        assert health["vendors"] == len({vendor_of(d) for d in history})


# ------------------------------------------------------------------ queue


class TestQueueTransitions:
    def test_a_clean_known_vendor_invoice_is_auto_accepted_and_accreted(
        self, world, history, fresh
    ):
        service = world["service"]
        before = len(service.knowledge.history)
        doc = pick(fresh, known_vendor(history, minimum=2))
        body = world["post"](doc).json()
        if body["status"] != "auto_accepted":
            pytest.skip(f"this invoice routed to review: {body['review_reasons']}")
        assert body["payable"] is True
        assert body["overall_verdict"] == "PASS"
        assert world["client"].get("/review/queue").json() == []
        assert len(service.knowledge.history) == before + 1

    def test_an_unknown_vendor_goes_to_review_and_is_accepted_by_a_person(
        self, world, history, fresh
    ):
        client = world["client"]
        doc = pick(fresh, unknown_vendor(history))
        body = world["post"](doc).json()
        assert body["overall_verdict"] == "REVIEW"
        assert body["status"] == "in_review"
        assert body["payable"] is False

        queue = client.get("/review/queue").json()
        items = [QueueItem.model_validate(i) for i in queue if i["doc_id"] == body["doc_id"]]
        doc_item = next(i for i in items if i.kind == "document")
        assert "No supplier record matches" in doc_item.explanation
        assert doc_item.image_url == f"/documents/{body['doc_id']}/image"
        assert doc_item.extracted["vendor"]["name"] == vendor_of(doc)

        resp = client.post(
            f"/review/documents/{body['doc_id']}", json={"reviewer": "asha", "action": "accept"}
        )
        assert resp.status_code == 200, resp.text
        record = resp.json()
        assert record["status"] == "accepted"
        assert record["payable"] is True
        assert record["decisions"][-1]["reviewer"] == "asha"
        assert not [i for i in client.get("/review/queue").json() if i["doc_id"] == body["doc_id"]]

    def test_decisions_are_final(self, world, history, fresh):
        doc = pick(fresh, unknown_vendor(history))
        doc_id = world["post"](doc).json()["doc_id"]
        url = f"/review/documents/{doc_id}"
        assert (
            world["client"].post(url, json={"reviewer": "a", "action": "accept"}).status_code == 200
        )
        second = world["client"].post(url, json={"reviewer": "b", "action": "reject"})
        assert second.status_code == 409
        assert world["client"].get(f"/documents/{doc_id}").json()["status"] == "accepted"

    def test_a_decision_needs_a_named_reviewer(self, world, history, fresh):
        doc = pick(fresh, unknown_vendor(history))
        doc_id = world["post"](doc).json()["doc_id"]
        for reviewer in ("", "   "):
            resp = world["client"].post(
                f"/review/documents/{doc_id}", json={"reviewer": reviewer, "action": "accept"}
            )
            assert resp.status_code == 422
        assert world["client"].get(f"/documents/{doc_id}").json()["payable"] is False

    def test_a_rejected_document_is_not_payable_and_not_learned_from(self, world, history, fresh):
        service = world["service"]
        doc = pick(fresh, unknown_vendor(history))
        doc_id = world["post"](doc).json()["doc_id"]
        before = len(service.knowledge.history)
        record = (
            world["client"]
            .post(
                f"/review/documents/{doc_id}",
                json={"reviewer": "asha", "action": "reject", "reason": "not our supplier"},
            )
            .json()
        )
        assert record["status"] == "rejected"
        assert record["payable"] is False
        assert len(service.knowledge.history) == before
        assert service.knowledge.master.get(vendor_of(doc)) is None

    def test_an_abstained_line_blocks_acceptance_until_it_is_categorised(
        self, world, history, fresh
    ):
        client, service = world["client"], world["service"]
        doc = pick(fresh, known_vendor(history))
        novel = "Floor Polisher"
        doc["line_items"][0]["description"] = novel
        body = world["post"](doc).json()
        decisions = body["categories"]
        if not decisions[0]["abstained"]:
            pytest.skip(f"retrieval did not abstain on {novel!r}")
        assert body["status"] in ("in_review", "blocked")

        line_items = [
            i
            for i in client.get("/review/queue").json()
            if i["doc_id"] == body["doc_id"] and i["kind"] == "line_category"
        ]
        assert [i["line_index"] for i in line_items] == [0]
        assert line_items[0]["extracted"]["description"] == novel, "renderer/extractor altered it"

        if body["status"] == "blocked":
            pytest.skip("document also failed vendor validation")

        early = client.post(
            f"/review/documents/{body['doc_id']}", json={"reviewer": "ravi", "action": "accept"}
        )
        assert early.status_code == 409
        assert "abstained" in early.json()["detail"]

        # Accepting an abstention is meaningless: there is no category to accept.
        resp = client.post(
            f"/review/lines/{line_items[0]['item_id']}",
            json={"reviewer": "ravi", "action": "accept"},
        )
        assert resp.status_code == 409

        resp = client.post(
            f"/review/lines/{line_items[0]['item_id']}",
            json={"reviewer": "ravi", "action": "correct", "category": "5050"},
        )
        assert resp.status_code == 200, resp.text
        record = client.post(
            f"/review/documents/{body['doc_id']}", json={"reviewer": "ravi", "action": "accept"}
        ).json()
        assert record["status"] == "accepted"

        # The correction is now ledger knowledge, retrieved as the top neighbour.
        assert any(
            e.description == novel and e.category == "5050" for e in service.knowledge.entries
        )

        # Later, *different* invoices carrying the same item. (Re-posting the
        # same invoice would correctly trip duplicate-payment detection.)
        others = iter(
            copy.deepcopy(d)
            for d in fresh
            if known_vendor(history)(d) and d["doc_id"] != doc["doc_id"]
        )

        def resubmit() -> dict:
            again = next(others)
            again["line_items"][0]["description"] = novel
            return world["post"](again).json()

        second = resubmit()
        top = second["categories"][0]["neighbours"][0]
        assert (top["description"], top["category"]) == (novel, "5050")
        assert top["score"] >= 0.8

        # FINDING, pinned rather than hidden: one correction is not enough. The
        # corrected line is the top neighbour, but it is outvoted by several
        # weak neighbours (~0.26-0.34, similar mainly because the vendor name
        # is part of the indexed text), so the next invoice abstains again.
        # Count how many reviewed corrections it takes before the item is
        # categorised without help; docs/phase5_service.md reports the number.
        corrections, current = 1, second
        while current["categories"][0]["abstained"] and corrections < 6:
            item = next(
                i
                for i in client.get("/review/queue").json()
                if i["doc_id"] == current["doc_id"] and i["kind"] == "line_category"
            )
            resp = client.post(
                f"/review/lines/{item['item_id']}",
                json={"reviewer": "ravi", "action": "correct", "category": "5050"},
            )
            assert resp.status_code == 200, resp.text
            action = "override" if current["status"] == "blocked" else "accept"
            resp = client.post(
                f"/review/documents/{current['doc_id']}",
                json={"reviewer": "ravi", "action": action, "reason": "test"},
            )
            assert resp.status_code == 200, resp.text
            corrections += 1
            current = resubmit()
        print(f"\nCORRECTIONS-UNTIL-LEARNED {corrections}")
        assert current["categories"][0]["category"] == "5050", (
            corrections,
            current["categories"][0],
        )
        assert corrections <= 5

    def test_an_invalid_category_correction_is_refused(self, world, history, fresh):
        doc = pick(fresh, unknown_vendor(history))
        doc_id = world["post"](doc).json()["doc_id"]
        resp = world["client"].post(
            f"/review/documents/{doc_id}",
            json={"reviewer": "a", "action": "accept", "categories": {"0": UNCATEGORISED}},
        )
        assert resp.status_code == 422


# ------------------------------------------------------------------- FAIL


def redirect_bank_account(doc: dict) -> dict:
    doc.setdefault("payment_info", {})["account_number"] = "99999999999999"
    return doc


class TestFailIsNeverAutoAccepted:
    @pytest.fixture
    def failed(self, world, history, fresh):
        doc = redirect_bank_account(pick(fresh, known_vendor(history, minimum=2)))
        body = world["post"](doc).json()
        assert body["overall_verdict"] == "FAIL", body["review_reasons"]
        return world, doc, body

    def test_a_redirected_bank_account_is_blocked(self, failed):
        world, doc, body = failed
        assert body["status"] == "blocked"
        assert body["payable"] is False
        assert "bank_account_changed" in {c["code"] for c in body["validation"]["checks"]}
        # Not learned from: the attacker's account must not enter the master.
        record = world["service"].knowledge.master.get(vendor_of(doc))
        assert "99999999999999" not in record.bank_accounts

    def test_accept_is_refused_on_a_fail(self, failed):
        world, _, body = failed
        resp = world["client"].post(
            f"/review/documents/{body['doc_id']}", json={"reviewer": "asha", "action": "accept"}
        )
        assert resp.status_code == 409
        record = world["client"].get(f"/documents/{body['doc_id']}").json()
        assert record["status"] == "blocked"
        assert record["payable"] is False

    def test_an_override_needs_a_reason(self, failed):
        world, _, body = failed
        for reason in (None, "", "  "):
            resp = world["client"].post(
                f"/review/documents/{body['doc_id']}",
                json={"reviewer": "asha", "action": "override", "reason": reason},
            )
            assert resp.status_code == 422
        assert world["client"].get(f"/documents/{body['doc_id']}").json()["payable"] is False

    def test_an_override_is_attributable(self, failed):
        world, _, body = failed
        reason = "Supplier confirmed new account by phone, ticket 4411"
        record = (
            world["client"]
            .post(
                f"/review/documents/{body['doc_id']}",
                json={"reviewer": "asha", "action": "override", "reason": reason},
            )
            .json()
        )
        assert record["status"] == "accepted_with_override"
        assert record["payable"] is True
        override = [d for d in record["decisions"] if d["action"] == "override"]
        assert len(override) == 1
        assert override[0]["reviewer"] == "asha"
        assert override[0]["reason"] == reason
        assert override[0]["old_value"] == "blocked"
        assert override[0]["created_at"]

    def test_no_routing_turns_a_fail_into_auto_acceptance(self, world, history, fresh):
        """Every document that reaches FAIL, across the whole fresh split,
        routes to `blocked` -- whatever else is or is not wrong with it."""
        service = world["service"]
        seen_fail = 0
        for doc in fresh[:30]:
            tampered = redirect_bank_account(copy.deepcopy(doc))
            body = world["post"](tampered).json()
            if body["validation"]["verdict"] == "FAIL":
                seen_fail += 1
                assert body["status"] == "blocked"
                assert body["payable"] is False
        assert seen_fail, "no tampered document reached FAIL"
        assert all(
            d["status"] != "auto_accepted"
            for d in (service.document_record(r["doc_id"]) for r in service.store.items(state=None))
            if d["overall_verdict"] == "FAIL"
        )

    def test_route_never_auto_accepts_a_fail(self, history, fresh):
        """The routing function itself, independent of the HTTP layer."""
        from invoice_extract.service.pipeline import run_chain_on_text

        knowledge = build_knowledge(history)
        doc = redirect_bank_account(pick(fresh, known_vendor(history, minimum=2)))
        _, words = render(doc)
        text = OcrResult(words, "fixture").text
        chain = run_chain_on_text(
            text, knowledge, doc_id="x", source=doc["source"], pages=doc["pages"], today=TODAY
        )
        overall, status, reasons, items = route(chain)
        assert overall == "FAIL"
        assert status == "blocked"
        assert ("document", None) in items
        assert any(r.startswith("FAIL") for r in reasons)


# --------------------------------------------------------------- feedback


class TestCorrectionsReachTheVendorMaster:
    def test_a_corrected_vendor_name_creates_the_supplier_record(self, world, history, fresh):
        """Exit criterion 4. OCR misreads a first-time supplier's name; the
        reviewer corrects it; the vendor master now holds the *corrected* name
        with that invoice's GSTIN and bank account; the next invoice from the
        same supplier resolves to it instead of going to review again."""
        client, service = world["client"], world["service"]
        names = {vendor_of(d) for d in history}
        new_vendor_docs = [d for d in fresh if vendor_of(d) not in names]
        by_vendor: dict[str, list[dict]] = {}
        for d in new_vendor_docs:
            by_vendor.setdefault(vendor_of(d), []).append(d)
        repeat = next((v for v in by_vendor.values() if len(v) >= 2), None)
        if repeat is None:
            pytest.skip("no first-time supplier invoices twice in the fresh split")
        first, second = copy.deepcopy(repeat[0]), copy.deepcopy(repeat[1])
        true_name = vendor_of(first)
        first_word = true_name.split()[0]

        assert service.knowledge.master.get(true_name) is None

        # An OCR error on the first word of the vendor name.
        garbled = first_word[:-2] + "##"
        body = world["post"](first, rename={first_word: garbled}).json()
        assert body["document"]["parties"]["vendor"]["name"] != true_name
        assert body["overall_verdict"] in ("REVIEW", "FAIL")
        assert "unknown_vendor" in {c["code"] for c in body["validation"]["checks"]}

        record = client.post(
            f"/review/documents/{body['doc_id']}",
            json={"reviewer": "meera", "action": "accept", "vendor_name": true_name},
        ).json()
        assert record["status"] == "accepted"
        # The machine's extraction is kept as it was; the correction is a decision.
        assert record["result"]["document"]["parties"]["vendor"]["name"] != true_name
        assert record["final_document"]["parties"]["vendor"]["name"] == true_name
        correction = next(d for d in record["decisions"] if d["field"] == "parties.vendor.name")
        assert correction["new_value"] == true_name
        assert correction["reviewer"] == "meera"

        # The vendor master changed.
        vendor = service.knowledge.master.get(true_name)
        assert vendor is not None
        assert vendor.invoice_count == 1
        assert first["parties"]["vendor"]["gstin"] in vendor.gstins
        assert first["payment_info"]["account_number"] in vendor.bank_accounts
        assert client.get(f"/vendors/{true_name}").status_code == 200
        assert service.knowledge.master.get(body["document"]["parties"]["vendor"]["name"]) is None

        # And it is used: the next invoice from this supplier resolves.
        nxt = world["post"](second).json()
        assert nxt["validation"]["resolution"]["resolved"]
        assert nxt["validation"]["resolution"]["vendor"] == true_name
        assert "unknown_vendor" not in {c["code"] for c in nxt["validation"]["checks"]}

    def test_corrections_survive_a_restart(self, world, history, fresh):
        doc = pick(fresh, unknown_vendor(history))
        doc_id = world["post"](doc).json()["doc_id"]
        world["client"].post(
            f"/review/documents/{doc_id}",
            json={"reviewer": "meera", "action": "accept", "vendor_name": "Renamed Supplier Ltd"},
        )
        restarted = ReviewService(
            ReviewStore(world["tmp"] / "review.sqlite3"),
            build_knowledge(history),
            lambda: world["engine"],
            image_dir=world["tmp"] / "images",
            today=lambda: TODAY,
        )
        assert restarted.knowledge.master.get("Renamed Supplier Ltd") is not None
        decisions = restarted.store.decisions(doc_id)
        assert any(d["new_value"] == "Renamed Supplier Ltd" for d in decisions)

    def test_the_decision_log_is_json(self, world, history, fresh):
        doc = pick(fresh, unknown_vendor(history))
        doc_id = world["post"](doc).json()["doc_id"]
        record = (
            world["client"]
            .post(f"/review/documents/{doc_id}", json={"reviewer": "a", "action": "accept"})
            .json()
        )
        json.dumps(record["decisions"])
