"""The HTTP surface. The only module that imports FastAPI.

Endpoints:

    GET  /health
    POST /documents                      one image, processed synchronously
    POST /jobs                           one image, processed in the background
    GET  /jobs/{job_id}                  poll a background job
    GET  /documents/{doc_id}             result, status, payable, decision log
    GET  /documents/{doc_id}/image       the page as uploaded
    GET  /review/queue                   open review items, self-contained
    POST /review/lines/{item_id}         categorise an abstained line
    POST /review/documents/{doc_id}      accept / override / reject a document
    GET  /vendors/{name}                 the vendor master record
    GET  /categories                     the chart of accounts
    GET  /review                         a minimal reviewer page over the API

**Synchronous vs job.** Real OCR is ~40 s/page on CPU with the fast models
(docs/phase1_preprocessing.md), well past what an HTTP request should hold
open. `POST /jobs` returns a handle immediately and is the endpoint to use with
real OCR; `POST /documents` exists for fast engines and for tests, and reports
its own stage timings so the cost is visible rather than assumed.

Run with `python scripts/run_service.py` (or `.\\run.ps1 serve`).
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from invoice_extract.rag.taxonomy import CHART_OF_ACCOUNTS
from invoice_extract.service.auth import (  # noqa: E402
    TOKEN_ENV,
    Authenticator,
    Principal,
    Scope,
    authorization_header,
    describe,
)
from invoice_extract.service.knowledge import Knowledge
from invoice_extract.service.pipeline import SchemaViolationError
from invoice_extract.service.review import ReviewError, ReviewService
from invoice_extract.service.review_page import REVIEW_PAGE
from invoice_extract.service.schemas import (
    DocumentDecisionRequest,
    DocumentRecord,
    JobHandle,
    JobStatus,
    LineDecisionRequest,
    ProcessResult,
    QueueItem,
)
from invoice_extract.service.store import ReviewStore


class JobRunner:
    """Background processing with one worker. OCR is CPU-bound and the engine
    is serialised anyway, so more workers would only queue inside the lock."""

    def __init__(self, service: ReviewService) -> None:
        self.service = service
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ie-job")
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def submit(self, data: bytes, filename: str | None) -> str:
        job_id = uuid.uuid4().hex[:16]
        with self._lock:
            self.jobs[job_id] = {"job_id": job_id, "status": "queued"}
        self.pool.submit(self._run, job_id, data, filename)
        return job_id

    def _run(self, job_id: str, data: bytes, filename: str | None) -> None:
        with self._lock:
            self.jobs[job_id]["status"] = "running"
        try:
            result = self.service.process_image(data, filename)
            update = {"status": "done", "doc_id": result["doc_id"], "result": result}
        except ReviewError as exc:
            update = {"status": "error", "error": exc.message}
        except SchemaViolationError as exc:
            update = {"status": "error", "error": str(exc)}
        except Exception as exc:  # surfaced to the caller, never swallowed
            update = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        with self._lock:
            self.jobs[job_id].update(update)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None


def default_service() -> ReviewService:
    """Build the service from environment variables (all optional)."""
    root = Path(os.environ.get("IE_SERVICE_DIR", "data/service"))
    corpus = Path(os.environ.get("IE_CORPUS", "data/processed/gst_in_synthetic"))
    ledger = Path(os.environ.get("IE_LEDGER", "data/processed/ledger"))
    engine_name = os.environ.get("IE_OCR_ENGINE", "paddle")
    fast = os.environ.get("IE_OCR_FAST", "1") not in ("0", "false", "False")

    def engine_factory():
        from invoice_extract.ocr.base import get_engine

        return (
            get_engine(engine_name, fast=fast)
            if engine_name == "paddle"
            else get_engine(engine_name)
        )

    return ReviewService(
        ReviewStore(root / "review.sqlite3"),
        Knowledge.bootstrap(corpus, ledger),
        engine_factory,
        image_dir=root / "images",
    )


log = logging.getLogger(__name__)


def _reviewer(who: Principal, claimed: str | None) -> str:
    """The name that goes in the audit trail.

    The credential decides, never the request body. A body that disagrees is
    rejected rather than ignored, so a client still sending its own name fails
    loudly instead of quietly writing a different one into the log than the
    one that was authenticated.

    With authentication disabled the caller is `anonymous` and the body's name
    is used, which is exactly the previous behaviour.
    """
    if who.anonymous:
        return claimed or "anonymous"
    if claimed and claimed != who.name:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"authenticated as {who.name!r}; cannot record a decision as {claimed!r}",
        )
    return who.name


def create_app(service: ReviewService | None = None, auth: Authenticator | None = None) -> FastAPI:
    app = FastAPI(
        title="invoice-extract-vlm",
        version="0.5.0",
        description="Phase 5: the extraction pipeline as a service, with a human-review path.",
    )
    authenticator = auth if auth is not None else Authenticator.from_env()
    if not authenticator.enabled:
        # Loud, because a security control that is silently absent is worse
        # than one that is absent. /health says the same thing.
        log.warning(
            "No %s configured: every caller is 'anonymous' with every scope. "
            "Set it before exposing this beyond localhost.",
            TOKEN_ENV,
        )
    app.state.authenticator = authenticator
    state: dict[str, Any] = {"service": service, "jobs": None}

    def principal_with(scope: Scope):
        """Dependency factory: resolve the caller and check one scope."""

        def dependency(
            authorization: str | None = Depends(authorization_header),
        ) -> Principal:
            who = authenticator.principal_for(authorization)
            authenticator.require(who, scope)
            return who

        return dependency

    # `Depends` in the DEFAULT, not inside `Annotated`. This module has
    # `from __future__ import annotations`, so every annotation is a string
    # resolved against module globals -- a type alias declared here, inside
    # the factory, is invisible to that lookup and FastAPI silently falls
    # back to treating the parameter as a query field. The symptom is a 422
    # demanding `who` as a query parameter.
    #
    # ruff's B008 (no function calls in argument defaults) is a false
    # positive for FastAPI: `Depends` is inert until the framework resolves
    # it, and this is the documented idiom.
    may_submit = Depends(principal_with(Scope.SUBMIT))  # noqa: B008
    may_review = Depends(principal_with(Scope.REVIEW))  # noqa: B008

    def svc() -> ReviewService:
        if state["service"] is None:
            state["service"] = default_service()
        return state["service"]

    def jobs() -> JobRunner:
        if state["jobs"] is None:
            state["jobs"] = JobRunner(svc())
        return state["jobs"]

    app.state.get_service = svc

    @app.exception_handler(ReviewError)
    async def _review_error(_: Request, exc: ReviewError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    @app.exception_handler(SchemaViolationError)
    async def _schema_error(_: Request, exc: SchemaViolationError) -> JSONResponse:
        # Not repaired, not coerced into the response model: reported.
        return JSONResponse(
            status_code=422,
            content={
                "detail": "extraction produced a document that violates the canonical schema",
                "doc_id": exc.doc_id,
                "problems": exc.problems,
            },
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        service = svc()
        return {
            "status": "ok",
            "history_documents": len(service.knowledge.history),
            "ledger_entries": len(service.knowledge.entries),
            "vendors": len(service.knowledge.master),
            "open_review_items": len(service.store.items(state="open")),
            **describe(authenticator),
        }

    @app.post("/documents", response_model=ProcessResult, response_model_exclude_unset=True)
    def process_document(
        file: Annotated[UploadFile, File()],
        who: Principal = may_submit,
    ) -> dict[str, Any]:
        # A plain `def`, so FastAPI runs it in a worker thread: a 40 s OCR call
        # inside an `async def` would stall every other request, health
        # checks and job polling included.
        data = file.file.read()
        return svc().process_image(data, file.filename)

    @app.post("/jobs", response_model=JobHandle, status_code=status.HTTP_202_ACCEPTED)
    def submit_job(
        file: Annotated[UploadFile, File()],
        who: Principal = may_submit,
    ) -> dict[str, Any]:
        data = file.file.read()
        job_id = jobs().submit(data, file.filename)
        return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}", response_model=JobStatus, response_model_exclude_unset=True)
    def job_status(job_id: str) -> dict[str, Any]:
        job = jobs().get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job

    @app.get(
        "/documents/{doc_id}", response_model=DocumentRecord, response_model_exclude_unset=True
    )
    def get_document(doc_id: str) -> dict[str, Any]:
        return svc().document_record(doc_id)

    @app.get("/documents/{doc_id}/image")
    def get_image(doc_id: str) -> FileResponse:
        path = svc().image_path(doc_id)
        if not path.exists():
            raise HTTPException(404, "image file missing")
        return FileResponse(path)

    @app.get("/review/queue", response_model=list[QueueItem])
    def review_queue(state_filter: str = "open") -> list[dict[str, Any]]:
        if state_filter not in ("open", "resolved", "all"):
            raise HTTPException(422, "state_filter must be open, resolved or all")
        return svc().queue(None if state_filter == "all" else state_filter)

    @app.post(
        "/review/lines/{item_id}", response_model=DocumentRecord, response_model_exclude_unset=True
    )
    def decide_line(
        item_id: int,
        body: LineDecisionRequest,
        who: Principal = may_review,
    ) -> dict[str, Any]:
        return svc().decide_line(
            item_id,
            reviewer=_reviewer(who, body.reviewer),
            action=body.action,
            category=body.category,
            note=body.note,
        )

    @app.post(
        "/review/documents/{doc_id}",
        response_model=DocumentRecord,
        response_model_exclude_unset=True,
    )
    def decide_document(
        doc_id: str,
        body: DocumentDecisionRequest,
        who: Principal = may_review,
    ) -> dict[str, Any]:
        # Clearing a FAIL is a separate right from reviewing. If the account
        # that uploaded an invoice can also clear its own block, the queue is
        # decoration -- so OVERRIDE is checked here rather than at the door,
        # where the verdict is known.
        record = svc().document_record(doc_id)
        if record.get("verdict") == "FAIL" and body.action == "accept":
            authenticator.require(who, Scope.OVERRIDE)
        return svc().decide_document(
            doc_id,
            reviewer=_reviewer(who, body.reviewer),
            action=body.action,
            reason=body.reason,
            vendor_name=body.vendor_name,
            categories=body.categories,
        )

    @app.get("/vendors/{name}")
    def get_vendor(name: str) -> dict[str, Any]:
        record = svc().knowledge.master.get(name)
        if record is None:
            raise HTTPException(404, f"no vendor record for {name!r}")
        return record.as_dict()

    @app.get("/categories")
    def categories() -> list[dict[str, str]]:
        return [
            {"code": a.code, "name": a.name, "description": a.description}
            for a in CHART_OF_ACCOUNTS.values()
        ]

    @app.get("/review", response_class=HTMLResponse)
    def review_page() -> str:
        return REVIEW_PAGE

    return app
