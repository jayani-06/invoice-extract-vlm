"""Response and request models for the service.

**The extracted document is not modelled here.** It is
`invoice_extract.data.models.CanonicalInvoiceDocument`, the pydantic mirror of
`schema/invoice_schema.json` that every other phase already validates against.
Writing a second document model for the API would give the project two schemas
that drift; reusing the existing one means a field added to the canonical
schema reaches the API with no edit here.

What *is* defined here is everything around the document -- categories, flags,
checks, review state -- which no existing schema covers. Those mirror the
`as_dict()` methods of the Phase 3/4 dataclasses field for field, and
`tests/test_service.py` validates real pipeline output through them, so a key
renamed in `as_dict()` fails a test instead of silently vanishing from the API.
All models forbid extra keys for the same reason.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from invoice_extract.data.models import CanonicalInvoiceDocument

Verdict = Literal["PASS", "REVIEW", "FAIL"]
Severity = Literal["high", "medium", "low"]

#: Lifecycle of a processed document. `payable` is derived from this, never
#: stored separately, so the two cannot disagree.
DocumentStatus = Literal[
    "auto_accepted",  # PASS, nothing abstained, no actionable flag
    "in_review",  # REVIEW, or abstained lines, or an actionable anomaly flag
    "blocked",  # FAIL: payable only through a recorded human override
    "accepted",  # a reviewer accepted it
    "accepted_with_override",  # a reviewer accepted a FAIL, with a reason
    "rejected",  # a reviewer rejected it
]
PAYABLE_STATUSES = frozenset({"auto_accepted", "accepted", "accepted_with_override"})


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------- Phase 3


class Neighbour(Strict):
    description: str | None
    category: str
    vendor: str | None
    score: float


class CategoryResult(Strict):
    line_index: int
    category: str
    category_name: str
    confidence: float
    abstained: bool
    reason: str
    votes: dict[str, float]
    neighbours: list[Neighbour]
    explanation: str


class AnomalyFlag(Strict):
    code: str
    severity: Severity
    signal: Literal["rule", "statistical", "semantic"]
    message: str
    evidence: dict[str, Any]


class AnomalyResult(Strict):
    doc_id: str
    clean: bool
    max_severity: Severity | None
    n_flags: int
    flags: list[AnomalyFlag]


# --------------------------------------------------------------- Phase 4


class ResolutionResult(Strict):
    resolved: bool
    vendor: str | None
    method: str
    score: float
    candidates: list[tuple[str, float]]


class CheckResult(Strict):
    code: str
    verdict: Verdict
    message: str
    evidence: dict[str, Any]


class ValidationResult(Strict):
    doc_id: str
    verdict: Verdict
    resolution: ResolutionResult
    checks: list[CheckResult]
    explanation: str


# --------------------------------------------------------------- Phase 5


class OcrSummary(Strict):
    engine: str | None = None
    n_words: int | None = None
    mean_confidence: float | None = None
    preprocess: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None


class ProcessResult(Strict):
    """Everything the pipeline concluded about one document, plus its routing."""

    doc_id: str
    document: CanonicalInvoiceDocument
    consistency_problems: list[str]
    categories: list[CategoryResult]
    anomalies: AnomalyResult
    validation: ValidationResult
    ocr: OcrSummary
    timings_ms: dict[str, float]
    overall_verdict: Verdict
    status: DocumentStatus
    payable: bool
    review_reasons: list[str]


class Decision(Strict):
    """One recorded reviewer action. Append-only: nothing is ever edited."""

    decision_id: int
    doc_id: str
    item_id: int | None
    reviewer: str
    action: str
    field: str | None
    old_value: Any = None
    new_value: Any = None
    reason: str | None
    created_at: str


class DocumentRecord(Strict):
    doc_id: str
    filename: str | None
    created_at: str
    status: DocumentStatus
    payable: bool
    overall_verdict: Verdict
    result: ProcessResult
    final_document: CanonicalInvoiceDocument | None
    decisions: list[Decision]


class QueueItem(Strict):
    """What a reviewer needs to decide one item, without a second request."""

    item_id: int
    doc_id: str
    kind: Literal["document", "line_category"]
    line_index: int | None
    state: Literal["open", "resolved"]
    overall_verdict: Verdict
    image_url: str
    #: The value under review: the vendor block for a document item, the line
    #: item for a category item.
    extracted: dict[str, Any]
    explanation: str
    neighbours: list[Neighbour] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    flags: list[AnomalyFlag] = Field(default_factory=list)


# ---------------------------------------------------------------- requests


class LineDecisionRequest(Strict):
    reviewer: str = Field(min_length=1)
    action: Literal["accept", "correct"]
    category: str | None = None
    note: str | None = None


class DocumentDecisionRequest(Strict):
    reviewer: str = Field(min_length=1)
    action: Literal["accept", "override", "reject"]
    reason: str | None = None
    #: Human corrections applied to the accepted copy. The machine's
    #: extraction is kept untouched beside it, and every change is a recorded
    #: decision with its old and new value.
    vendor_name: str | None = None
    categories: dict[int, str] = Field(default_factory=dict)


class JobHandle(Strict):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    status_url: str


class JobStatus(Strict):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    doc_id: str | None = None
    result: ProcessResult | None = None
    error: str | None = None
