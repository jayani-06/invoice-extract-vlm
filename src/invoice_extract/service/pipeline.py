"""Image to verdict, as one call.

This is the chain `tests/test_pipeline_end_to_end.py` drives stage by stage,
packaged so the service and the tests run *the same code*:

    image -> preprocess -> OCR -> reading order -> extract -> coerce+validate
          -> categorise (Phase 3) -> anomaly flags (Phase 3)
          -> vendor resolution + record validation (Phase 4)

Two rules from the rest of the project carry over unchanged:

- **Nothing is repaired.** `finalize` coerces types (a string "1,234.00" to a
  float) exactly as every evaluation did; arithmetic problems come back as a
  report beside the document, never as an edited total.
- **A schema-invalid extraction is an error, not a result.** If the document
  coming out of extraction does not validate against `schema/`, the chain
  raises `SchemaViolationError` with the problems, rather than returning
  something the response model would have to bend to accept.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np

from invoice_extract.data.validate import validate_jsonschema
from invoice_extract.models.postprocess import ConsistencyReport, finalize
from invoice_extract.models.registry import get_extractor
from invoice_extract.ocr.base import OcrEngine
from invoice_extract.preprocess.pipeline import Pipeline, get_config
from invoice_extract.rag.anomaly import AnomalyReport, detect_anomalies
from invoice_extract.rag.categorize import CategoryDecision, categorize_document
from invoice_extract.rag.validate_vendor import ValidationReport, validate_against_master
from invoice_extract.service.knowledge import Knowledge

#: The Phase 1 pipeline with the best word accuracy on degraded pages at n=15
#: (docs/phase1_preprocessing.md), and the one the end-to-end OCR test uses.
DEFAULT_PREPROCESS = "full(no-binarize)"


class SchemaViolationError(ValueError):
    """Extraction produced a document the canonical schema rejects."""

    def __init__(self, doc_id: str, problems: list[str]) -> None:
        super().__init__(f"{doc_id}: extracted document violates the schema: {problems[:3]}")
        self.doc_id = doc_id
        self.problems = problems


@dataclass
class ChainResult:
    doc_id: str
    document: dict[str, Any]
    consistency: ConsistencyReport
    categories: list[CategoryDecision]
    anomalies: AnomalyReport
    validation: ValidationReport
    ocr: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The same shape the end-to-end test round-trips through JSON, plus
        the per-line explanation a reviewer reads and the stage timings."""
        categories = []
        for i, decision in enumerate(self.categories):
            row = decision.as_dict()
            row["line_index"] = i
            row["explanation"] = decision.explain()
            categories.append(row)
        return {
            "doc_id": self.doc_id,
            "document": self.document,
            "consistency_problems": list(self.consistency.problems),
            "categories": categories,
            "anomalies": self.anomalies.as_dict(),
            "validation": self.validation.as_dict(),
            "ocr": self.ocr,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


def run_chain_on_text(
    text: str,
    knowledge: Knowledge,
    *,
    doc_id: str,
    source: dict[str, Any],
    pages: list[dict[str, Any]],
    extractor: str = "rules",
    today: date | None = None,
    timings_ms: dict[str, float] | None = None,
    ocr: dict[str, Any] | None = None,
) -> ChainResult:
    """Everything after OCR. Used directly by tests and by `run_chain_on_image`."""
    timings = dict(timings_ms or {})

    t = time.perf_counter()
    raw = get_extractor(extractor).predict_from_text(text)
    document, consistency = finalize(raw, doc_id, source, pages)
    timings["extract"] = (time.perf_counter() - t) * 1000

    problems = validate_jsonschema(document)
    if problems:
        raise SchemaViolationError(doc_id, problems)

    t = time.perf_counter()
    categories = categorize_document(document, knowledge.retriever, vendor_stats=knowledge.stats)
    anomalies = detect_anomalies(
        document,
        knowledge.retriever,
        knowledge.stats,
        today=today,
        seen_invoice_numbers=knowledge.seen_invoice_numbers,
    )
    timings["enrich"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    validation = validate_against_master(document, knowledge.master, history=knowledge.history)
    timings["validate"] = (time.perf_counter() - t) * 1000

    return ChainResult(
        doc_id=doc_id,
        document=document,
        consistency=consistency,
        categories=categories,
        anomalies=anomalies,
        validation=validation,
        ocr=dict(ocr or {}),
        timings_ms=timings,
    )


def run_chain_on_image(
    image: np.ndarray,
    engine: OcrEngine,
    knowledge: Knowledge,
    *,
    doc_id: str,
    source: dict[str, Any],
    pages: list[dict[str, Any]],
    preprocess: str = DEFAULT_PREPROCESS,
    extractor: str = "rules",
    today: date | None = None,
) -> ChainResult:
    """The whole chain from pixels. `image` is a grayscale uint8 array."""
    timings: dict[str, float] = {}

    t = time.perf_counter()
    prepared = Pipeline(get_config(preprocess))(image)
    timings["preprocess"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    recognised = engine.recognize(prepared.image)
    timings["ocr"] = (time.perf_counter() - t) * 1000

    ocr = {
        "engine": recognised.engine,
        "n_words": len(recognised.words),
        "mean_confidence": recognised.mean_confidence,
        "preprocess": preprocess,
        "meta": recognised.meta,
        "text": recognised.text,
    }
    return run_chain_on_text(
        recognised.text,
        knowledge,
        doc_id=doc_id,
        source=source,
        pages=pages,
        extractor=extractor,
        today=today,
        timings_ms=timings,
        ocr=ocr,
    )
