"""Phase 5: the pipeline as a callable service, plus the human-review path.

Layering, so that only one module needs the `service` extra:

- `knowledge`  what the RAG stages retrieve against, and how it accretes
- `pipeline`   image -> verdict, the same chain `tests/test_pipeline_end_to_end.py`
               runs, returned as one JSON-serialisable result
- `schemas`    pydantic response models; the document itself is the existing
               `data.models.CanonicalInvoiceDocument`, not a second schema
- `store`      SQLite persistence for results, the review queue and every
               reviewer decision (stdlib `sqlite3`, no new dependency)
- `review`     queue routing, the decision rules, and feedback into the
               vendor master and ledger
- `api`        the FastAPI app -- the only module that imports FastAPI
"""
