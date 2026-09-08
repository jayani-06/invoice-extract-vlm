import json
from pathlib import Path

import pytest

from invoice_extract.data.models import CanonicalInvoiceDocument
from invoice_extract.data.validate import load_schema, validate_jsonschema

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "schema" / "examples" / "example_invoice.json"


@pytest.fixture(scope="module")
def example_doc() -> dict:
    return json.loads(EXAMPLE_PATH.read_text())


def test_example_validates_against_jsonschema(example_doc):
    errors = validate_jsonschema(example_doc, load_schema())
    assert errors == []


def test_example_validates_against_pydantic_model(example_doc):
    # Raises on failure; also checked explicitly for a clearer assertion message.
    doc = CanonicalInvoiceDocument.model_validate(example_doc)
    assert doc.doc_id == example_doc["doc_id"]


def test_missing_required_field_is_rejected(example_doc):
    broken = dict(example_doc)
    del broken["totals"]
    errors = validate_jsonschema(broken, load_schema())
    assert any("totals" in e for e in errors)


def test_wrong_schema_version_is_rejected(example_doc):
    broken = {**example_doc, "schema_version": "0.9.0"}
    errors = validate_jsonschema(broken, load_schema())
    assert any("schema_version" in e for e in errors)


def test_invalid_gstin_format_is_rejected(example_doc):
    broken = json.loads(json.dumps(example_doc))  # deep copy
    broken["parties"]["vendor"]["gstin"] = "not-a-gstin"
    errors = validate_jsonschema(broken, load_schema())
    assert any("gstin" in e for e in errors)


def test_additional_properties_rejected(example_doc):
    broken = {**example_doc, "unexpected_field": "surprise"}
    errors = validate_jsonschema(broken, load_schema())
    assert any("additional" in e.lower() for e in errors)


def test_fixture_gold_and_pred_docs_all_validate():
    schema = load_schema()
    for fixture in ["gold.jsonl", "pred.jsonl"]:
        path = REPO_ROOT / "tests" / "fixtures" / fixture
        for line in path.read_text().splitlines():
            doc = json.loads(line)
            errors = validate_jsonschema(doc, schema)
            assert errors == [], f"{fixture}:{doc['doc_id']} -> {errors}"
