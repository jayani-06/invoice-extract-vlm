"""The offline path that turns a finished GPU run into scored predictions.

The VLM runs on Colab and the scoring happens here, which puts a file transfer
between the model and its result. Three things can go wrong in that gap, and
each one silently flatters or misreports the model rather than failing loudly:

- **documents the model never answered get dropped**, shrinking the denominator
  so a model that answered 40 of 100 pages scores as if it answered 40 of 40
- **a vision-only run gets stamped as a perfect-OCR ceiling**, because
  `--ocr-source` defaults to `gold` and the replay does not use OCR at all --
  which mislabels the project's only genuine end-to-end number
- **the baseline is scored on different documents**, which makes the
  side-by-side table meaningless no matter how carefully each half was measured

None of those raise. They produce a plausible number, which is why they are
pinned here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data/processed/gst_in_synthetic"

pytestmark = pytest.mark.skipif(
    not (CORPUS / "manifest.jsonl").exists(), reason="image corpus not built"
)


def manifest_doc_ids(limit: int) -> list[str]:
    rows = [
        json.loads(ln) for ln in (CORPUS / "manifest.jsonl").read_text().splitlines() if ln.strip()
    ]
    return [r["doc_id"] for r in rows if r["profile"] == "clean"][:limit]


def gold_response(doc_id: str) -> str:
    """A response that reproduces the document exactly.

    Scored, it should come back near-perfect -- which is what makes it a usable
    control: anything less means the replay path is losing information rather
    than the model being wrong.
    """
    doc = json.loads((CORPUS / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8"))
    # Every top-level key the prompt template asks for. Leaving one out makes
    # the control score below the rules baseline, which looks like the replay
    # path losing information -- `payment_info` was missing here at first and
    # cost exactly the 3 fields it carries.
    keep = (
        "document_type",
        "currency",
        "parties",
        "invoice_meta",
        "line_items",
        "tax_lines",
        "payment_info",
    )
    reply = {k: doc[k] for k in keep if k in doc}
    reply["totals"] = doc.get("totals") or {}
    # Models wrap JSON in a markdown fence; carry that through so the parser is
    # exercised the way it will be in production.
    return "```json\n" + json.dumps(reply) + "\n```"


@pytest.fixture
def selection(tmp_path) -> Path:
    doc_ids = manifest_doc_ids(4)
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "selections": [
                    {
                        "corpus": "gst_in_synthetic",
                        "profile": "clean",
                        "n": len(doc_ids),
                        "doc_ids": doc_ids,
                    },
                    # A second corpus in the same record: the runner must take
                    # only the ids that belong to the corpus it was given.
                    {
                        "corpus": "fatura2",
                        "profile": "clean",
                        "n": 1,
                        "doc_ids": ["fatura2_test_00000"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def write_outputs(tmp_path: Path, responses: dict[str, str], *, ocr_hint: bool = False) -> Path:
    path = tmp_path / "vlm_outputs.json"
    path.write_text(
        json.dumps(
            {
                "model": "test-model",
                "model_id": "test/model",
                "ocr_hint": ocr_hint,
                "profile": "clean",
                "outputs": {
                    doc_id: {"raw_response": text, "elapsed_ms": 1234.0, "error": None}
                    for doc_id, text in responses.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def replay(tmp_path: Path, outputs: Path, selection: Path, out: str = "out") -> dict:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/run_extraction.py"),
            "--corpus",
            str(CORPUS),
            "--extractor",
            "vlm:test",
            "--replay",
            str(outputs),
            "--doc-ids",
            str(selection),
            "--out",
            str(tmp_path / out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    run = json.loads((tmp_path / out / "run.json").read_text(encoding="utf-8"))
    run["_stderr"] = result.stderr
    return run


class TestDenominator:
    def test_unanswered_documents_are_scored_as_misses(self, tmp_path, selection):
        """The failure that would most flatter the model: answering one page out
        of four and being scored out of one."""
        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(tmp_path, {doc_ids[0]: gold_response(doc_ids[0])})

        run = replay(tmp_path, outputs, selection)

        assert run["n_docs"] == 4, "denominator shrank to the documents the model answered"
        assert run["parse_failures"] == 3
        assert "scored as misses" in run["_stderr"]

    def test_unparseable_and_empty_responses_count_as_failures(self, tmp_path, selection):
        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(
            tmp_path,
            {
                doc_ids[0]: gold_response(doc_ids[0]),
                doc_ids[1]: "I am unable to read this invoice.",
                doc_ids[2]: "",
                doc_ids[3]: '{"totals": {"grand_total": 1.0}}',
            },
        )

        run = replay(tmp_path, outputs, selection)

        assert run["n_docs"] == 4
        assert run["parse_failures"] == 2, "prose-only and empty must both count"

    def test_a_faithful_response_scores_near_perfectly(self, tmp_path, selection):
        """The control. If a response that restates the document does not score
        well, the replay path is losing information and no model number taken
        through it means anything."""
        from invoice_extract.eval.harness import read_jsonl
        from invoice_extract.eval.metrics import aggregate, compare_documents

        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(tmp_path, {d: gold_response(d) for d in doc_ids})
        replay(tmp_path, outputs, selection)

        gold = read_jsonl(tmp_path / "out" / "gold.jsonl")
        pred = read_jsonl(tmp_path / "out" / "pred.jsonl")
        overall = aggregate([compare_documents(gold[d], pred[d]) for d in gold])

        assert overall.doc_critical_accuracy == 1.0
        assert overall.macro_field_accuracy > 0.95


class TestLabelling:
    def test_a_vision_only_run_is_not_labelled_an_ocr_ceiling(self, tmp_path, selection):
        """`--ocr-source` defaults to `gold`, and every other Phase 2 number is
        a perfect-OCR ceiling because of it. A replayed vision-only run read
        nothing but pixels, so inheriting that default would file the project's
        first end-to-end measurement under its ceiling figures."""
        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(tmp_path, {d: gold_response(d) for d in doc_ids}, ocr_hint=False)

        run = replay(tmp_path, outputs, selection)

        assert run["is_ocr_ceiling"] is False
        assert run["ocr_source"] == "none"
        assert run["vision_only"] is True

    def test_a_hybrid_run_is_labelled_a_ceiling(self, tmp_path, selection):
        """The converse: a run whose prompt carried gold OCR text IS a ceiling,
        and must say so even though nothing on the command line mentions OCR."""
        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(tmp_path, {d: gold_response(d) for d in doc_ids}, ocr_hint=True)

        run = replay(tmp_path, outputs, selection)

        assert run["is_ocr_ceiling"] is True
        assert run["vision_only"] is False

    def test_the_replayed_model_is_recorded(self, tmp_path, selection):
        """`--extractor` is a label the caller types; the model id comes from the
        payload. A report must not be able to attribute a run to the wrong
        model because someone mistyped the flag."""
        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(tmp_path, {d: gold_response(d) for d in doc_ids})

        run = replay(tmp_path, outputs, selection)

        assert run["replay_model"] == "test/model"
        assert run["replayed_from"].endswith("vlm_outputs.json")


class TestDocumentSelection:
    def test_the_same_record_drives_both_corpora(self, tmp_path, selection):
        """One selection file covers every corpus in the run. Asked for
        gst_in_synthetic it must take the 4 gst ids, not the fatura2 one."""
        doc_ids = manifest_doc_ids(4)
        outputs = write_outputs(tmp_path, {d: gold_response(d) for d in doc_ids})

        replay(tmp_path, outputs, selection)
        gold = [
            json.loads(ln)
            for ln in (tmp_path / "out" / "gold.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert [g["doc_id"] for g in gold] == doc_ids

    def test_a_record_naming_no_matching_corpus_fails_loudly(self, tmp_path):
        """Selecting nothing must be an error. Falling through to the whole
        corpus would score the two extractors on different document sets, which
        is the one thing the flag exists to prevent."""
        record = tmp_path / "other.json"
        record.write_text(
            json.dumps({"selections": [{"corpus": "sroie", "doc_ids": ["sroie_0001"]}]}),
            encoding="utf-8",
        )
        outputs = write_outputs(tmp_path, {manifest_doc_ids(1)[0]: "{}"})

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "scripts/run_extraction.py"),
                "--corpus",
                str(CORPUS),
                "--extractor",
                "vlm:test",
                "--replay",
                str(outputs),
                "--doc-ids",
                str(record),
                "--out",
                str(tmp_path / "nope"),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode != 0
        assert "selects no documents" in result.stderr


class TestPageMetadata:
    def test_predictions_validate_against_the_schema_without_a_dpi(self, tmp_path):
        """FATURA2's manifest records no dpi, and the schema types dpi as an
        integer. Emitting `"dpi": null` made all 50 FATURA2 predictions fail
        schema validation -- which reads in a report as the extractor producing
        malformed documents, not as a corpus that does not record dpi."""
        fatura2 = ROOT / "data/processed/fatura2"
        if not (fatura2 / "manifest.jsonl").exists():
            pytest.skip("fatura2 not converted")

        rows = [
            json.loads(ln)
            for ln in (fatura2 / "manifest.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert rows[0].get("dpi") is None, "fatura2 now records a dpi; this test is obsolete"

        doc_ids = [r["doc_id"] for r in rows[:3]]
        record = tmp_path / "f2.json"
        record.write_text(
            json.dumps({"selections": [{"corpus": "fatura2", "doc_ids": doc_ids}]}),
            encoding="utf-8",
        )
        outputs = write_outputs(tmp_path, {d: '{"totals": {"grand_total": 1.0}}' for d in doc_ids})

        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "scripts/run_extraction.py"),
                "--corpus",
                str(fatura2),
                "--extractor",
                "vlm:test",
                "--replay",
                str(outputs),
                "--doc-ids",
                str(record),
                "--out",
                str(tmp_path / "f2out"),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stderr
        run = json.loads((tmp_path / "f2out" / "run.json").read_text(encoding="utf-8"))
        assert run["schema_failures"] == 0, run
