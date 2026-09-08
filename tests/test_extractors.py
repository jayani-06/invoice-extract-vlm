"""Tests for prompting, the rules baseline, the VLM wrapper and the registry.

The VLM is exercised through an injected fake model and processor. That covers
the part of the pipeline most likely to break silently -- prompt construction,
response parsing, coercion into the canonical schema -- on a machine with no
GPU, which is the actual development environment here. What it deliberately
does not cover is whether the transformers generation call itself is correct;
only a real Colab/Kaggle run can establish that, and the docs say so.

The rules baseline is tested against text drawn from the real corpus rather
than hand-written strings, so a change to the renderer that breaks extraction
shows up here instead of in a much later ablation.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import jsonschema
import pytest

from invoice_extract.data.validate import validate_jsonschema
from invoice_extract.models.postprocess import finalize
from invoice_extract.models.prompting import (
    EXTRACTION_SCHEMA,
    SYSTEM_PROMPT,
    TARGET_SKELETON,
    build_prompt,
    extract_json_object,
)
from invoice_extract.models.registry import available_extractors, get_extractor
from invoice_extract.models.rules import NullExtractor, RuleBasedExtractor
from invoice_extract.models.vlm import (
    DEFAULT_PRESET,
    MODEL_PRESETS,
    VlmExtractor,
    VlmPrediction,
)
from invoice_extract.reading_order import reading_order_text

CORPUS = Path("data/processed/gst_in_synthetic")


def corpus_texts(limit: int = 8) -> list[tuple[str, str]]:
    paths = sorted(glob.glob(str(CORPUS / "ocr_gold" / "*.clean.json")))[:limit]
    out = []
    for path in paths:
        gold = json.loads(Path(path).read_text(encoding="utf-8"))
        out.append(
            (
                gold["doc_id"],
                reading_order_text(gold["words"], lambda w: tuple(w["bbox"]), lambda w: w["text"]),
            )
        )
    return out


# --------------------------------------------------------------- prompting


class TestExtractionSchema:
    def test_is_a_valid_json_schema(self):
        jsonschema.Draft202012Validator.check_schema(EXTRACTION_SCHEMA)

    def test_the_skeleton_validates_against_it(self):
        """The template shown to the model must itself be legal output, or the
        model is being asked to imitate something invalid."""
        skeleton = json.loads(TARGET_SKELETON)
        jsonschema.Draft202012Validator(EXTRACTION_SCHEMA).validate(skeleton)

    def test_it_excludes_caller_owned_identity_fields(self):
        """A model must not invent doc_id/source/pages -- those are provenance,
        and a hallucinated one is worse than a missing one."""
        for forbidden in ("doc_id", "source", "pages", "grounding", "schema_version"):
            assert forbidden not in EXTRACTION_SCHEMA["properties"]

    def test_a_coerced_skeleton_survives_the_real_schema(self):
        doc, _ = finalize(
            json.loads(TARGET_SKELETON),
            "d1",
            {"dataset": "other", "original_id": "d1", "split": "test"},
            [{"page_index": 0, "image_path": "x.png", "width": 10, "height": 10}],
        )
        assert validate_jsonschema(doc) == []


class TestBuildPrompt:
    def test_vision_only_has_no_ocr_block(self):
        assert "<ocr>" not in build_prompt()

    def test_hybrid_includes_the_ocr_text(self):
        prompt = build_prompt("TAX INVOICE\nAcme Traders")
        assert "<ocr>" in prompt and "Acme Traders" in prompt

    def test_ocr_is_framed_as_fallible(self):
        """Presented as authoritative, a model copies OCR errors verbatim
        instead of reading the pixels."""
        prompt = build_prompt("some text").lower()
        assert "may" in prompt and "trust the image" in prompt

    def test_long_ocr_is_truncated_from_the_middle(self):
        """Headers and totals sit at the two ends of a page; truncating the
        tail would throw away the totals block."""
        text = "HEAD " + ("x" * 9000) + " TAIL"
        prompt = build_prompt(text, max_ocr_chars=400)
        assert "HEAD" in prompt and "TAIL" in prompt and "..." in prompt
        assert len(prompt) < 2500

    def test_empty_ocr_is_treated_as_absent(self):
        assert "<ocr>" not in build_prompt("")


class TestExtractJsonObject:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('```json\n{"a": 2}\n```', {"a": 2}),
            ('```\n{"a": 3}\n```', {"a": 3}),
            ('Here is the JSON:\n{"a": 4}', {"a": 4}),
            ('{"a": 5,}', {"a": 5}),
            ('[{"a": 6}]', {"a": 6}),
            ('{"a": 7}\nHope that helps!', {"a": 7}),
        ],
    )
    def test_recovers_from_real_model_habits(self, raw, expected):
        assert extract_json_object(raw) == expected

    def test_braces_inside_string_values_do_not_truncate_the_object(self):
        """Invoice text contains braces; a non-greedy regex stops at the first
        one and silently returns a truncated document."""
        raw = '{"description": "item {A} and }brace", "total": 10}'
        assert extract_json_object(raw) == {"description": "item {A} and }brace", "total": 10}

    def test_escaped_quotes_are_handled(self):
        assert extract_json_object(r'{"note": "he said \"hi\""}') == {"note": 'he said "hi"'}

    @pytest.mark.parametrize("raw", ["", "   ", "I cannot read this image.", "[1, 2, 3]", None])
    def test_unrecoverable_returns_none(self, raw):
        assert extract_json_object(raw) is None

    def test_a_nested_object_is_returned_whole(self):
        raw = '{"parties": {"vendor": {"name": "Acme"}}, "totals": {"grand_total": 1}}'
        assert extract_json_object(raw)["parties"]["vendor"]["name"] == "Acme"


# ------------------------------------------------------------ rules baseline


@pytest.mark.skipif(
    not glob.glob(str(CORPUS / "ocr_gold" / "*.clean.json")), reason="corpus not built"
)
class TestRuleBasedExtractor:
    def test_extracts_the_critical_fields_from_real_corpus_text(self):
        extractor = RuleBasedExtractor()
        for doc_id, text in corpus_texts(8):
            gold = json.loads((CORPUS / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8"))
            raw = extractor.predict_from_text(text)
            doc, _ = finalize(raw, doc_id, gold["source"], gold["pages"])

            assert doc["invoice_meta"]["invoice_number"] == gold["invoice_meta"]["invoice_number"]
            assert doc["invoice_meta"]["invoice_date"] == gold["invoice_meta"]["invoice_date"]
            assert doc["totals"]["grand_total"] == pytest.approx(gold["totals"]["grand_total"])

    def test_finds_every_line_item_row(self):
        extractor = RuleBasedExtractor()
        for doc_id, text in corpus_texts(8):
            gold = json.loads((CORPUS / "canonical" / f"{doc_id}.json").read_text(encoding="utf-8"))
            raw = extractor.predict_from_text(text)
            assert len(raw["line_items"]) == len(gold["line_items"]), doc_id

    def test_numbers_embedded_in_descriptions_survive(self):
        """ "A4 Copier Paper (500 sheets)" must not become "A Copier Paper
        ( sheets)" -- only whitespace-delimited numbers are column values."""
        extractor = RuleBasedExtractor()
        text = (
            "# Description HSN/SAC Qty Rate Tax % Amount\n"
            "1 A4 Copier Paper (500 sheets) 4802 13 295.64 5% 4035.49\n"
            "Subtotal: 3843.32\n"
        )
        items = extractor.predict_from_text(text)["line_items"]
        assert items and items[0]["description"] == "A4 Copier Paper (500 sheets)"

    def test_row_index_is_not_mistaken_for_a_quantity(self):
        extractor = RuleBasedExtractor()
        text = (
            "# Description Qty Rate Amount\n"
            "1 Wireless Mouse 12 861.88 12204.22\n"
            "Subtotal: 10342.56\n"
        )
        item = extractor.predict_from_text(text)["line_items"][0]
        assert item["quantity"] == pytest.approx(12.0)
        assert item.get("line_no") == 1

    def test_label_alternation_does_not_break_value_capture(self):
        """`\\bifsc\\b|\\bswift\\b` concatenated with a value pattern used to
        bind the value to only the last branch and crash."""
        extractor = RuleBasedExtractor()
        info = extractor.predict_from_text("IFSC: MHYR0237088\nBank: HDFC Bank")
        assert info["payment_info"]["ifsc_or_swift"] == "MHYR0237088"
        assert info["payment_info"]["bank_name"] == "HDFC Bank"

    def test_output_is_always_schema_valid_after_finalize(self):
        extractor = RuleBasedExtractor()
        for doc_id, text in corpus_texts(8):
            raw = extractor.predict_from_text(text)
            doc, _ = finalize(
                raw,
                doc_id,
                {"dataset": "other", "original_id": doc_id, "split": "test"},
                [{"page_index": 0, "image_path": "x.png", "width": 1, "height": 1}],
            )
            assert validate_jsonschema(doc) == [], doc_id

    def test_empty_and_garbage_input_do_not_raise(self):
        extractor = RuleBasedExtractor()
        for text in ("", "   ", "!!!", "no invoice here at all"):
            result = extractor.predict_from_text(text)
            assert isinstance(result["line_items"], list)

    def test_needs_an_engine_to_read_an_image(self):
        with pytest.raises(RuntimeError, match="OCR engine"):
            RuleBasedExtractor().predict([Path("nonexistent.png")])


class TestNullExtractor:
    def test_predicts_nothing(self):
        result = NullExtractor().predict([])
        assert result["line_items"] == []
        assert result["totals"]["grand_total"] is None

    def test_survives_finalize_and_schema_validation(self):
        doc, _ = finalize(
            NullExtractor().predict([]),
            "d",
            {"dataset": "other", "original_id": "d", "split": "test"},
            [{"page_index": 0, "image_path": "x.png", "width": 1, "height": 1}],
        )
        assert validate_jsonschema(doc) == []


# ------------------------------------------------------------------- VLM


class FakeProcessor:
    """Minimal stand-in for a transformers processor."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_images = None
        self.last_text = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for message in messages:
            for chunk in message["content"]:
                if chunk.get("type") == "text":
                    parts.append(chunk["text"])
        self.last_text = "\n".join(parts)
        return self.last_text

    def __call__(self, text=None, images=None, return_tensors=None, padding=None):
        import torch

        self.last_images = images
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

    def decode(self, tokens, skip_special_tokens=True):
        return self.response


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.generate_kwargs = None

    def generate(self, **kwargs):
        import torch

        self.generate_kwargs = kwargs

        class Output:
            sequences = torch.zeros((1, 8), dtype=torch.long)
            scores = None

        return Output()


torch = pytest.importorskip("torch", reason="VLM wrapper tests need torch tensors")


class TestVlmExtractor:
    def _extractor(self, response: str, **kwargs) -> VlmExtractor:
        return VlmExtractor(
            _model=FakeModel(), _processor=FakeProcessor(response), use_ocr_hint=False, **kwargs
        )

    def test_parses_a_well_formed_response(self, tmp_path):
        image = tmp_path / "page.png"
        _write_blank_png(image)
        extractor = self._extractor('{"totals": {"grand_total": 100}}')
        prediction = extractor.predict_verbose([image])
        assert isinstance(prediction, VlmPrediction)
        assert prediction.parsed == {"totals": {"grand_total": 100}}
        assert not prediction.parse_failed

    def test_a_malformed_response_is_a_result_not_an_exception(self, tmp_path):
        """A parse failure must score as a missed document, not abort a run
        halfway through a corpus."""
        image = tmp_path / "page.png"
        _write_blank_png(image)
        prediction = self._extractor("I cannot read this.").predict_verbose([image])
        assert prediction.parse_failed
        assert prediction.raw_response == "I cannot read this."
        assert self._extractor("I cannot read this.").predict([image]) == {}

    def test_predict_returns_raw_fields_without_coercing(self, tmp_path):
        """Coercion is the caller's job so every extractor goes through the
        same postprocess path -- otherwise ablation differences could come from
        wrappers rather than models."""
        image = tmp_path / "page.png"
        _write_blank_png(image)
        raw = self._extractor('{"invoice_meta": {"invoice_date": "12/01/2026"}}').predict([image])
        assert raw["invoice_meta"]["invoice_date"] == "12/01/2026"  # not yet ISO

    def test_the_system_prompt_and_template_reach_the_processor(self, tmp_path):
        image = tmp_path / "page.png"
        _write_blank_png(image)
        extractor = self._extractor('{"a": 1}')
        extractor.predict_verbose([image])
        sent = extractor._processor.last_text
        assert SYSTEM_PROMPT in sent
        assert '"line_items"' in sent

    def test_name_reflects_the_hybrid_flag(self):
        assert self._extractor("{}").name.endswith("vision-only")
        hybrid = VlmExtractor(_model=FakeModel(), _processor=FakeProcessor("{}"), use_ocr_hint=True)
        assert hybrid.name.endswith("+ocr")

    def test_unknown_preset_is_rejected(self):
        with pytest.raises(KeyError, match="unknown preset"):
            VlmExtractor(preset="not-a-model")

    def test_presets_follow_the_compute_plan_quantization_rule(self):
        """fp16 below 3B, 4-bit NF4 at or above -- see configs/compute_plan.md."""
        for name, preset in MODEL_PRESETS.items():
            if preset.params_billion >= 3.0:
                assert preset.four_bit, f"{name} should be 4-bit at {preset.params_billion}B"
            else:
                assert not preset.four_bit, f"{name} should be fp16 at {preset.params_billion}B"

    def test_default_preset_exists(self):
        assert DEFAULT_PRESET in MODEL_PRESETS


def _write_blank_png(path: Path) -> None:
    from PIL import Image

    Image.new("L", (32, 32), 255).save(path)


# -------------------------------------------------------------- registry


class TestRegistry:
    def test_builds_the_non_model_extractors(self):
        assert isinstance(get_extractor("rules"), RuleBasedExtractor)
        assert isinstance(get_extractor("null"), NullExtractor)

    def test_lists_rules_null_and_the_vlm_presets(self):
        names = available_extractors()
        assert "rules" in names and "null" in names
        assert any(n.startswith("vlm:") for n in names)

    def test_unknown_name_is_rejected(self):
        with pytest.raises(KeyError, match="unknown extractor"):
            get_extractor("gpt-9")

    def test_vlm_names_select_a_preset_without_loading_a_model(self):
        """Constructing must not download weights -- loading is deferred to the
        first generate() call."""
        extractor = get_extractor(f"vlm:{DEFAULT_PRESET}")
        assert extractor.preset_name == DEFAULT_PRESET
        assert extractor._model is None
