"""Vision-Language Model extractor.

Torch, transformers and bitsandbytes are optional dependencies (`pip install
-e ".[vlm]"`). Importing this module never requires them, so the schema,
dataset, rules-baseline and evaluation work all stay runnable on a laptop with
no GPU -- which is the actual development environment for this project, with
model inference happening on Colab/Kaggle T4s.

Model choice follows `configs/compute_plan.md`: Qwen2-VL-2B-Instruct in fp16
is the recommended first baseline (it fits a T4 comfortably and avoids
quantization error on a small model), with 4-bit NF4 reserved for anything
>= 3B. The preset table below encodes that decision rather than leaving it to
be re-litigated in each notebook.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from invoice_extract.models.prompting import (
    SYSTEM_PROMPT,
    build_prompt,
    extract_json_object,
)

INSTALL_HINT = (
    "VLM inference needs the optional 'vlm' extra:\n"
    "  pip install -e \".[vlm]\"\n"
    "On a T4 (Colab/Kaggle) this pulls torch, transformers, accelerate and "
    "bitsandbytes. See configs/compute_plan.md for the memory budget."
)


class VlmUnavailableError(RuntimeError):
    """torch/transformers are not installed, or no GPU is available."""


@dataclass(frozen=True)
class ModelPreset:
    """A model and how to run it on the project's target hardware."""

    model_id: str
    params_billion: float
    four_bit: bool
    max_new_tokens: int = 1536
    notes: str = ""


#: Presets encode the compute-plan decision: fp16 below 3B, 4-bit NF4 at or
#: above it. `estimate_weight_memory_gb` in utils/quantization.py can
#: sanity-check any addition here before anything is downloaded.
MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen2-vl-2b": ModelPreset(
        "Qwen/Qwen2-VL-2B-Instruct", 2.0, four_bit=False,
        notes="Recommended first baseline per configs/compute_plan.md; fp16 fits a T4.",
    ),
    "qwen2.5-vl-3b": ModelPreset(
        "Qwen/Qwen2.5-VL-3B-Instruct", 3.0, four_bit=True,
        notes="Newer generation, stronger document grounding; 4-bit at 3B per the plan.",
    ),
    "qwen2-vl-7b": ModelPreset(
        "Qwen/Qwen2-VL-7B-Instruct", 7.0, four_bit=True,
        notes="Try only if the 2B ceiling is clearly the limiter; 2-3x slower per document.",
    ),
    "internvl2-2b": ModelPreset(
        "OpenGVLab/InternVL2-2B", 2.0, four_bit=False,
        notes="Alternative to Qwen at the same size; useful as a second data point.",
    ),
    "florence-2-large": ModelPreset(
        "microsoft/Florence-2-large", 0.77, four_bit=False, max_new_tokens=1024,
        notes="Very light, strong OCR grounding, weaker instruction following.",
    ),
}

DEFAULT_PRESET = "qwen2-vl-2b"


@dataclass
class VlmPrediction:
    """One model call: the parsed object plus everything needed to debug it."""

    raw_response: str
    parsed: dict[str, Any] | None
    elapsed_ms: float
    field_confidence: dict[str, float] = field(default_factory=dict)
    mean_logprob: float | None = None

    @property
    def parse_failed(self) -> bool:
        return self.parsed is None


class VlmExtractor:
    """Prompted VLM extraction, optionally with an OCR text hint.

    `use_ocr_hint` is a flag rather than a hard-coded choice because whether
    hybrid input beats vision-only is exactly what the Phase 2 ablation is
    meant to measure on our data. Refs [3] and [7] predict it wins; predicting
    is not measuring.
    """

    def __init__(
        self,
        preset: str = DEFAULT_PRESET,
        *,
        model_id: str | None = None,
        four_bit: bool | None = None,
        use_ocr_hint: bool = True,
        ocr_engine: Any = None,
        preprocess: Any = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        constrained: bool = True,
        _model: Any = None,
        _processor: Any = None,
    ) -> None:
        if preset not in MODEL_PRESETS and model_id is None:
            raise KeyError(f"unknown preset {preset!r}; have {sorted(MODEL_PRESETS)}")

        base = MODEL_PRESETS.get(preset)
        self.preset_name = preset
        self.model_id = model_id or base.model_id
        self.four_bit = base.four_bit if four_bit is None else four_bit
        self.max_new_tokens = max_new_tokens or (base.max_new_tokens if base else 1536)
        self.temperature = temperature
        self.use_ocr_hint = use_ocr_hint
        self.ocr_engine = ocr_engine
        self.preprocess = preprocess
        self.constrained = constrained

        # Injectable for tests: a fake model/processor pair exercises the whole
        # prompt -> response -> parse -> coerce path with no GPU.
        self._model = _model
        self._processor = _processor

    @property
    def name(self) -> str:
        suffix = "+ocr" if self.use_ocr_hint else "+vision-only"
        return f"vlm:{self.preset_name}{suffix}"

    # ------------------------------------------------------------- loading

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from invoice_extract.utils.quantization import load_quantized_vlm
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise VlmUnavailableError(INSTALL_HINT) from exc
        try:
            self._model, self._processor = load_quantized_vlm(self.model_id, self.four_bit)
        except ImportError as exc:
            raise VlmUnavailableError(f"{exc}\n\n{INSTALL_HINT}") from exc

    # ---------------------------------------------------------------- OCR

    def _ocr_text(self, image_paths: list[Path]) -> str | None:
        if not self.use_ocr_hint or self.ocr_engine is None:
            return None
        import numpy as np
        from PIL import Image

        texts = []
        for path in image_paths:
            image = np.array(Image.open(path).convert("L"))
            if self.preprocess is not None:
                image = self.preprocess(image).image
            texts.append(self.ocr_engine.recognize(image).text)
        return "\n".join(t for t in texts if t) or None

    # ----------------------------------------------------------- inference

    def generate(self, image_paths: list[Path], ocr_text: str | None = None) -> VlmPrediction:
        """Run the model and parse its response. Never raises on a bad response.

        A parse failure returns `parsed=None` with the raw text retained. That
        is deliberate: a malformed response is a *result* the ablation must
        count (it scores as a fully missed document), not an exception that
        aborts a run halfway through a corpus.
        """
        self._ensure_loaded()
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in image_paths]
        prompt = build_prompt(ocr_text)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
            },
        ]

        t0 = time.perf_counter()
        response, confidence, mean_logprob = self._generate_once(messages, images)
        elapsed = (time.perf_counter() - t0) * 1000

        return VlmPrediction(
            raw_response=response,
            parsed=extract_json_object(response),
            elapsed_ms=elapsed,
            field_confidence=confidence,
            mean_logprob=mean_logprob,
        )

    def _generate_once(self, messages, images) -> tuple[str, dict[str, float], float | None]:
        import torch

        processor, model = self._processor, self._model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                return_dict_in_generate=True,
                output_scores=True,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output.sequences[0][prompt_len:]
        response = processor.decode(generated, skip_special_tokens=True)

        confidence, mean_logprob = self._token_confidence(output, generated, processor)
        return response, confidence, mean_logprob

    def _token_confidence(self, output, generated, processor) -> tuple[dict[str, float], float | None]:
        """Per-field confidence from generation log-probabilities.

        Confidence is averaged over the tokens that produced each JSON *value*,
        which is what Phase 5's review UI needs: knowing the model was unsure
        about `grand_total` specifically is actionable, whereas one number for
        the whole document is not.
        """
        import torch

        scores = getattr(output, "scores", None)
        if not scores:
            return {}, None

        logprobs = []
        for step, token_id in enumerate(generated):
            if step >= len(scores):
                break
            step_logprobs = torch.log_softmax(scores[step][0].float(), dim=-1)
            logprobs.append(float(step_logprobs[token_id]))
        if not logprobs:
            return {}, None

        # Walk the decoded text alongside the tokens, attributing each token to
        # whichever JSON key was most recently opened.
        pieces = [processor.decode([t], skip_special_tokens=True) for t in generated]
        confidence: dict[str, list[float]] = {}
        current_key: str | None = None
        buffer = ""
        in_value = False

        for piece, logprob in zip(pieces, logprobs, strict=False):
            buffer += piece
            if '"' in piece and not in_value:
                # Track the most recent quoted token as a candidate key.
                parts = buffer.split('"')
                if len(parts) >= 2:
                    current_key = parts[-2].strip() or current_key
            if ":" in piece:
                in_value = True
                buffer = ""
                continue
            if piece.strip() in {",", "}", "]"}:
                in_value = False
                buffer = ""
                continue
            if in_value and current_key:
                confidence.setdefault(current_key, []).append(logprob)

        import math

        return (
            {k: math.exp(sum(v) / len(v)) for k, v in confidence.items() if v},
            sum(logprobs) / len(logprobs),
        )

    # ------------------------------------------------------------ protocol

    def predict(self, image_paths: list[Path]) -> dict[str, Any]:
        """Return raw extracted fields. Post-processing is the caller's job.

        Keeping `predict` free of coercion means the same
        `postprocess.finalize` runs over every extractor's output -- so a
        difference in the ablation table is a difference in the model, not in
        how carefully each wrapper cleaned up after itself.
        """
        prediction = self.predict_verbose(image_paths)
        return prediction.parsed or {}

    def predict_verbose(self, image_paths: list[Path]) -> VlmPrediction:
        return self.generate(image_paths, ocr_text=self._ocr_text(image_paths))
