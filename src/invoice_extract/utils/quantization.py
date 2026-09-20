"""4-bit (NF4) model loading helper for T4-class GPUs.

Kept isolated from any specific model's inference code so the
quantization decision (see configs/compute_plan.md) is a one-line switch,
not something re-implemented per notebook.

Requires the optional `vlm` extra (`pip install -e ".[vlm]"`) — torch/
transformers/bitsandbytes are not core dependencies since schema/dataset/
eval work doesn't need them.
"""

from __future__ import annotations

from typing import Any


def load_quantized_vlm(model_id: str, four_bit: bool = True) -> tuple[Any, Any]:
    """Load a HF VLM checkpoint, 4-bit NF4 quantized by default.

    Returns (model, processor). Set four_bit=False for models <3B params
    where fp16 already fits comfortably on a T4 (see configs/compute_plan.md
    for the size threshold and rationale).
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    # `AutoModelForVision2Seq` was renamed to `AutoModelForImageTextToText` and
    # then removed in transformers v5. Probing keeps this working across both
    # generations -- pinning either name breaks on the other, and the failure
    # only surfaces on a GPU box, which is the worst place to discover it.
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # transformers < 4.46
        from transformers import AutoModelForVision2Seq as AutoVLM

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if four_bit
        else None
    )

    model = AutoVLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def estimate_weight_memory_gb(num_params_billion: float, four_bit: bool) -> float:
    """Rough weight-only memory estimate in GB, for sanity-checking a model
    choice against the T4's 16GB budget before downloading anything."""
    bytes_per_param = (
        0.5 if four_bit else 2.0
    )  # NF4 ~4 bits/param (+overhead), fp16 = 2 bytes/param
    return round(num_params_billion * 1e9 * bytes_per_param / (1024**3), 2)
