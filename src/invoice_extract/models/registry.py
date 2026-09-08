"""Extractor registry.

Mirrors `invoice_extract.ocr.base`'s factory pattern for the same reason: the
runner should name an extractor as a string, and adding a model should not mean
editing the runner. Construction is lazy so importing this module never pulls
in torch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from invoice_extract.models.rules import NullExtractor, RuleBasedExtractor

_FACTORIES: dict[str, Callable[..., Any]] = {
    "null": NullExtractor,
    "rules": RuleBasedExtractor,
}


def register_extractor(name: str, factory: Callable[..., Any]) -> None:
    _FACTORIES[name] = factory


def available_extractors() -> list[str]:
    return sorted(_FACTORIES) + [f"vlm:{p}" for p in _vlm_presets()]


def _vlm_presets() -> list[str]:
    from invoice_extract.models.vlm import MODEL_PRESETS

    return sorted(MODEL_PRESETS)


def get_extractor(name: str, **kwargs) -> Any:
    """Construct an extractor by name.

    `vlm:<preset>` selects a VLM preset (see `vlm.MODEL_PRESETS`); a bare
    `vlm` uses the default preset from the compute plan.
    """
    if name in _FACTORIES:
        return _FACTORIES[name](**kwargs)

    if name == "vlm" or name.startswith("vlm:"):
        from invoice_extract.models.vlm import DEFAULT_PRESET, VlmExtractor

        preset = name.split(":", 1)[1] if ":" in name else DEFAULT_PRESET
        return VlmExtractor(preset=preset, **kwargs)

    raise KeyError(f"unknown extractor {name!r}; have {available_extractors()}")
