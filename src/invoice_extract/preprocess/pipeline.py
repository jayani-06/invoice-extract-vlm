"""Composable preprocessing pipeline plus the named configurations the
Phase 1 ablation sweeps over.

A `PipelineConfig` is a declarative description (which stages, with which
parameters) rather than a list of constructed objects, so configurations are
serialisable, hashable, and can be written straight into a results table --
which matters when the deliverable is an ablation you have to defend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from invoice_extract.preprocess.stages import (
    IDENTITY,
    Binarize,
    CorrectPerspective,
    Denoise,
    Deskew,
    NormalizeIllumination,
    RemoveScanBorder,
    StageResult,
    ToGrayscale,
    UpscaleForOcr,
)

Array = np.ndarray

# Canonical execution order. A config names a *subset* of these; Pipeline
# always runs them in this sequence, so a config cannot express a physically
# wrong ordering. Geometry before binarisation, illumination before denoising
# -- see the module docstring in stages.py for why.
DEFAULT_ORDER = [
    "grayscale",
    "remove_border",
    "correct_perspective",
    "deskew",
    "normalize_illumination",
    "denoise",
    "upscale",
    "binarize",
]

#: Perspective correction is deliberately *not* part of the default ladder.
#: It is the one stage that can catastrophically mis-fire (warping a whole
#: page off a stray rectangle), and the flatbed-scan case it targets is a
#: minority of the corpus, so it is evaluated as an opt-in variant instead.
FULL_STAGES = [s for s in DEFAULT_ORDER if s != "correct_perspective"]

_CONSTRUCTORS = {
    "grayscale": ToGrayscale,
    "remove_border": RemoveScanBorder,
    "correct_perspective": CorrectPerspective,
    "deskew": Deskew,
    "normalize_illumination": NormalizeIllumination,
    "denoise": Denoise,
    "upscale": UpscaleForOcr,
    "binarize": Binarize,
}


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    stages: tuple[str, ...]
    params: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...] = ()

    @classmethod
    def build(cls, name: str, stages: list[str], params: dict[str, dict[str, Any]] | None = None):
        params = params or {}
        return cls(
            name=name,
            stages=tuple(stages),
            params=tuple((k, tuple(sorted(v.items()))) for k, v in sorted(params.items())),
        )

    def param_dict(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self.params}

    def describe(self) -> str:
        p = self.param_dict()
        parts = []
        for s in self.stages:
            if s in p:
                inner = ",".join(f"{k}={v}" for k, v in p[s].items())
                parts.append(f"{s}({inner})")
            else:
                parts.append(s)
        return " -> ".join(parts)


@dataclass
class PipelineResult:
    image: Array
    homography: Array
    stage_info: dict[str, dict] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    def map_boxes(
        self, boxes: list[tuple[float, float, float, float]]
    ) -> list[tuple[float, float, float, float]]:
        """Map boxes from input space into preprocessed space."""
        from invoice_extract.degrade.transforms import transform_boxes

        return transform_boxes(boxes, self.homography)


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        params = config.param_dict()
        # Honour DEFAULT_ORDER regardless of the order stages were listed in, so
        # a config cannot accidentally specify a physically wrong sequence.
        ordered = [s for s in DEFAULT_ORDER if s in config.stages]
        unknown = set(config.stages) - set(DEFAULT_ORDER)
        if unknown:
            raise ValueError(f"unknown stage(s): {sorted(unknown)}")
        self.stages = [_CONSTRUCTORS[s](**params.get(s, {})) for s in ordered]

    def __call__(self, img: Array) -> PipelineResult:
        h_total = IDENTITY.copy()
        info: dict[str, dict] = {}
        timings: dict[str, float] = {}
        out = img

        for stage in self.stages:
            t0 = time.perf_counter()
            res: StageResult = stage(out)
            timings[stage.name] = (time.perf_counter() - t0) * 1000
            out = res.image
            h_total = res.homography @ h_total
            info[stage.name] = res.info

        return PipelineResult(image=out, homography=h_total, stage_info=info, timings_ms=timings)


# --------------------------------------------------------------- configs

#: The ablation ladder. Each rung adds exactly one capability to the one
#: before it, so the delta between adjacent rows attributes cleanly to that
#: stage -- the whole point of reporting an ablation rather than a single
#: end-to-end number.
ABLATION_CONFIGS: list[PipelineConfig] = [
    PipelineConfig.build("raw", []),
    PipelineConfig.build("gray", ["grayscale"]),
    PipelineConfig.build("gray+border", ["grayscale", "remove_border"]),
    PipelineConfig.build("gray+border+deskew", ["grayscale", "remove_border", "deskew"]),
    PipelineConfig.build(
        "gray+border+deskew+illum",
        ["grayscale", "remove_border", "deskew", "normalize_illumination"],
    ),
    PipelineConfig.build(
        "gray+border+deskew+illum+denoise",
        ["grayscale", "remove_border", "deskew", "normalize_illumination", "denoise"],
        {"denoise": {"mode": "median", "strength": 3}},
    ),
    PipelineConfig.build(
        "full(no-binarize)",
        ["grayscale", "remove_border", "deskew", "normalize_illumination", "denoise", "upscale"],
        {"denoise": {"mode": "median", "strength": 3}},
    ),
    PipelineConfig.build(
        "full+sauvola",
        FULL_STAGES,
        {"denoise": {"mode": "median", "strength": 3}, "binarize": {"method": "sauvola"}},
    ),
]

#: Single-axis variants, held against `full(no-binarize)` as the control, to
#: answer "which method for this stage" without re-running the whole ladder.
VARIANT_CONFIGS: list[PipelineConfig] = [
    PipelineConfig.build(
        "binarize=otsu",
        FULL_STAGES,
        {"denoise": {"mode": "median", "strength": 3}, "binarize": {"method": "otsu"}},
    ),
    PipelineConfig.build(
        "binarize=adaptive",
        FULL_STAGES,
        {"denoise": {"mode": "median", "strength": 3}, "binarize": {"method": "adaptive"}},
    ),
    PipelineConfig.build(
        "deskew=min_area_rect",
        ["grayscale", "remove_border", "deskew", "normalize_illumination", "denoise", "upscale"],
        {"deskew": {"method": "min_area_rect"}, "denoise": {"mode": "median", "strength": 3}},
    ),
    PipelineConfig.build(
        "denoise=nlm",
        ["grayscale", "remove_border", "deskew", "normalize_illumination", "denoise", "upscale"],
        {"denoise": {"mode": "nlm", "strength": 3}},
    ),
    PipelineConfig.build(
        "denoise=bilateral",
        ["grayscale", "remove_border", "deskew", "normalize_illumination", "denoise", "upscale"],
        {"denoise": {"mode": "bilateral", "strength": 3}},
    ),
    PipelineConfig.build(
        "with_perspective",
        DEFAULT_ORDER,
        {"denoise": {"mode": "median", "strength": 3}, "binarize": {"method": "sauvola"}},
    ),
]

CONFIGS_BY_NAME: dict[str, PipelineConfig] = {c.name: c for c in ABLATION_CONFIGS + VARIANT_CONFIGS}


def get_config(name: str) -> PipelineConfig:
    if name not in CONFIGS_BY_NAME:
        raise KeyError(f"unknown pipeline config {name!r}; have {sorted(CONFIGS_BY_NAME)}")
    return CONFIGS_BY_NAME[name]
