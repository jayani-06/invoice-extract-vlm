"""Decide whether a page needs preprocessing, and which parts of it.

The Phase 1 ablation produced a result the pipeline then ignored: the full
chain minus binarisation raises word accuracy by 14.1 points on `heavy` pages
and 6.4 on `medium`, and *loses* 8.9 points on `clean` ones (0.5733 raw versus
0.4842 preprocessed). Every caller nonetheless ran one fixed config over
everything, so clean pages were paying that penalty for no reason.

The missing piece is a decision, and it has to be made from the page itself:
at inference time nothing knows which degradation profile a scan came from, or
whether it was degraded at all.

So each stage is gated on a signal that stage actually addresses:

| Signal | Measured by | Enables |
|---|---|---|
| skew | projection-profile angle search | `deskew` |
| dark edge band | mean intensity of a thin border frame | `remove_border` |
| uneven lighting | spread of a coarse background estimate | `normalize_illumination` |
| grain / speckle | residual after a median filter, on background only | `denoise` |
| softness, small glyphs | Laplacian variance and estimated text height | `upscale` |

`binarize` is never enabled. It is the worst row in every CER column of the
ablation (PaddleOCR's detector is trained on continuous-tone images), so there
is no page condition under which this policy should reach for it.

**The thresholds are calibrated, not guessed** -- `scripts/calibrate_triage.py`
fits them on the *train* split of the paired corpus, where the true profile of
every page is known, and reports how well they separate. They are not tuned
against the evaluation split, and the calibration report is checked in so the
numbers can be audited.

What this does not do: it cannot recover a judgement the signals do not carry.
A page that is noisy *and* already sharp gets denoised either way, because
nothing here measures whether denoising will help that particular page -- only
whether the condition it targets is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from invoice_extract.preprocess.pipeline import PipelineConfig
from invoice_extract.preprocess.stages import _ensure_gray, _text_mask, estimate_skew_projection

Array = np.ndarray

#: Thresholds, in the units each signal is measured in. Defaults come from
#: reports/phase1_triage/calibration.json, fitted on the train split.
#:
#: Each is the midpoint between the clean distribution and the degraded one on
#: that signal, which is the decision boundary that minimises total error when
#: the two are roughly equally spread. A signal whose distributions overlap
#: heavily gets a threshold anyway -- the calibration report says which those
#: are, and for those the gate is close to a coin flip and should be read as
#: "no information" rather than as a judgement.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "skew_deg": 0.5417,
    "border_darkness": 0.2937,
    "illumination_spread": 0.1475,
    "noise_sigma": 2.081,
    "laplacian_variance": 399.6,
}


#: The configuration a page gets when nothing fires. Measured best on clean
#: pages: 0.5733 word accuracy, against 0.4842 for the full chain -- the
#: 8.9-point penalty this whole module exists to avoid paying.
CLEAN_CONFIG = "raw"

#: And when anything fires. Measured best on both degraded profiles: 0.4918 on
#: `medium` and 0.3062 on `heavy`, against `raw`'s 0.4281 and 0.1651.
#:
#: `full+sauvola` is deliberately not reachable. It is the worst row in every
#: CER column measured, because PaddleOCR's detector is trained on
#: continuous-tone images and binarisation discards what it uses.
DEGRADED_CONFIG = "full(no-binarize)"


@dataclass
class PageCondition:
    """What the page looks like, in the units the thresholds are set in."""

    skew_deg: float
    border_darkness: float
    illumination_spread: float
    noise_sigma: float
    laplacian_variance: float
    text_height_px: float
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "skew_deg": self.skew_deg,
            "border_darkness": self.border_darkness,
            "illumination_spread": self.illumination_spread,
            "noise_sigma": self.noise_sigma,
            "laplacian_variance": self.laplacian_variance,
            "text_height_px": self.text_height_px,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class TriageDecision:
    """Which stages to run, and the evidence for each."""

    condition: PageCondition
    stages: tuple[str, ...]
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def degraded(self) -> bool:
        """True when any signal says this page needs work."""
        return bool(self.reasons)

    @property
    def chosen(self) -> str:
        """The configuration this page gets.

        Deliberately one of two **measured** configurations rather than the set
        of stages whose gates happened to fire. `ABLATION_CONFIGS` is a
        cumulative ladder and the +14 points of word accuracy belongs to the
        whole chain; an independently-composed subset of a ladder gets less
        than the ladder. Composing stages scored 0.4432 on `medium` where the
        full chain scored 0.4918, and no threshold adjustment fixes that,
        because the shortfall is structural rather than a mis-set gate.
        """
        return DEGRADED_CONFIG if self.degraded else CLEAN_CONFIG

    def config(self, name: str | None = None) -> PipelineConfig:
        from invoice_extract.preprocess.pipeline import get_config

        chosen = get_config(self.chosen)
        if name is None:
            return chosen
        return PipelineConfig.build(name, list(chosen.stages), chosen.param_dict())

    def explain(self) -> str:
        if not self.reasons:
            return f"page looks clean; applying {CLEAN_CONFIG}"
        why = "; ".join(f"{stage}: {reason}" for stage, reason in sorted(self.reasons.items()))
        return f"applying {DEGRADED_CONFIG} because {why}"

    def as_dict(self) -> dict:
        return {
            "condition": self.condition.as_dict(),
            "chosen_config": self.chosen,
            "stages": list(self.config().stages),
            "evidence": dict(self.reasons),
            "degraded": self.degraded,
        }


def _background_estimate(gray: Array, grid: int = 16) -> Array:
    """Coarse per-tile background level, ink excluded.

    The 90th percentile per tile rather than the mean: a tile that happens to
    contain a dense table would otherwise read as "dark", and uneven lighting
    would be confused with uneven ink coverage.
    """
    h, w = gray.shape
    th, tw = max(1, h // grid), max(1, w // grid)
    tiles = []
    for y in range(0, h - th + 1, th):
        row = [
            float(np.percentile(gray[y : y + th, x : x + tw], 90)) for x in range(0, w - tw + 1, tw)
        ]
        if row:
            tiles.append(row)
    return np.asarray(tiles, dtype=np.float64) if tiles else np.asarray([[float(gray.mean())]])


def _text_height(mask: Array) -> float:
    """Median height of connected ink components, in pixels.

    Stands in for effective resolution. A page downscaled to 40% has the same
    skew and the same noise as the original but glyphs too small for the
    recogniser, and that is what `upscale` is for. Components outside a
    plausible glyph range are dropped so rules and table borders do not
    dominate the median.
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return float("nan")
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
    keep = (heights >= 3) & (heights <= 80) & (widths >= 2) & (widths <= 80)
    if not keep.any():
        return float("nan")
    return float(np.median(heights[keep]))


def assess(img: Array) -> PageCondition:
    """Measure the five page-condition signals. No reference image needed."""
    import time

    t0 = time.perf_counter()
    gray = _ensure_gray(img)
    mask = _text_mask(gray)

    # Skew: the same projection-profile search `Deskew` uses, so the gate and
    # the stage cannot disagree about whether a page is straight.
    skew = abs(estimate_skew_projection(mask, limit=8.0))

    # Border: a thin frame around the edge. A scan border is a dark band, so a
    # frame much darker than the page interior is the signal.
    h, w = gray.shape
    band = max(2, min(h, w) // 50)
    frame = np.concatenate(
        [
            gray[:band, :].ravel(),
            gray[-band:, :].ravel(),
            gray[:, :band].ravel(),
            gray[:, -band:].ravel(),
        ]
    )
    interior = gray[band:-band, band:-band] if h > 2 * band and w > 2 * band else gray
    interior_level = float(np.percentile(interior, 90)) or 1.0
    border_darkness = max(0.0, 1.0 - float(frame.mean()) / interior_level)

    # Illumination: how much the background level moves across the page,
    # normalised so it is a fraction rather than a grey level.
    bg = _background_estimate(gray)
    illumination_spread = float(np.ptp(bg)) / max(float(bg.mean()), 1.0)

    # Noise: residual after a median filter, measured on background only. On
    # ink the residual is edge energy, not grain, and including it would make
    # every text-dense page look noisy.
    blurred = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.int16) - blurred.astype(np.int16)
    background = mask == 0
    sample = residual[background] if background.any() else residual
    # 1.4826 * MAD estimates sigma for Gaussian noise while ignoring the
    # outliers that salt-and-pepper speckle would otherwise contribute.
    noise_sigma = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))

    # Both remaining signals are measured on a denoised copy, because on the
    # raw page both of them measure noise instead of what they claim to:
    #
    # - Laplacian variance is the standard blur metric, and grain is
    #   high-frequency energy too. Measured raw, a `heavy` page scored 9340
    #   against 2408 for the same page clean -- so the "is this soft" gate was
    #   ordering pages by how noisy they were, backwards, and never fired.
    # - Connected-component height is the resolution proxy, and speckle
    #   contributes thousands of 3-pixel components. Measured raw, the median
    #   glyph height on `heavy` read 3.0px against 22.0px clean, which fired
    #   `upscale` for the right page for entirely the wrong reason.
    #
    # A 3x median filter removes the speckle and leaves glyph strokes intact at
    # 200 dpi, which is enough to make both signals mean what they say.
    denoised = cv2.medianBlur(gray, 3)
    laplacian_variance = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    text_height_px = _text_height(_text_mask(denoised))

    return PageCondition(
        skew_deg=skew,
        border_darkness=border_darkness,
        illumination_spread=illumination_spread,
        noise_sigma=noise_sigma,
        laplacian_variance=laplacian_variance,
        text_height_px=text_height_px,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
    )


def triage(img: Array, thresholds: dict[str, float] | None = None) -> TriageDecision:
    """Measure the page, then choose the stages its condition calls for.

    A signal absent from `thresholds` has its gate **disabled**, which is how a
    signal that fails calibration is retired. `text_height_px` is the live
    example: it was meant to detect downscaling, separates the pages that were
    actually rescaled from the rest by 0.35 SD, and before it was dropped it
    fired `upscale` on 19 of 30 clean pages on its own.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    c = assess(img)

    def gate(signal: str, value: float, higher_is_worse: bool) -> bool:
        limit = t.get(signal)
        if limit is None or value != value:  # retired gate, or unmeasurable
            return False
        return value > limit if higher_is_worse else value < limit

    # `stages` records which stages the evidence points at, for the review UI
    # and the calibration report. It is no longer what builds the pipeline --
    # see TriageDecision.chosen.
    stages = ["grayscale"]
    reasons: dict[str, str] = {}

    if gate("border_darkness", c.border_darkness, True):
        stages.append("remove_border")
        reasons["remove_border"] = (
            f"edge band {c.border_darkness:.3f} darker than interior " f"(> {t['border_darkness']})"
        )

    if gate("skew_deg", c.skew_deg, True):
        stages.append("deskew")
        reasons["deskew"] = f"skew {c.skew_deg:.2f} deg (> {t['skew_deg']})"

    if gate("illumination_spread", c.illumination_spread, True):
        stages.append("normalize_illumination")
        reasons["normalize_illumination"] = (
            f"background varies {c.illumination_spread:.3f} across the page "
            f"(> {t['illumination_spread']})"
        )

    if gate("noise_sigma", c.noise_sigma, True):
        stages.append("denoise")
        reasons["denoise"] = f"background noise sigma {c.noise_sigma:.2f} (> {t['noise_sigma']})"

    soft = gate("laplacian_variance", c.laplacian_variance, False)
    small = gate("text_height_px", c.text_height_px, False)
    if soft or small:
        stages.append("upscale")
        why = []
        if soft:
            why.append(
                f"laplacian variance {c.laplacian_variance:.0f} (< {t['laplacian_variance']})"
            )
        if small:
            why.append(f"median glyph height {c.text_height_px:.1f}px (< {t['text_height_px']})")
        reasons["upscale"] = " and ".join(why)

    # `binarize` is deliberately unreachable. See the module docstring: it is
    # the worst row in every CER column of the Phase 1 ablation, so no page
    # condition should select it.

    return TriageDecision(condition=c, stages=tuple(stages), reasons=reasons)


class AdaptivePipeline:
    """Triage each page, then run only the stages it needs.

    Interface-compatible with `Pipeline.__call__`, so it drops into the
    ablation runner, the extraction runner and the demo cache builder without
    any of them knowing the config is now per page. `last_decision` carries the
    evidence for the page just processed, which is what a review UI needs in
    order to show *why* an image was altered.
    """

    name = "adaptive"

    def __init__(self, thresholds: dict[str, float] | None = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.last_decision: TriageDecision | None = None

    def __call__(self, img: Array):
        from invoice_extract.preprocess.pipeline import Pipeline

        decision = triage(img, self.thresholds)
        self.last_decision = decision
        # One of two measured configurations, chosen by whether the page shows
        # any sign of degradation.
        result = Pipeline(decision.config())(img)
        # Triage is not free, and a cost table that omitted it would understate
        # the adaptive pipeline against the fixed ones it is compared to.
        result.timings_ms["triage"] = decision.condition.elapsed_ms
        result.stage_info["triage"] = decision.as_dict()
        return result
