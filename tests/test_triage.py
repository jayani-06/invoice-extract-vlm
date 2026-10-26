"""The decision the Phase 1 ablation showed was missing.

Preprocessing raises word accuracy 14.1 points on `heavy` pages and *loses*
8.9 on `clean` ones, and every caller ran one fixed config over everything. So
the pipeline was paying the clean-page penalty on every clean page for no
reason, and the fix is a decision made from the page itself -- at inference
time nothing knows which degradation profile a scan came from.

What matters here is not that the signals are clever but that they are
*honest*. Two of them were measuring noise while claiming to measure something
else, and a third carried no information at all while confidently firing a
stage. None of those failed loudly; they produced plausible numbers. So the
tests pin direction, calibration and the retirement mechanism rather than
exact values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from invoice_extract.preprocess.pipeline import ADAPTIVE, make_pipeline
from invoice_extract.preprocess.triage import (
    DEFAULT_THRESHOLDS,
    AdaptivePipeline,
    assess,
    triage,
)

CORPUS = Path("data/processed/gst_in_synthetic")
pytestmark = pytest.mark.skipif(
    not (CORPUS / "manifest.jsonl").exists(), reason="image corpus not built"
)


def manifest_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in (CORPUS / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def pages() -> dict[str, list[np.ndarray]]:
    """A few pages per profile, as grayscale arrays."""
    from PIL import Image

    out: dict[str, list[np.ndarray]] = {}
    for row in manifest_rows():
        bucket = out.setdefault(row["profile"], [])
        if len(bucket) < 4:
            bucket.append(np.array(Image.open(CORPUS / row["image_path"]).convert("L")))
    return out


class TestSignalsPointTheRightWay:
    def test_a_clean_page_needs_nothing(self, pages):
        """The whole point. A clean page must come out of triage with no stage
        beyond the free grayscale conversion, or it pays the 8.9-point
        word-accuracy penalty the ablation measured."""
        decisions = [triage(img) for img in pages["clean"]]
        touched = [d for d in decisions if d.degraded]
        assert not touched, [d.explain() for d in touched]

    def test_a_heavy_page_needs_work(self, pages):
        for img in pages["heavy"]:
            decision = triage(img)
            assert decision.degraded, decision.condition.as_dict()

    def test_degradation_is_monotonic_across_profiles(self, pages):
        """More damage must not produce fewer stages. This is weaker than
        asserting specific stages and that is deliberate -- degradations are
        applied probabilistically, so an individual `medium` page may genuinely
        need less work than an individual `light` one. The *mean* must still
        order correctly."""
        means = {
            profile: np.mean([len(triage(img).stages) for img in imgs])
            for profile, imgs in pages.items()
        }
        assert means["clean"] <= means["light"] <= means["medium"]
        assert means["clean"] < means["heavy"]

    def test_blur_is_measured_on_a_denoised_page(self, pages):
        """Laplacian variance is the standard blur metric and grain is
        high-frequency energy too. Measured on the raw page, a `heavy` scan
        scored 9340 against 2408 for the same page clean -- so the gate ordered
        pages by noise, backwards, and never fired. It must now read lower on
        damaged pages, not higher."""
        clean = np.mean([assess(img).laplacian_variance for img in pages["clean"]])
        medium = np.mean([assess(img).laplacian_variance for img in pages["medium"]])
        assert medium < clean, f"blur signal inverted: clean={clean:.0f} medium={medium:.0f}"

    def test_noise_is_measured_on_background_not_ink(self, pages):
        """On ink the median-filter residual is edge energy. If the sample
        included it, every text-dense page would look noisy regardless of
        actual grain."""
        clean = np.mean([assess(img).noise_sigma for img in pages["clean"]])
        heavy = np.mean([assess(img).noise_sigma for img in pages["heavy"]])
        assert clean < 1.0, f"clean pages should read as essentially noiseless, got {clean:.2f}"
        assert heavy > clean


class TestRetiringASignal:
    def test_a_threshold_absent_means_the_gate_is_off(self, pages):
        """How a signal that fails calibration is retired. `text_height_px`
        separated the pages that were actually rescaled by 0.35 SD -- no
        information -- and on its own fired `upscale` on 19 of 30 clean pages.
        It is dropped by being absent from DEFAULT_THRESHOLDS, so that has to
        disable the gate rather than raise."""
        assert "text_height_px" not in DEFAULT_THRESHOLDS
        for img in pages["clean"]:
            assert "upscale" not in triage(img).stages

    def test_an_unmeasurable_signal_does_not_fire_a_stage(self):
        """A blank page has no connected components, so glyph height is NaN.
        NaN comparisons are false in Python, which is the right answer here but
        only by accident -- pinned so a later rewrite cannot turn it into a
        stage firing on no evidence."""
        blank = np.full((400, 300), 255, dtype=np.uint8)
        decision = triage(blank, {**DEFAULT_THRESHOLDS, "text_height_px": 11.0})
        assert decision.condition.text_height_px != decision.condition.text_height_px
        assert "upscale" not in decision.stages or "laplacian" in decision.reasons.get(
            "upscale", ""
        )

    def test_binarize_is_unreachable(self, pages):
        """It is the worst row in every CER column of the ablation, so no page
        condition should select it."""
        for imgs in pages.values():
            for img in imgs:
                assert "binarize" not in triage(img).stages


class TestCalibrationIsHonest:
    def test_thresholds_match_the_checked_in_calibration(self):
        """The defaults must be the fitted values, not hand-edited ones. A
        threshold nudged by hand to make a table look better is exactly the
        kind of thing this project is otherwise careful about."""
        report = Path("reports/phase1_triage/calibration.json")
        if not report.exists():
            pytest.skip("calibration not run")
        fitted = json.loads(report.read_text(encoding="utf-8"))["fitted_thresholds"]
        assert pytest.approx(fitted) == DEFAULT_THRESHOLDS

    def test_calibration_used_only_the_train_split(self):
        report = Path("reports/phase1_triage/calibration.json")
        if not report.exists():
            pytest.skip("calibration not run")
        assert json.loads(report.read_text(encoding="utf-8"))["split"] == "train"

    def test_every_live_gate_cleared_the_separation_floor(self):
        """A gate kept below the floor is worse than no gate, because it fires
        on evidence it cannot actually see."""
        report = Path("reports/phase1_triage/calibration.json")
        if not report.exists():
            pytest.skip("calibration not run")
        payload = json.loads(report.read_text(encoding="utf-8"))
        for signal in payload["signals"]:
            if signal["signal"] in DEFAULT_THRESHOLDS:
                assert signal["separation_sd"] >= 0.8, signal


class TestAdaptivePipeline:
    def test_it_substitutes_for_a_fixed_pipeline(self, pages):
        """Interface compatibility is the point: the ablation runner, the
        extraction runner and the demo cache all build pipelines by name, and
        none of them should need to know one option decides per page."""
        pipeline = make_pipeline(ADAPTIVE)
        assert isinstance(pipeline, AdaptivePipeline)

        result = pipeline(pages["medium"][0])
        assert result.image is not None
        assert result.homography.shape == (3, 3)
        assert pipeline.last_decision is not None

    def test_triage_cost_is_reported(self, pages):
        """A cost table that omitted triage would understate the adaptive
        pipeline against the fixed ones it is compared to."""
        pipeline = make_pipeline(ADAPTIVE)
        result = pipeline(pages["clean"][0])
        assert result.timings_ms["triage"] > 0
        assert result.total_ms >= result.timings_ms["triage"]

    def test_the_decision_is_serialisable_and_explained(self, pages):
        """A review UI has to be able to show *why* a page was altered."""
        pipeline = make_pipeline(ADAPTIVE)
        result = pipeline(pages["heavy"][0])
        json.dumps(result.stage_info["triage"])
        assert pipeline.last_decision.explain()

    def test_boxes_still_map_through_an_adaptive_run(self, pages):
        """Every geometric stage contributes to one homography and gold boxes
        follow the glyphs through it. A per-page stage list must not break
        that, or the ablation's detection metrics become meaningless."""
        pipeline = make_pipeline(ADAPTIVE)
        img = pages["heavy"][0]
        result = pipeline(img)
        h, w = img.shape
        mapped = result.map_boxes([(0.0, 0.0, float(w), float(h))])
        assert len(mapped) == 1
        assert all(v == v for v in mapped[0])
