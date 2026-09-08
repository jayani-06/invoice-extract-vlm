"""Tests for the OpenCV preprocessing stages and the ablation pipeline.

Two things are worth testing here and one thing is not. Worth testing: that
each stage does what its name claims on an input where the effect is
measurable (deskew recovers a *known* injected angle; illumination
normalisation actually flattens a gradient), and that the composed homography
survives the full render -> degrade -> preprocess chain so gold boxes still
land on ink. Not worth testing: exact pixel values, which are OpenCV version
detail and would make the suite brittle without catching real defects.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from invoice_extract.degrade.transforms import MEDIUM, PROFILES, degrade, rotate
from invoice_extract.preprocess.pipeline import (
    ABLATION_CONFIGS,
    CONFIGS_BY_NAME,
    DEFAULT_ORDER,
    FULL_STAGES,
    Pipeline,
    PipelineConfig,
    get_config,
)
from invoice_extract.preprocess.stages import (
    IDENTITY,
    Binarize,
    Denoise,
    Deskew,
    NormalizeIllumination,
    RemoveScanBorder,
    ToGrayscale,
    UpscaleForOcr,
    _text_mask,
    estimate_skew_min_area_rect,
    estimate_skew_projection,
)
from invoice_extract.render.invoice_renderer import InvoiceRenderer


@pytest.fixture(scope="module")
def page():
    doc = {
        "doc_id": "preproc_fixture",
        "document_type": "invoice",
        "parties": {
            "vendor": {"name": "Meridian Supplies", "address": "88 Residency Road, Bengaluru"},
            "buyer": {"name": "Client Industries", "address": "3 Park Street, Kolkata"},
        },
        "invoice_meta": {
            "invoice_number": "INV-2200",
            "invoice_date": "2026-02-14",
            "due_date": "2026-03-14",
        },
        "line_items": [
            {
                "description": "Steel Fastener Set",
                "quantity": 10,
                "unit_price": 90.0,
                "tax_rate": 18,
                "line_total": 900.0,
            },
            {
                "description": "Industrial Adhesive",
                "quantity": 3,
                "unit_price": 410.0,
                "tax_rate": 18,
                "line_total": 1230.0,
            },
            {
                "description": "Safety Gloves",
                "quantity": 20,
                "unit_price": 45.0,
                "tax_rate": 12,
                "line_total": 900.0,
            },
        ],
        "tax_lines": [
            {"type": "CGST", "rate": 9, "amount": 274.5},
            {"type": "SGST", "rate": 9, "amount": 274.5},
        ],
        "totals": {"subtotal": 3030.0, "grand_total": 3579.0},
    }
    return InvoiceRenderer(dpi=150).render(doc, seed=17)


def ink_fraction(image: np.ndarray, bbox, threshold: int = 170) -> float:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = image[y0:y1, x0:x1]
    return float((patch < threshold).mean()) if patch.size else 0.0


class TestSkewEstimation:
    @pytest.mark.parametrize("angle", [-4.0, -2.0, -0.8, 0.8, 2.0, 4.0])
    def test_projection_profile_recovers_a_known_angle(self, page, angle):
        """The strongest available check: we injected the skew, so we know the
        answer, and the estimator must return its negation to within a
        fraction of a degree."""
        skewed, _ = rotate(np.array(page.image), angle)
        estimated = estimate_skew_projection(_text_mask(skewed))
        assert estimated == pytest.approx(
            -angle, abs=0.35
        ), f"injected {angle}, estimated {estimated}"

    def test_projection_profile_leaves_a_level_page_alone(self, page):
        estimated = estimate_skew_projection(_text_mask(np.array(page.image)))
        assert abs(estimated) < 0.35

    def test_a_scan_border_defeats_deskew_unless_removed_first(self, page):
        """This is why `remove_border` precedes `deskew` in DEFAULT_ORDER.

        A scanner border is added to an already-skewed page, so its edges are
        axis-aligned while the text is not. Those full-width edges contribute
        an enormous spike to the projection profile, which is maximised at 0
        degrees -- so the estimator locks onto the border and reports no skew.
        Removing the border first is not a tidiness step; without it the
        geometry stage silently does nothing.
        """
        angle = 3.0
        skewed, _ = rotate(np.array(page.image), angle)
        bordered = skewed.copy()
        bordered[:40, :] = 0
        bordered[-40:, :] = 0
        bordered[:, :40] = 0
        bordered[:, -40:] = 0

        with_border = estimate_skew_projection(_text_mask(bordered))
        assert abs(with_border - (-angle)) > 1.0, "expected the border to mislead the estimator"

        cleaned = RemoveScanBorder()(bordered).image
        recovered = estimate_skew_projection(_text_mask(cleaned))
        assert recovered == pytest.approx(
            -angle, abs=0.5
        ), f"border removal should restore deskew: got {recovered}, want {-angle}"

    def test_min_area_rect_is_less_accurate_than_projection_on_clean_text(self, page):
        """The default choice, justified rather than asserted: on a plain
        skewed page with no border, fitting a rectangle to the whole ink mask
        is measurably worse than maximising the projection profile."""
        angle = 3.0
        skewed, _ = rotate(np.array(page.image), angle)
        mask = _text_mask(skewed)

        projection_err = abs(estimate_skew_projection(mask) - (-angle))
        min_rect_err = abs(estimate_skew_min_area_rect(mask) - (-angle))
        assert projection_err < 0.35
        assert projection_err < min_rect_err

    def test_min_area_rect_handles_an_empty_mask(self):
        assert estimate_skew_min_area_rect(np.zeros((50, 50), dtype=np.uint8)) == 0.0


class TestStages:
    def test_grayscale_passes_through_and_converts(self, page):
        gray = np.array(page.image)
        assert ToGrayscale()(gray).image.ndim == 2
        colour = np.stack([gray] * 3, axis=-1)
        assert ToGrayscale()(colour).image.shape == gray.shape

    def test_remove_border_whitens_without_moving_pixels(self, page):
        img = np.array(page.image)
        bordered = img.copy()
        bordered[:30, :] = 0
        result = RemoveScanBorder()(bordered)
        assert np.allclose(result.homography, IDENTITY), "border removal must not shift coordinates"
        assert result.image[:30, :].mean() > 200
        assert result.info["trimmed"].get("top", 0) > 0

    def test_remove_border_leaves_a_clean_page_untouched(self, page):
        img = np.array(page.image)
        result = RemoveScanBorder()(img)
        assert result.info["trimmed"] == {}
        assert np.array_equal(result.image, img)

    def test_normalize_illumination_flattens_a_gradient(self, page):
        img = np.array(page.image)
        h, w = img.shape
        gradient = np.linspace(0.45, 1.0, w, dtype=np.float32)[None, :]
        lit = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

        def background_spread(a: np.ndarray) -> float:
            # Compare the bright (paper) end of each half; the text is unchanged.
            left = np.percentile(a[:, : w // 2], 90)
            right = np.percentile(a[:, w // 2 :], 90)
            return abs(float(left) - float(right))

        before = background_spread(lit)
        after = background_spread(NormalizeIllumination()(lit).image)
        assert after < before / 2, f"illumination spread {before:.1f} -> {after:.1f}"

    @pytest.mark.parametrize("mode", ["median", "nlm", "bilateral", "none"])
    def test_denoise_modes_run_and_preserve_geometry(self, page, mode):
        img = np.array(page.image)
        result = Denoise(mode=mode)(img)
        assert result.image.shape == img.shape
        assert np.allclose(result.homography, IDENTITY)

    def test_denoise_reduces_salt_and_pepper(self, page):
        img = np.array(page.image)
        rng = np.random.default_rng(0)
        noisy = img.copy()
        mask = rng.random(img.shape)
        noisy[mask < 0.02] = 0
        noisy[mask > 0.98] = 255

        def speckle(a):
            return float(np.abs(a.astype(np.int16) - img.astype(np.int16)).mean())

        assert speckle(Denoise(mode="median")(noisy).image) < speckle(noisy)

    def test_denoise_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown denoise mode"):
            Denoise(mode="wavelet")

    def test_deskew_corrects_and_reports_the_angle(self, page):
        img = np.array(page.image)
        skewed, _ = rotate(img, 3.0)
        result = Deskew()(skewed)
        assert result.info["angle"] == pytest.approx(-3.0, abs=0.35)
        # After correction the page should be level again.
        assert abs(estimate_skew_projection(_text_mask(result.image))) < 0.4

    def test_deskew_is_a_no_op_below_the_threshold(self, page):
        result = Deskew(min_angle=1.5)(np.array(page.image))
        assert result.info["angle"] == 0.0
        assert np.allclose(result.homography, IDENTITY)

    def test_deskew_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="unknown deskew method"):
            Deskew(method="hough")

    def test_upscale_raises_small_pages_and_reports_scale(self):
        small = np.full((400, 300), 255, dtype=np.uint8)
        small[100:120, 50:250] = 0
        result = UpscaleForOcr(target_min_dim=900)(small)
        assert result.image.shape[1] > 300
        assert result.info["factor"] > 1
        assert result.homography[0, 0] == pytest.approx(result.info["factor"])

    def test_upscale_leaves_large_pages_alone(self, page):
        img = np.array(page.image)
        result = UpscaleForOcr(target_min_dim=100)(img)
        assert result.info["factor"] == 1.0
        assert np.allclose(result.homography, IDENTITY)

    @pytest.mark.parametrize("method", ["otsu", "adaptive", "sauvola"])
    def test_binarize_produces_two_levels(self, page, method):
        out = Binarize(method=method)(np.array(page.image)).image
        assert set(np.unique(out)).issubset({0, 255})

    def test_binarize_none_keeps_grayscale(self, page):
        img = np.array(page.image)
        assert np.array_equal(Binarize(method="none")(img).image, img)

    def test_binarize_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="unknown binarize method"):
            Binarize(method="niblack")

    def test_sauvola_survives_a_shadow_that_defeats_otsu(self, page):
        """The textbook case for local thresholding: under a hard shadow, a
        single global threshold sacrifices a whole region."""
        img = np.array(page.image).astype(np.float32)
        h, w = img.shape
        img[:, : w // 2] *= 0.4  # heavy shade over the left half
        shaded = img.astype(np.uint8)

        def ink_ratio(a, region):
            return float((a[:, region] < 128).mean())

        otsu = Binarize(method="otsu")(shaded).image
        sauvola = Binarize(method="sauvola")(shaded).image
        left = slice(0, w // 2)
        # Otsu turns the shaded half almost entirely to ink; Sauvola should not.
        assert ink_ratio(otsu, left) > ink_ratio(sauvola, left) * 2


class TestPipeline:
    def test_stages_run_in_canonical_order_regardless_of_config_order(self):
        cfg = PipelineConfig.build("scrambled", ["binarize", "deskew", "grayscale"])
        names = [s.name for s in Pipeline(cfg).stages]
        assert names == ["grayscale", "deskew", "binarize"]

    def test_unknown_stage_is_rejected(self):
        with pytest.raises(ValueError, match="unknown stage"):
            Pipeline(PipelineConfig.build("bad", ["grayscale", "sharpen"]))

    def test_empty_pipeline_is_an_identity(self, page):
        img = np.array(page.image)
        result = Pipeline(get_config("raw"))(img)
        assert np.array_equal(result.image, img)
        assert np.allclose(result.homography, IDENTITY)
        assert result.total_ms == 0

    def test_config_params_reach_the_stages(self):
        cfg = get_config("deskew=min_area_rect")
        deskew = next(s for s in Pipeline(cfg).stages if s.name == "deskew")
        assert deskew.method == "min_area_rect"

    def test_full_stages_excludes_perspective(self):
        assert "correct_perspective" not in FULL_STAGES
        assert "correct_perspective" in DEFAULT_ORDER
        assert "correct_perspective" in get_config("with_perspective").stages

    def test_every_named_config_builds_and_runs(self, page):
        img = np.array(page.image)
        for name, cfg in CONFIGS_BY_NAME.items():
            result = Pipeline(cfg)(img)
            assert result.image.ndim == 2, name
            assert result.image.size > 0, name

    def test_ablation_ladder_is_strictly_cumulative(self):
        """Each rung must add stages without removing any, or the row-to-row
        delta cannot be attributed to the stage named in the row."""
        ladder = [set(c.stages) for c in ABLATION_CONFIGS]
        for previous, current in zip(ladder, ladder[1:], strict=False):
            assert previous <= current, f"{previous - current} disappeared between rungs"

    def test_timings_are_recorded_per_stage(self, page):
        result = Pipeline(get_config("full+sauvola"))(np.array(page.image))
        assert set(result.timings_ms) == set(
            s for s in DEFAULT_ORDER if s in get_config("full+sauvola").stages
        )
        assert result.total_ms > 0

    def test_config_describe_is_readable(self):
        described = get_config("full+sauvola").describe()
        assert "grayscale -> " in described
        assert "binarize(method=sauvola)" in described


class TestGeometryEndToEnd:
    """The property the whole ablation rests on: after render -> degrade ->
    preprocess, gold boxes mapped through the composed homographies must still
    sit on their glyphs. Everything else can be right and the numbers still
    meaningless if this is wrong."""

    @pytest.mark.parametrize("profile_name", ["light", "medium", "heavy"])
    def test_boxes_track_ink_through_the_full_chain(self, page, profile_name):
        img = np.array(page.image)
        boxes = [w.bbox for w in page.words]
        pipeline = Pipeline(get_config("full(no-binarize)"))

        covered_all = []
        for seed in range(3):
            deg = degrade(img, PROFILES[profile_name], random.Random(seed))
            pre = pipeline(deg.image)
            mapped = pre.map_boxes(deg.map_boxes(boxes))
            covered_all.append(np.mean([ink_fraction(pre.image, b) > 0.01 for b in mapped]))

        mean_covered = float(np.mean(covered_all))
        assert (
            mean_covered > 0.80
        ), f"{profile_name}: only {mean_covered:.0%} of gold boxes land on ink after preprocessing"

    def test_deskew_recovers_the_injected_rotation(self, page):
        """End-to-end sign check. A sign error in the homography composition
        would leave the image correct but every mapped box wrong, so verify
        the recovered angle against the one we injected."""
        img = np.array(page.image)
        for angle in (-3.5, -1.5, 1.5, 3.5):
            skewed, _ = rotate(img, angle)
            result = Pipeline(PipelineConfig.build("d", ["grayscale", "deskew"]))(skewed)
            assert result.stage_info["deskew"]["angle"] == pytest.approx(-angle, abs=0.4)

    def test_preprocessing_does_not_destroy_the_text(self, page):
        """A pipeline can trivially minimise CER-adjacent metrics by wiping the
        page. Assert ink survives at a plausible density."""
        img = np.array(page.image)
        deg = degrade(img, MEDIUM, random.Random(1))
        for name in ("full(no-binarize)", "full+sauvola", "binarize=otsu"):
            out = Pipeline(get_config(name))(deg.image).image
            ink = float((out < 128).mean())
            assert 0.002 < ink < 0.35, f"{name} produced implausible ink density {ink:.4f}"
