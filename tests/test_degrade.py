"""Tests for scan-artifact simulation.

The load-bearing property here is the homography contract: every geometric
degradation must report a transform that maps the clean page's word boxes
onto their new positions. If that contract breaks, the ablation still runs and
still prints numbers -- they are just wrong, in a way no amount of staring at
a CER table reveals. So these tests check box tracking directly against the
pixels, not against the matrices.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from invoice_extract.degrade.transforms import (
    CLEAN,
    HEAVY,
    IDENTITY,
    LIGHT,
    MEDIUM,
    PROFILES,
    bleed_through,
    degrade,
    gaussian_noise,
    illumination_gradient,
    ink_spread,
    jpeg_artifacts,
    motion_blur,
    perspective,
    rescale,
    rotate,
    salt_pepper,
    scan_border,
    shadow,
    transform_boxes,
)
from invoice_extract.render.invoice_renderer import InvoiceRenderer


@pytest.fixture(scope="module")
def page():
    doc = {
        "doc_id": "degrade_fixture",
        "document_type": "invoice",
        "parties": {"vendor": {"name": "Northwind Traders", "address": "12 Harbour Road, Chennai"}},
        "invoice_meta": {"invoice_number": "INV-9001", "invoice_date": "2026-03-04"},
        "line_items": [
            {
                "description": "Cable Assembly",
                "quantity": 4,
                "unit_price": 250.0,
                "line_total": 1000.0,
            },
            {
                "description": "Mounting Bracket",
                "quantity": 2,
                "unit_price": 125.5,
                "line_total": 251.0,
            },
        ],
        "tax_lines": [{"type": "IGST", "rate": 18, "amount": 225.18}],
        "totals": {"subtotal": 1251.0, "grand_total": 1476.18},
    }
    return InvoiceRenderer(dpi=120).render(doc, seed=11)


def ink_fraction(image: np.ndarray, bbox, threshold: int = 170) -> float:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = image[y0:y1, x0:x1]
    return float((patch < threshold).mean()) if patch.size else 0.0


class TestIndividualTransforms:
    def test_rotate_reports_a_transform_that_tracks_the_ink(self, page):
        img = np.array(page.image)
        out, h = rotate(img, 4.0)
        assert out.shape == img.shape

        boxes = [w.bbox for w in page.words]
        moved = transform_boxes(boxes, h)
        assert not np.allclose(h, IDENTITY)

        # The boxes must follow the glyphs, so mapped boxes still cover ink
        # while the original (unmapped) positions largely no longer do.
        tracked = np.mean([ink_fraction(out, b) > 0.02 for b in moved])
        untracked = np.mean([ink_fraction(out, b) > 0.02 for b in boxes])
        assert tracked > 0.9, f"only {tracked:.0%} of mapped boxes still cover ink"
        assert tracked > untracked

    def test_rotation_homography_is_invertible(self):
        _, h = rotate(np.zeros((200, 200), dtype=np.uint8), 3.0)
        assert np.linalg.inv(h) @ h == pytest.approx(IDENTITY, abs=1e-9)

    def test_box_round_trip_encloses_the_original(self, page):
        """A box round-tripped through rotate/unrotate must *contain* the
        original but need not equal it: each hop re-bounds to an axis-aligned
        rectangle, and the enclosing rectangle of a rotated rectangle is
        strictly larger. Growth is bounded and one-directional -- ground truth
        gets looser, never wrong."""
        boxes = [w.bbox for w in page.words][:20]
        _, h = rotate(np.array(page.image), 3.0)
        back = transform_boxes(transform_boxes(boxes, h), np.linalg.inv(h))

        for (ox0, oy0, ox1, oy1), (rx0, ry0, rx1, ry1) in zip(boxes, back, strict=True):
            assert rx0 <= ox0 + 1e-6 and ry0 <= oy0 + 1e-6
            assert rx1 >= ox1 - 1e-6 and ry1 >= oy1 - 1e-6
            # A 3-degree round trip should not inflate a box beyond ~2x height.
            assert (ry1 - ry0) < (oy1 - oy0) * 2 + 12

    def test_perspective_tracks_boxes(self, page):
        img = np.array(page.image)
        out, h = perspective(img, 0.02, random.Random(3))
        moved = transform_boxes([w.bbox for w in page.words], h)
        tracked = np.mean([ink_fraction(out, b) > 0.02 for b in moved])
        assert tracked > 0.85

    def test_rescale_is_geometrically_neutral(self, page):
        """Down- then up-sampling returns to the original canvas, so boxes must
        not move -- only detail is lost."""
        img = np.array(page.image)
        out, h = rescale(img, 0.5)
        assert out.shape == img.shape
        assert np.allclose(h, IDENTITY)
        assert out.std() < img.std() * 1.05  # detail lost, not gained

    @pytest.mark.parametrize(
        "fn,args",
        [
            (gaussian_noise, (8.0,)),
            (salt_pepper, (0.01,)),
            (illumination_gradient, (0.3,)),
            (shadow, (0.3,)),
            (scan_border, (0.04,)),
            (ink_spread, (1,)),
            (bleed_through, (0.1,)),
        ],
    )
    def test_photometric_transforms_are_geometrically_neutral(self, page, fn, args):
        img = np.array(page.image)
        out, h = fn(img, *args, random.Random(5))
        assert np.allclose(h, IDENTITY), f"{fn.__name__} must not move pixels"
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_jpeg_and_blur_are_geometrically_neutral(self, page):
        img = np.array(page.image)
        for out, h in (jpeg_artifacts(img, 40), motion_blur(img, 7, 30.0)):
            assert np.allclose(h, IDENTITY)
            assert out.shape == img.shape

    def test_scan_border_darkens_an_edge(self, page):
        img = np.array(page.image)
        out, _ = scan_border(img, 0.06, random.Random(1))
        assert out.min() < 70

    def test_bleed_through_only_darkens(self, page):
        img = np.array(page.image)
        out, _ = bleed_through(img, 0.2, random.Random(2))
        assert (out <= img).all(), "bleed-through must never brighten the page"

    def test_illumination_gradient_is_not_uniform(self, page):
        img = np.array(page.image)
        out, _ = illumination_gradient(img, 0.4, random.Random(4))
        h, w = out.shape
        quadrants = [
            out[: h // 2, : w // 2].mean(),
            out[: h // 2, w // 2 :].mean(),
            out[h // 2 :, : w // 2].mean(),
            out[h // 2 :, w // 2 :].mean(),
        ]
        assert max(quadrants) - min(quadrants) > 3.0


class TestProfiles:
    def test_clean_profile_is_a_no_op(self, page):
        img = np.array(page.image)
        result = degrade(img, CLEAN, random.Random(0))
        assert result.applied == []
        assert np.array_equal(result.image, img)
        assert np.allclose(result.homography, IDENTITY)

    @pytest.mark.parametrize("profile", [LIGHT, MEDIUM, HEAVY])
    def test_profiles_apply_effects_and_stay_valid_images(self, page, profile):
        img = np.array(page.image)
        result = degrade(img, profile, random.Random(9))
        assert result.applied, f"{profile.name} applied nothing"
        assert result.image.shape == img.shape
        assert result.image.dtype == np.uint8

    def test_severity_is_monotonic_in_pixel_distortion(self, page):
        """Averaged over seeds, heavier profiles must actually damage the page
        more -- otherwise the ablation's x-axis is meaningless."""
        img = np.array(page.image).astype(np.float32)
        distortion = {}
        for profile in (LIGHT, MEDIUM, HEAVY):
            diffs = [
                float(
                    np.abs(
                        degrade(np.array(page.image), profile, random.Random(s)).image.astype(
                            np.float32
                        )
                        - img
                    ).mean()
                )
                for s in range(6)
            ]
            distortion[profile.name] = sum(diffs) / len(diffs)
        assert distortion["light"] < distortion["medium"] < distortion["heavy"], distortion

    def test_degradation_tracks_boxes_end_to_end(self, page):
        """The composed homography across a whole profile must still land the
        boxes on ink. This is the property the ablation actually depends on."""
        img = np.array(page.image)
        boxes = [w.bbox for w in page.words]
        for seed in range(4):
            result = degrade(img, MEDIUM, random.Random(seed))
            moved = result.map_boxes(boxes)
            covered = np.mean([ink_fraction(result.image, b) > 0.015 for b in moved])
            assert covered > 0.85, f"seed {seed}: only {covered:.0%} of boxes track the ink"

    def test_profiles_are_reproducible_from_a_seed(self, page):
        img = np.array(page.image)
        a = degrade(img, HEAVY, random.Random(21))
        b = degrade(img, HEAVY, random.Random(21))
        assert a.applied == b.applied
        assert np.array_equal(a.image, b.image)

    def test_registry_exposes_every_profile(self):
        assert set(PROFILES) == {"clean", "light", "medium", "heavy"}


class TestTransformBoxes:
    def test_identity_is_a_pass_through(self):
        boxes = [(1.0, 2.0, 3.0, 4.0)]
        assert transform_boxes(boxes, IDENTITY) == boxes

    def test_empty_input(self):
        assert transform_boxes([], IDENTITY) == []

    def test_translation(self):
        h = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float64)
        assert transform_boxes([(0.0, 0.0, 2.0, 2.0)], h)[0] == pytest.approx((10, 5, 12, 7))

    def test_rotated_box_becomes_its_enclosing_rectangle(self):
        """A rotated rectangle is no longer axis-aligned; the honest
        axis-aligned answer is the bounding box, which is necessarily larger."""
        _, h = rotate(np.zeros((100, 100), dtype=np.uint8), 45.0)
        x0, y0, x1, y1 = transform_boxes([(40.0, 40.0, 60.0, 50.0)], h)[0]
        assert (x1 - x0) > 20 and (y1 - y0) > 10
