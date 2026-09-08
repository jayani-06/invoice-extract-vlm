"""Individual OpenCV preprocessing stages.

Every stage is an independent, toggleable unit with the same signature, which
is what makes the Phase 1 ablation possible: a pipeline configuration is just
a subset of these, and the ablation table is the cross product of subsets and
degradation severities.

Each stage returns the homography it applied so gold word boxes can be mapped
into preprocessed space. Stages that only touch pixel values return identity;
deskew and perspective correction return real transforms.

Ordering rationale (see `pipeline.DEFAULT_ORDER`): geometry must be fixed
before binarisation, because a local threshold computed on a skewed page
leaks neighbouring lines into each window; and illumination must be
normalised before denoising, because a denoiser tuned on a globally dark
region over-smooths a well-lit one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

Array = np.ndarray
IDENTITY = np.eye(3, dtype=np.float64)


@dataclass
class StageResult:
    image: Array
    homography: Array
    info: dict


class Stage(Protocol):
    name: str

    def __call__(self, img: Array) -> StageResult: ...


def _ensure_gray(img: Array) -> Array:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _text_mask(img: Array) -> Array:
    """Binary mask of probable ink (255 = ink), robust to uneven lighting.

    Used by the geometry stages, which need to reason about where the text is
    without committing to a final binarisation.
    """
    gray = _ensure_gray(img)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    return mask


# ------------------------------------------------------------- grayscale


class ToGrayscale:
    name = "grayscale"

    def __call__(self, img: Array) -> StageResult:
        return StageResult(_ensure_gray(img), IDENTITY.copy(), {})


# --------------------------------------------------------- border removal


class RemoveScanBorder:
    """Whiten the dark frame a scanner lid leaves around an undersized page.

    This runs before deskew on purpose. A `minAreaRect` deskew fitted over all
    dark pixels will happily lock onto a black border band and report an angle
    that has nothing to do with the text -- one of the concrete failure modes
    the ablation is built to expose.
    """

    name = "remove_border"

    def __init__(
        self, dark_threshold: int = 70, max_band_frac: float = 0.12, row_dark_frac: float = 0.85
    ) -> None:
        self.dark_threshold = dark_threshold
        self.max_band_frac = max_band_frac
        self.row_dark_frac = row_dark_frac

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        h, w = gray.shape
        dark = gray < self.dark_threshold
        out = gray.copy()
        trimmed = {}

        # Scan inward from each edge while rows/columns stay overwhelmingly dark.
        top = 0
        while top < int(h * self.max_band_frac) and dark[top].mean() > self.row_dark_frac:
            top += 1
        bottom = 0
        while (
            bottom < int(h * self.max_band_frac)
            and dark[h - 1 - bottom].mean() > self.row_dark_frac
        ):
            bottom += 1
        left = 0
        while left < int(w * self.max_band_frac) and dark[:, left].mean() > self.row_dark_frac:
            left += 1
        right = 0
        while (
            right < int(w * self.max_band_frac)
            and dark[:, w - 1 - right].mean() > self.row_dark_frac
        ):
            right += 1

        # Fill rather than crop: cropping would shift every coordinate and make
        # the gold boxes wrong for a stage whose job is not geometric.
        fill = int(np.percentile(gray, 90))
        if top:
            out[:top, :] = fill
            trimmed["top"] = top
        if bottom:
            out[h - bottom :, :] = fill
            trimmed["bottom"] = bottom
        if left:
            out[:, :left] = fill
            trimmed["left"] = left
        if right:
            out[:, w - right :] = fill
            trimmed["right"] = right

        return StageResult(out, IDENTITY.copy(), {"trimmed": trimmed})


# --------------------------------------------------- illumination / shading


class NormalizeIllumination:
    """Flatten uneven lighting by dividing out a morphological background estimate.

    A large closing operator on dark-on-light text erases the glyphs and leaves
    only the slowly-varying page illumination; dividing the original by that
    estimate yields a page lit as if evenly. This is what rescues global
    thresholding on shadowed and gradient-lit scans.
    """

    name = "normalize_illumination"

    def __init__(self, kernel_frac: float = 0.03) -> None:
        self.kernel_frac = kernel_frac

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        h, w = gray.shape
        k = max(15, int(min(h, w) * self.kernel_frac) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        background = cv2.GaussianBlur(background, (k, k), 0)
        norm = cv2.divide(gray, background, scale=255)
        return StageResult(norm, IDENTITY.copy(), {"kernel": k})


# ------------------------------------------------------------- denoising


class Denoise:
    """Suppress sensor noise and speckle.

    `median` is the right tool for salt-and-pepper (it is order-statistic based,
    so an isolated extreme pixel is discarded outright), while `nlm` preserves
    thin strokes better under Gaussian noise. `bilateral` is the cheap middle
    ground. The mode is a config knob precisely because which one wins depends
    on the degradation, and that dependence is a result worth reporting.
    """

    name = "denoise"

    def __init__(self, mode: str = "median", strength: int = 3) -> None:
        if mode not in {"median", "nlm", "bilateral", "none"}:
            raise ValueError(f"unknown denoise mode: {mode}")
        self.mode = mode
        self.strength = strength

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        if self.mode == "none":
            return StageResult(gray, IDENTITY.copy(), {"mode": "none"})
        if self.mode == "median":
            k = max(3, self.strength | 1)
            out = cv2.medianBlur(gray, k)
        elif self.mode == "nlm":
            out = cv2.fastNlMeansDenoising(
                gray, None, h=float(self.strength * 3), templateWindowSize=7, searchWindowSize=21
            )
        else:
            out = cv2.bilateralFilter(
                gray, d=5, sigmaColor=self.strength * 15, sigmaSpace=self.strength * 5
            )
        return StageResult(out, IDENTITY.copy(), {"mode": self.mode})


# --------------------------------------------------------------- deskew


def estimate_skew_projection(
    mask: Array, limit: float = 8.0, coarse: float = 1.0, fine: float = 0.1
) -> float:
    """Estimate skew by maximising the variance of the horizontal projection profile.

    When a page is level, every text line contributes a sharp peak to the row-sum
    profile and the variance is maximal; skew smears those peaks together. This
    is slower than fitting a `minAreaRect`, but it keys on the text lines
    themselves rather than on whatever the largest dark blob happens to be,
    which makes it far more robust on documents with borders, stamps, or
    heavy table rules.

    Two-pass coarse-then-fine search keeps it affordable.
    """
    small = cv2.resize(mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    def score(angle: float) -> float:
        h, w = small.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        profile = rot.sum(axis=1, dtype=np.float64)
        return float(np.var(profile))

    coarse_angles = np.arange(-limit, limit + coarse, coarse)
    best = max(coarse_angles, key=score)
    fine_angles = np.arange(best - coarse, best + coarse + fine, fine)
    return float(max(fine_angles, key=score))


def estimate_skew_min_area_rect(mask: Array) -> float:
    """Classic baseline: fit a rotated rectangle to all ink pixels.

    Kept as an explicit alternative so the ablation can quantify how much the
    projection-profile method actually buys, rather than asserting it.
    """
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return float(angle)


class Deskew:
    name = "deskew"

    def __init__(
        self, method: str = "projection", limit: float = 8.0, min_angle: float = 0.05
    ) -> None:
        if method not in {"projection", "min_area_rect"}:
            raise ValueError(f"unknown deskew method: {method}")
        self.method = method
        self.limit = limit
        self.min_angle = min_angle

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        mask = _text_mask(gray)
        if self.method == "projection":
            angle = estimate_skew_projection(mask, limit=self.limit)
        else:
            angle = estimate_skew_min_area_rect(mask)
            angle = float(np.clip(angle, -self.limit, self.limit))

        if abs(angle) < self.min_angle:
            return StageResult(gray, IDENTITY.copy(), {"angle": 0.0, "method": self.method})

        h, w = gray.shape
        border = int(np.percentile(gray, 90))
        m2x3 = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(
            gray,
            m2x3,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        return StageResult(
            out, np.vstack([m2x3, [0, 0, 1]]), {"angle": angle, "method": self.method}
        )


# ------------------------------------------------- perspective correction


class CorrectPerspective:
    """Find the page quadrilateral and warp it back to a rectangle.

    Only fires when a convincing four-sided contour covering most of the frame
    is found; a false positive here is far more damaging than a miss, since it
    warps the entire page on the strength of a stray rectangle.
    """

    name = "correct_perspective"

    def __init__(self, min_area_frac: float = 0.35, max_area_frac: float = 0.995) -> None:
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac

    def _find_quad(self, gray: Array) -> Array | None:
        h, w = gray.shape
        small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = small.shape[0] * small.shape[1]

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if not (self.min_area_frac * page_area <= area <= self.max_area_frac * page_area):
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32) * 4.0
        return None

    @staticmethod
    def _order_corners(pts: Array) -> Array:
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        return np.float32(
            [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]]
        )

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        quad = self._find_quad(gray)
        if quad is None:
            return StageResult(gray, IDENTITY.copy(), {"applied": False})

        h, w = gray.shape
        src = self._order_corners(quad)
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        m = cv2.getPerspectiveTransform(src, dst)
        border = int(np.percentile(gray, 90))
        out = cv2.warpPerspective(
            gray,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        return StageResult(out, m.astype(np.float64), {"applied": True})


# ------------------------------------------------------------ upscaling


class UpscaleForOcr:
    """Bring small text up to the ~30 px cap-height that OCR engines expect.

    Tesseract in particular degrades sharply below roughly 20 px of x-height,
    and a low-DPI scan lands there routinely. Upscaling cannot restore lost
    information, but it does stop the engine's own downsampling from
    compounding the loss.
    """

    name = "upscale"

    def __init__(self, target_min_dim: int = 1700, max_factor: float = 3.0) -> None:
        self.target_min_dim = target_min_dim
        self.max_factor = max_factor

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        h, w = gray.shape
        if min(h, w) >= self.target_min_dim:
            return StageResult(gray, IDENTITY.copy(), {"factor": 1.0})
        factor = min(self.max_factor, self.target_min_dim / min(h, w))
        out = cv2.resize(
            gray, (round(w * factor), round(h * factor)), interpolation=cv2.INTER_CUBIC
        )
        scale = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, 1]], dtype=np.float64)
        return StageResult(out, scale, {"factor": factor})


# ------------------------------------------------------------ binarisation


class Binarize:
    """Produce a bilevel image.

    - `otsu`: one global threshold. Fast and clean on evenly lit pages, and the
      classic failure case under a shadow or gradient, where it sacrifices a
      whole region.
    - `adaptive`: per-window Gaussian mean. Handles uneven lighting, but
      amplifies noise in blank areas.
    - `sauvola`: local mean and standard deviation. The standard choice for
      degraded documents; it is the most expensive of the three.
    - `none`: leave grayscale. Modern OCR engines do their own binarisation,
      so forcing ours can actively hurt -- the ablation reports whether it does.
    """

    name = "binarize"

    def __init__(self, method: str = "sauvola", window: int = 31, k: float = 0.2) -> None:
        if method not in {"otsu", "adaptive", "sauvola", "none"}:
            raise ValueError(f"unknown binarize method: {method}")
        self.method = method
        self.window = window
        self.k = k

    def __call__(self, img: Array) -> StageResult:
        gray = _ensure_gray(img)
        if self.method == "none":
            return StageResult(gray, IDENTITY.copy(), {"method": "none"})
        if self.method == "otsu":
            _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif self.method == "adaptive":
            out = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                max(3, self.window | 1),
                10,
            )
        else:
            thresh = threshold_sauvola(gray, window_size=max(3, self.window | 1), k=self.k)
            out = ((gray > thresh) * 255).astype(np.uint8)
        return StageResult(out, IDENTITY.copy(), {"method": self.method})


STAGE_REGISTRY: dict[str, type] = {
    ToGrayscale.name: ToGrayscale,
    RemoveScanBorder.name: RemoveScanBorder,
    NormalizeIllumination.name: NormalizeIllumination,
    Denoise.name: Denoise,
    Deskew.name: Deskew,
    CorrectPerspective.name: CorrectPerspective,
    UpscaleForOcr.name: UpscaleForOcr,
    Binarize.name: Binarize,
}
