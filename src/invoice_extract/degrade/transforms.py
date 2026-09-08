"""Scan-artifact simulation: turn a clean rendered page into something that
looks like it came off a flatbed scanner, a photocopier, or a phone camera.

Two design rules drive this module:

1. **Geometry is tracked, not lost.** Every transform returns the 3x3
   homography it applied (identity for photometric ones). The pipeline
   composes them, so the renderer's word boxes can be mapped into degraded
   image space and remain valid ground truth. Without this, a deskew ablation
   could only be scored on text, never on localisation.

2. **Degradations are sampled from a profile, not hard-coded.** A severity
   profile is a distribution over parameters, so `light`/`medium`/`heavy`
   produce a spread of realistic documents rather than one canonical "noisy"
   image that the preprocessing could overfit to.

Images are handled as uint8 grayscale numpy arrays throughout.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

Array = np.ndarray
IDENTITY = np.eye(3, dtype=np.float64)


def _as_gray(img: Array) -> Array:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# --------------------------------------------------------------- geometric


def rotate(img: Array, angle_deg: float, border: int = 255) -> tuple[Array, Array]:
    """Rotate about the image centre, keeping the canvas size fixed.

    Real scanner skew is small (a page nudged against the guide), so callers
    should pass fractions of a degree up to a few degrees, not arbitrary
    rotations.
    """
    h, w = img.shape[:2]
    m2x3 = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    out = cv2.warpAffine(
        img,
        m2x3,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return out, np.vstack([m2x3, [0, 0, 1]])


def perspective(
    img: Array, strength: float, rng: random.Random, border: int = 255
) -> tuple[Array, Array]:
    """Simulate a page photographed off-axis by jittering the four corners.

    `strength` is the max corner displacement as a fraction of page size.
    """
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = lambda span: rng.uniform(-span, span)  # noqa: E731
    dx, dy = strength * w, strength * h
    dst = np.float32(
        [
            [jitter(dx), jitter(dy)],
            [w + jitter(dx), jitter(dy)],
            [w + jitter(dx), h + jitter(dy)],
            [jitter(dx), h + jitter(dy)],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        img,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return out, m.astype(np.float64)


def rescale(img: Array, factor: float) -> tuple[Array, Array]:
    """Downsample then upsample back to the original size.

    This is how a low-DPI scan loses stroke detail: the information is gone,
    and no amount of later upscaling brings it back. Because the canvas
    returns to its original dimensions, the net homography is the identity.
    """
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA
    )
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out, IDENTITY.copy()


# ------------------------------------------------------------ photometric


def gaussian_blur(img: Array, sigma: float) -> tuple[Array, Array]:
    k = max(3, int(sigma * 6) | 1)  # odd kernel covering +/-3 sigma
    return cv2.GaussianBlur(img, (k, k), sigma), IDENTITY.copy()


def motion_blur(img: Array, length: int, angle_deg: float) -> tuple[Array, Array]:
    """Directional blur from a moving scanner head or an unsteady hand."""
    length = max(3, length | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, m, (length, length))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return cv2.filter2D(img, -1, kernel), IDENTITY.copy()


def gaussian_noise(img: Array, sigma: float, rng: random.Random) -> tuple[Array, Array]:
    gen = np.random.default_rng(rng.getrandbits(32))
    noisy = img.astype(np.float32) + gen.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8), IDENTITY.copy()


def salt_pepper(img: Array, amount: float, rng: random.Random) -> tuple[Array, Array]:
    """Speckle from dust on the platen and from thresholding a poor original."""
    gen = np.random.default_rng(rng.getrandbits(32))
    out = img.copy()
    r = gen.random(img.shape)
    out[r < amount / 2] = 0
    out[r > 1 - amount / 2] = 255
    return out, IDENTITY.copy()


def illumination_gradient(img: Array, strength: float, rng: random.Random) -> tuple[Array, Array]:
    """Uneven lighting: a smooth multiplicative field across the page.

    This is the degradation that breaks global (Otsu) thresholding, and the
    reason the preprocessing pipeline offers local binarisation and background
    normalisation as separate, ablatable stages.
    """
    h, w = img.shape[:2]
    cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((xs / w - cx) ** 2 + (ys / h - cy) ** 2) / 2.0)
    field = 1.0 - strength * (dist / max(dist.max(), 1e-6))
    out = img.astype(np.float32) * field
    return np.clip(out, 0, 255).astype(np.uint8), IDENTITY.copy()


def shadow(img: Array, strength: float, rng: random.Random) -> tuple[Array, Array]:
    """A hard-edged shade band, as cast by a phone or an open book spine."""
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    vertical = rng.random() < 0.5
    span = rng.uniform(0.15, 0.45)
    start = rng.uniform(0.0, 1.0 - span)
    if vertical:
        a, b = int(start * w), int((start + span) * w)
        mask[:, a:b] = 1.0 - strength
    else:
        a, b = int(start * h), int((start + span) * h)
        mask[a:b, :] = 1.0 - strength
    blur_k = max(3, (int(min(h, w) * 0.03) | 1))
    mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    out = img.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8), IDENTITY.copy()


def jpeg_artifacts(img: Array, quality: int) -> tuple[Array, Array]:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("L")), IDENTITY.copy()


def ink_spread(img: Array, amount: int, rng: random.Random) -> tuple[Array, Array]:
    """Photocopy generation loss: strokes thicken (toner bleed) or thin (fade)."""
    k = max(1, amount) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Text is dark on light, so eroding the image thickens the glyphs.
    if rng.random() < 0.6:
        return cv2.erode(img, kernel, iterations=1), IDENTITY.copy()
    return cv2.dilate(img, kernel, iterations=1), IDENTITY.copy()


def paper_texture(img: Array, strength: float, rng: random.Random) -> tuple[Array, Array]:
    """Low-frequency paper grain, so the background is never a flat 255."""
    h, w = img.shape[:2]
    gen = np.random.default_rng(rng.getrandbits(32))
    small = gen.normal(0, strength * 255, (max(2, h // 24), max(2, w // 24))).astype(np.float32)
    grain = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    out = img.astype(np.float32) + grain
    return np.clip(out, 0, 255).astype(np.uint8), IDENTITY.copy()


def scan_border(img: Array, max_frac: float, rng: random.Random) -> tuple[Array, Array]:
    """The dark band a scanner lid leaves when the page is smaller than the bed.

    Included specifically because it is what a naive `minAreaRect` deskew
    latches onto instead of the text, so the ablation can show border removal
    earning its place.
    """
    h, w = img.shape[:2]
    out = img.copy()
    for side in ("top", "bottom", "left", "right"):
        if rng.random() < 0.6:
            frac = rng.uniform(0.005, max_frac)
            value = rng.randint(0, 60)
            if side == "top":
                out[: int(h * frac), :] = value
            elif side == "bottom":
                out[h - int(h * frac) :, :] = value
            elif side == "left":
                out[:, : int(w * frac)] = value
            else:
                out[:, w - int(w * frac) :] = value
    return out, IDENTITY.copy()


def bleed_through(img: Array, strength: float, rng: random.Random) -> tuple[Array, Array]:
    """Faint mirrored text from the reverse side of a thin sheet."""
    flipped = cv2.flip(img, 1)
    shift = rng.randint(-6, 6)
    flipped = np.roll(flipped, shift, axis=1)
    out = img.astype(np.float32) * (1 - strength) + flipped.astype(np.float32) * strength
    # Only darken; bleed-through never brightens the page.
    return np.minimum(img, np.clip(out, 0, 255).astype(np.uint8)), IDENTITY.copy()


# ------------------------------------------------------------ box mapping


def transform_boxes(
    boxes: list[tuple[float, float, float, float]], h_matrix: Array
) -> list[tuple[float, float, float, float]]:
    """Map axis-aligned boxes through a homography.

    All four corners are transformed and re-bounded, because a rotated or
    warped box is no longer axis-aligned; the enclosing rectangle is the
    honest axis-aligned representation of where the word now sits.
    """
    if not boxes:
        return []
    if np.allclose(h_matrix, IDENTITY):
        return list(boxes)

    corners = np.array(
        [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]] for x0, y0, x1, y1 in boxes], dtype=np.float32
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, h_matrix.astype(np.float64)).reshape(-1, 4, 2)
    return [
        (float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max()))
        for q in mapped
    ]


# ----------------------------------------------------------------- profile

Step = Callable[[Array, random.Random], tuple[Array, Array]]


@dataclass
class DegradationProfile:
    """A named severity level: a probability and a parameter range per effect.

    Ordering matters and mirrors physical reality: the page is printed, then
    photocopied (ink spread), then physically warped, then lit unevenly, then
    sensed (noise), then compressed. Applying JPEG before blur, for instance,
    would produce artifacts no real pipeline ever creates.
    """

    name: str
    p_ink_spread: float = 0.0
    ink_spread_amount: tuple[int, int] = (1, 1)
    p_paper_texture: float = 0.0
    paper_texture_strength: tuple[float, float] = (0.005, 0.02)
    p_bleed_through: float = 0.0
    bleed_strength: tuple[float, float] = (0.03, 0.10)
    p_rotate: float = 0.0
    rotate_deg: tuple[float, float] = (-1.0, 1.0)
    p_perspective: float = 0.0
    perspective_strength: tuple[float, float] = (0.005, 0.02)
    p_rescale: float = 0.0
    rescale_factor: tuple[float, float] = (0.5, 0.85)
    p_illumination: float = 0.0
    illumination_strength: tuple[float, float] = (0.10, 0.35)
    p_shadow: float = 0.0
    shadow_strength: tuple[float, float] = (0.10, 0.35)
    p_scan_border: float = 0.0
    scan_border_frac: tuple[float, float] = (0.01, 0.04)
    p_blur: float = 0.0
    blur_sigma: tuple[float, float] = (0.5, 1.5)
    p_motion_blur: float = 0.0
    motion_blur_len: tuple[int, int] = (3, 9)
    p_gaussian_noise: float = 0.0
    gaussian_noise_sigma: tuple[float, float] = (3.0, 12.0)
    p_salt_pepper: float = 0.0
    salt_pepper_amount: tuple[float, float] = (0.001, 0.008)
    p_jpeg: float = 0.0
    jpeg_quality: tuple[int, int] = (35, 85)


CLEAN = DegradationProfile(name="clean")

LIGHT = DegradationProfile(
    name="light",
    p_paper_texture=0.7,
    paper_texture_strength=(0.004, 0.012),
    p_rotate=0.6,
    rotate_deg=(-0.7, 0.7),
    p_illumination=0.4,
    illumination_strength=(0.05, 0.15),
    p_blur=0.5,
    blur_sigma=(0.4, 0.9),
    p_gaussian_noise=0.6,
    gaussian_noise_sigma=(2.0, 6.0),
    p_jpeg=0.5,
    jpeg_quality=(70, 92),
)

MEDIUM = DegradationProfile(
    name="medium",
    p_ink_spread=0.35,
    ink_spread_amount=(1, 1),
    p_paper_texture=0.8,
    paper_texture_strength=(0.008, 0.020),
    p_bleed_through=0.2,
    bleed_strength=(0.03, 0.08),
    p_rotate=0.8,
    rotate_deg=(-2.5, 2.5),
    p_perspective=0.3,
    perspective_strength=(0.004, 0.012),
    p_rescale=0.35,
    rescale_factor=(0.6, 0.85),
    p_illumination=0.6,
    illumination_strength=(0.12, 0.30),
    p_shadow=0.25,
    shadow_strength=(0.10, 0.25),
    p_scan_border=0.35,
    scan_border_frac=(0.01, 0.03),
    p_blur=0.6,
    blur_sigma=(0.6, 1.4),
    p_motion_blur=0.15,
    motion_blur_len=(3, 7),
    p_gaussian_noise=0.75,
    gaussian_noise_sigma=(4.0, 12.0),
    p_salt_pepper=0.35,
    salt_pepper_amount=(0.001, 0.005),
    p_jpeg=0.7,
    jpeg_quality=(45, 80),
)

HEAVY = DegradationProfile(
    name="heavy",
    p_ink_spread=0.6,
    ink_spread_amount=(1, 2),
    p_paper_texture=0.9,
    paper_texture_strength=(0.012, 0.030),
    p_bleed_through=0.4,
    bleed_strength=(0.05, 0.14),
    p_rotate=0.9,
    rotate_deg=(-5.0, 5.0),
    p_perspective=0.6,
    perspective_strength=(0.008, 0.025),
    p_rescale=0.6,
    rescale_factor=(0.4, 0.7),
    p_illumination=0.8,
    illumination_strength=(0.20, 0.45),
    p_shadow=0.5,
    shadow_strength=(0.18, 0.40),
    p_scan_border=0.6,
    scan_border_frac=(0.02, 0.06),
    p_blur=0.75,
    blur_sigma=(0.9, 2.0),
    p_motion_blur=0.3,
    motion_blur_len=(5, 11),
    p_gaussian_noise=0.9,
    gaussian_noise_sigma=(8.0, 20.0),
    p_salt_pepper=0.6,
    salt_pepper_amount=(0.003, 0.012),
    p_jpeg=0.85,
    jpeg_quality=(25, 60),
)

PROFILES: dict[str, DegradationProfile] = {p.name: p for p in (CLEAN, LIGHT, MEDIUM, HEAVY)}


@dataclass
class DegradationResult:
    image: Array
    homography: Array
    applied: list[str]

    def map_boxes(
        self, boxes: list[tuple[float, float, float, float]]
    ) -> list[tuple[float, float, float, float]]:
        return transform_boxes(boxes, self.homography)


def degrade(img: Array, profile: DegradationProfile, rng: random.Random) -> DegradationResult:
    """Apply a profile's effects in physical order, composing the homography."""
    img = _as_gray(img)
    h_total = IDENTITY.copy()
    applied: list[str] = []

    def maybe(p: float) -> bool:
        return p > 0 and rng.random() < p

    def step(name: str, out: Array, h_matrix: Array) -> Array:
        nonlocal h_total
        h_total = h_matrix @ h_total
        applied.append(name)
        return out

    if maybe(profile.p_ink_spread):
        img = step("ink_spread", *ink_spread(img, rng.randint(*profile.ink_spread_amount), rng))
    if maybe(profile.p_paper_texture):
        img = step(
            "paper_texture", *paper_texture(img, rng.uniform(*profile.paper_texture_strength), rng)
        )
    if maybe(profile.p_bleed_through):
        img = step("bleed_through", *bleed_through(img, rng.uniform(*profile.bleed_strength), rng))
    if maybe(profile.p_rotate):
        img = step("rotate", *rotate(img, rng.uniform(*profile.rotate_deg)))
    if maybe(profile.p_perspective):
        img = step(
            "perspective", *perspective(img, rng.uniform(*profile.perspective_strength), rng)
        )
    if maybe(profile.p_rescale):
        img = step("rescale", *rescale(img, rng.uniform(*profile.rescale_factor)))
    if maybe(profile.p_illumination):
        img = step(
            "illumination",
            *illumination_gradient(img, rng.uniform(*profile.illumination_strength), rng),
        )
    if maybe(profile.p_shadow):
        img = step("shadow", *shadow(img, rng.uniform(*profile.shadow_strength), rng))
    if maybe(profile.p_scan_border):
        img = step("scan_border", *scan_border(img, rng.uniform(*profile.scan_border_frac), rng))
    if maybe(profile.p_blur):
        img = step("blur", *gaussian_blur(img, rng.uniform(*profile.blur_sigma)))
    if maybe(profile.p_motion_blur):
        img = step(
            "motion_blur",
            *motion_blur(img, rng.randint(*profile.motion_blur_len), rng.uniform(0, 180)),
        )
    if maybe(profile.p_gaussian_noise):
        img = step(
            "gaussian_noise", *gaussian_noise(img, rng.uniform(*profile.gaussian_noise_sigma), rng)
        )
    if maybe(profile.p_salt_pepper):
        img = step("salt_pepper", *salt_pepper(img, rng.uniform(*profile.salt_pepper_amount), rng))
    if maybe(profile.p_jpeg):
        img = step("jpeg", *jpeg_artifacts(img, rng.randint(*profile.jpeg_quality)))

    return DegradationResult(image=img, homography=h_total, applied=applied)
