"""Cross-platform TrueType font resolution for the invoice renderer.

The renderer needs real TTFs (PIL's built-in bitmap font has no metrics worth
speaking of, and word-level bounding boxes have to be pixel-accurate for the
OCR ground truth to mean anything). Rather than vendoring fonts, probe the
usual system font directories and fall back gracefully.

Font *family* is itself a layout-diversity axis: FATURA's point is that
template-tuned extractors break on unseen layouts, so the synthetic corpus
varies typeface alongside geometry.
"""

from __future__ import annotations

import functools
from pathlib import Path

from PIL import ImageFont

# (family_key, [candidate filenames]) — first hit wins on any given machine.
FONT_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "sans": {
        "regular": ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
        "bold": ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
    },
    "humanist": {
        "regular": ["calibri.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"],
        "bold": ["calibrib.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"],
    },
    "serif": {
        "regular": [
            "times.ttf",
            "Times New Roman.ttf",
            "DejaVuSerif.ttf",
            "LiberationSerif-Regular.ttf",
        ],
        "bold": ["timesbd.ttf", "DejaVuSerif-Bold.ttf", "LiberationSerif-Bold.ttf"],
    },
    "mono": {
        "regular": ["cour.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"],
        "bold": ["courbd.ttf", "DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"],
    },
}

FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path.home() / "Library/Fonts",
]

FAMILIES = list(FONT_CANDIDATES)


class FontUnavailableError(RuntimeError):
    """No usable TrueType font was found anywhere on this machine."""


@functools.cache
def _find_font_file(filename: str) -> Path | None:
    for d in FONT_DIRS:
        if not d.is_dir():
            continue
        direct = d / filename
        if direct.is_file():
            return direct
        # Linux nests fonts in per-family subdirectories.
        for hit in d.rglob(filename):
            return hit
    return None


@functools.cache
def resolve(family: str = "sans", weight: str = "regular") -> Path:
    """Return a path to a usable TTF for (family, weight).

    Falls back across weights and then across families before giving up, so a
    machine with only one font still renders (just with less visual variety).
    """
    tried: list[str] = []
    order = [(family, weight), (family, "regular")]
    order += [(f, w) for f in FAMILIES for w in ("regular", "bold")]
    for fam, wt in order:
        for name in FONT_CANDIDATES.get(fam, {}).get(wt, []):
            tried.append(name)
            hit = _find_font_file(name)
            if hit is not None:
                return hit
    raise FontUnavailableError(
        f"No TrueType font found in {[str(d) for d in FONT_DIRS]}; tried {sorted(set(tried))}"
    )


@functools.lru_cache(maxsize=256)
def load(family: str = "sans", weight: str = "regular", size: int = 14) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(resolve(family, weight)), size=size)


def available_families() -> list[str]:
    """Families that actually resolve to a distinct file on this machine."""
    seen: dict[str, str] = {}
    for fam in FAMILIES:
        try:
            seen[fam] = str(resolve(fam, "regular"))
        except FontUnavailableError:
            continue
    # Collapse families that fell back onto the same physical file.
    unique: dict[str, str] = {}
    for fam, path in seen.items():
        unique.setdefault(path, fam)
    return sorted(unique.values())
