"""Group positioned text into visual lines, in reading order.

This is shared by the renderer, the OCR adapters and the evaluation harness on
purpose: all three must agree on what "a line" means, or they produce
incomparable text.

The bug this module exists to prevent was real. The renderer originally
assigned a line id per *draw call*, so a right-aligned value drawn separately
from its left-aligned label landed on its own line -- producing gold text like

    Invoice No:
    INV-2026-4733

for a row that any OCR engine reads as a single line. Comparing engine output
against that reference inflates CER and WER with pure line-break noise that has
nothing to do with recognition quality. Lines are therefore derived from
geometry, never from how the text happened to be emitted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

Box = tuple[float, float, float, float]


def _median(values: list[float]) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def group_lines(
    items: Sequence[Any],
    get_bbox: Callable[[Any], Box],
    tolerance_frac: float = 0.6,
) -> list[list[Any]]:
    """Group items into visual lines, each sorted left-to-right.

    Two items share a line when their vertical centres sit within
    `tolerance_frac` of the median item height. Scaling the tolerance by the
    median height keeps the rule resolution-independent, so it behaves
    identically at 150 and 300 dpi.

    Returned lines are ordered top-to-bottom by their topmost edge, with the
    leftmost edge breaking ties -- the tie-break matters for side-by-side
    blocks that start at the same y.
    """
    if not items:
        return []

    heights = [get_bbox(i)[3] - get_bbox(i)[1] for i in items]
    tolerance = (_median(heights) or 1.0) * tolerance_frac

    ordered = sorted(items, key=lambda i: ((get_bbox(i)[1] + get_bbox(i)[3]) / 2, get_bbox(i)[0]))

    lines: list[list[Any]] = []
    current: list[Any] = []
    centre = None
    for item in ordered:
        box = get_bbox(item)
        item_centre = (box[1] + box[3]) / 2
        if centre is None or abs(item_centre - centre) <= tolerance:
            current.append(item)
            # Track a running mean so a line does not drift as it accumulates
            # slightly-offset words.
            centre = item_centre if centre is None else (centre + item_centre) / 2
        else:
            lines.append(current)
            current = [item]
            centre = item_centre
    if current:
        lines.append(current)

    for line in lines:
        line.sort(key=lambda i: get_bbox(i)[0])

    lines.sort(key=lambda ln: (min(get_bbox(i)[1] for i in ln), min(get_bbox(i)[0] for i in ln)))
    return lines


def reading_order_text(
    items: Sequence[Any],
    get_bbox: Callable[[Any], Box],
    get_text: Callable[[Any], str],
    tolerance_frac: float = 0.6,
) -> str:
    """Render positioned items as plain text, one visual line per output line."""
    lines = group_lines(items, get_bbox, tolerance_frac)
    return "\n".join(" ".join(get_text(i) for i in line) for line in lines)
