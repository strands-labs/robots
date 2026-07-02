"""Decide whether the bed is "made" — a tolerant, **pillow-anchored** corner test.

Issue #2 wants the goal kept intentionally imperfect ("good enough", not
Figure-perfect). The robots draw the sheet's two head-side corners up toward the
head of the bed; the bed counts as made when each of those corners has been
pulled **up into the head region** — i.e. within a generous radius of the
head-end mattress corner, a radius that **extends out to the pillow**. So a cover
drawn up to the pillows passes; a cover left bunched at mid-mattress does not.

This reads the sheet-corner positions straight from the PhysX tensor cloth view
(:func:`cloth.corner_world_positions`) — **no top-down overseer camera**, which
would break sim-to-real (issue #2: each robot only has its own head sensors). The
test is purely geometric on the live cloth, so it is an honest physical goal, not
a trust-based tally of how many ``putDownBedSheet`` calls were made.

The legacy colour-based top-down coverage estimate (:func:`estimate_coverage`)
is kept below for reference but is no longer the goal signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

XYZ = Tuple[float, float, float]

# How far past the mattress-corner→pillow span a sheet corner may sit and still
# count as "drawn up to the head". The robots grip the head EDGE inboard of the
# true corner (the corner itself is at the robot's far side, out of a planted
# reach), so the corner trails the gripped point; this pad keeps the tolerant
# goal honest about that lag while still rejecting a cover left at mid-bed.
DEFAULT_PAD_M = 0.25


@dataclass
class CornerPlacement:
    corner: str        # head-side sheet/bed corner label (NW or SW)
    distance: float    # m from the head-end mattress corner (xy plane)
    radius: float      # acceptance radius (reaches the mattress corner out to the pillow)
    placed: bool       # within the radius → drawn up to the head


@dataclass
class BedMade:
    placements: Dict[str, CornerPlacement] = field(default_factory=dict)

    @property
    def placed(self) -> Dict[str, bool]:
        return {k: p.placed for k, p in self.placements.items()}

    @property
    def placed_count(self) -> int:
        return sum(1 for p in self.placements.values() if p.placed)

    @property
    def made(self) -> bool:
        # Made when every head-side corner we evaluated is drawn up.
        return bool(self.placements) and all(p.placed for p in self.placements.values())

    def as_dict(self) -> dict:
        return {
            "bed_made": self.made,
            "placed": self.placed,
            "placed_count": self.placed_count,
            "corners": {k: {"distance_m": round(p.distance, 3),
                            "radius_m": round(p.radius, 3), "placed": p.placed}
                        for k, p in self.placements.items()},
        }


def _xy(p: XYZ) -> Tuple[float, float]:
    return float(p[0]), float(p[1])


def _dist_xy(a: XYZ, b: XYZ) -> float:
    (ax, ay), (bx, by) = _xy(a), _xy(b)
    return math.hypot(ax - bx, ay - by)


def pillow_anchored_radius(mattress_corner: XYZ, pillow: XYZ, pad: float = DEFAULT_PAD_M) -> float:
    """Acceptance radius around a head-end mattress corner that reaches out to the
    pillow (plus a tolerance ``pad`` for the gripped-edge-vs-true-corner lag)."""
    return _dist_xy(mattress_corner, pillow) + pad


def _nearest_pillow(corner: XYZ, pillows: Dict[str, XYZ]) -> Optional[XYZ]:
    if not pillows:
        return None
    # Same-side pillow: the one closest to the head corner (matches y-side).
    return min(pillows.values(), key=lambda p: _dist_xy(corner, p))


def evaluate_bed_made(
    sheet_corners: Dict[str, XYZ],
    head_corners: Dict[str, XYZ],
    pillows: Dict[str, XYZ],
    *,
    pad: float = DEFAULT_PAD_M,
) -> BedMade:
    """Tolerant, pillow-anchored "made" verdict.

    ``sheet_corners``  {label: (x,y,z)} live cloth corners (cloth.corner_world_positions).
    ``head_corners``   {label: (x,y,z)} the head-end mattress corners to test (e.g. NW, SW).
    ``pillows``        {name: (x,y,z)} pillow centres; the acceptance radius reaches the
                       nearest pillow so a cover pulled up to the pillows counts as made.

    A head-side sheet corner is *placed* when its xy distance to the matching
    head-end mattress corner is within that pillow-reaching radius.
    """
    out = BedMade()
    for label, mattress in head_corners.items():
        sc = sheet_corners.get(label)
        if sc is None:
            continue
        pillow = _nearest_pillow(mattress, pillows)
        radius = pillow_anchored_radius(mattress, pillow, pad) if pillow else (0.5 + pad)
        d = _dist_xy(sc, mattress)
        out.placements[label] = CornerPlacement(label, d, radius, d <= radius)
    return out


# ── Legacy colour-based top-down coverage (reference only; not the goal) ───────
@dataclass
class Coverage:
    coverage: float            # fraction of the bed top covered by the sheet
    centerline_aligned: bool   # sheet straddles the head->foot centreline
    midpoint_covered: bool     # the centre of the bed is covered
    good_enough: bool
    bed_px: int

    def as_dict(self) -> dict:
        return {
            "coverage_pct": round(self.coverage * 100, 1),
            "centerline_aligned": self.centerline_aligned,
            "midpoint_covered": self.midpoint_covered,
            "good_enough": self.good_enough,
        }


def _masks(rgb):
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    # Sheet: light and slightly bluish (high in all channels, blue >= red).
    sheet = (r > 140) & (g > 140) & (b > 150) & (b >= r - 10)
    # Bare bed: brown (red dominant, clearly warmer than blue).
    bed = (r > 60) & (r < 210) & (g < r) & (b < g) & ((r - b) > 25)
    return sheet, bed


def estimate_coverage(rgb, coverage_threshold: float = 0.5) -> Coverage:
    """Estimate bed coverage from a top-down RGB image framed on the bed (legacy)."""

    sheet, bed = _masks(rgb)
    bedtop = sheet | bed
    bed_px = int(bedtop.sum())
    coverage = float(sheet.sum()) / max(1, bed_px)

    H, W = sheet.shape
    cy0, cy1 = int(H * 0.42), int(H * 0.58)
    cx0, cx1 = int(W * 0.42), int(W * 0.58)
    centre = sheet[cy0:cy1, cx0:cx1]
    midpoint_covered = bool(centre.mean() > 0.5)

    by0, by1 = int(H * 0.45), int(H * 0.55)
    band = sheet[by0:by1, :]
    col_has_sheet = band.any(axis=0)
    centerline_aligned = bool(col_has_sheet.mean() > 0.6)

    good_enough = (coverage >= coverage_threshold) or (centerline_aligned and midpoint_covered)
    return Coverage(coverage, centerline_aligned, midpoint_covered, good_enough, bed_px)
