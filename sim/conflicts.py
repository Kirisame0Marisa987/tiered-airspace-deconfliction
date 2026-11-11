"""Four-dimensional conflict screening for airspace blocks."""
from __future__ import annotations

from itertools import combinations

from . import geometry as geo
from .doctrine import default_classification, severity
from .schema import AirspaceBlock, Conflict, Scenario

DEFAULT_SEP_H_KM = 5.0
DEFAULT_SEP_TIME_MIN = 0.0
DEFAULT_SEP_ALT_FT = 0.0


def _bbox_gap(a, b) -> float:
    ax0, ay0, ax1, ay1 = geo.bbox(max(a.shape.parts, key=geo.polygon_area))
    bx0, by0, bx1, by1 = geo.bbox(max(b.shape.parts, key=geo.polygon_area))
    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    return (dx * dx + dy * dy) ** 0.5


def pair_conflict(
    a,
    b,
    cid: str = "",
    sep_h_km: float = DEFAULT_SEP_H_KM,
    sep_time_min: float = DEFAULT_SEP_TIME_MIN,
    sep_alt_ft: float = DEFAULT_SEP_ALT_FT,
) -> Conflict | None:
    """Return the conflict for one pair, or None when the pair is separated."""
    if a.time_end_min <= b.time_start_min + sep_time_min:
        return None
    if b.time_end_min <= a.time_start_min + sep_time_min:
        return None
    alt_overlap = a.altitude_overlap(b, sep_alt_ft)
    if alt_overlap <= 0:
        return None
    # Cheap bounding-box reject before exact geometry.
    if _bbox_gap(a, b) > sep_h_km + 2.0:
        return None
    gap = geo.shape_distance(a.shape, b.shape)
    if gap >= sep_h_km:
        return None
    inter = geo.shape_intersection_area(a.shape, b.shape)
    min_area = max(1e-6, min(geo.shape_area(a.shape), geo.shape_area(b.shape)))
    if inter > 0:
        ratio = min(1.0, inter / min_area)
    else:
        ratio = 0.10 * max(0.0, 1.0 - gap / max(sep_h_km, 1e-6))
    t_overlap = a.time_overlap(b, sep_time_min)
    if t_overlap <= 0:
        return None
    c = Conflict(
        id=cid,
        a=a.id,
        b=b.id,
        horizontal_gap_km=round(gap, 3),
        alt_overlap_ft=round(alt_overlap, 1),
        time_overlap_min=round(t_overlap, 1),
        overlap_ratio=round(ratio, 4),
        classification=default_classification(a, b),
    )
    c.severity = severity(c)
    return c


def detect_conflicts(
    scenario: Scenario,
    sep_h_km: float = DEFAULT_SEP_H_KM,
    sep_time_min: float = DEFAULT_SEP_TIME_MIN,
    sep_alt_ft: float = DEFAULT_SEP_ALT_FT,
) -> list[Conflict]:
    blocks = scenario.blocks
    out: list[Conflict] = []
    cid = 0
    for a, b in combinations(blocks, 2):
        c = pair_conflict(a, b, f"C{cid:04d}", sep_h_km, sep_time_min, sep_alt_ft)
        if c is not None:
            out.append(c)
            cid += 1
    # Most severe first, then largest overlap.
    out.sort(key=lambda c: (-c.severity, -c.overlap_ratio, c.id))
    return out


def hard_conflicts(conflicts: list[Conflict]) -> list[Conflict]:
    return [c for c in conflicts if c.classification == "hard" and not c.resolved]


def unresolved(conflicts: list[Conflict]) -> list[Conflict]:
    return [c for c in conflicts if not c.resolved and c.classification != "compatible"]
