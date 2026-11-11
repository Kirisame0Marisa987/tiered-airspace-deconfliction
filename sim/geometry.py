"""Planar geometry for airspace footprints.

Every footprint is a list of convex polygons (``parts``).  This keeps the
implementation dependency-free (no shapely) while still supporting circles,
sectors, racetracks, polygonal areas and buffered corridors.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from .schema import Point, Shape

EPS = 1e-9


# ---------------------------------------------------------------- basics
def polygon_area(poly: Sequence[Point]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_centroid(poly: Sequence[Point]) -> Point:
    # Use the signed area so clockwise and counter-clockwise rings both work.
    cross_sum = 0.0
    cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cross_sum += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(cross_sum) < EPS:
        xs = [p[0] for p in poly] or [0.0]
        ys = [p[1] for p in poly] or [0.0]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    return cx / (3 * cross_sum), cy / (3 * cross_sum)


def bbox(poly: Sequence[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + EPS) + x1):
            inside = not inside
    return inside


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    d1 = _orient(c, d, a)
    d2 = _orient(c, d, b)
    d3 = _orient(a, b, c)
    d4 = _orient(a, b, d)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    ):
        return True
    return False


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < EPS:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polygons_intersect(p: Sequence[Point], q: Sequence[Point]) -> bool:
    if polygon_area(p) < EPS or polygon_area(q) < EPS:
        return False
    for pt in p:
        if point_in_polygon(pt, q):
            return True
    for pt in q:
        if point_in_polygon(pt, p):
            return True
    n, m = len(p), len(q)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(m):
            c, d = q[j], q[(j + 1) % m]
            if segments_intersect(a, b, c, d):
                return True
    return False


def _segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def polygon_distance(p: Sequence[Point], q: Sequence[Point]) -> float:
    if polygons_intersect(p, q):
        return 0.0
    best = float("inf")
    n, m = len(p), len(q)
    for i in range(n):
        a, b = p[i], p[(i + 1) % n]
        for j in range(m):
            c, d = q[j], q[(j + 1) % m]
            best = min(best, _segment_distance(a, b, c, d))
    return best


def _bbox_gap_box(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(b[0] - a[2], a[0] - b[2]))
    dy = max(0.0, max(b[1] - a[3], a[1] - b[3]))
    return math.hypot(dx, dy)


def shape_distance(a: Shape, b: Shape) -> float:
    """Minimum distance between two multi-part footprints.

    Part bounding boxes give a cheap lower bound, so distant parts are never
    passed to the exact segment-distance routine.
    """
    best = float("inf")
    bboxes_a = [(p, bbox(p)) for p in a.parts]
    bboxes_b = [(p, bbox(p)) for p in b.parts]
    for pa, ba in bboxes_a:
        for pb, bb in bboxes_b:
            if _bbox_gap_box(ba, bb) >= best:
                continue
            d = polygon_distance(pa, pb)
            if d < best:
                best = d
                if best <= EPS:
                    return 0.0
    return 0.0 if best == float("inf") else best


# ---------------------------------------------------------------- clipping
def _clip_polygon(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Sutherland-Hodgman: clip a convex subject by a convex clip polygon (CCW)."""
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % n]

        def inside(p: Point) -> bool:
            return _orient(a, b, p) >= -EPS

        def intersect(p: Point, q: Point) -> Point:
            x1, y1 = p
            x2, y2 = q
            x3, y3 = a
            x4, y4 = b
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < EPS:
                return q
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)

        input_list = output
        output = []
        s = input_list[-1]
        for e in input_list:
            if inside(e):
                if not inside(s):
                    output.append(intersect(s, e))
                output.append(e)
            elif inside(s):
                output.append(intersect(s, e))
            s = e
    return output


def _ensure_ccw(poly: Sequence[Point]) -> list[Point]:
    signed = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        signed += x1 * y2 - x2 * y1
    return list(poly) if signed >= 0 else list(reversed(poly))


def shape_intersection_area(a: Shape, b: Shape) -> float:
    total = 0.0
    for pa in a.parts:
        pa = _ensure_ccw(pa)
        for pb in b.parts:
            pb = _ensure_ccw(pb)
            if not polygons_intersect(pa, pb):
                continue
            inter = _clip_polygon(pa, pb)
            total += polygon_area(inter)
    return total


# ---------------------------------------------------------------- builders
def _regular_polygon(center: Point, radius: float, n: int = 36) -> list[Point]:
    cx, cy = center
    return [
        (cx + radius * math.cos(2 * math.pi * i / n), cy + radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def circle_shape(center: Point, radius_km: float) -> Shape:
    return Shape("circle", [_regular_polygon(center, radius_km)], {"center": center, "radius": radius_km})


def sector_shape(center: Point, radius_km: float, start_deg: float, end_deg: float) -> Shape:
    """Sector with angles in degrees, 0 = +x, CCW positive; may wrap 360."""
    cx, cy = center
    if end_deg < start_deg:
        end_deg += 360.0
    span = end_deg - start_deg
    n = max(6, int(span / 10) + 2)
    pts: list[Point] = [(cx, cy)]
    for i in range(n + 1):
        ang = math.radians(start_deg + span * i / n)
        pts.append((cx + radius_km * math.cos(ang), cy + radius_km * math.sin(ang)))
    return Shape("sector", [pts], {"center": center, "radius": radius_km,
                                   "start_deg": start_deg, "end_deg": end_deg})


def racetrack_shape(center: Point, length_km: float, width_km: float, orient_deg: float = 0.0) -> Shape:
    """Stadium / racetrack orbit: two half-circles joined by straight sides."""
    cx, cy = center
    r = width_km / 2.0
    half = max(0.0, (length_km - width_km) / 2.0)
    ang = math.radians(orient_deg)
    ux, uy = math.cos(ang), math.sin(ang)
    nx, ny = -uy, ux
    pts: list[Point] = []
    # right cap at +half
    for i in range(13):
        t = -math.pi / 2 + math.pi * i / 12
        pts.append((cx + ux * half + r * math.cos(t + ang), cy + uy * half + r * math.sin(t + ang)))
    # left cap at -half
    for i in range(13):
        t = math.pi / 2 + math.pi * i / 12
        pts.append((cx - ux * half + r * math.cos(t + ang), cy - uy * half + r * math.sin(t + ang)))
    return Shape("racetrack", [pts], {"center": center, "length": length_km,
                                      "width": width_km, "orient_deg": orient_deg})


def corridor_shape(polyline: Sequence[Point], width_km: float) -> Shape:
    """Buffered polyline: one convex rectangle per segment plus a disc at joints."""
    half = width_km / 2.0
    parts: list[list[Point]] = []
    for i in range(len(polyline) - 1):
        (x1, y1), (x2, y2) = polyline[i], polyline[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < EPS:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        parts.append([
            (x1 + nx * half, y1 + ny * half),
            (x2 + nx * half, y2 + ny * half),
            (x2 - nx * half, y2 - ny * half),
            (x1 - nx * half, y1 - ny * half),
        ])
        parts.append(_regular_polygon((x2, y2), half, n=16))
    if not parts:
        parts.append(_regular_polygon(tuple(polyline[0]), half, n=16))
    return Shape("corridor", parts, {"polyline": list(polyline), "width": width_km})


def polygon_shape(vertices: Sequence[Point]) -> Shape:
    verts = _ensure_ccw(vertices)
    # assume input is convex or near-convex; for non-convex, fan triangulation
    if _is_convex(verts):
        return Shape("polygon", [verts], {})
    parts = []
    for i in range(1, len(verts) - 1):
        parts.append([verts[0], verts[i], verts[i + 1]])
    return Shape("polygon", parts, {})


def _is_convex(poly: Sequence[Point]) -> bool:
    n = len(poly)
    sign = 0
    for i in range(n):
        o = _orient(poly[i], poly[(i + 1) % n], poly[(i + 2) % n])
        if abs(o) < EPS:
            continue
        s = 1 if o > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


# ---------------------------------------------------------------- transforms
def translate_shape(shape: Shape, dx: float, dy: float) -> Shape:
    parts = [[(x + dx, y + dy) for x, y in part] for part in shape.parts]
    meta = dict(shape.meta)
    if "center" in meta:
        cx, cy = meta["center"]
        meta["center"] = (cx + dx, cy + dy)
    if "polyline" in meta:
        meta["polyline"] = [(x + dx, y + dy) for x, y in meta["polyline"]]
    return Shape(shape.kind, parts, meta)


def scale_shape(shape: Shape, ratio: float) -> Shape:
    """Scale a footprint about its centroid by ``ratio`` (ratio <= 1 shrinks)."""
    if ratio >= 1.0 - EPS:
        return shape
    # use the largest part's centroid as the anchor
    anchor_poly = max(shape.parts, key=polygon_area)
    ax, ay = polygon_centroid(anchor_poly)
    parts = [[(ax + (x - ax) * ratio, ay + (y - ay) * ratio) for x, y in part] for part in shape.parts]
    meta = dict(shape.meta)
    if "center" in meta:
        meta["center"] = (ax + (meta["center"][0] - ax) * ratio, ay + (meta["center"][1] - ay) * ratio)
    if "radius" in meta:
        meta["radius"] = meta["radius"] * ratio
    if "polyline" in meta:
        meta["polyline"] = [(ax + (x - ax) * ratio, ay + (y - ay) * ratio) for x, y in meta["polyline"]]
    return Shape(shape.kind, parts, meta)


def shape_area(shape: Shape) -> float:
    return sum(polygon_area(p) for p in shape.parts)


def shape_centroid(shape: Shape) -> Point:
    best = max(shape.parts, key=polygon_area)
    return polygon_centroid(best)


def lateral_offset(shape: Shape, distance_km: float, bearing_deg: float) -> Shape:
    ang = math.radians(bearing_deg)
    return translate_shape(shape, distance_km * math.cos(ang), distance_km * math.sin(ang))
