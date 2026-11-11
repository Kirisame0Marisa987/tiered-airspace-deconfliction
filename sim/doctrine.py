"""Doctrine knowledge base: compatibility matrix, separation rules, priority model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import AirspaceBlock, Conflict

AREA_TYPES = {"CAP", "AAR", "AEW", "CAS", "RECCE", "ROZ", "HIDACZ"}
CORRIDOR_TYPES = {"MRR", "LLTR", "TC", "AIRCOR"}
GIVEN_TYPES = {"JEZ", "LMEZ", "SHORAD", "ACA", "RFA"}

# Default qualitative classification for a geometrically overlapping pair.
# "soft" means a doctrine rule can normally separate them; "hard" means the
# pair cannot share the same four-dimensional volume.
_COMPAT = {
    ("AREA", "AREA"): "soft",
    ("AREA", "CORRIDOR"): "soft",
    ("CORRIDOR", "CORRIDOR"): "soft",
    ("AREA", "GIVEN"): "hard",
    ("CORRIDOR", "GIVEN"): "hard",
    ("GIVEN", "GIVEN"): "hard",
}


def category(acm_type: str) -> str:
    if acm_type in AREA_TYPES:
        return "AREA"
    if acm_type in CORRIDOR_TYPES:
        return "CORRIDOR"
    if acm_type in GIVEN_TYPES:
        return "GIVEN"
    return "AREA"


def default_classification(a: AirspaceBlock, b: AirspaceBlock) -> str:
    if a.user == b.user and a.acm_type == b.acm_type:
        return "compatible"
    if a.user == b.user and a.parent and a.parent == b.parent:
        return "compatible"
    ca, cb = category(a.acm_type), category(b.acm_type)
    key = tuple(sorted((ca, cb)))
    return _COMPAT.get(key, "soft")


DOCTRINE_CLAUSES = {
    "D1": "Airspace blocks assigned to the same mission package may be shared when the using units coordinate their deconfliction.",
    "D2": "Blocks of different users shall maintain at least 1,000 ft vertical separation or 15 min time separation.",
    "D3": "Parallel corridors shall maintain at least 5 km lateral separation.",
    "D4": "Air-defense engagement zones and restricted firing areas are fixed by higher authority and shall not be adjusted.",
    "D5": "An AEW orbit shall not be displaced if the displacement creates a sensor coverage gap.",
    "D6": "An AAR track shall remain outside threat engagement zones and within 30 min of the receiver corridor.",
    "D7": "A CAP station shall be positioned between the threat axis and the protected asset.",
    "D8": "A corridor shall not be routed through an active engagement zone.",
}

# AHP weights for the priority model (importance, urgency, inverse adjustability).
PRIORITY_WEIGHTS = (0.50, 0.30, 0.20)
# Adjustment-cost weights (time, altitude, lateral, area).
COST_WEIGHTS = (0.30, 0.25, 0.30, 0.15)

# Normalisation scales.
TIME_SCALE_MIN = 60.0
ALT_SCALE_FT = 6000.0
LATERAL_SCALE_KM = 20.0


@dataclass
class RuleMatch:
    rule_id: str
    strategy: str
    delta: float
    doctrine_ref: str
    description: str
    bearing: float = 0.0


def compute_priority(block: AirspaceBlock, horizon_min: float) -> float:
    urgency = max(0.0, min(1.0, 1.0 - block.time_start_min / max(horizon_min, 1e-6)))
    inv_adjust = 1.0 - block.adjustability
    w1, w2, w3 = PRIORITY_WEIGHTS
    return round(w1 * block.importance + w2 * urgency + w3 * inv_adjust, 4)


def adjustment_cost(delta_time: float, delta_alt: float, delta_lateral: float, shrink: float) -> float:
    w1, w2, w3, w4 = COST_WEIGHTS
    return round(
        w1 * abs(delta_time) / TIME_SCALE_MIN
        + w2 * abs(delta_alt) / ALT_SCALE_FT
        + w3 * abs(delta_lateral) / LATERAL_SCALE_KM
        + w4 * abs(shrink) / 0.20,
        4,
    )


def rule_candidates(a: AirspaceBlock, b: AirspaceBlock) -> list[RuleMatch]:
    """Return doctrine rules that could separate this pair, cheapest first."""
    out: list[RuleMatch] = []
    # Same package: no action needed.
    if default_classification(a, b) == "compatible":
        return [RuleMatch("D1", "none", 0.0, "D1", "same mission package may share the volume")]

    # A given block cannot be moved; the other side must adjust.
    given = a if a.given else (b if b.given else None)
    if given is not None:
        mover = b if given is a else a
    else:
        mover = a if a.priority <= b.priority else b
    other = b if mover is a else a
    if given is not None and mover.adjustability <= 1e-6:
        return []

    # Altitude layering.
    lo, hi = 0.0, 30000.0
    if mover.alt_low_ft - 1000.0 >= lo:
        out.append(RuleMatch("D2-alt-down", "altitude", -1000.0, "D2",
                             "lower the block by 1,000 ft"))
    if mover.alt_high_ft + 1000.0 <= hi:
        out.append(RuleMatch("D2-alt-up", "altitude", 1000.0, "D2",
                             "raise the block by 1,000 ft"))
    # Time separation: both directions, subject to the horizon being checked later.
    out.append(RuleMatch("D2-time", "time", 15.0, "D2", "shift the block 15 min later"))
    out.append(RuleMatch("D2-time", "time", -15.0, "D2", "shift the block 15 min earlier"))
    # Lateral separation: offset directly away from the other block.
    import math
    ax, ay = _shape_anchor(mover)
    bx, by = _shape_anchor(other)
    bearing = math.degrees(math.atan2(ay - by, ax - bx)) if (ax, ay) != (bx, by) else 90.0
    out.append(RuleMatch("D3-lateral", "lateral", 5.0, "D3",
                         "offset the block 5 km away from the other block",
                         bearing=bearing))
    # Area shrinkage for large area blocks.
    if a.acm_type in AREA_TYPES or b.acm_type in AREA_TYPES:
        out.append(RuleMatch("area-shrink", "shrink", 0.10, "D2", "shrink the block by 10%"))

    # Rank by adjustment cost.
    out.sort(key=lambda r: adjustment_cost(
        r.delta if r.strategy == "time" else 0.0,
        r.delta if r.strategy == "altitude" else 0.0,
        r.delta if r.strategy == "lateral" else 0.0,
        r.delta if r.strategy == "shrink" else 0.0,
    ))
    return out


def _shape_anchor(block: AirspaceBlock) -> tuple[float, float]:
    from . import geometry as geo
    return geo.shape_centroid(block.shape)


def doctrine_context(a: AirspaceBlock, b: AirspaceBlock) -> str:
    """Compact doctrine text injected into an LLM prompt."""
    keys = ["D1", "D2", "D3"]
    if "AEW" in (a.acm_type, b.acm_type):
        keys.append("D5")
    if "AAR" in (a.acm_type, b.acm_type):
        keys.append("D6")
    if "CAP" in (a.acm_type, b.acm_type):
        keys.append("D7")
    if category(a.acm_type) == "CORRIDOR" or category(b.acm_type) == "CORRIDOR":
        keys.append("D8")
    if a.given or b.given:
        keys.append("D4")
    return "\n".join(f"[{k}] {DOCTRINE_CLAUSES[k]}" for k in keys)


def severity(conflict: Conflict) -> int:
    """Severity from overlap ratio, altitude and time overlap."""
    if conflict.classification == "compatible":
        return 1
    score = 0
    if conflict.overlap_ratio >= 0.30:
        score += 1
    if conflict.alt_overlap_ft >= 2000:
        score += 1
    if conflict.time_overlap_min >= 30:
        score += 1
    if conflict.classification == "hard":
        score += 1
    return max(1, min(3, score))
