"""Data model for the tiered-deconfliction simulation.

All spatial units are kilometres, altitude in feet, time in minutes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]


@dataclass
class Shape:
    """A block's horizontal footprint, represented as one or more convex parts."""

    kind: str                      # circle | sector | racetrack | corridor | polygon
    parts: list[list[Point]]       # convex polygons (CCW)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AirspaceBlock:
    id: str
    acm_type: str                  # CAP, AAR, AEW, CAS, RECCE, ROZ, HIDACZ, MRR, LLTR, TC, AIRCOR, JEZ, LMEZ, SHORAD, ACA, RFA
    user: str                      # using unit / mission package
    mission: str                   # mission description (free text)
    shape: Shape
    alt_low_ft: float
    alt_high_ft: float
    time_start_min: float
    time_end_min: float
    given: bool = False            # imposed by higher authority (not adjustable)
    importance: float = 0.5        # 0..1 task importance
    adjustability: float = 0.5     # 0..1; 0 = fixed, 1 = freely adjustable
    priority: float = 0.0          # computed by priority model
    parent: str | None = None      # for corridors: the mission package
    doctrine_ref: str = ""         # clause supporting the block

    def altitude_overlap(self, other: "AirspaceBlock", sep_ft: float = 0.0) -> float:
        lo = max(self.alt_low_ft, other.alt_low_ft)
        hi = min(self.alt_high_ft, other.alt_high_ft)
        return max(0.0, hi - lo - sep_ft)

    def time_overlap(self, other: "AirspaceBlock", sep_min: float = 0.0) -> float:
        lo = max(self.time_start_min, other.time_start_min)
        hi = min(self.time_end_min, other.time_end_min)
        return max(0.0, hi - lo - sep_min)


@dataclass
class Conflict:
    id: str
    a: str
    b: str
    horizontal_gap_km: float       # min distance between footprints
    alt_overlap_ft: float
    time_overlap_min: float
    overlap_ratio: float           # horizontal intersection area / min(area)
    severity: int = 1              # 1 advisory, 2 affects feasibility, 3 makes plan infeasible
    classification: str = "unknown"  # compatible | soft | hard
    rule_id: str = ""
    rationale: str = ""
    doctrine_ref: str = ""
    resolved_by: str = ""
    resolved: bool = False
    round: int = 0
    chain: bool = False            # discovered during incremental recheck


@dataclass
class Adjustment:
    block_id: str
    strategy: str                  # time | altitude | lateral | shrink
    delta_time_min: float = 0.0
    delta_alt_ft: float = 0.0
    delta_lateral_km: float = 0.0
    shrink_ratio: float = 0.0
    bearing_deg: float = 0.0
    cost: float = 0.0


@dataclass
class ResolutionRecord:
    conflict_id: str
    tier: str                      # rule | negotiation | arbitration | escalated
    adjustments: list[Adjustment]
    rounds: int
    rationale: str
    doctrine_ref: str
    cost: float


@dataclass
class Scenario:
    id: str
    seed: int
    region_km: tuple[float, float]
    blocks: list[AirspaceBlock]
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
