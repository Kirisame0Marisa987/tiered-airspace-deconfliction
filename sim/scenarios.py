"""Synthetic four-dimensional airspace scenarios.

Coordinates are kilometres in a local planar frame.  The scenarios are
deliberately adversarial: area blocks, corridors and higher-authority zones
overlap in space, altitude and time so that the deconfliction stack has work
to do.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

from . import geometry as geo
from .doctrine import AREA_TYPES, CORRIDOR_TYPES, GIVEN_TYPES, compute_priority
from .schema import AirspaceBlock, Scenario

USERS = ["BLUE-1", "BLUE-2", "BLUE-3", "BLUE-4", "RED", "HQ"]
MISSIONS = {
    "CAP": "protect the defended asset against the eastern threat axis",
    "AAR": "refuel the strike package before its ingress",
    "AEW": "maintain continuous wide-area radar coverage",
    "CAS": "provide close air support to the forward battalion",
    "RECCE": "conduct tactical reconnaissance of the eastern corridor",
    "ROZ": "restrict artillery fire while friendly aircraft transit",
    "HIDACZ": "concentrate helicopter operations below the transit layer",
    "MRR": "support the main supply route",
    "LLTR": "provide a low-level transit route",
    "TC": "connect the rear base to the forward operating base",
    "AIRCOR": "route the strike package to the ingress point",
    "JEZ": "deny the eastern engagement zone to friendly aircraft",
    "LMEZ": "protect the northern missile engagement zone",
    "SHORAD": "protect the forward air-defense battery",
    "ACA": "coordinate airspace for the coalition boundary",
    "RFA": "restrict firing during the civil evacuation",
}


def _priority(b: AirspaceBlock, horizon: float) -> None:
    b.priority = compute_priority(b, horizon)


def typical_scenario(seed: int = 0, extra: int = 0) -> Scenario:
    """A defended-asset scenario with a full mix of ACM types."""
    horizon = 240.0
    blocks: list[AirspaceBlock] = []

    def add(bid: str, t: str, user: str, shape, lo: float, hi: float,
            t0: float, t1: float, given: bool = False,
            importance: float = 0.5, adjustability: float = 0.5,
            parent: str | None = None, mission: str | None = None) -> None:
        blocks.append(AirspaceBlock(
            id=bid, acm_type=t, user=user, mission=mission or MISSIONS.get(t, t),
            shape=shape, alt_low_ft=lo, alt_high_ft=hi,
            time_start_min=t0, time_end_min=t1, given=given,
            importance=importance, adjustability=adjustability, parent=parent,
        ))

    # Area blocks in the defended-asset package at (200, 150).
    add("B-CAP-1", "CAP", "BLUE-1", geo.circle_shape((170, 150), 30), 15000, 25000,
        0, 180, importance=0.90, adjustability=0.40, parent="PKG-A")
    add("B-AEW-1", "AEW", "BLUE-2", geo.racetrack_shape((115, 190), 120, 24, 0), 24000, 30000,
        0, 240, importance=0.95, adjustability=0.15, parent="PKG-A")
    add("B-AAR-1", "AAR", "BLUE-3", geo.racetrack_shape((85, 105), 90, 20, 0), 18000, 26000,
        20, 220, importance=0.75, adjustability=0.55, parent="PKG-A")
    add("B-CAS-1", "CAS", "BLUE-1", geo.circle_shape((235, 128), 18), 3000, 12000,
        30, 150, importance=0.85, adjustability=0.45, parent="PKG-A")
    add("B-RECCE-1", "RECCE", "BLUE-3", geo.polygon_shape([(250, 190), (300, 195), (305, 225),
                                                          (255, 230)]), 12000, 20000,
        40, 200, importance=0.60, adjustability=0.70)
    add("B-ROZ-1", "ROZ", "BLUE-4", geo.circle_shape((95, 235), 28), 0, 8000,
        0, 240, importance=0.55, adjustability=0.65)
    add("B-HIDACZ-1", "HIDACZ", "BLUE-4", geo.circle_shape((250, 50), 20), 0, 3000,
        20, 200, importance=0.50, adjustability=0.75)

    # Corridors.
    add("B-MRR-1", "MRR", "BLUE-2", geo.corridor_shape([(10, 120), (120, 115), (240, 110), (390, 95)], 12),
        0, 10000, 0, 240, importance=0.70, adjustability=0.55)
    add("B-LLTR-1", "LLTR", "BLUE-3", geo.corridor_shape([(10, 60), (140, 70), (270, 85), (390, 100)], 10),
        0, 6000, 0, 240, importance=0.65, adjustability=0.60)
    add("B-TC-1", "TC", "BLUE-4", geo.corridor_shape([(50, 240), (150, 230), (250, 245), (310, 235)], 12),
        5000, 14000, 0, 220, importance=0.60, adjustability=0.65)
    add("B-AIRCOR-1", "AIRCOR", "BLUE-1", geo.corridor_shape([(175, 5), (190, 90), (200, 160), (210, 270)], 8),
        10000, 22000, 10, 230, importance=0.80, adjustability=0.50, parent="PKG-A")

    # Higher-authority blocks: not adjustable.
    add("G-JEZ-1", "JEZ", "RED", geo.circle_shape((345, 155), 30), 0, 30000,
        0, 240, given=True, importance=1.00, adjustability=0.0)
    add("G-LMEZ-1", "LMEZ", "RED", geo.circle_shape((310, 40), 25), 0, 28000,
        0, 240, given=True, importance=0.95, adjustability=0.0)
    add("G-SHORAD-1", "SHORAD", "RED", geo.circle_shape((285, 140), 14), 0, 18000,
        0, 240, given=True, importance=0.90, adjustability=0.0)
    add("G-ACA-1", "ACA", "HQ", geo.polygon_shape([(330, 180), (395, 180), (395, 240), (335, 245)]),
        0, 30000, 0, 240, given=True, importance=0.85, adjustability=0.0)
    add("G-RFA-1", "RFA", "HQ", geo.circle_shape((245, 270), 18), 0, 15000,
        0, 200, given=True, importance=0.80, adjustability=0.0)

    rng = random.Random(seed)
    types = ["CAP", "AEW", "AAR", "CAS", "RECCE", "ROZ", "HIDACZ",
             "MRR", "LLTR", "TC", "AIRCOR"]
    for i in range(extra):
        t = rng.choice(types)
        user = rng.choice(USERS[:4])
        cx, cy = rng.uniform(60, 340), rng.uniform(40, 250)
        if t in AREA_TYPES:
            shape = geo.circle_shape((cx, cy), rng.uniform(12, 28))
            lo = rng.choice([0, 3000, 9000, 15000, 21000])
            hi = lo + rng.choice([5000, 8000, 10000])
            t0 = rng.uniform(0, 120)
        elif t in CORRIDOR_TYPES:
            pts = [(max(5, cx - rng.uniform(40, 90)), cy)]
            for k in range(2):
                pts.append((min(395, pts[-1][0] + rng.uniform(50, 110)),
                            max(10, min(280, pts[-1][1] + rng.uniform(-60, 60)))))
            shape = geo.corridor_shape(pts, rng.uniform(6, 14))
            lo = rng.choice([0, 3000, 8000])
            hi = lo + rng.choice([5000, 9000])
            t0 = rng.uniform(0, 60)
        else:
            shape = geo.circle_shape((cx, cy), rng.uniform(15, 35))
            lo, hi = 0, 30000
            t0 = 0
        add(f"X-{t}-{i:02d}", t, user, shape, lo, hi, t0,
            min(240.0, t0 + rng.uniform(80, 200)),
            given=t in GIVEN_TYPES,
            importance=rng.uniform(0.4, 0.9),
            adjustability=0.0 if t in GIVEN_TYPES else rng.uniform(0.3, 0.9))

    for b in blocks:
        _priority(b, horizon)
    return Scenario(
        id=f"typical-s{seed}", seed=seed, region_km=(0.0, 0.0, 400.0, 300.0),
        blocks=blocks,
        description="Defended asset with CAP/AEW/AAR/CAS, four corridors and five higher-authority zones.",
        meta={"horizon_min": horizon, "scenario_family": "typical"},
    )


ALT_BANDS = {
    "CAP": (15000, 25000), "AEW": (24000, 30000), "AAR": (18000, 26000),
    "CAS": (3000, 12000), "RECCE": (12000, 20000), "ROZ": (0, 8000),
    "HIDACZ": (0, 3000), "MRR": (0, 10000), "LLTR": (0, 6000),
    "TC": (5000, 14000), "AIRCOR": (10000, 22000),
    "JEZ": (0, 30000), "LMEZ": (0, 28000), "SHORAD": (0, 18000),
    "ACA": (0, 30000), "RFA": (0, 15000),
}
TYPE_WEIGHTS = {
    "CAP": 7, "AEW": 3, "AAR": 3, "CAS": 5, "RECCE": 4, "ROZ": 3, "HIDACZ": 3,
    "MRR": 4, "LLTR": 4, "TC": 3, "AIRCOR": 3,
    "JEZ": 1, "LMEZ": 1, "SHORAD": 1, "ACA": 1, "RFA": 1,
}


def _make_random_block(rng: random.Random, t: str, idx: int) -> AirspaceBlock:
    given = t in GIVEN_TYPES
    if given:
        # Higher-authority zones sit near the periphery, not on top of the package.
        ang = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(120, 175)
        cx = max(25.0, min(375.0, 200 + radius * math.cos(ang)))
        cy = max(25.0, min(275.0, 150 + 0.75 * radius * math.sin(ang)))
    else:
        cx = rng.uniform(45, 355)
        cy = rng.uniform(35, 265)
    if t in AREA_TYPES:
        shape = geo.circle_shape((cx, cy), rng.uniform(14, 26))
        t0 = rng.uniform(0, 100)
        dur = rng.uniform(80, 180)
    elif t in CORRIDOR_TYPES:
        pts = [(max(5.0, cx - rng.uniform(60, 110)), cy)]
        for _ in range(2):
            pts.append((min(395.0, pts[-1][0] + rng.uniform(65, 120)),
                        max(10.0, min(285.0, pts[-1][1] + rng.uniform(-55, 55)))))
        shape = geo.corridor_shape(pts, rng.uniform(6, 12))
        t0 = rng.uniform(0, 70)
        dur = rng.uniform(100, 200)
    else:
        shape = geo.circle_shape((cx, cy), rng.uniform(18, 34))
        t0 = 0.0
        dur = 240.0
    lo, hi = ALT_BANDS.get(t, (0.0, 30000.0))
    return AirspaceBlock(
        id=f"R{idx:03d}-{t}", acm_type=t,
        user="HQ" if t in {"ACA", "RFA"} else ("RED" if given else rng.choice(USERS[:4])),
        mission=MISSIONS.get(t, t), shape=shape,
        alt_low_ft=float(lo), alt_high_ft=float(hi),
        time_start_min=t0, time_end_min=min(240.0, t0 + dur),
        given=given, importance=rng.uniform(0.45, 0.95),
        adjustability=0.0 if given else rng.uniform(0.3, 0.9),
    )


def random_scenario(n_blocks: int = 40, seed: int = 0, density: float = 1.0) -> Scenario:
    """Scalable scenario with realistic altitude bands and a bounded conflict rate.

    New blocks are rejected when they would overlap too many accepted blocks,
    so the conflict graph stays sparse (roughly O(n) edges), as in a staffed
    airspace control order rather than a random pile-up.
    """
    horizon = 240.0
    rng = random.Random(seed * 100003 + n_blocks)
    types = list(TYPE_WEIGHTS)
    weights = list(TYPE_WEIGHTS.values())
    allowed = 0 if density < 0.8 else (1 if density < 1.2 else 2)
    blocks: list[AirspaceBlock] = []
    for i in range(n_blocks):
        # Give the first block of each given type a chance to appear.
        t = types[i] if i < len(types) and i % 4 == 0 else rng.choices(types, weights=weights, k=1)[0]
        best: AirspaceBlock | None = None
        best_conf = 10**6
        for _ in range(14):
            cand = _make_random_block(rng, t, i)
            conf = 0
            for b in blocks:
                if cand.time_end_min <= b.time_start_min or b.time_end_min <= cand.time_start_min:
                    continue
                if cand.altitude_overlap(b, 0) <= 0:
                    continue
                if geo.shape_distance(cand.shape, b.shape) < 5.0:
                    conf += 1
            if conf < best_conf:
                best, best_conf = cand, conf
            if conf <= allowed:
                break
        assert best is not None
        blocks.append(best)
    for b in blocks:
        _priority(b, horizon)
    return Scenario(
        id=f"random-n{n_blocks}-s{seed}-d{density}", seed=seed,
        region_km=(0.0, 0.0, 400.0, 300.0), blocks=blocks,
        description=f"Random scenario with {n_blocks} blocks (density {density}).",
        meta={"horizon_min": horizon, "scenario_family": "random", "density": density},
    )
