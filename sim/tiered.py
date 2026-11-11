"""Tiered deconfliction: doctrine rules -> agent negotiation -> arbitration."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Iterable

from . import geometry as geo
from .agents import DecisionLayer, Proposal
from .conflicts import detect_conflicts, pair_conflict
from .doctrine import (adjustment_cost, category, default_classification,
                       rule_candidates, severity)
from .schema import Adjustment, AirspaceBlock, Conflict, ResolutionRecord, Scenario


@dataclass
class ResolveConfig:
    mode: str = "tiered"          # tiered | rule-only | sequential
    use_llm: bool = True
    max_rounds: int = 3           # negotiation rounds K
    max_steps: int = 400
    max_attempts_per_pair: int = 2
    max_block_adjustments: int = 2
    sep_h_km: float = 5.0
    sep_alt_ft: float = 0.0
    sep_time_min: float = 0.0
    rule_cost_limit: float = 0.20
    candidate_budget: int = 12     # per side, after cost sorting
    alt_ceiling_ft: float = 30000.0
    record_rationale: bool = True


@dataclass
class Outcome:
    scenario: Scenario
    initial_conflicts: list[Conflict]
    final_conflicts: list[Conflict]
    records: list[ResolutionRecord]
    escalated: list[Conflict]
    runtime_s: float
    stats: dict[str, Any]

    @property
    def resolved_count(self) -> int:
        return len([r for r in self.records if r.tier != "escalated"])

    @property
    def total_cost(self) -> float:
        return round(sum(r.cost for r in self.records), 4)


# --------------------------------------------------------------- pair helpers
def _pair_conflict(a: AirspaceBlock, b: AirspaceBlock, cfg: ResolveConfig) -> bool:
    if a.time_end_min <= b.time_start_min + cfg.sep_time_min:
        return False
    if b.time_end_min <= a.time_start_min + cfg.sep_time_min:
        return False
    if a.altitude_overlap(b, cfg.sep_alt_ft) <= 0:
        return False
    ax0, ay0, ax1, ay1 = geo.bbox(max(a.shape.parts, key=geo.polygon_area))
    bx0, by0, bx1, by1 = geo.bbox(max(b.shape.parts, key=geo.polygon_area))
    dx = max(0.0, max(bx0 - ax1, ax0 - bx1))
    dy = max(0.0, max(by0 - ay1, ay0 - by1))
    if (dx * dx + dy * dy) ** 0.5 >= cfg.sep_h_km:
        return False
    return geo.shape_distance(a.shape, b.shape) < cfg.sep_h_km


def _apply(block: AirspaceBlock, adj: Adjustment) -> AirspaceBlock:
    """Return a new block with one adjustment applied."""
    if adj.strategy == "none":
        return block
    if adj.strategy == "time":
        return replace(block,
                       time_start_min=max(0.0, block.time_start_min + adj.delta_time_min),
                       time_end_min=max(0.0, block.time_end_min + adj.delta_time_min))
    if adj.strategy == "altitude":
        lo = max(0.0, block.alt_low_ft + adj.delta_alt_ft)
        hi = min(30000.0, block.alt_high_ft + adj.delta_alt_ft)
        if hi - lo < 500.0:
            return block
        return replace(block, alt_low_ft=lo, alt_high_ft=hi)
    if adj.strategy == "lateral":
        shape = geo.lateral_offset(block.shape, adj.delta_lateral_km, adj.bearing_deg)
        return replace(block, shape=shape)
    if adj.strategy == "shrink":
        shape = geo.scale_shape(block.shape, max(0.5, 1.0 - adj.shrink_ratio))
        return replace(block, shape=shape)
    return block


def _conflict_involving(blocks: dict[str, AirspaceBlock], moved_id: str,
                        cfg: ResolveConfig) -> int:
    """Number of non-compatible conflicts involving one block after a trial move."""
    b = blocks[moved_id]
    n = 0
    for oid, o in blocks.items():
        if oid == moved_id:
            continue
        if _pair_conflict(b, o, cfg) and default_classification(b, o) != "compatible":
            n += 1
    return n


def _hard_involving(blocks: dict[str, AirspaceBlock], moved_id: str,
                    cfg: ResolveConfig) -> int:
    b = blocks[moved_id]
    n = 0
    for oid, o in blocks.items():
        if oid == moved_id:
            continue
        if not _pair_conflict(b, o, cfg):
            continue
        if default_classification(b, o) == "hard":
            n += 1
    return n


def _pair_clear(blocks: dict[str, AirspaceBlock], a_id: str, b_id: str,
                cfg: ResolveConfig) -> bool:
    return not _pair_conflict(blocks[a_id], blocks[b_id], cfg)


# ------------------------------------------------------------ candidate sets
def _candidate_adjustments(block: AirspaceBlock, horizon: float,
                           cfg: ResolveConfig, *, extended: bool = False,
                           region: tuple[float, float, float, float] = (0.0, 0.0, 400.0, 300.0)) -> list[Adjustment]:
    """Geometrically feasible single-block adjustments, cheapest first."""
    if block.given or block.adjustability <= 1e-6:
        return []
    out: list[Adjustment] = []
    # Time shifts.
    for dt in ([15.0, 30.0, 45.0, 60.0, -15.0, -30.0] if extended else [15.0, 30.0, -15.0, -30.0]):
        if block.time_start_min + dt < 0 or block.time_end_min + dt > horizon:
            continue
        out.append(Adjustment(block.id, "time", delta_time_min=dt,
                              cost=adjustment_cost(dt, 0, 0, 0)))
    # Altitude layering.
    for dh in ([1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
                -1000.0, -2000.0, -3000.0, -4000.0]
               if extended else [1000.0, 2000.0, 3000.0, -1000.0, -2000.0, -3000.0]):
        if block.alt_low_ft + dh < 0 or block.alt_high_ft + dh > cfg.alt_ceiling_ft:
            continue
        out.append(Adjustment(block.id, "altitude", delta_alt_ft=dh,
                              cost=adjustment_cost(0, dh, 0, 0)))
    # Lateral offsets in eight directions.
    for km in ([5.0, 10.0, 15.0, 20.0, 25.0, 30.0] if extended
               else [5.0, 10.0, 15.0, 20.0]):
        for bearing in range(0, 360, 45):
            moved = geo.lateral_offset(block.shape, km, bearing)
            cx, cy = geo.shape_centroid(moved)
            if not (region[0] + 2 <= cx <= region[2] - 2 and region[1] + 2 <= cy <= region[3] - 2):
                continue
            out.append(Adjustment(block.id, "lateral", delta_lateral_km=km,
                                  bearing_deg=float(bearing),
                                  cost=adjustment_cost(0, 0, km, 0)))
    # Area shrinkage applies to area footprints only; shrinking a corridor
    # about its centroid would distort the route rather than narrow it.
    if category(block.acm_type) == "AREA":
        for ratio in ([0.10, 0.20, 0.30] if extended else [0.10, 0.20]):
            out.append(Adjustment(block.id, "shrink", shrink_ratio=ratio,
                                  cost=adjustment_cost(0, 0, 0, ratio)))
    out.sort(key=lambda x: x.cost)
    return out


def _goal_adjustments(block: AirspaceBlock, other: AirspaceBlock, horizon: float,
                      cfg: ResolveConfig,
                      region: tuple[float, float, float, float] = (0.0, 0.0, 400.0, 300.0),
                      ) -> list[Adjustment]:
    """Adjustments sized to actually separate this pair, cheapest first.

    Generic +/- steps often stop short of the separation needed by two large
    footprints.  These candidates are derived from the current overlap and are
    therefore the ones a planner would consider first.
    """
    out: list[Adjustment] = []
    # Vertical: move fully above or below the other block's band.
    up = other.alt_high_ft - block.alt_low_ft
    if up > 0 and block.alt_high_ft + up <= cfg.alt_ceiling_ft:
        out.append(Adjustment(block.id, "altitude", delta_alt_ft=up,
                              cost=adjustment_cost(0, up, 0, 0)))
    down = block.alt_high_ft - other.alt_low_ft
    if down > 0 and block.alt_low_ft - down >= 0:
        out.append(Adjustment(block.id, "altitude", delta_alt_ft=-down,
                              cost=adjustment_cost(0, down, 0, 0)))
    # Time: move fully after or before the other window.
    later = other.time_end_min - block.time_start_min
    if later > 0 and block.time_end_min + later <= horizon:
        out.append(Adjustment(block.id, "time", delta_time_min=later,
                              cost=adjustment_cost(later, 0, 0, 0)))
    earlier = block.time_end_min - other.time_start_min
    if earlier > 0 and block.time_start_min - earlier >= 0:
        out.append(Adjustment(block.id, "time", delta_time_min=-earlier,
                              cost=adjustment_cost(earlier, 0, 0, 0)))
    # Lateral: search outward along the separation vector and its diagonals,
    # taking the first distance that clears the pair in each direction.
    cx, cy = geo.shape_centroid(block.shape)
    ox, oy = geo.shape_centroid(other.shape)
    base = math.degrees(math.atan2(cy - oy, cx - ox))
    seen: set[tuple[int, int]] = set()
    for off in (0, 45, -45, 90, -90, 135, -135, 180):
        bearing = (base + off) % 360
        key = (int(round(bearing / 45)) % 8, 0)
        if key in seen:
            continue
        seen.add(key)
        for km in range(5, 65, 5):
            moved = geo.lateral_offset(block.shape, km, bearing)
            mx, my = geo.shape_centroid(moved)
            if not (region[0] + 2 <= mx <= region[2] - 2 and region[1] + 2 <= my <= region[3] - 2):
                break
            if geo.shape_distance(moved, other.shape) >= cfg.sep_h_km:
                out.append(Adjustment(block.id, "lateral", delta_lateral_km=float(km),
                                      bearing_deg=bearing,
                                      cost=adjustment_cost(0, 0, km, 0)))
                break
    # Area shrink: first ratio that clears the pair.
    if category(block.acm_type) == "AREA":
        for ratio in (0.10, 0.20, 0.30, 0.40, 0.50):
            moved = geo.scale_shape(block.shape, 1.0 - ratio)
            if geo.shape_distance(moved, other.shape) >= cfg.sep_h_km:
                out.append(Adjustment(block.id, "shrink", shrink_ratio=ratio,
                                      cost=adjustment_cost(0, 0, 0, ratio)))
                break
    out.sort(key=lambda a: a.cost)
    return out


def _proposals(block: AirspaceBlock, blocks: dict[str, AirspaceBlock],
               cfg: ResolveConfig, start_id: int = 0, *,
               horizon: float | None = None, extended: bool = False,
               region: tuple[float, float, float, float] = (0.0, 0.0, 400.0, 300.0),
               allowed: set[str] | None = None,
               partner_id: str | None = None) -> tuple[list[Proposal], int]:
    """Convert adjustments into proposals annotated with residual hard conflicts."""
    if allowed is not None and block.id not in allowed:
        return [], start_id
    props: list[Proposal] = []
    pid = start_id
    h = horizon if horizon is not None else blocks_meta_horizon(blocks)
    adjustments = _candidate_adjustments(block, h, cfg, extended=extended, region=region)
    if partner_id is not None and partner_id in blocks:
        goal = _goal_adjustments(block, blocks[partner_id], h, cfg, region)
        seen_keys = set()
        merged: list[Adjustment] = []
        for adj in goal + adjustments:
            key = (adj.strategy, round(adj.delta_time_min, 3), round(adj.delta_alt_ft, 3),
                   round(adj.delta_lateral_km, 3), round(adj.bearing_deg, 3),
                   round(adj.shrink_ratio, 3))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(adj)
        adjustments = merged
    zero = 0
    for adj in adjustments:
        newb = _apply(block, adj)
        trial = dict(blocks)
        trial[block.id] = newb
        # Local residual: does this proposal still conflict with the partner,
        # or create a new hard conflict with any third block?
        residual = 0
        for oid, o in trial.items():
            if oid == block.id:
                continue
            if partner_id is not None and oid == partner_id:
                continue
            if _pair_conflict(newb, o, cfg) and default_classification(newb, o) == "hard":
                residual += 1
        if partner_id is not None and partner_id in trial:
            if _pair_conflict(newb, trial[partner_id], cfg):
                residual += 1
        if residual == 0:
            zero += 1
        props.append(Proposal(
            id=pid, block_id=block.id, strategy=adj.strategy,
            delta_time_min=adj.delta_time_min, delta_alt_ft=adj.delta_alt_ft,
            delta_lateral_km=adj.delta_lateral_km,
            bearing_deg=adj.bearing_deg,
            shrink_ratio=adj.shrink_ratio, cost=adj.cost, residual_hard=residual,
        ))
        pid += 1
        # Cheap candidates are generated first, so stop once enough feasible
        # options exist; this bounds the O(n) trial evaluation per conflict.
        if zero >= max(4, cfg.candidate_budget // 3) or len(props) >= cfg.candidate_budget * 2:
            break
    # Keep the budget: feasible and cheapest first, but preserve strategy diversity.
    props.sort(key=lambda p: (p.residual_hard, p.cost))
    props = props[: cfg.candidate_budget]
    for i, p in enumerate(props):
        p.id = i
    return props, len(props)


_HORIZON: dict[int, float] = {}


def blocks_meta_horizon(blocks: dict[str, AirspaceBlock]) -> float:
    if blocks:
        return max(b.time_end_min for b in blocks.values())
    return 240.0


# --------------------------------------------------------------- the resolver
class TieredResolver:
    def __init__(self, decision: DecisionLayer, cfg: ResolveConfig) -> None:
        self.decision = decision
        self.cfg = cfg
        self._class_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._horizon = 240.0
        self._region = (0.0, 0.0, 400.0, 300.0)

    def _classify(self, a: AirspaceBlock, b: AirspaceBlock, c: Conflict) -> dict[str, Any]:
        key = tuple(sorted((a.id, b.id)))
        if key in self._class_cache:
            return self._class_cache[key]
        res = self.decision.classify(a, b, c)
        self._class_cache[key] = res
        return res

    def _rule_tier(self, a: AirspaceBlock, b: AirspaceBlock,
                   blocks: dict[str, AirspaceBlock]) -> ResolutionRecord | None:
        """Cheapest doctrine rule that clears the pair and creates no hard conflict."""
        movable = [x for x in (a, b) if self._can_move(x)]
        if not movable:
            return None
        mover = min(movable, key=lambda x: x.priority)
        other = b if mover is a else a
        for rm in rule_candidates(a, b):
            if rm.strategy == "none":
                return ResolutionRecord("", "rule", [], 0, "same package may share", rm.doctrine_ref, 0.0)
            adj = _adjustment_from_rule(rm, mover.id)
            if adj is None:
                continue
            if adj.cost > self.cfg.rule_cost_limit:
                continue
            trial = dict(blocks)
            trial[mover.id] = _apply(mover, adj)
            if not _pair_conflict(trial[mover.id], trial[other.id], self.cfg) and \
                    _hard_involving(trial, mover.id, self.cfg) == 0:
                return ResolutionRecord("", "rule", [adj], 0,
                                        f"{rm.rule_id}: {rm.description}",
                                        rm.doctrine_ref, adj.cost)
        return None

    def _negotiation(self, a: AirspaceBlock, b: AirspaceBlock,
                     c: Conflict, blocks: dict[str, AirspaceBlock]
                     ) -> ResolutionRecord | None:
        """Bilateral negotiation, at most K rounds."""
        movable = [x for x in (a, b) if self._can_move(x)]
        if not movable:
            return None
        proposer = min(movable, key=lambda x: x.priority)
        responder = b if proposer is a else a
        p_props, _ = _proposals(proposer, blocks, self.cfg, 0,
                                horizon=self._horizon, region=self._region,
                                allowed={proposer.id}, partner_id=responder.id)
        # A proposer should offer actual changes, not "do nothing".
        p_props = [p for p in p_props if p.strategy != "none"]
        if not p_props:
            return None
        rejected: set[int] = set()
        for rnd in range(1, self.cfg.max_rounds + 1):
            live = [p for p in p_props if p.id not in rejected]
            if not live:
                return None
            pick = self.decision.propose(proposer, responder, c, live)
            pid = int(pick.get("proposal_id", -1))
            prop = next((p for p in live if p.id == pid), None)
            if prop is None:
                prop = live[0]
            resp = self.decision.respond(responder, proposer, c, prop)
            if str(resp.get("decision", "")).lower() == "accept":
                adj = Adjustment(proposer.id, prop.strategy,
                                 delta_time_min=prop.delta_time_min,
                                 delta_alt_ft=prop.delta_alt_ft,
                                 delta_lateral_km=prop.delta_lateral_km,
                                 shrink_ratio=prop.shrink_ratio, cost=prop.cost)
                adj.bearing_deg = prop.bearing_deg
                trial = dict(blocks)
                trial[proposer.id] = _apply(proposer, adj)
                if not _pair_conflict(trial[proposer.id], trial[responder.id], self.cfg) and \
                        _hard_involving(trial, proposer.id, self.cfg) == 0:
                    return ResolutionRecord("", "negotiation", [adj], rnd,
                                            f"proposer: {pick.get('rationale','')} | responder: {resp.get('rationale','')}",
                                            "D1-D3", adj.cost)
            rejected.add(prop.id)
        return None

    def _arbitration(self, a: AirspaceBlock, b: AirspaceBlock,
                     c: Conflict, blocks: dict[str, AirspaceBlock]
                     ) -> ResolutionRecord | None:
        pa, _ = _proposals(a, blocks, self.cfg, 0,
                           horizon=self._horizon, region=self._region,
                           allowed={a.id} if self._can_move(a) else set(),
                           partner_id=b.id)
        pb, nxt = _proposals(b, blocks, self.cfg, len(pa) + 1,
                             horizon=self._horizon, region=self._region,
                             allowed={b.id} if self._can_move(b) else set(),
                             partner_id=a.id)
        pa = [p for p in pa if p.strategy != "none"][:6] + [
            Proposal(-1, a.id, "none", cost=0.0, residual_hard=0)]
        pb = [p for p in pb if p.strategy != "none"][:6] + [
            Proposal(-2, b.id, "none", cost=0.0, residual_hard=0)]
        denom = max(1e-6, a.priority + b.priority)
        best: tuple[float, Proposal, Proposal] | None = None
        for p in pa:
            for q in pb:
                if p.strategy == "none" and q.strategy == "none":
                    continue
                trial = dict(blocks)
                if p.strategy != "none":
                    trial[a.id] = _apply(a, _adj_from_proposal(p))
                if q.strategy != "none":
                    trial[b.id] = _apply(b, _adj_from_proposal(q))
                if _pair_conflict(trial[a.id], trial[b.id], self.cfg):
                    continue
                if _hard_involving(trial, a.id, self.cfg) or _hard_involving(trial, b.id, self.cfg):
                    continue
                wc = (a.priority * p.cost + b.priority * q.cost) / denom
                if best is None or wc < best[0]:
                    best = (wc, p, q)
        if best is None:
            return None
        wc, p, q = best
        chosen = [x for x in (p, q) if x.strategy != "none"]
        rationale = self.decision.explain(a, b, c, chosen, round(wc, 4))
        adjs = [_adj_from_proposal(x) for x in chosen]
        return ResolutionRecord("", "arbitration", adjs, 0, rationale, "D2-D3", round(wc, 4))

    def _can_move(self, block: AirspaceBlock) -> bool:
        return (not block.given and block.adjustability > 1e-6
                and self._adjust_count.get(block.id, 0) < self.cfg.max_block_adjustments)

    def _escalate(self, c: Conflict, records: list[ResolutionRecord]) -> None:
        c.resolved = False
        c.resolved_by = "escalated"
        if self.cfg.record_rationale:
            records.append(ResolutionRecord(c.id, "escalated", [], 0,
                                            "no feasible adjustment within the block-movement budget; "
                                            "escalate to the JFACC",
                                            "D4", 0.0))

    def _classify_into(self, c: Conflict, blocks: dict[str, AirspaceBlock]) -> None:
        a, b = blocks[c.a], blocks[c.b]
        res = self._classify(a, b, c)
        c.classification = res["classification"]
        c.rule_id = res.get("rule_id", "")
        c.rationale = res.get("rationale", "")
        c.severity = severity(c)

    def run(self, scenario: Scenario) -> Outcome:
        t0 = time.perf_counter()
        cfg = self.cfg
        self._horizon = float(scenario.meta.get("horizon_min", 240.0))
        self._region = tuple(scenario.region_km)  # type: ignore[assignment]
        blocks = {b.id: b for b in scenario.blocks}
        self._adjust_count = {b.id: 0 for b in scenario.blocks}
        initial = detect_conflicts(scenario, cfg.sep_h_km, cfg.sep_time_min, cfg.sep_alt_ft)
        for c in initial:
            self._classify_into(c, blocks)
            if c.classification == "compatible":
                c.resolved = True
                c.resolved_by = "compatible"
        records: list[ResolutionRecord] = []
        escalated: list[Conflict] = []
        attempts: dict[tuple[str, str], int] = {}
        blocked: set[tuple[str, str]] = set()
        queue: list[Conflict] = [c for c in initial if c.classification != "compatible"]
        steps = 0
        cid_counter = len(initial)

        while queue and steps < cfg.max_steps:
            queue.sort(key=lambda c: (-c.severity, -c.overlap_ratio))
            c = queue.pop(0)
            key = tuple(sorted((c.a, c.b)))
            if key in blocked:
                continue
            # The pair may already be clear because another block moved.
            cur = pair_conflict(blocks[c.a], blocks[c.b], c.id,
                                cfg.sep_h_km, cfg.sep_time_min, cfg.sep_alt_ft)
            if cur is None:
                continue
            c = cur
            self._classify_into(c, blocks)
            if c.classification == "compatible":
                continue
            if attempts.get(key, 0) >= cfg.max_attempts_per_pair:
                blocked.add(key)
                self._escalate(c, records)
                escalated.append(c)
                continue
            attempts[key] = attempts.get(key, 0) + 1
            a, b = blocks[c.a], blocks[c.b]

            rec: ResolutionRecord | None = None
            if cfg.mode == "arbitration-only":
                rec = self._arbitration(a, b, c, blocks)
            elif cfg.mode != "sequential":
                rec = self._rule_tier(a, b, blocks)
                if rec is None and cfg.mode == "tiered":
                    rec = self._negotiation(a, b, c, blocks)
                if rec is None and cfg.mode in {"tiered", "rule-arbitration"}:
                    rec = self._arbitration(a, b, c, blocks)

            if rec is None or (rec.adjustments and not self._record_applies(rec, blocks, c)):
                steps += 1
                if attempts[key] >= cfg.max_attempts_per_pair:
                    blocked.add(key)
                    self._escalate(c, records)
                    escalated.append(c)
                else:
                    queue.append(c)
                continue

            rec.conflict_id = c.id
            adjusted: list[str] = []
            for adj in rec.adjustments:
                blocks[adj.block_id] = _apply(blocks[adj.block_id], adj)
                adjusted.append(adj.block_id)
                self._adjust_count[adj.block_id] = self._adjust_count.get(adj.block_id, 0) + 1
            records.append(rec)
            c.resolved = True
            c.resolved_by = rec.tier
            steps += 1
            # Incremental recheck: only pairs involving a block that just moved.
            for bid in adjusted:
                for oid, other in blocks.items():
                    if oid == bid:
                        continue
                    cid_counter += 1
                    nc = pair_conflict(blocks[bid], other, f"C{cid_counter:04d}",
                                       cfg.sep_h_km, cfg.sep_time_min, cfg.sep_alt_ft)
                    if nc is None:
                        continue
                    k2 = tuple(sorted((nc.a, nc.b)))
                    if k2 in blocked:
                        continue
                    nc.chain = True
                    self._classify_into(nc, blocks)
                    if nc.classification != "compatible":
                        queue.append(nc)

        final_scn = replace(scenario, blocks=list(blocks.values()))
        final = detect_conflicts(final_scn, cfg.sep_h_km, cfg.sep_time_min, cfg.sep_alt_ft)
        for c in final:
            self._classify_into(c, blocks)
            c.resolved = c.classification == "compatible"
        runtime = time.perf_counter() - t0
        client = self.decision.client if self.decision.use_llm else None
        api_calls = getattr(client, "calls", 0) if client else 0
        cache_hits = getattr(client, "cache_hits", 0) if client else 0
        stats = {
            "steps": steps,
            "llm_calls": api_calls,
            "llm_decisions": api_calls + cache_hits,
            "llm_tokens": getattr(client, "tokens", 0) if client else 0,
            "cache_hits": cache_hits,
            "llm_fallbacks": getattr(self.decision, "fallbacks", 0),
            "blocks_adjusted": len({a.block_id for r in records for a in r.adjustments}),
        }
        return Outcome(final_scn, initial, final, records, escalated, runtime, stats)

    def _record_applies(self, rec: ResolutionRecord, blocks: dict[str, AirspaceBlock],
                        c: Conflict) -> bool:
        trial = dict(blocks)
        for adj in rec.adjustments:
            trial[adj.block_id] = _apply(trial[adj.block_id], adj)
        return not _pair_conflict(trial[c.a], trial[c.b], self.cfg)


def _adjustment_from_rule(rm, block_id: str) -> Adjustment | None:
    if rm.strategy == "time":
        return Adjustment(block_id, "time", delta_time_min=rm.delta,
                          cost=adjustment_cost(rm.delta, 0, 0, 0))
    if rm.strategy == "altitude":
        return Adjustment(block_id, "altitude", delta_alt_ft=rm.delta,
                          cost=adjustment_cost(0, rm.delta, 0, 0))
    if rm.strategy == "lateral":
        return Adjustment(block_id, "lateral", delta_lateral_km=rm.delta,
                          bearing_deg=rm.bearing, cost=adjustment_cost(0, 0, rm.delta, 0))
    if rm.strategy == "shrink":
        return Adjustment(block_id, "shrink", shrink_ratio=rm.delta,
                          cost=adjustment_cost(0, 0, 0, rm.delta))
    if rm.strategy == "none":
        return Adjustment(block_id, "none", cost=0.0)
    return None


def _adj_from_proposal(p: Proposal) -> Adjustment:
    adj = Adjustment(p.block_id, p.strategy, delta_time_min=p.delta_time_min,
                     delta_alt_ft=p.delta_alt_ft, delta_lateral_km=p.delta_lateral_km,
                     shrink_ratio=p.shrink_ratio, cost=p.cost)
    adj.bearing_deg = p.bearing_deg  # type: ignore[attr-defined]
    return adj


# ---------------------------------------------------------- sequential baseline
def resolve_sequential(scenario: Scenario, cfg: ResolveConfig) -> Outcome:
    """Priority-ordered greedy planner: each block is adjusted around accepted ones."""
    import time as _time
    t0 = _time.perf_counter()
    order = sorted(scenario.blocks, key=lambda b: -b.priority)
    accepted: dict[str, AirspaceBlock] = {}
    records: list[ResolutionRecord] = []
    initial = detect_conflicts(scenario, cfg.sep_h_km)
    horizon = scenario.meta.get("horizon_min", 240.0)
    region = tuple(scenario.region_km)
    for b in order:
        cands: list[Adjustment] = []
        for o in accepted.values():
            # Goal candidates are only meaningful for pairs that actually conflict;
            # skipping the rest avoids a full lateral search against every block.
            if _pair_conflict(b, o, cfg) is None:
                continue
            cands.extend(_goal_adjustments(b, o, horizon, cfg, region))
        cands.extend(_candidate_adjustments(b, horizon, cfg, extended=True, region=region))
        cands.sort(key=lambda a: a.cost)
        best = b
        for adj in cands:
            cand = _apply(b, adj)
            if any(_pair_conflict(cand, o, cfg) for o in accepted.values()):
                continue
            best = cand
            records.append(ResolutionRecord("", "sequential", [adj], 0,
                                            "priority-ordered greedy placement", "", adj.cost))
            break
        accepted[b.id] = best
    final_scn = replace(scenario, blocks=list(accepted.values()))
    final = detect_conflicts(final_scn, cfg.sep_h_km, cfg.sep_time_min, cfg.sep_alt_ft)
    return Outcome(final_scn, initial, final, records, [], _time.perf_counter() - t0,
                   {"steps": len(order), "llm_calls": 0, "llm_tokens": 0,
                    "cache_hits": 0,
                    "blocks_adjusted": len({a.block_id for r in records for a in r.adjustments})})
