"""Evaluation metrics for deconfliction outcomes."""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .schema import Conflict, ResolutionRecord
from .tiered import Outcome


def _active(conflicts: Iterable[Conflict]) -> list[Conflict]:
    return [c for c in conflicts if c.classification != "compatible"]


def summarize(outcome: Outcome, label: str = "") -> dict[str, Any]:
    initial = _active(outcome.initial_conflicts)
    final = _active(outcome.final_conflicts)
    given = {b.id: b.given for b in outcome.scenario.blocks}
    inherent = [c for c in initial if given.get(c.a) and given.get(c.b)]
    actionable = [c for c in initial if not (given.get(c.a) and given.get(c.b))]
    initial_ids = {tuple(sorted((c.a, c.b))) for c in initial}
    final_ids = {tuple(sorted((c.a, c.b))) for c in final}
    resolved = len(initial_ids - final_ids)
    actionable_ids = {tuple(sorted((c.a, c.b))) for c in actionable}
    actionable_resolved = len(actionable_ids - final_ids)
    tiers = Counter(r.tier for r in outcome.records)
    hard_final = len([c for c in final if c.classification == "hard"])
    return {
        "label": label,
        "initial_conflicts": len(initial),
        "initial_hard": len([c for c in initial if c.classification == "hard"]),
        "final_conflicts": len(final),
        "final_hard": hard_final,
        "resolved_conflicts": resolved,
        "resolution_rate": round(resolved / max(1, len(initial_ids)), 4),
        "actionable_conflicts": len(actionable),
        "inherent_conflicts": len(inherent),
        "actionable_resolved": actionable_resolved,
        "actionable_rate": round(actionable_resolved / max(1, len(actionable_ids)), 4),
        "zero_hard": int(hard_final == 0),
        "escalated": len(outcome.escalated),
        "rule_count": tiers.get("rule", 0),
        "negotiation_count": tiers.get("negotiation", 0),
        "arbitration_count": tiers.get("arbitration", 0),
        "escalation_count": tiers.get("escalated", 0),
        "total_cost": outcome.total_cost,
        "mean_cost": round(outcome.total_cost / max(1, len(outcome.records)), 4),
        "blocks_adjusted": outcome.stats.get("blocks_adjusted", 0),
        "llm_calls": outcome.stats.get("llm_calls", 0),
        "llm_decisions": outcome.stats.get("llm_decisions", 0),
        "llm_tokens": outcome.stats.get("llm_tokens", 0),
        "cache_hits": outcome.stats.get("cache_hits", 0),
        "runtime_s": round(outcome.runtime_s, 3),
    }


def displacement(outcome: Outcome) -> float:
    """Total normalised adjustment magnitude."""
    total = 0.0
    for r in outcome.records:
        for a in r.adjustments:
            total += (abs(a.delta_time_min) / 60.0 + abs(a.delta_alt_ft) / 6000.0
                      + abs(a.delta_lateral_km) / 20.0 + abs(a.shrink_ratio) / 0.20)
    return round(total, 4)


def stability(prev: Outcome, cur: Outcome) -> dict[str, float]:
    """How much of the previous plan had to change after a dynamic update."""
    old = {b.id: b for b in prev.scenario.blocks}
    new = {b.id: b for b in cur.scenario.blocks}
    changed = 0
    for bid, b in new.items():
        if bid not in old:
            changed += 1
            continue
        o = old[bid]
        if (abs(o.alt_low_ft - b.alt_low_ft) > 1e-6
                or abs(o.alt_high_ft - b.alt_high_ft) > 1e-6
                or abs(o.time_start_min - b.time_start_min) > 1e-6
                or abs(o.time_end_min - b.time_end_min) > 1e-6
                or o.shape.meta.get("center") != b.shape.meta.get("center")):
            changed += 1
    return {
        "changed_blocks": changed,
        "change_ratio": round(changed / max(1, len(new)), 4),
    }
