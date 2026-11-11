"""Agent decision layer: LLM-backed classification, negotiation and explanation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .doctrine import adjustment_cost, default_classification, doctrine_context
from .llm import LLMClient, LLMError
from .schema import AirspaceBlock, Conflict

VALID_CLASSES = {"compatible", "soft", "hard"}


def _block_desc(b: AirspaceBlock) -> dict[str, Any]:
    return {
        "id": b.id,
        "type": b.acm_type,
        "user": b.user,
        "mission": b.mission,
        "altitude_ft": [b.alt_low_ft, b.alt_high_ft],
        "time_min": [b.time_start_min, b.time_end_min],
        "given_by_higher_authority": b.given,
        "adjustability": b.adjustability,
        "priority": b.priority,
    }


@dataclass
class Proposal:
    id: int
    block_id: str
    strategy: str
    delta_time_min: float = 0.0
    delta_alt_ft: float = 0.0
    delta_lateral_km: float = 0.0
    bearing_deg: float = 0.0
    shrink_ratio: float = 0.0
    cost: float = 0.0
    residual_hard: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "block": self.block_id,
            "strategy": self.strategy,
            "delta_time_min": self.delta_time_min,
            "delta_alt_ft": self.delta_alt_ft,
            "delta_lateral_km": self.delta_lateral_km,
            "bearing_deg": self.bearing_deg,
            "shrink_ratio": self.shrink_ratio,
            "cost": self.cost,
            "residual_hard_if_applied": self.residual_hard,
        }


class DecisionLayer:
    """All LLM calls go through here; deterministic fallbacks keep ablations runnable."""

    def __init__(self, client: LLMClient | None, use_llm: bool = True) -> None:
        self.client = client
        self.use_llm = use_llm and client is not None
        self.last_error: str | None = None
        self.fallbacks = 0

    # ---------------------------------------------------------- classification
    def classify(self, a: AirspaceBlock, b: AirspaceBlock, c: Conflict) -> dict[str, Any]:
        default = default_classification(a, b)
        if not self.use_llm:
            return {"classification": default, "rule_id": "", "rationale": "compatibility matrix", "confidence": 0.6}
        system = (
            "You are a tactical airspace deconfliction officer. Two four-dimensional airspace blocks "
            "overlap. Decide whether the overlap is compatible (the blocks may share the volume), "
            "soft (a doctrine rule can separate them), or hard (they cannot share the volume). "
            "Apply the doctrine clauses exactly. A block imposed by higher authority cannot be moved. "
            "Return JSON with keys classification, rule_id, rationale, confidence."
        )
        user = json.dumps({
            "block_a": _block_desc(a),
            "block_b": _block_desc(b),
            "overlap": {
                "horizontal_gap_km": c.horizontal_gap_km,
                "altitude_overlap_ft": c.alt_overlap_ft,
                "time_overlap_min": c.time_overlap_min,
                "area_overlap_ratio": c.overlap_ratio,
            },
            "compatibility_matrix_default": default,
            "doctrine": doctrine_context(a, b),
        }, ensure_ascii=False, indent=1)
        schema = '{"classification":"compatible|soft|hard","rule_id":"D1..D8 or empty","rationale":"one or two sentences","confidence":0.0}'
        try:
            out = self.client.complete_json(system, user, schema)
        except LLMError as exc:
            self.fallbacks += 1
            self.last_error = str(exc)
            return {"classification": default, "rule_id": "", "rationale": "LLM unavailable; matrix fallback", "confidence": 0.5}
        cls = str(out.get("classification", "")).strip().lower()
        if cls not in VALID_CLASSES:
            cls = default
        # Doctrine constraints override the model: same-package blocks are
        # compatible by D1, and a higher-authority block cannot be co-used by
        # a different user.
        if default == "compatible":
            cls = "compatible"
        if (a.given or b.given) and a.user != b.user and cls == "compatible":
            cls = "hard"
        return {
            "classification": cls,
            "rule_id": str(out.get("rule_id", ""))[:16],
            "rationale": str(out.get("rationale", ""))[:400],
            "confidence": float(out.get("confidence", 0.6) or 0.6),
        }

    # ------------------------------------------------------------- negotiation
    def propose(self, a: AirspaceBlock, b: AirspaceBlock, c: Conflict,
                proposals: list[Proposal]) -> dict[str, Any]:
        """Lower-priority agent selects a proposal; deterministic fallback = cheapest."""
        if not proposals:
            return {"proposal_id": -1, "rationale": "no feasible adjustment"}
        if not self.use_llm:
            best = min(proposals, key=lambda p: (p.residual_hard, p.cost))
            return {"proposal_id": best.id, "rationale": "minimum-cost feasible adjustment"}
        system = (
            "You are the planning agent for the lower-priority airspace block. Select one adjustment "
            "from the candidate list. Prefer the lowest operational cost that still removes the conflict. "
            "Reject a candidate that creates a sensor coverage gap, violates doctrine, or breaks mission timing. "
            "Return JSON with proposal_id and rationale."
        )
        user = json.dumps({
            "your_block": _block_desc(a),
            "other_block": _block_desc(b),
            "conflict": {
                "horizontal_gap_km": c.horizontal_gap_km,
                "altitude_overlap_ft": c.alt_overlap_ft,
                "time_overlap_min": c.time_overlap_min,
                "classification": c.classification,
            },
            "doctrine": doctrine_context(a, b),
            "candidates": [p.as_dict() for p in proposals],
        }, ensure_ascii=False, indent=1)
        schema = '{"proposal_id": 0, "rationale": "one or two sentences"}'
        try:
            out = self.client.complete_json(system, user, schema)
            pid = int(out.get("proposal_id", -1))
        except (LLMError, ValueError, TypeError):
            self.fallbacks += 1
            best = min(proposals, key=lambda p: (p.residual_hard, p.cost))
            return {"proposal_id": best.id, "rationale": "LLM unavailable; minimum-cost fallback"}
        if pid not in {p.id for p in proposals}:
            best = min(proposals, key=lambda p: (p.residual_hard, p.cost))
            pid = best.id
        return {"proposal_id": pid, "rationale": str(out.get("rationale", ""))[:400]}

    def respond(self, a: AirspaceBlock, b: AirspaceBlock, c: Conflict,
                proposal: Proposal) -> dict[str, Any]:
        """Higher-priority agent accepts or rejects; fallback = accept if no residual hard conflict."""
        if not self.use_llm:
            ok = proposal.residual_hard == 0
            return {"decision": "accept" if ok else "reject",
                    "rationale": "feasible" if ok else "creates residual hard conflict"}
        system = (
            "You are the planning agent for the higher-priority airspace block. Decide whether to accept "
            "the other side's adjustment. Accept only if it removes the conflict without creating a coverage "
            "gap, violating doctrine, or breaking your mission timing. Return JSON with decision and rationale."
        )
        user = json.dumps({
            "your_block": _block_desc(a),
            "other_block": _block_desc(b),
            "conflict": {
                "horizontal_gap_km": c.horizontal_gap_km,
                "altitude_overlap_ft": c.alt_overlap_ft,
                "time_overlap_min": c.time_overlap_min,
            },
            "doctrine": doctrine_context(a, b),
            "proposal": proposal.as_dict(),
        }, ensure_ascii=False, indent=1)
        schema = '{"decision":"accept|reject","rationale":"one or two sentences"}'
        try:
            out = self.client.complete_json(system, user, schema)
            dec = str(out.get("decision", "")).strip().lower()
        except LLMError:
            self.fallbacks += 1
            ok = proposal.residual_hard == 0
            return {"decision": "accept" if ok else "reject",
                    "rationale": "LLM unavailable; feasibility fallback"}
        if dec not in {"accept", "reject"}:
            dec = "accept" if proposal.residual_hard == 0 else "reject"
        return {"decision": dec, "rationale": str(out.get("rationale", ""))[:400]}

    # -------------------------------------------------------------- arbitration
    def explain(self, a: AirspaceBlock, b: AirspaceBlock, c: Conflict,
                chosen: list[Proposal], cost: float) -> str:
        if not self.use_llm:
            return "minimum weighted adjustment cost among feasible combinations"
        system = (
            "You are the deconfliction arbitrator. The geometric optimiser has selected the feasible "
            "combination with minimum priority-weighted adjustment cost. Explain in one or two sentences "
            "why this combination is operationally acceptable, citing doctrine. Return JSON with rationale."
        )
        user = json.dumps({
            "block_a": _block_desc(a),
            "block_b": _block_desc(b),
            "chosen_adjustments": [p.as_dict() for p in chosen],
            "weighted_cost": cost,
            "doctrine": doctrine_context(a, b),
        }, ensure_ascii=False, indent=1)
        try:
            out = self.client.complete_json(system, user, '{"rationale":"one or two sentences"}')
            return str(out.get("rationale", ""))[:400]
        except LLMError:
            self.fallbacks += 1
            return "minimum weighted adjustment cost among feasible combinations"
