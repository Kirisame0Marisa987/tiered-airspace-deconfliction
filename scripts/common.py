"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.agents import DecisionLayer
from sim.llm import LLMClient, LLMError
from sim.metrics import summarize
from sim.scenarios import random_scenario, typical_scenario
from sim.tiered import (Outcome, ResolveConfig, TieredResolver,
                        resolve_sequential)

RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    import yaml
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def make_client(cfg: dict[str, Any]) -> LLMClient | None:
    llm = cfg.get("llm", {})
    path_cache = ROOT / llm.get("cache", "results/llm-cache.jsonl")
    path_transcript = ROOT / llm.get("transcript", "results/llm-transcript.jsonl")
    try:
        client = LLMClient(
            model=llm.get("model", "deepseek-chat"),
            base_url=llm.get("base_url", "https://api.deepseek.com"),
            api_key_env=llm.get("api_key_env", "DEEPSEEK_API_KEY"),
            temperature=float(llm.get("temperature", 0.0)),
            max_tokens=int(llm.get("max_tokens", 1024)),
            cache_path=path_cache,
            transcript_path=path_transcript,
        )
    except Exception:  # noqa: BLE001
        return None
    return client if client.available() else None


def make_scenario(name: str, cfg: dict[str, Any], seed: int = 1):
    if name == "typical":
        return typical_scenario(0, extra=int(cfg.get("typical_extra", 0)))
    if name.startswith("random-"):
        n = int(name.rsplit("-", 1)[1])
        return random_scenario(n, seed=seed)
    raise ValueError(name)


def run_one(scenario, method: str, cfg: dict[str, Any], client: LLMClient | None = None) -> dict[str, Any]:
    spec = cfg["methods"][method]
    mode = spec["mode"]
    use_llm = bool(spec["use_llm"])
    if use_llm:
        client = make_client(cfg)
        if client is None:
            return {"label": f"{scenario.id}|{method}", "method": method,
                    "scenario": scenario.id, "seed": scenario.seed,
                    "n_blocks": len(scenario.blocks), "mode": mode,
                    "status": "skipped_no_api_key"}
    rcfg = ResolveConfig(
        mode=mode,
        use_llm=use_llm,
        max_rounds=int(cfg.get("max_rounds", 3)),
        max_block_adjustments=int(cfg.get("max_block_adjustments", 3)),
        max_steps=max(40, int(cfg.get("max_steps_factor", 6)) * len(scenario.blocks)),
    )
    if mode == "sequential":
        outcome: Outcome = resolve_sequential(scenario, rcfg)
        fallbacks = 0
    else:
        decision = DecisionLayer(client if use_llm else None, use_llm=use_llm)
        outcome = TieredResolver(decision, rcfg).run(scenario)
        fallbacks = decision.fallbacks
    row = summarize(outcome, label=f"{scenario.id}|{method}")
    row.update({
        "method": method,
        "scenario": scenario.id,
        "seed": scenario.seed,
        "n_blocks": len(scenario.blocks),
        "mode": mode,
        "llm_fallbacks": fallbacks,
        "total_displacement": __import__("sim.metrics", fromlist=["displacement"]).displacement(outcome),
    })
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def event_scenario(base, kind: str, seed: int = 0):
    """Apply one dynamic event and return (scenario, description)."""
    import random
    from sim import geometry as geo
    from sim.schema import AirspaceBlock
    from sim.doctrine import compute_priority

    rng = random.Random(seed * 7919 + 11)
    blocks = [replace(b) for b in base.blocks]
    horizon = float(base.meta.get("horizon_min", 240.0))

    if kind == "popup_jez":
        # A pop-up engagement zone appears near the forward edge of the package.
        target = next((b for b in blocks if b.acm_type == "CAP"), blocks[0])
        cx, cy = geo.shape_centroid(target.shape)
        cx += rng.uniform(25, 45)
        cy += rng.uniform(-25, 25)
        new = AirspaceBlock(
            id="E-POPUP-JEZ", acm_type="JEZ", user="RED",
            mission="pop-up engagement zone reported by the intelligence cell",
            shape=geo.circle_shape((cx, cy), rng.uniform(22, 32)),
            alt_low_ft=0, alt_high_ft=30000, time_start_min=0, time_end_min=horizon,
            given=True, importance=1.0, adjustability=0.0,
        )
        blocks.append(new)
        desc = "pop-up JEZ appears adjacent to the CAP station"
    elif kind == "time_shift":
        # A package slips by 30 min; one area and one corridor move with it.
        movable = [b for b in blocks if not b.given and b.acm_type in {"CAP", "AAR", "AIRCOR"}]
        for b in rng.sample(movable, k=min(2, len(movable))):
            b.time_start_min = min(horizon - 20, b.time_start_min + 30)
            b.time_end_min = min(horizon, b.time_end_min + 30)
        desc = "strike package slips 30 min; dependent blocks shift"
    elif kind == "cap_relocate":
        caps = [b for b in blocks if b.acm_type == "CAP" and not b.given]
        if caps:
            b = caps[0]
            b.shape = geo.lateral_offset(b.shape, 20.0, rng.uniform(0, 360))
        desc = "CAP station relocated 20 km after a threat-axis update"
    else:
        raise ValueError(kind)

    for b in blocks:
        b.priority = compute_priority(b, horizon)
    return replace(base, id=f"{base.id}-{kind}-s{seed}", blocks=blocks), desc
