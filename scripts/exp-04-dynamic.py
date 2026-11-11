"""Dynamic update: incremental repair from the published ACO versus full replanning."""
from __future__ import annotations

from dataclasses import replace

from common import RESULTS, event_scenario, load_config, make_scenario, write_csv

from sim.agents import DecisionLayer
from sim.metrics import stability, summarize
from sim.tiered import ResolveConfig, TieredResolver

EVENTS = ["popup_jez", "time_shift", "cap_relocate"]


def solve(scenario, cfg):
    rcfg = ResolveConfig(mode="tiered", use_llm=False,
                         max_block_adjustments=int(cfg.get("max_block_adjustments", 3)),
                         max_steps=max(40, 6 * len(scenario.blocks)))
    return TieredResolver(DecisionLayer(None, use_llm=False), rcfg).run(scenario)


def main() -> None:
    cfg = load_config()
    rows = []
    bases = [make_scenario("typical", cfg)] + [
        make_scenario("random-40", cfg, seed=s) for s in cfg.get("main_seeds", [1, 2, 3])[:2]
    ]
    for base in bases:
        prev = solve(base, cfg)
        published = prev.scenario
        for event in EVENTS:
            for seed in range(3):
                updated_incremental, desc = event_scenario(published, event, seed)
                updated_full, _ = event_scenario(base, event, seed)

                inc = solve(updated_incremental, cfg)
                full = solve(updated_full, cfg)

                inc_row = summarize(inc, label=f"{base.id}|{event}|incremental")
                inc_row.update({"base": base.id, "event": event, "event_seed": seed,
                                "strategy": "incremental", "description": desc})
                inc_row.update(stability(prev, inc))
                full_row = summarize(full, label=f"{base.id}|{event}|full")
                full_row.update({"base": base.id, "event": event, "event_seed": seed,
                                 "strategy": "full", "description": desc})
                full_row.update(stability(prev, full))
                rows += [inc_row, full_row]
                print(
                    f"{base.id:18s} {event:13s} s{seed} "
                    f"inc: changed={inc_row['changed_blocks']:3d} rt={inc_row['runtime_s']:.1f}s "
                    f"hard={inc_row['final_hard']} | "
                    f"full: changed={full_row['changed_blocks']:3d} rt={full_row['runtime_s']:.1f}s "
                    f"hard={full_row['final_hard']}",
                    flush=True,
                )
    write_csv(RESULTS / "exp-04-dynamic.csv", rows)


if __name__ == "__main__":
    main()
