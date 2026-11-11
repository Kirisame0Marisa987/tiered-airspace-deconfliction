"""Main comparison: sequential / rule / rule+arbitration / tiered-det / tiered-LLM."""
from __future__ import annotations

from common import RESULTS, load_config, make_client, make_scenario, run_one, write_csv

METHODS = ["sequential", "rule-only", "rule-arbitration", "tiered-det", "tiered-llm"]


def main() -> None:
    cfg = load_config()
    available = make_client(cfg) is not None
    print(f"LLM available: {available}", flush=True)

    scenarios = [make_scenario("typical", cfg)]
    scenarios += [make_scenario("random-40", cfg, seed=s) for s in cfg.get("main_seeds", [1, 2, 3])]

    rows = []
    for sc in scenarios:
        for method in METHODS:
            if method == "tiered-llm" and not available:
                print(f"skip {method} on {sc.id}: no API key", flush=True)
                continue
            row = run_one(sc, method, cfg)
            rows.append(row)
            if row.get("status"):
                print(f"{sc.id:22s} {method:18s} {row['status']}", flush=True)
                continue
            print(
                f"{sc.id:22s} {method:18s} "
                f"act={row['actionable_resolved']}/{row['actionable_conflicts']} "
                f"rate={row['actionable_rate']:.2f} hard={row['final_hard']} "
                f"esc={row['escalated']} cost={row['total_cost']:.2f} "
                f"rt={row['runtime_s']:.1f}s llm={row['llm_calls']}",
                flush=True,
            )
    write_csv(RESULTS / "exp-01-main.csv", rows)


if __name__ == "__main__":
    main()
