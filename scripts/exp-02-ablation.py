"""Component ablation at a fixed scenario size."""
from __future__ import annotations

from common import RESULTS, load_config, make_client, make_scenario, run_one, write_csv

METHODS = ["rule-only", "rule-arbitration", "arbitration-only", "tiered-det", "tiered-llm"]


def main() -> None:
    cfg = load_config()
    available = make_client(cfg) is not None
    scenarios = [make_scenario("random-40", cfg, seed=s) for s in cfg.get("main_seeds", [1, 2, 3])]
    rows = []
    for sc in scenarios:
        for method in METHODS:
            if method == "tiered-llm" and not available:
                continue
            row = run_one(sc, method, cfg)
            rows.append(row)
            if row.get("status"):
                print(f"{sc.id} {method}: {row['status']}", flush=True)
                continue
            print(
                f"{sc.id:18s} {method:18s} "
                f"rate={row['actionable_rate']:.2f} hard={row['final_hard']} "
                f"rule={row['rule_count']} neg={row['negotiation_count']} "
                f"arb={row['arbitration_count']} cost={row['total_cost']:.2f} "
                f"rt={row['runtime_s']:.1f}s",
                flush=True,
            )
    write_csv(RESULTS / "exp-02-ablation.csv", rows)


if __name__ == "__main__":
    main()
