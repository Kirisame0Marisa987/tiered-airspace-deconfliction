"""Scalability: conflict detection and resolution time versus the number of blocks."""
from __future__ import annotations

import time

from common import RESULTS, load_config, make_scenario, run_one, write_csv

from sim.conflicts import detect_conflicts

METHODS = ["sequential", "tiered-det"]


def main() -> None:
    cfg = load_config()
    rows = []
    for n in cfg.get("random_sizes", [20, 40, 60, 80]):
        for seed in cfg.get("seeds", [1, 2, 3]):
            sc = make_scenario(f"random-{n}", cfg, seed=seed)
            t0 = time.perf_counter()
            conflicts = detect_conflicts(sc)
            detect_s = time.perf_counter() - t0
            for method in METHODS:
                row = run_one(sc, method, cfg)
                row["detect_s"] = round(detect_s, 4)
                row["initial_k"] = len(conflicts)
                row["conflict_density"] = round(len(conflicts) / max(1, n), 3)
                rows.append(row)
                print(
                    f"n={n:3d} seed={seed} {method:12s} "
                    f"k={len(conflicts):3d} det={detect_s:.3f}s "
                    f"rate={row['actionable_rate']:.2f} rt={row['runtime_s']:.1f}s",
                    flush=True,
                )
    write_csv(RESULTS / "exp-03-scale.csv", rows)


if __name__ == "__main__":
    main()
