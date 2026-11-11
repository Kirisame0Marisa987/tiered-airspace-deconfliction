"""Build manuscript tables (Markdown) from the experiment CSVs."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
OUT = ROOT / "results-table.md"

METHOD_LABEL = {
    "rule-only": "Rule only",
    "rule-arbitration": "Rule + arbitration",
    "arbitration-only": "Arbitration only",
    "sequential": "Sequential",
    "tiered-det": "Tiered (deterministic)",
    "tiered-llm": "Tiered (LLM)",
}


def rows(name: str) -> list[dict]:
    with (RES / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def agg(rs: list[dict], col: str) -> tuple[float, float]:
    xs = [float(r[col]) for r in rs if r.get(col) not in (None, "")]
    return (mean(xs), pstdev(xs)) if xs else (0.0, 0.0)


def fmt(rs: list[dict], col: str, nd: int = 2) -> str:
    m, s = agg(rs, col)
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def table2() -> list[str]:
    data = [r for r in rows("exp-01-main.csv") if r["scenario"].startswith("random")]
    out = ["### TABLE II — Main comparison (random 40 blocks, 3 seeds)", "",
           "| Method | Actionable rate | Residual hard | Escalations | Cost | Time (s) |",
           "|---|---|---|---|---|---|"]
    for m in ["rule-only", "rule-arbitration", "sequential", "tiered-det", "tiered-llm"]:
        rs = [r for r in data if r["method"] == m]
        out.append(f"| {METHOD_LABEL[m]} | {fmt(rs,'actionable_rate')} | {fmt(rs,'final_hard',1)} | "
                   f"{fmt(rs,'escalated',1)} | {fmt(rs,'total_cost')} | {fmt(rs,'runtime_s',1)} |")
    typ = [r for r in rows("exp-01-main.csv") if r["scenario"].startswith("typical")]
    out += ["", "Typical scenario (16 blocks):", "",
            "| Method | Actionable rate | Residual hard | Escalations | Cost |",
            "|---|---|---|---|---|"]
    for m in ["rule-only", "rule-arbitration", "sequential", "tiered-det", "tiered-llm"]:
        rs = [r for r in typ if r["method"] == m]
        if rs:
            out.append(f"| {METHOD_LABEL[m]} | {rs[0]['actionable_rate']} | {rs[0]['final_hard']} | "
                       f"{rs[0]['escalated']} | {float(rs[0]['total_cost']):.2f} |")
    return out


def table3() -> list[str]:
    data = rows("exp-02-ablation.csv")
    out = ["### TABLE III — Component ablation (random 40 blocks, 3 seeds)", "",
           "| Configuration | Actionable rate | Residual hard | Escalations | Cost |",
           "|---|---|---|---|---|"]
    for m in ["rule-only", "arbitration-only", "rule-arbitration", "tiered-det", "tiered-llm"]:
        rs = [r for r in data if r["method"] == m]
        out.append(f"| {METHOD_LABEL[m]} | {fmt(rs,'actionable_rate')} | {fmt(rs,'final_hard',1)} | "
                   f"{fmt(rs,'escalated',1)} | {fmt(rs,'total_cost')} |")
    return out


def table4() -> list[str]:
    path = RES / "exp-04-dynamic.csv"
    if not path.exists():
        return ["### TABLE IV — dynamic (pending)"]
    data = rows("exp-04-dynamic.csv")
    out = ["### TABLE IV — Incremental repair versus full replanning", "",
           "| Event | Strategy | Changed blocks | Time (s) | Residual hard |",
           "|---|---|---|---|---|"]
    for e in ["popup_jez", "time_shift", "cap_relocate"]:
        for s in ["incremental", "full"]:
            rs = [r for r in data if r["event"] == e and r["strategy"] == s]
            if rs:
                out.append(f"| {e} | {s} | {fmt(rs,'changed_blocks',1)} | {fmt(rs,'runtime_s',2)} | "
                           f"{fmt(rs,'final_hard',1)} |")
    return out


def table5() -> list[str]:
    data = rows("exp-03-scale.csv")
    out = ["### TABLE V — Scalability", "",
           "| Blocks | Method | Detection (s) | Resolution (s) | Actionable rate |",
           "|---|---|---|---|---|"]
    for n in [20, 40, 60, 80]:
        for m in ["sequential", "tiered-det"]:
            rs = [r for r in data if r["method"] == m and int(r["n_blocks"]) == n]
            out.append(f"| {n} | {METHOD_LABEL[m]} | {agg(rs,'detect_s')[0]:.3f} | "
                       f"{fmt(rs,'runtime_s',1)} | {fmt(rs,'actionable_rate')} |")
    return out


def main() -> None:
    parts = ["# Results tables", "", *table2(), "", *table3(), "", *table5(), "", *table4()]
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
