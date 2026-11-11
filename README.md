# Tiered airspace deconfliction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24135751.svg)](https://doi.org/10.5281/zenodo.24135751)

Simulation code and experimental data for the manuscript **“LLM-Based Multi-Agent
Tiered Deconfliction for Tactical Airspace in ACO Planning”** (submitted to
*IEEE Access*).

An airspace control order assigns mission areas, corridors and restricted
volumes to using units. Because those volumes overlap in space and time, staff
must remove conflicts under doctrinal rules, unit priorities and a planning
clock. This repository models each airspace measure as a **four-dimensional
block with an owner and a priority**, and resolves conflicts in three tiers:

1. **Doctrinal rules** — configurable clauses `D1–D8` settle routine overlaps
   without communication.
2. **Bounded negotiation** — units exchange adjustment proposals; a large
   language model classifies ambiguous cases and drafts candidates that a
   deterministic checker validates.
3. **Priority-weighted arbitration** — minimises weighted cost when negotiation
   stalls; infeasible pairs escalate to the airspace control authority with a
   recorded rationale.

An incremental repair path re-examines only the block pairs affected by a
dynamic event.

## Contents

```
config.yaml          scenario sizes, seeds, rule set, LLM settings
design.md            experimental design and metric definitions
sim/                 simulator
  scenarios.py       synthetic ACO scenario generator
  doctrine.py        rules D1–D8 and the compatibility model
  geometry.py        4D block geometry and overlap tests
  conflicts.py       conflict detection and severity classification
  tiered.py          rule / negotiation / arbitration tiers
  agents.py          unit agents and bounded proposals
  llm.py             LLM client, disk cache and transcript writer
  metrics.py         resolution rate, cost, residual conflicts
  schema.py          shared data structures
scripts/             one script per experiment
results/             raw CSV results and LLM cache/transcripts
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments

```bash
# optional: only the LLM tier needs a key
export DEEPSEEK_API_KEY=...

python scripts/exp-01-main.py       # main comparison   (Table II, Fig. 3)
python scripts/exp-02-ablation.py   # component ablation (Table III, Fig. 4)
python scripts/exp-03-scale.py      # 20–80 blocks       (Fig. 5)
python scripts/exp-04-dynamic.py    # dynamic events     (Fig. 6)
```

Without `DEEPSEEK_API_KEY` the LLM method is skipped and the scripts report
`skipped_no_api_key`; deterministic methods still run. Cached responses in
`results/llm-cache.jsonl` can be replayed to reproduce the reported numbers
offline — the cache is keyed by prompt hash and temperature is fixed at 0.

All runs use the seeds in `config.yaml`; values reported in the manuscript are
means over three seeds per configuration.

## Data

Scenarios are synthetic. They are constructed from doctrinal constraints and
typical ACO layouts, not from operational data. `results/llm-transcript.jsonl`
contains the full prompts and responses behind the worked example.

## Contributors

- **Kejun Xu** — problem formulation, simulator design and implementation,
  experiments, analysis.
- **DeepSeek** (`deepseek-chat`) — code generation assistance during
  implementation; all generated code was reviewed, tested and is covered by the
  deterministic checker used in the experiments.

## Citation

The manuscript is under review. If you use this code, please cite the paper once
it is published; until then, cite the archived software:

> Xu, K. (2026). *Tiered airspace deconfliction: simulation code and experimental
> data* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24135751

## License

MIT — see [LICENSE](LICENSE).
