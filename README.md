# WorldFlow Research Harness

WorldFlow investigates whether an indoor mobile robot can maintain a persistent
metric world state and predict how that state changes under hypothetical future
actions. Habitat-GS supplies controlled geometry, RGB/depth, pose, and
counterfactual rollouts; it is infrastructure, not the paper contribution.

## Start here

1. Read `AGENTS.md` for the execution contract.
2. Read `docs/RESEARCH_CONTRACT.md` for immutable research scope.
3. Read the sole active plan under `docs/exec-plans/active/`.
4. Run `python scripts/sanity_check.py` before substantial work.

Research phases and their durable status live in `experiments/registry.yaml`.
Formal runs must be launched through `scripts/run_experiment.py`, which writes
an immutable run record and updates the registry.

Existing `tools/experiment_00*.py` and `outputs/exp00*/` are preserved as
preliminary S0--S2 evidence. They are not the formal benchmark harness.

## Layout

- `docs/`: research contract, specifications, plans, and result notes
- `configs/`: versioned benchmark/model/experiment configurations
- `src/`: future project-owned implementation, separate from Habitat-GS
- `scripts/`: preflight and formal-run utilities
- `experiments/`: registry plus immutable formal run manifests
- `outputs/`: generated run data and diagnostics
- `paper_assets/`: publication-quality figures, videos, and tables
