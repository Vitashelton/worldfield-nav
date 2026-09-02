# B1 Pilot: Multi-Scene Counterfactual Benchmark

## Authority and objective

This is the only active plan. It is authorized by the research contract and
benchmark specification. Build a small, deterministic, resumable pilot harness
for formal world-field benchmarking; do not train a learned model in this plan.

The pilot's purpose is to prove that the dataset/evaluation harness can produce
paper-grade matched counterfactual evidence before any full dataset is made.

## Fixed pilot contract

- Split target: 5 train, 2 validation, 3 unseen-test Habitat-GS scenes.
- Branches use a fixed 3.0s duration and share an initial field, start pose,
  seed, and sampling clock. `straight`, `left`, `right`, and `mixed_turn` are
  action-family labels only: the stored control is a continuous ordered
  sequence of `(v, omega, dt)` tuples.
- Branch construction targets matched translational-motion budgets; the actual
  integrated translation and its mismatch are required metadata and must be
  reported per action family.
- Starts span open space, turn/corner, and doorway/occlusion contexts. Corner
  and doorway candidates must satisfy the P0 revelation criterion; trajectories
  with nearly zero unknown-to-observed area are not formal episodes.
- Targets: O/H/V/A snapshots at exact 0.5/1.0/2.0/3.0s horizons from a fixed
  sampling rate of at least 5Hz. The formal benchmark may not omit 0.5s.
- Baseline implemented and evaluated in this plan: M0-Transport only.
- Required metadata: scene asset provenance, start pose, seed, continuous
  controls, action-family label, timestamps, integrated poses and translation,
  simulator configuration, field origin, config path/hash, and code revision
  when a Git revision is available.

## Phases

### P0 — inventory and freeze the split

1. Inspect the locally installed Habitat-GS scene inventory and select as many
   of the ten target scenes as it can supply.
2. If fewer than ten valid scenes are local, download only the exact additional
   official Habitat-GS assets selected to complete this pilot split. Do not
   download a full scene collection. Record source URL/version/checksum and
   local path for every targeted asset.
3. Run deterministic probe trajectories across each candidate context and
   measure the distribution of unknown-to-observed area at the benchmark
   horizons. This probe is selection evidence, not a formal benchmark run.
4. Freeze a quantitative revelation threshold and its distributional rationale
   in `configs/benchmark/b1_pilot.yaml` before formal episode selection. The
   threshold decision must be explicitly recorded; do not silently choose one.
5. Freeze the exact 5/2/3 split and eligible starts in the config. Reject
   corner/doorway candidates below the revelation threshold. If targeted assets
   cannot produce eligible starts, register B1 blocked with the evidence.

### P1 — deterministic episode contract

1. Implement a resumable episode-manifest builder under `src/datasets/`.
2. Create a tiny smoke manifest: one revelation-qualified start/context per
   selected scene and all four matched branches. It must be seed-stable across
   two invocations.
3. Integrate every `(v, omega, dt)` sequence into future poses; validate the
   fixed duration, >=5Hz timestamps, exact 0.5/1/2/3s samples, and recorded
   translational-budget mismatch before episode generation.
4. Do not store duplicate RGB/depth when a reproducible source reference is
   sufficient; preserve O/H/V/A ground truth and required metadata.

### P2 — pilot generation and B0 evaluation

1. Generate only the smoke manifest episodes into a run-specific output path.
2. Run M0-Transport from current Phi and kinematically integrated action/pose
   history; future observations are comparison-only.
3. Produce per-horizon, per-split, per-action metrics and paper-ready figures:
   same current world with multiple actions; GT vs M0-Transport; and exact
   0.5/1/2/3s rollouts. Decompose metrics into persistent-known and
   newly-revealed regions. Do not use a learned model.

## Acceptance criteria

- Exact split, asset provenance, revelation distribution, and frozen threshold
  are in config, or the plan is registered blocked with evidence.
- Two manifest builds are byte-equivalent except for explicitly recorded run
  timestamps; all episode IDs, seeds, starts, controls, and integrated poses
  match.
- Each generated episode contains all required metadata and finite O/H/V/A.
- Every formal episode has exact 0.5/1/2/3s GT; missing 0.5s is a failure, not
  an unsupported value.
- M0-Transport metrics exist per horizon for persistent-known and
  newly-revealed regions, including Known-IoU, Reveal-IoU/F1, Persistent Drift,
  and applicable revelation latent similarity.
- Required figures and tables are in `paper_assets/` and the formal run is
  registered via `scripts/run_experiment.py`.
- A B1 result note records limitations, storage use, generation throughput,
  action-budget mismatch, and the revelation distribution.

## Stop condition

When all acceptance criteria are met, move this plan to `completed/`, set B1
completed in the registry, and evaluate the explicit B1-to-B2 scale-up gate:
revelation distribution, action matching, required paper figures, disk budget,
and generation throughput must all pass. Only then may a B2-scale dataset be
generated. Do not start model training or another research phase automatically.
