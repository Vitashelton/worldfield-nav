# C1 Pilot — Sequential Multimodal World-Field Dataset

## Authority and objective

This is the only active plan. It implements the frozen WorldFlow research
contract by producing a small, deterministic sequential dataset for persistent
multimodal world-state updates. It does not train a model, extract DINOv3,
start Nav2, or continue the archived B1 counterfactual benchmark.

## Fixed C1 contract

- Reuse the frozen 5/2/3, ten-scene Habitat-GS split. Do not download scenes
  or unrelated data.
- Generate three deterministic trajectories per scene: 30 total, each 15--30
  seconds at 10Hz.
- Store timestamp, absolute robot pose, RGB, depth, `P_t^sim-lidar`,
  `G_lidar`, `G_rgbd`, `V`, `A`, causal fused field, oracle reference field,
  world-field origin, scene ID, and trajectory ID per frame.
- `P_t^sim-lidar` is a deterministic sparse geometric point observation made
  from simulator depth/geometry. It is an interface proxy for future real
  `/livox/lidar`, not an exact Livox scan simulation.
- `causal_field_t` uses observations through time `t` only. `oracle_field_t`
  may use the full completed trajectory solely as a reference target.
- Data generation is deterministic and resumable. A completed trajectory is
  immutable on rerun; partial trajectories may be regenerated.

## Required implementation

1. Add a single PyTorch Dataset/DataLoader reader for C1 trajectory files.
2. Generate the 30 trajectories under a C1-specific output root through
   `scripts/run_experiment.py`.
3. Validate finite pose and O/H/V/A values, exact 10Hz timestamps, modality
   schema, causal/oracle separation, and Dataset/DataLoader readability.
4. Generate `persistent_field_rollout.mp4`,
   `occlusion_revisit_example.mp4`, and `causal_vs_oracle.png` in
   `paper_assets/`.
5. Report trajectory/frame count, disk size, generation FPS, finite-field
   ratio, causal coverage growth, and causal-versus-oracle completeness.

## Acceptance criteria

- All 30 trajectories are complete, with absolute poses and finite O/H/V/A.
- Causal and oracle field sources are explicitly separated in metadata.
- At least one valid visible -> occluded -> visible world-cell case exists.
- All three required visual outputs exist.
- A unified PyTorch Dataset/DataLoader reads the completed dataset.
- A second invocation skips completed trajectories without overwriting or
  duplicating them.
- Metrics, result note, and registry/run record are saved.

## Stop condition

When C1 acceptance criteria are met, write the result note, move this plan to
`completed/`, update the registry, and stop. Do not begin C2, DINOv3,
learned-model training, Nav2, or predictive rollout work.
