# C1 — Sequential Multimodal World-Field Dataset Pilot

## Outcome

C1 completed and passed all acceptance criteria on 2026-09-03. It is a
deterministic, resumable data pilot for the persistent WorldFlow state; it does
not include DINOv3 feature extraction, learned-model training, Nav2, or an
action-conditioned rollout experiment.

## Dataset and validation

| Check | Result |
|---|---:|
| Completed trajectories | 30 / 30 |
| Frames | 4,500 |
| O/H/V/A finite ratio | 1.000 |
| Dataset disk size | 1,347,681,797 bytes (1.35 GB decimal) |
| Generation throughput | 8.097 frames/s |
| Mean causal coverage growth | 203.944 m² |
| Mean causal/oracle geometry completeness | 0.7346 |
| Final-frame causal/oracle completeness | 1.0000 |
| Dataset/DataLoader | PASS (4,500 frames; batch size 2) |
| Second invocation | PASS (30 complete trajectories skipped) |

Each trajectory stores absolute pose, timestamped RGB/depth, deterministic
simulated LiDAR-like geometry, separate `G_lidar`, `G_rgbd`, `V`, and `A`
arrays, the causal online field, and a separately labeled oracle-only reference
field. The PyTorch reader exposes the oracle only as
`oracle_field_reference_only`.

## Qualitative evidence

- `paper_assets/videos/persistent_field_rollout.mp4`
- `paper_assets/videos/occlusion_revisit_example.mp4`
- `paper_assets/figures/causal_vs_oracle.png`

The selected occlusion/revisit case is
`interior_0405_840145_traj00`, world cell `(-43, -57)`: visible at frame 8,
occluded at frame 15, and revisited at frame 23.
