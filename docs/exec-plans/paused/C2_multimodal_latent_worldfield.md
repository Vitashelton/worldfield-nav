# C2 — Multimodal Latent World Field

## Authority and objective

C2 is the only active plan. It establishes the paper's frozen multimodal
latent world representation on completed C1 data. It does not modify C1 data,
train a WorldFlow update network, run navigation, or execute B1.

## Fixed representation

`Phi_t = [G_lidar, G_rgbd, Z_visual, V, A]`

- `G_lidar`: C1 simulated LiDAR-like metric geometry; it remains primary.
- `G_rgbd`: C1 RGB-D metric geometry.
- `Z_visual`: frozen DINOv3 dense feature lifted with depth and absolute pose.
- `V`: causal visibility state.
- `A`: deterministic information age.

## Authorized work

1. Extract frozen DINOv3 Small or Base patch features for all C1 RGB frames.
2. Apply one fitted, fixed 32--64 channel projection; do not retain raw high-D features.
3. Lift visual features using C1 depth, intrinsics, and absolute pose to the
   existing 128x128, 10m x 10m world grid without future-observation leakage.
4. Add a unified PyTorch Dataset/DataLoader sample for all five field branches.

Use frozen weights only; record backbone, revision, input and patch resolution,
latent dimensionality, extraction throughput, and incremental disk consumption.

## Required evidence

- Cross-view latent cosine consistency for shared physical world cells.
- LiDAR-only, RGB-D-only, and combined geometry completeness/occupancy metrics
  relative to the C1 oracle geometry reference.
- The existing `interior_0405_840145_traj00` frame 8 -> 15 -> 23
  visible -> occluded -> revisited latent-memory case.

## Required paper assets

- `paper_assets/figures/c2_multimodal_worldfield.png`
- `paper_assets/figures/c2_crossview_consistency.png`
- `paper_assets/figures/c2_modality_complementarity.png`
- `paper_assets/videos/c2_occlusion_revisit_latent.mp4`

## Acceptance criteria

- All 4,500 C1 frames have readable `Z_visual`, or a documented pilot expands
  to all frames before acceptance.
- `Z_visual` alignment to the 128x128 world grid validates without future leak.
- A unified Dataset/DataLoader returns LiDAR, RGB-D, visual, visibility, and age.
- Cross-view and modality-complementarity metrics are saved.
- The specified occlusion/revisit latent video and all four paper assets exist.
- No WorldFlow neural update is trained and no navigation experiment is run.

## Stop condition
