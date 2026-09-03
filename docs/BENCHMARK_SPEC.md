# GeoAnchor Benchmark Specification

## G1 audit set

G1 uses only a small representative subset of completed C1 trajectories from
train, validation, and unseen scenes. It includes small, medium, and large
viewpoint change; translation; rotation; doorway/corner views; and the known
`interior_0405_840145_traj00` frame 8 -> 15 -> 23 occlusion/revisit case.

No semantic labels are required and no full-dataset feature extraction is
authorized.

## Geometry validation

Every correspondence records source/target pixels, projected depth, observed
target depth, depth residual, world-coordinate residual, and optional
sim-LiDAR agreement. Report these errors so a visual-feature failure cannot be
attributed to invalid geometry labels.

## Metrics

Report positive cosine, hard-negative cosine, positive-negative margin, R@1,
R@5, and retrieved world-position error for small/medium/large/revisit and
train/validation/unseen categories.

## Required assets

- `paper_assets/figures/g1_crossview_correspondence.png`
- `paper_assets/figures/g1_dinov3_viewpoint_degradation.png`
- `paper_assets/figures/g1_revisit_failure.png`
- `paper_assets/tables/g1_dinov3_baseline.csv`

The G1 result must make an explicit GO or NO-GO recommendation against its
preregistered threshold; it cannot introduce adapter training.
