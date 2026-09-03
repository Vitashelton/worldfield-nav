# G1 — Cross-View Foundation Feature Audit

## Authority

This completed plan validated whether frozen DINOv3 features are physically
consistent under robot motion before any GeoAnchor adapter could be considered.
It did not authorize training, full-C1 extraction, navigation, Nav2,
real-robot work, LLM planning, or C1 modification.

## Executed scope

- Ran the authorized two-image GPU smoke for
  `timm/vit_small_patch16_dinov3.lvd1689m`.
- Used only representative completed-C1 trajectories across train, validation,
  and unseen scenes.
- Included small, medium, and large translation/rotation changes plus the
  known `interior_0405_840145_traj00` 8 → 15 → 23 occlusion/revisit sequence.
- Mined positives exclusively by depth, intrinsics, absolute pose,
  reprojection/depth/occlusion checks, world residual, and optional
  sim-LiDAR-like geometry checks. Appearance never created a positive pair.
- Evaluated frozen DINOv3 only. No training or batch extraction occurred.

## Required outputs

- `paper_assets/figures/g1_crossview_correspondence.png`
- `paper_assets/figures/g1_dinov3_viewpoint_degradation.png`
- `paper_assets/figures/g1_revisit_failure.png`
- `paper_assets/tables/g1_dinov3_baseline.csv`
- `docs/results/G1_CROSSVIEW_FOUNDATION_FEATURE_AUDIT.md`

## Decision

Completed with **GO**: the geometry-valid multi-scene large-view failure met
the pre-registered degradation condition. This makes a future adapter plan
eligible, but does not create or authorize one.
