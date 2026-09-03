# G1 — Frozen DINOv3 Cross-View Foundation Feature Audit

## Decision

**GO (problem validation only).**  This result authorizes consideration of a
future, explicitly planned GeoAnchor adapter experiment.  It does not itself
claim that an adapter works, and it does not authorize training.

## Fixed audit scope

- Backbone: `hf_hub:timm/vit_small_patch16_dinov3.lvd1689m`, frozen.
- Source: completed C1 trajectories only; 14 fixed frame pairs across train,
  validation, and unseen scenes.
- Positive pairs: depth back-projection, absolute-pose reprojection, target
  depth/occlusion validation, and world-coordinate residual threshold; image
  appearance was never used to create positives.
- Hard negatives: target patches at least 0.50 m from the physical surface.

## Smoke

On NVIDIA GeForce RTX 4080 SUPER, the two-image smoke completed using 256 ×
256 input. `forward_features()` returned `[2, 261, 384]`: 5 prefix tokens, 0
register tokens, and 256 dense patch tokens (`16 × 16 × 384`). First forward
time was 0.127 s and peak allocated GPU memory was 101.0 MiB.

## Geometry validity and coverage

The final audit mined 406 valid cross-view physical correspondences. Mean
depth residual was 0.0296 m, mean world-coordinate residual was 0.0405 m, and
mean nearest sim-LiDAR-like residual was 0.1377 m. Large-view positives were
available in all audit roles: train (24), validation (3), unseen (9).

## Frozen-feature result

| Regime | Correspondences | Positive cosine | Hard-negative cosine | Margin | R@1 | R@5 | World error (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | 204 | 0.957 | 0.958 | -0.001 | 0.461 | 0.848 | 0.629 |
| Medium | 67 | 0.887 | 0.906 | -0.019 | 0.194 | 0.493 | 1.564 |
| Large | 36 | 0.797 | 0.826 | -0.030 | 0.056 | 0.250 | 2.139 |
| Revisit (8 → 23) | 29 | 0.859 | 0.859 | 0.000 | 0.034 | 0.241 | 0.269 |

Large relative to small therefore loses 40.5 percentage points R@1 and 0.160
absolute positive cosine similarity, while the metric correspondence residual
remains approximately 4 cm. This satisfies the pre-registered G1 GO condition
and appears in train, validation, and unseen scenes rather than one selected
example.

Unseen-scene aggregate: 96 correspondences, positive cosine 0.934, R@1 0.344,
R@5 0.677, world-position error 0.978 m.

## Assets and exact result

- `outputs/formal/G1/audit_final/metrics.json`
- `paper_assets/figures/g1_crossview_correspondence.png`
- `paper_assets/figures/g1_dinov3_viewpoint_degradation.png`
- `paper_assets/figures/g1_revisit_failure.png`
- `paper_assets/tables/g1_dinov3_baseline.csv`

The known sequence `interior_0405_840145_traj00` frame 8 → 15 → 23 is retained
in the audit. The 8 → 23 revisit comparison contains 29 geometry-valid
correspondences; the 8 → 15 occlusion-transition check contains 70 locally
visible physical correspondences and is recorded separately, not folded into
the revisit metric.

## Boundary

No model was trained. No all-C1-frame feature extraction, navigation, Nav2,
real-robot, LLM, or adapter work was started. There is intentionally no active
post-G1 plan; the next work requires explicit research authorization.
