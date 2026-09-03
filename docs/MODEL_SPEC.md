# GeoAnchor Model Specification

## G1 status

No GeoAnchor adapter is authorized. G1 evaluates only frozen public
`timm/vit_small_patch16_dinov3.lvd1689m` features on physically validated C1
cross-view correspondences.

## Physical correspondence miner

For a source RGB patch at frame `t`, use depth, intrinsics, and absolute camera
pose to back-project a world point `X`. Reproject `X` into frame `t+k` and
accept it only if the projected pixel is in bounds, target depth agrees with
the projected depth, the visibility/occlusion test passes, and the reconstructed
world residual is below a recorded threshold. Where available, sim-LiDAR-like
geometry is an additional geometric check.

Hard negatives are nearby projected image patches whose metric world surfaces
are physically distinct. Appearance similarity is never used to form positives.

## Frozen-feature evaluation

The model runs in eval/no-grad mode. Inspect and record actual
`forward_features()` output, prefix/register tokens, patch-token count, dense
feature shape, inference timing, and peak GPU memory before mining pairs.

For valid pairs, report physical positive cosine, hard-negative cosine, margin,
R@1, R@5, and retrieved world-position error by viewpoint regime, split, and
revisit. G1 does not extract all 4,500 frames and does not train an adapter.
