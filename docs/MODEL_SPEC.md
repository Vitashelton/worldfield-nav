# MetricAnchor Model Specification

## Frozen visual input

`timm/vit_small_patch16_dinov3.lvd1689m` receives RGB resized and normalized by
its timm data config. It emits a 16 × 16 grid of 384-dimensional patch tokens
after removal of 5 prefix tokens. Backbone parameters remain frozen.

## Residual adapter

For dense input `F ∈ R^(16×16×384)`:

`A(F) = normalize(F + W2(GELU(DWConv3×3(GELU(W1(F))))))`

where `W1: 384→128`, depthwise spatial mixing uses 128 channels, and
`W2: 128→384`. The adapter is below 2M trainable parameters.

## Methods

- **M0 Frozen DINOv3:** L2-normalized cached tokens.
- **M1 Vanilla Cross-View Adapter:** identical adapter, metric positive pairs
  with random negatives only.
- **M2 MetricAnchor:** M1 plus physical hard negatives from distinct metric
  surfaces.
- **M3 MetricAnchor-Full:** M2 plus three-view consistency and feature
  preservation.

All use the identical cache, correspondence manifests, descriptor pooling, and
evaluation protocol. A learned method never re-runs the backbone.

## Losses

Contrastive positive-vs-candidate InfoNCE is used for M1/M2. M2 candidates
include physically distinct hard surfaces; M1 samples random negatives. M3 adds
a three-view cosine-consistency term and an L2 feature-preservation term.
