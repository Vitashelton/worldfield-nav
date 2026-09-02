# Research Contract

## Working title

**WorldFlow: Persistent Multimodal World Models for Indoor Robot Perception and Navigation**

## Scientific question

Can an indoor robot use RGB-D, LiDAR geometry, pose, and frozen foundation visual features to maintain a world-aligned metric latent field over time, and can a motion-aware learned update make that state more stable and complete than frame-wise perception or naive memory under viewpoint change, occlusion, and revisit?

## Central object

`Phi_t = [G_t, Z_t, V_t, A_t]`

- `G`: metric geometry. LiDAR point clouds are the primary metric-geometry stream; RGB-D supplies complementary dense local geometry.
- `Z`: world-aligned visual latent lifted from frozen DINOv3 RGB features.
- `V`: observation and visibility state.
- `A`: deterministic information age / memory freshness.

`Phi_t` is the persistent internal world state. Pose does not constitute a second world representation: it aligns all observation streams in the shared world frame.

## Core method

1. RGB is encoded by frozen DINOv3 dense features.
2. LiDAR-like points, RGB-D geometry, and visual features are lifted into a shared metric observation field `X_t` with pose.
3. The preceding field is deterministically transported using the measured pose delta.
4. A learned WorldFlow update fuses transported memory and the new multimodal observation field: `Phi_t = U_theta(T(Phi_(t-1), Delta p_t), X_t)`.
5. Perception, world understanding, and navigation query the same `Phi_t`.

## Simulator-to-robot modality contract

Habitat-GS C1 generates RGB, depth, absolute pose, and `P_t^sim-lidar`: a deterministic LiDAR-like geometric observation constructed from simulator depth/geometry. C1 does not attempt an exact Livox scan pattern. The data interface preserves separate LiDAR, RGB-D, and visual branches so the real Ranger Mini stream can replace `P_t^sim-lidar` with `/livox/lidar` without redesigning the field pipeline.

## Optional predictive extension

Action-conditioned future rollout is an optional planning-query mode: `Phi_t + a_(t:t+H) -> Phi_hat_(t+1:t+H)`. It is not the current central scientific question, training requirement, or paper identity.

## Not the paper

- A counterfactual-action benchmark paper.
- Scalar trajectory scoring, human forecasting, autonomous driving, or manipulation.
- Pure occupancy forecasting or visual video generation.
- Habitat-GS itself.

## Final evidence

The paper must demonstrate persistent multimodal state quality under viewpoint change, occlusion, and revisit; comparison against frame-only and memory baselines; navigation value from the resulting state; and Ranger Mini real-robot validation. Predictive rollout may be added only after this core evidence is established.
