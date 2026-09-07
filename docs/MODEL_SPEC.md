# Trajectory Grounding P1 Model Specification

For each decision state a deterministic local planner proposes K=8 short polar
trajectory corridors from current robot pose. Proposal generation observes only
robot-centric local geometry and heading; it never reads the hidden goal pose,
goal direction or NavMesh.

Frozen DINOv3-S/16 (`timm/vit_small_patch16_dinov3.lvd1689m`) extracts a
16x16x384 dense map from the current 256x256 RGB and a global goal-image
descriptor. Candidate corridors are depth/intrinsics projected into the current
image and pool valid DINO tokens along each corridor. Their normalized visual
similarity to the goal descriptor is the DINO trajectory-goal score.

Frozen cached DeepSeek Vision sees only goal image, current RGB and a numbered
candidate-trajectory overlay, yielding a semantic ranking/score per candidate.
It has no coordinate, NavMesh or outcome access.

Planner geometry features (free-space support, clearance, trajectory length,
observed ratio), VLM score and DINO score are fused by a 1--5M-parameter
lightweight trajectory ranker. The ranker predicts normalized geodesic progress
and eventual candidate success/failure, and selects a trajectory. No VLM or
DINO weight is trained. The same fixed Habitat executor runs every method.
