# ExecField P1 Model Specification — Frozen DINOv3 Spatial Fusion

Frozen `timm/vit_small_patch16_dinov3.lvd1689m` receives 256x256 current RGB
and emits `16x16x384` dense tokens. Aligned depth, intrinsics and robot/camera
pose project valid token centers into robot-centric metric cells. Multiple
tokens per cell use normalized mean pooling; empty cells have an explicit mask.

The local map contains DINO spatial features, occupancy/obstacle evidence,
free-space evidence, clearance distance, explored-valid mask, rasterized cached
VLM semantic scores, selected/failed subgoal history and visitation history.
A 1–5M parameter lightweight multi-scale spatial fusion encoder-decoder
outputs `F_reach`, `F_prog` and `F_fail`; it is not claimed as an architecture
contribution. `F_exec=alpha*F_reach+beta*F_prog-gamma*F_fail` is queried only
at fixed candidate positions and combined with frozen VLM scores.

Candidate-position masked losses are BCE reachability, SmoothL1 normalized
geodesic progress and BCE failure. DINOv3, VLM and executor stay frozen.

The geometry channels are obstacle occupancy/evidence, free-space evidence,
clearance distance and explored-validity. The semantic prior is formed by
Gaussian splatting cached candidate scores. History contains prior selected,
failed/no-progress and recent-visit evidence; P1 uses it only as a basic
static-history ablation, not as a dynamic recovery policy.
