# WorldFlow Model Specification

## Status

This is the target model contract. C2 authorizes frozen DINOv3 extraction, fixed projection, deterministic visual lifting, and representation diagnostics on completed C1 data. It does not authorize learned-model training, real LiDAR acquisition, or navigation execution.

## Persistent multimodal field

The field is world-aligned on Habitat world X-Z; Habitat world Y is height. The initial local crop is 128x128 over 10m x 10m (0.078125m/cell):

`Phi_t = [G_lidar, G_rgbd, Z_visual, V, A]`

- `G_t`: metric geometry. The LiDAR branch is primary; RGB-D provides complementary dense local geometry.
- `Z_t`: frozen-DINOv3 visual features lifted with RGB-D and pose into world field cells. It is absent from C1 feature extraction.
- `V_t`: observation state, including which cells received a modality observation.
- `A_t`: deterministic age since last observation. It is updated by time and visibility events, not a principal neural prediction target.

Every crop records its world origin. The crop may follow the robot, but cell coordinates retain world correspondence through that origin.

## C2 field instantiation

`G_lidar` and `G_rgbd` remain separate metric-geometry tensors. `Z_visual` is
a frozen DINOv3 dense feature after one dataset-level fixed projection and
causal depth-plus-pose world lifting. `V` and `A` retain their C1 definitions.
C2 adds no learned fusion: its persistent visual state is deterministic
transport-and-accumulation evidence for the later learned update stage.

## Observation branches and update graph

`RGB -> frozen DINOv3 dense feature -> visual branch`

`Depth + pose -> dense local geometry -> RGB-D branch`

`P_t^lidar + pose -> sparse metric geometry -> LiDAR branch`

`[visual, RGB-D, LiDAR] -> geometry-aware world lifting -> X_t`

`Phi_(t-1) + measured pose delta -> deterministic world-field transport -> Phi_transport`

`Phi_transport + X_t -> learned WorldFlow Update -> Phi_t`

In C1, `P_t^sim-lidar` is a deterministic, sparse point subset constructed from Habitat-GS depth/geometry. The data schema persists `G_lidar` and `G_rgbd` separately, then records fused causal `V`, `A`, and O/H/V/A. It is an interface-compatible proxy, not a claim of accurate Mid-360S/Livox scan simulation. On Ranger Mini it is replaced by the real `/livox/lidar` stream.

The update operator, rather than a per-frame reconstruction, is the method core. It decides what persistent information to retain, what multimodal observation to inject, and how visibility and freshness condition the update.

## Causal and oracle separation

`causal_field_t` may use observations `0:t` only and is the sole eligible online/model input. `oracle_field_t` is built offline from all observations in the completed trajectory solely as a geometry-completeness reference target. Oracle fields must never be passed to online updates, models, planners, or causal metrics.

## Baselines

- `M0 Frame-Only`: current RGB-D and LiDAR geometry lifted at the current frame, with no persistent temporal memory.
- `M1 Geometric Memory`: deterministic RGB-D/LiDAR/pose fusion without learned update.
- `M2 ConvGRU Memory`: a sequential learned memory baseline.
- `M3 WorldFlow`: transport plus learned multimodal visual-geometric field update.

## Non-negotiable properties

- The learned output is a persistent field, never only a navigation score.
- Pose and all geometry use an explicit shared world frame.
- LiDAR remains a first-class metric branch, not merely a localization aid.
- DINOv3 is frozen by default and is not the claimed contribution.
- Information age remains deterministic.
- Future observations and oracle fields cannot leak into online inputs.
