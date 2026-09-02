# Benchmark Specification

## Scope

WorldFlow is evaluated as a sequential multimodal world-state task, not as a counterfactual rollout benchmark. The first dataset stage is C1, a small, deterministic Habitat-GS pilot that validates data integrity, three observation branches, and causal/oracle separation before any learned model is trained.

## Fixed C1 split and trajectories

Reuse the frozen Habitat-GS assets and split:

- Train: `scene01`, `scene02`, `scene03`, `scene09`, `interior_0405_840145`
- Validation: `scene04`, `scene05`
- Unseen test: `scene56`, `scene57`, `scene58`

C1 contains three deterministic continuous robot trajectories per scene, for 30 trajectories total. Each is 15--30 seconds at 10Hz and should naturally include forward motion, turns, doorway/corner views, occlusion, and revisit where the scene permits. This pilot does not require artificial context balancing and does not rerun B1 revelation probes.

## Per-frame data contract

Every frame stores timestamp, absolute robot pose, RGB data or reproducible reference, depth data or reproducible reference, `P_t^sim-lidar`, causal O/H/V/A field, oracle O/H/V/A field, world-field origin, scene ID, and trajectory ID. Fields and poses must be finite. The on-disk schema must be readable through a single PyTorch Dataset/DataLoader implementation.

`P_t^sim-lidar` is a sparse, deterministic geometric observation derived from Habitat-GS geometry/depth. It supports the LiDAR branch from the first data stage without claiming a precise Mid-360S scan model. Its schema is deliberately compatible with replacement by real `/livox/lidar` points.

## Causal/oracle contract

The causal field at `t` fuses observations up to and including `t`; it is the only field an online method may use. The oracle field is an offline fusion of the complete trajectory and serves only as a completeness/reference target. The schema, metadata, and evaluation code must make this separation explicit.

## Future formal model evaluation

Once a learned stage is authorized, compare M0 Frame-Only, M1 Geometric Memory, M2 ConvGRU Memory, and M3 WorldFlow. Evaluate geometry completeness, temporal stability/revisit consistency, visibility-aware state quality, cross-modal ablations, and navigation value by split. Action-conditioned rollout is optional evidence, not a required benchmark axis.

## C1 non-final diagnostics

Report trajectory/frame count, storage, throughput, finite O/H/V/A ratio, causal coverage growth, causal-versus-oracle geometry completeness, and a verified visible-to-occluded-to-revisited case. These are pilot diagnostics, not final paper claims.
