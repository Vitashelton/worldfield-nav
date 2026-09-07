# GeoAnchor Research Risks and Reviewer Defense

This memo is part of the project memory. It records the objections that must
be addressed before presenting GeoAnchor as a paper contribution.

## Current positioning

GeoAnchor is a geometry-anchored persistent visual place representation for
indoor robot relocalization under viewpoint change, occlusion, revisit, and
repetitive structure. It is not a claim of a new SLAM system, a new planner,
or a purely RGB relocalizer.

## Reviewer objections to guard against

1. **Incremental-fusion objection.** A reviewer may see GeoAnchor as merely
   DINO features averaged or stored in a 3-D map. The paper must specify the
   physical-anchor memory, multi-view aggregation/update rule, and geometric
   candidate verification. Required comparisons include frozen features,
   nearest-keyframe matching, appearance-only adaptation, 3-D memory without
   adaptation, and geometric re-ranking.

2. **Train/inference mismatch.** Depth and metric pose are not privileged
   training labels that disappear at test time. They are normal robot inputs:
   D435i depth and Mid-360S/LIO pose participate in anchor construction and
   retrieval. The claim is geometry-assisted visual place relocalization.

3. **Pose/depth leakage objection.** Results must include modest translation,
   yaw, and depth-noise tests plus an ablation without geometric validation,
   showing the method is not only exploiting perfect simulator state.

4. **Semantic mismatch.** DINOv3 supplies visual identity, not text-aligned
   object semantics. Do not claim scene-language understanding from DINOv3.
   If object names or relations are needed, add a separately identified
   CLIP/SigLIP (or curated-label) semantic channel.

5. **Simulation-to-real objection.** Habitat-GS establishes the controlled
   mechanism; a small Ranger Mini 2.0 experiment must test viewpoint change,
   occlusion/revisit, repetitive structures, and the same Frozen-vs-GeoAnchor
   comparison through localization and navigation.

6. **Scale objection.** The study is a feasibility evaluation, not a large
   benchmark. State the limited scene/route count and avoid broad claims.

7. **Downstream-value objection.** Feature similarity alone is insufficient.
   Report metric localization and at least one navigation loop using the same
   planner: success rate, SPL/path length, and wrong-place arrival rate.

8. **Tuning/complexity objection.** Keep DINOv3 frozen, keep the adapter or
   aggregation lightweight, use a fixed protocol, and report parameters,
   runtime, and controlled ablations. Improvements must be attributed to
   physical correspondence and persistent 3-D aggregation.

## Practical acceptance bar

The project is viable for a mid-tier journal if simulation shows repeatable
large-view/revisit gains, the real robot reproduces the direction of the
effect, and the improvement reaches localization or navigation. Do not claim
that GeoAnchor replaces mature visual-inertial/SLAM relocalization; position
it as a geometry-assisted visual place memory that complements those systems.
