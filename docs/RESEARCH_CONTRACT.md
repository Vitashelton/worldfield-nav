# Research Contract — Goal-Pose Field

## Working title

**Vision-Language-Guided Semantic Goal-Pose Refinement for Mobile Robot Navigation**

## Scientific question

After a robot identifies a semantic target, how can it select an executable,
safe and task-appropriate robot goal pose rather than navigate to the target's
estimated geometric center?

## Boundary

This is not generic ImageNav, language-to-object localization, SLAM, a new
planner, planner-parameter tuning, end-to-end VLA control or simulator-only
outcome learning. A target center can lie inside furniture, behind a door or
inside an elevator; the robot instead needs a valid approach pose `q=(x,y,theta)`.

## Core computation

Frozen VLM maps goal image and optional language to a bounded task/approach
template. Frozen DINOv3 maps goal/current RGB to visual target evidence.
Depth, LiDAR and pose give a metric target hypothesis and local geometry. A
candidate-pose lattice is scored by Goal-Pose Field `E(x,y,theta)` and its best
pose is sent to the fixed Nav2 executor.

The field combines physical freedom/reachability, clearance, target visibility,
task-conditioned standoff/orientation and target-hypothesis uncertainty.

## Roles

The VLM outputs only a bounded template (`observe_doorway`, `approach_doorway`,
`wait_at_elevator`, `inspect_target`), never coordinates/actions/trajectories.
DINOv3-S/16 is frozen visual evidence, not text semantics, metric geometry or
reliable global localization.

Habitat-GS supplies controlled NavMesh, visibility and clearance evaluation. It
does not provide real-robot truth. Ranger Mini later validates the frozen
algorithm on a small held-out physical set, not a large training dataset.

## Evidence

The claim is supported only if Goal-Pose Field improves task-appropriate,
executable arrivals over target-center and standard geometric refinements in
seen and scene-disjoint unseen indoor scenes, then in controlled Ranger tests.
