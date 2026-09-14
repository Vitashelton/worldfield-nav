# Model Specification — Relation-Conditioned Spatial Goal Field

## Inputs

The state contains entity evidence, relation, phase, observed local geometry,
history and latest navigation event. Relations are APPROACH, CROSS, ENTER and
OBSERVE. Entity evidence is an upstream visual hypothesis or curated simulator
anchor, never hidden world coordinates at inference.

Inputs are frozen DINOv3 dense features, aligned RGB-D/LiDAR metric lifting,
relation token, entity visual evidence and history channels.

## Learned field

Geometry-aware lifting projects frozen visual features and depth into a
robot-centric grid. A lightweight relation-conditioned spatial decoder predicts
a dense executable target field. The relation token modulates the decoder; no
VLM, DINO backbone or planner is trained. Trainable parameters must remain
below 5M.

## Supervision

Offline curated portal planes, area anchors, landmark anchors and NavMesh
define relation-goal masks. APPROACH uses source-side approach cells; CROSS
uses destination-side cells; ENTER uses cells inside a target area; OBSERVE
uses cells with required landmark visibility. Dense BCE/Dice supervises masks.

## Transition and recovery

CROSS uses signed portal-side relation before/after execution as its completion
guard. Failure retains entity/relation, adds failed local regions to history
and re-queries the field. The controller never emits low-level controls.
