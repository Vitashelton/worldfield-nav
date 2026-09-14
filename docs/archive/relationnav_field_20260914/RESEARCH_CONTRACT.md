# Research Contract — RelationNav

## Working title

**RelationNav: Task-Relation-Grounded Spatial Transition Navigation for Indoor Mobile Robots**

## Scientific question

Given a known spatial entity and active task relation, can a robot predict an
executable local target region, verify the intended spatial transition after
execution, and recover while preserving the same entity-relation intent?

Semantic entity plus task relation implies an executable spatial goal region,
not one fixed coordinate.

## Task

A task contains nodes such as APPROACH(portal), CROSS(portal), ENTER(area) and
OBSERVE(landmark). Each has an entity, relation, region-level completion guard
and event-conditioned transition. CROSS completes only after the robot changes
from the source side of a portal plane to its destination side.

## Method

RelationNav predicts a local relation-goal field from entity visual evidence,
relation, RGB-D geometry, pose and history. A fixed planner executes the
selected local goal. Arrival, relation verification, collision, blockage and
no-progress update task state. Failure retains the entity and relation,
suppresses failed choices and re-grounds a replacement region.

## Roles

VLM is a low-frequency intent interface. DINOv3 is frozen visual evidence.
Depth/LiDAR/pose give geometry. RelationNav is the only learned method; the
planner is fixed. Habitat-GS provides privileged labels and evaluation, not
real-world transfer proof.

## Evidence

Compare fixed portal offsets, nearest-free goals, VLM direct candidate choice,
geometry-only grounding, RelationNav and evaluation-only oracle on identical
scene-disjoint multi-stage episodes. Report region prediction, relation
verification, complete task success and relation-preserving recovery. Ranger
Mini later provides small physical feasibility validation.
