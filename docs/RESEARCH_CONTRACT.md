# Research Contract — RelationNav: Relation-Verified Execution

## Working title

**RelationNav: Relation-Verified Execution for Multi-Stage Indoor Mobile Robot Navigation**

## Scientific question

Given a known spatial entity and active task relation, can a robot verify that
the intended physical relation was actually completed after navigation, and
recover from a failed realization while preserving that entity-relation intent?

Goal regions and relation-to-geometry compilation are established planning
tools. They are inputs to this work, not claimed contributions.

## Task

A task contains nodes such as APPROACH(portal), CROSS(portal), ENTER(area) and
OBSERVE(landmark). Each is an execution contract
``(entity, relation, admissible region, completion predicate, failure
predicate, recovery rule)``. CROSS completes only after the robot changes from
the source side of a portal plane to its destination side; ENTER only after
area containment; OBSERVE only after geometric visibility. Planner arrival is
never itself a stage-completion event.

## Method

RelationNav uses a relation verifier and a relation-preserving recovery
protocol around a fixed planner. Arrival, relation verification, collision,
blockage and no-progress update task state. Failure retains the entity and
relation, suppresses only the failed realization, and selects another
admissible realization from the same relation goal region.

## Roles

VLM is a low-frequency intent interface. DINOv3 is optional frozen visual
evidence, not a spatial-field predictor. Depth/LiDAR/pose give geometry. The
planner is fixed. Habitat-GS provides privileged labels and evaluation, not
real-world transfer proof.

## Evidence

Compare arrival-only navigation, same-goal retry, relation-verified execution,
relation-preserving recovery and an evaluation-only oracle on identical
scene-disjoint multi-stage episodes. Report false completion, wrong-stage
advance, relation completion, complete task success and recovery success.
Ranger Mini later provides small physical feasibility validation.
