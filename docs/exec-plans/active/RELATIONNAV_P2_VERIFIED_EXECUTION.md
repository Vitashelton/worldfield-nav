# RelationNav P2 — Verified Relation-Stage Execution

## Objective

Evaluate whether a fixed indoor navigation executor can falsely declare a
language-derived spatial stage complete, and whether relation-specific
verification plus relation-preserving recovery improves complete multi-stage
task execution. This plan does **not** claim novelty for goal regions, spatial
relation compilation, Nav2, Habitat-GS, or a learned relation field.

## Inputs and scope

Reuse the four audited InteriorGS scenes, curated portal/area/landmark
annotations and deterministic P1 manifests. No scene download, VLM training,
DINO training, planner modification or learned spatial field is allowed.

A stage is a contract:

``(entity, relation, admissible_goal_region, done, fail, recover)``.

The supported relations are APPROACH, CROSS, ENTER and OBSERVE.

## Methods

- **B0 Arrival-only**: fixed executor arrival advances the task state.
- **B1 Same-goal retry**: a failed realization is retried without semantic
  verification.
- **B2 Relation-verified**: task state advances only when the relation-specific
  predicate is true.
- **Ours Relation-verified + preserving recovery**: on failure retain the same
  ``(entity, relation)``, blacklist the failed realization and choose another
  predeclared admissible representative.
- **Oracle**: privileged best representative, evaluation upper bound only.

All methods share starts, regions, executor, budgets and termination rules.

## Required predicates

- APPROACH: source-side portal-neighborhood condition.
- CROSS: signed portal side changes from source to destination.
- ENTER: robot is contained in the declared area polygon.
- OBSERVE: landmark passes the RGB-D visibility test.

## Evidence

Report by scene-disjoint development/held-out split:

- relation completion rate;
- false-completion rate (arrival but predicate false);
- wrong-stage advance rate;
- recovery success rate and repeated-failure count;
- full multi-stage task success, SR/SPL, final DTG and path length.

Required assets are a task-contract architecture figure, an episode-level
false-completion-to-recovery matrix, three indoor cases (doorway crossing,
room entry, observation) and a readable Habitat-GS video. Every qualitative
case must show robot pose, phase, goal realization, predicate value, event and
trajectory in one metric coordinate system.

## Acceptance

A reproducible evaluator, scene-disjoint results, all metrics and the required
paper assets exist. Write a limitations note stating that the contract is an
engineering execution layer and not a new general planner. Archive this plan
then stop.
