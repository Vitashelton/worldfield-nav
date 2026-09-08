# Research Contract — Failure-Aware Rolling Subgoal Selection

## Working title

**Failure-Aware Rolling Subgoal Selection for VLM-Guided Indoor Image-Goal Navigation**

## Problem

Semantic relevance does not make a local direction physically executable: a
candidate can collide, stall, fail to progress or become blocked. The problem
is selecting and revising short-horizon executable subgoals while the fixed
navigator retains responsibility for motion execution.

## Method boundary

At each observation, a goal-independent K=8 candidate generator proposes local
subgoals. Frozen DINOv3 encodes current and goal imagery; geometry and
failure/visit history supply local state. A lightweight scorer predicts
candidate execution value from reached, collision, stuck, progress and
path-length supervision collected in Habitat-GS. On abort/stuck, the selected
failure is written to history and candidates are re-scored. No BEV field, new
planner, trajectory-image grounding or foundation-model training is part of the
paper.

VLM is deliberately outside the control loop. Standard ImageNav needs no VLM:
the goal image is the task condition. For an optional product-level language or
ambiguous-image interface, a frozen VLM may parse the task once into a
structured semantic constraint. It cannot see hidden pose/geometry or output
actions, waypoints, coordinates, candidate rankings, or planner state. The
continuous local loop remains `candidate generator -> scorer -> fixed
executor -> failure history -> re-score`.

## Method identity

The paper's learned component is the **Execution Bridge**: a lightweight,
goal-conditioned branch-value model trained from matched Habitat-GS outcomes
for the same state and multiple planner proposals.  It predicts whether a
proposal is worth executing (reach, progress, collision/stuck risk); it does
not generate a trajectory or replace the planner.  Failure memory is a
robot-maintained record of rejected/failed proposal directions used by the
rolling selector to avoid repeating a demonstrated local failure.

The optional **Semantic Intent Bridge** is a product-system interface.  It
maps an image/language task through a frozen VLM to a structured semantic task
description once per task or rare semantic ambiguity.  It neither supplies
physical control nor changes the core P1 benchmark.  Therefore the paper's
central causal claim is `learned execution bridge > handcrafted geometry
rules` under identical starts, goals, proposals and executor.

## Evidence

P0 remains motivation. P1 reports static indoor and scene-disjoint unseen
ImageNav; dynamic Habitat-GS recovery then evaluates whether history-aware
reselection improves a shared executor. Ranger Mini evidence is restricted to
logged recoverable Nav2 abort/costmap cases until a later authorized plan.
