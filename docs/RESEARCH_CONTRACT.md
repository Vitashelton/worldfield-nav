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

## Evidence

P0 remains motivation. P1 reports static indoor and scene-disjoint unseen
ImageNav; dynamic Habitat-GS recovery then evaluates whether history-aware
reselection improves a shared executor. Ranger Mini evidence is restricted to
logged recoverable Nav2 abort/costmap cases until a later authorized plan.
