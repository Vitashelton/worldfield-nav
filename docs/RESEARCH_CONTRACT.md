# Research Contract — Failure-Aware Rolling Subgoal Selection

## Working title

**Failure-Aware Rolling Subgoal Selection for VLM-Guided Indoor Image-Goal Navigation**

## Problem

A VLM may judge a local direction semantically compatible with a goal image,
but that candidate can collide, stall, fail to progress or become blocked.
The problem is selecting and revising short-horizon executable subgoals while
the fixed navigator retains responsibility for motion execution.

## Method boundary

At each observation, a goal-independent K=8 candidate generator proposes local
subgoals. Frozen DINOv3 encodes current and goal imagery; cached VLM supplies a
low-frequency semantic prior; geometry and failure/visit history supply local
state. A lightweight scorer predicts candidate execution value from reached,
collision, stuck, progress and path-length supervision collected in Habitat-GS.
On abort/stuck, the selected failure is written to history and candidates are
re-scored. No BEV field, new planner, trajectory-image grounding or foundation
model training is part of the paper.

## Evidence

P0 remains motivation. P1 reports static indoor and scene-disjoint unseen
ImageNav; dynamic Habitat-GS recovery then evaluates whether history-aware
reselection improves a shared executor. Ranger Mini evidence is restricted to
logged recoverable Nav2 abort/costmap cases until a later authorized plan.
