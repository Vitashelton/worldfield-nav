# ExecNav P0 — Oracle Executability Kill Test

## Question

Before training any critic, does perfect knowledge of subgoal executability
improve image-goal navigation over frozen-VLM semantic proposals alone?

## Scope

Use only existing Habitat-GS scenes/assets and its ImageNav/NavMesh executor.
Do not train a VLM, critic, policy, or ImageNav agent. Do not download a new
checkpoint or generate a large dataset.

## Methods

- **M0 semantic-free:** deterministic geometric candidate.
- **M1 VLM-only:** frozen/cached semantic subgoal proposal, with no critic.
- **M1+Oracle:** the identical M1 candidates, scored by privileged simulator
  executability labels (reachability, collision-free path, clearance, target
  visibility, and eventual ImageNav success).

The oracle is an upper bound only, never an online method or paper result.
If a live VLM backend is unavailable, use explicitly-labelled cached/curated
semantic proposals so the geometry/executability protocol can be tested.

## Pilot

Create 15–30 deterministic episodes over 3–5 existing scenes. Include ordinary
goals, occlusion/revisit, repeated structures, and deliberately invalid or
blocked semantic subgoals. Use the same candidates, starts, goal images and
Habitat executor for M1 and M1+Oracle.

## Acceptance criteria

Save a manifest, candidate/teacher-label table, and metrics for executable
goal rate, ImageNav success, SPL/path length, wrong-goal arrival, and failure
reason. Generate one VLM-only versus oracle figure. The direction is GO only
if oracle selection yields a clear, repeatable success/SPL gain and reduces
wrong-goal/invalid-subgoal failures. Otherwise archive ExecNav without training
the critic.

## Stop

After P0, write the decision and stop. Do not train the critic, add recovery,
start real robot work, or begin P1 without an explicit new plan.
