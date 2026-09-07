# ExecNav Model Specification

## VLM task parser

Input: goal RGB image plus optional natural-language instruction. Output:
semantic target, relation, confidence, and 2-D visual prior. The backend is
replaceable and may be cached or remote. It cannot output world coordinates or
low-level actions.

## Metric grounding and critic

Depth/geometry and robot pose map the semantic prior to candidate 2-D subgoals.
Candidates carry free-space, clearance, visibility, reachability, and relation
features. `C_theta(candidate, context)` predicts physical executability and
likely image-goal success from privileged Habitat outcomes. Inference uses only
robot-available observations. Recovery re-scores alternatives after rejection
or failure.

## Fixed comparisons

- M0 semantic-free nearest-free-cell/geometric baseline;
- M1 VLM semantic subgoal without critic;
- M2 VLM plus handcrafted feasibility checks;
- Ours: VLM plus learned executability critic and recovery.

NavMesh/Nav2 remains the unchanged executor; no end-to-end action policy is
trained in this plan.
