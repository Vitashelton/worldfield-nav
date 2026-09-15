# AGENTS.md — TopoNav Harness

## Mission

The active paper studies **Semantic-Topology-Grounded Closed-Loop VLM Planning
for Indoor Mobile Robot Navigation**.  A frozen VLM is a low-frequency,
high-level agent.  It selects named semantic-topology transitions; it never
outputs coordinates, waypoints, controls, NavMesh information, hidden goal
poses or planner parameters.

The contribution is the executable protocol around the VLM:

1. task-relevant topology and selective visual-context compilation;
2. topology-grounded, typed tool calls;
3. deterministic validation and metric execution;
4. typed execution feedback and relation-preserving recovery.

This is not a new SLAM system, end-to-end VLN/VLA model, costmap, local
planner, world model or general semantic-topology representation.

## Source of truth

Read this file, `docs/RESEARCH_CONTRACT.md`, `docs/MODEL_SPEC.md`,
`docs/BENCHMARK_SPEC.md`, and exactly one plan in `docs/exec-plans/active/`.
The active plan is the only authorized research work.

## Runtime boundary

Habitat-GS supplies RGB-D, agent state, NavMesh execution and evaluation-only
ground truth.  It never gives the VLM an oracle route or target coordinates.
Qwen and DeepSeek are interchangeable frozen VLM backends; cache every request.
The local Qwen backend is used for reproducible simulation.  The real robot
later uses the same tool protocol with DeepSeek API, D435i RGB, and
Mid-360S/LIO plus Nav2.  LIO localizes the *current robot* for control but is
never passed to a VLM as a hidden goal location.

## Evidence policy

All reported videos must replay actual Habitat execution logs.  Do not claim
dynamic-avatar evidence unless official dynamic assets and true collision or
progress logs were used.  Do not fabricate blocked edges or successful
recoveries.  Preserve all WorldFlow, GeoAnchor, ExecField, RelationNav and
other preliminary assets.

## Definition of done

An experiment is done only with fixed manifests, cached VLM calls, per-episode
logs, metrics, paper-ready figures/video and an explicit limitations note.
Stop when the active plan's acceptance criteria are met.
