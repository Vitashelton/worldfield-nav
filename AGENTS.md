# AGENTS.md — Goal-Pose Field Research Harness

## Mission

The active paper studies **VLM-guided semantic goal-pose refinement for indoor
mobile robot navigation**. The central distinction is between a semantic target
location and an executable, task-appropriate robot goal pose `q=(x,y,theta)`.
This project does not propose a new SLAM system, Nav2 planner, end-to-end VLA
controller, world model or generic ImageNav benchmark.

## Source of truth

Read, in order: this file; `docs/RESEARCH_CONTRACT.md`; `docs/MODEL_SPEC.md`;
`docs/BENCHMARK_SPEC.md`; and the single plan in `docs/exec-plans/active/`.
If an implementation conflicts with them, stop and report the conflict.

## System boundary

Frozen VLM parses a user request into a bounded task/approach template. It does
not output metric coordinates, waypoints, trajectories, NavMesh data, planner
parameters or controls.

Frozen DINOv3-S/16 supplies dense visual evidence between goal/current images.
It is not a global-localization oracle and is never trained here.

Depth/LiDAR/pose/costmap provide reachability, free space, clearance and metric
coordinates. Goal-Pose Field ranks feasible poses around a semantic target
using task-conditioned approach, visibility and geometric executability. Nav2
or the Habitat agent executes the selected pose and is not a contribution.

## Runtime and data policy

Project root: `/root/autodl-tmp/worldfield_nav`. Reuse installed Habitat-GS,
DINOv3, downloaded scene assets and prior outputs. Do not reinstall PyTorch,
CUDA or Habitat-GS, rebuild extensions, or audit the environment broadly.

Habitat-GS is used for controlled geometry, rendering, NavMesh evaluation and
dynamic-avatar stress tests. Its outputs are not proof of real-robot transfer;
Ranger evaluation is required for any transfer claim. Do not fabricate semantic
labels: curated target anchors must be recorded as curated.

## Evaluation standard

Every formal method shares target hypotheses, candidate-pose lattice, geometry,
executor and episodes. Compare target-center, nearest-free, clearance heuristic,
Goal-Pose Field and evaluation-only oracle. Report executable/reachable goal
rate, clearance, target visibility, task-relation satisfaction, SR/SPL, path
length, final position and orientation errors, separately for seen/unseen.

## Status and stopping

WorldFlow, GeoAnchor, ExecField, TrajectoryGrounding and Failure-Aware P1 are
preliminary/reusable infrastructure only. Do not rerun or extend them. The sole
authorized task is the active Goal-Pose Field plan. When it passes, write
results, update registry, archive the plan and stop.
