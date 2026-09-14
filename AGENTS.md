# AGENTS.md — RelationNav Research Harness

## Mission

The active paper studies RelationNav: task-relation-grounded spatial transition
navigation for indoor mobile robots. A long task is a sequence of spatial
relations such as APPROACH, CROSS, ENTER and OBSERVE. The robot grounds the
current entity and relation into an executable local target region, verifies
the spatial transition, and retains the relation after execution failure.

This is not a new SLAM system, Nav2 planner, end-to-end VLA controller, generic
ObjectNav benchmark or universal long-horizon planner.

## Source of truth

Read this file, docs/RESEARCH_CONTRACT.md, docs/MODEL_SPEC.md,
docs/BENCHMARK_SPEC.md, and the single plan in docs/exec-plans/active/.

## System boundary

Frozen VLM supplies only an interchangeable low-frequency entity/relation
intent. It never outputs metric coordinates, candidate IDs, waypoints, routes,
NavMesh data, planner parameters or controls. Frozen DINOv3-S/16 is visual
evidence. RGB-D/LiDAR/pose provide observed geometry. A learned
relation-conditioned spatial goal field predicts local cells satisfying the
active relation. The fixed planner/executor consumes the selected goal.

## Data and evaluation

Offline NavMesh and curated portal/landmark annotations create supervision and
evaluate simulation outcomes; neither may select an online candidate. Curated
entities must be recorded explicitly. Current dynamic-avatar assets are outdoor
scene09 only; do not claim indoor dynamic evaluation.

## Status

WorldFlow, GeoAnchor, ExecField, TrajectoryGrounding, Failure-Aware and
GoalPose work are preliminary infrastructure. The sole authorized work is the
active RelationNav plan. Finish it, archive it, then stop.
