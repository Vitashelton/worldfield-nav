# AGENTS.md — RelationNav Research Harness

## Mission

The active paper studies RelationNav: relation-verified execution for
multi-stage indoor mobile robots. A long task is a sequence of spatial
relations such as APPROACH, CROSS, ENTER and OBSERVE. The contribution is not
the conversion of a relation into a region. It is the explicit execution
semantics that decides whether a relation was physically completed, and keeps
the same relation active after a recoverable execution failure.

This is not a new SLAM system, Nav2 planner, end-to-end VLA controller, generic
ObjectNav benchmark or universal long-horizon planner.

## Source of truth

Read this file, docs/RESEARCH_CONTRACT.md, docs/MODEL_SPEC.md,
docs/BENCHMARK_SPEC.md, and the single plan in docs/exec-plans/active/.

## System boundary

Frozen VLM supplies only an interchangeable low-frequency entity/relation
intent. It never outputs metric coordinates, candidate IDs, waypoints, routes,
NavMesh data, planner parameters or controls. RGB-D/LiDAR/pose provide
observed geometry. A declarative relation contract supplies completion,
failure and recovery predicates. The fixed planner/executor consumes an
ordinary admissible goal from a pre-existing region.

## Data and evaluation

Offline NavMesh and curated portal/landmark annotations create supervision and
evaluate simulation outcomes; neither may select an online candidate. Curated
entities must be recorded explicitly. Current dynamic-avatar assets are outdoor
scene09 only; do not claim indoor dynamic evaluation.

## Status

WorldFlow, GeoAnchor, ExecField, TrajectoryGrounding, Failure-Aware, GoalPose
and the RelationNav learned-field formulation are preliminary infrastructure.
The sole authorized work is the active RelationNav execution-semantics plan.
Finish it, archive it, then stop.
