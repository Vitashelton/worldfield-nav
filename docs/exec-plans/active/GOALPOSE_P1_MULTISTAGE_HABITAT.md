# GoalPose P1 — Multi-stage Indoor Transport Benchmark

## Objective

Evaluate whether task-conditioned local goal-pose refinement improves complete
indoor transport success when a single high-level image goal requires multiple
spatial stages. This is the formal Habitat-GS benchmark; P0 is retained only
as mechanism evidence.

## Fixed system boundary

Qwen supplies a cached, low-frequency qualitative task template only. It never
returns coordinates, actions, routes or planner parameters. Frozen DINOv3 may
provide visual target evidence. Candidate ranking uses only robot-observable
depth/LiDAR/pose-derived local geometry and execution history. Habitat NavMesh
and hidden target poses are labels/evaluation only. The low-level executor is
fixed across methods.

## Episodes

Create deterministic, scene-disjoint indoor transport episodes with three to
four recorded phases drawn from: corridor approach, doorway approach, doorway
crossing, room entry, and final target observation. Each episode retains one
high-level goal intent. The benchmark must include seen validation and unseen
scenes, at least 100 episodes per split, and preserve all seeds, starts,
candidate lattices and executor settings.

## Methods

M0 target-center terminal pose; M1 nearest observed free pose; M2
clearance-only geometric refinement; M3 task-conditioned GoalPose Field; M4
evaluation-only oracle. All share semantic target hypotheses, candidate
lattices, local executor and termination.

## Required metrics

Terminal pose: valid-goal rate, reachability, clearance, target visibility,
standoff error, yaw error and task-relation satisfaction.

Phase: phase success, handoff success, wrong-terminal-pose rate, collision,
timeout and decision count.

Complete task: transport success, SPL, normalized geodesic progress, final
DTG, path length, replan count and execution time. Compute bootstrap 95%
confidence intervals and scene-disjoint seen/unseen tables.

## Required assets

- paper_assets/tables/goalpose_p1_terminal_quality.csv
- paper_assets/tables/goalpose_p1_phase_and_transport.csv
- paper_assets/tables/goalpose_p1_unseen.csv
- paper_assets/figures/goalpose_p1_episode_outcomes.png
- paper_assets/figures/goalpose_p1_terminal_quality.png
- paper_assets/figures/goalpose_p1_failure_cases.png
- paper_assets/videos/goalpose_p1_multistage_transport.mp4

The video must show current RGB, phase/current goal intent, candidate terminal
poses, selected local goal, fixed-executor trajectory and phase transition.
Include at least corridor/doorway/room-entry cases and one failure comparison.

## Acceptance

1. No hidden-goal or NavMesh information is used for candidate selection.
2. Deterministic manifest and scene-disjoint protocol are saved.
3. M3 is compared to M0–M2 and M4 on identical episodes.
4. Results include complete-task as well as terminal/phase metrics.
5. At least one readable real Habitat-GS multi-stage video and three
paper-quality qualitative cases are saved.

After P1, write results and stop. Do not start real-robot work in this plan.
