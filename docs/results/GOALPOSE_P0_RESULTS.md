# GoalPose P0 — Controlled Indoor Problem Signal

## Scope

P0 is a small, controlled feasibility study, not the formal multi-stage
transport benchmark. It uses 12 provisional visual target anchors across four
previously verified indoor Habitat-GS scenes. Each anchor is paired with a
fixed lattice of 60 candidate terminal robot poses. The candidate lattice is
independent of the hidden evaluation goal.

The target RGB center was visually reviewed as a provisional anchor. Where its
depth was invalid, the nearest valid observed pixel was recorded explicitly.
Habitat NavMesh, obstacle clearance and rendered target visibility are used
only as privileged evaluation checks in this P0 study; they are not claimed as
deployment inputs.

## Semantic interface

Local Qwen3-VL 8B Instruct produced a cached qualitative template for each
scene pair, such as approach_doorway or room_entrance, plus a standoff class
and whether the robot should face the target. It did not receive or output
coordinates, candidate IDs, routes, controls, depth, or NavMesh data. The VLM
cache is a task-intent input rather than ground truth or a contribution claim.

## Result

| Method | Executable goal | Target visible | Task relation satisfied | Mean clearance |
|---|---:|---:|---:|---:|
| M0 target center | 8.3% | 41.7% | 0.0% | 0.024 m |
| M1 nearest free | 0.0% | 25.0% | 0.0% | 0.015 m |
| M2 clearance heuristic | 91.7% | 50.0% | 33.3% | 0.938 m |
| M3 GoalPose field | 91.7% | 66.7% | 66.7% | 0.625 m |
| M4 privileged oracle | 91.7% | 75.0% | 66.7% | 0.699 m |

M3 preserves the executable-goal rate of M2 while improving target visibility
by 16.7 percentage points and task-relation satisfaction by 33.4 percentage
points. The result supports the narrow problem signal: a semantic surface
center, or a clearance-only free pose, is not necessarily a task-appropriate
terminal robot pose.

## What this does and does not establish

P0 demonstrates a static, single-stage endpoint distinction under controlled
simulation checks. It does **not** establish semantic-target detection,
multi-stage task completion, dynamic recovery, a learned policy, or real-robot
transfer. The next formal Habitat benchmark must hold the high-level task
intent across multiple spatial stages and measure whether repeated local
goal-pose refinement improves complete task success.

## Assets

- Candidate manifest: outputs/formal/GoalPose/P0/manifest.json
- Per-method table: paper_assets/tables/goalpose_p0_summary.csv
- Qualitative cases: paper_assets/figures/goalpose_p0_problem_cases.png
- Field examples: paper_assets/figures/goalpose_p0_field_examples.png
