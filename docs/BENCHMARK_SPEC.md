# Goal-Pose Field Benchmark

Each curated case has a goal image, semantic target anchor, bounded approach
template, robot start pose and hidden evaluation target pose/area. Anchors and
templates are explicitly curated, never described as automatic labels.

P0 reuses verified indoor scenes and a small curated set. Formal expansion is
authorized only after P0 accepts the problem signal. Splits are scene-disjoint.

Methods: M0 target center; M1 target plus nearest free cell; M2 target plus
clearance/reachability heuristic; M3 Task-Conditioned Goal-Pose Field; M4
evaluation-only privileged oracle. All share target hypothesis, lattice,
executor, footprint and termination.

Report executable-goal rate, reachable-goal rate, selected-pose clearance,
target visibility, task-relation satisfaction, SR/SPL, path length, final
position error and orientation error, with seen/unseen results separated.

P0 must show at least three indoor cases where center/nearest-free selection is
executable-poor or task-inappropriate while a field-selected pose is objectively
executable and task-satisfying. If this requires fabricated annotations or
privileged deployment input, stop.
