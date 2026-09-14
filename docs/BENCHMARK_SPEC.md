# Goal-Pose Field Benchmark

Each curated case has a goal image, semantic target anchor, bounded approach
template, robot start pose and hidden evaluation target pose/area. Anchors and
templates are explicitly curated, never described as automatic labels.

P0 reuses verified indoor scenes and a small curated set. It is retained as a
mechanism/motivation study only. Formal expansion uses deterministic,
scene-disjoint multi-stage indoor transport episodes. A single high-level
intent persists across corridor, doorway, room-entry and final-observation
phases; a local goal pose is recomputed only at phase boundaries.

Methods: M0 target center; M1 target plus nearest free cell; M2 target plus
clearance/reachability heuristic; M3 Task-Conditioned Goal-Pose Field; M4
evaluation-only privileged oracle. All share target hypothesis, lattice,
executor, footprint and termination.

At terminal-pose level report valid-goal rate, reachability, selected-pose
clearance, target visibility, standoff error, heading error and task-relation
satisfaction. At phase level report phase success, handoff success,
wrong-terminal-pose rate, collision and timeout. At task level report complete
transport success, SPL, normalized geodesic progress, final DTG, path length,
replan count and execution time. Report scene-disjoint seen/unseen results,
episode-level outcome matrices and bootstrap 95% confidence intervals.

Candidate ranking at deployment may use only depth/LiDAR/pose-derived local
geometry, VLM task intent, frozen visual evidence and execution history.
NavMesh/hidden target poses may label or evaluate simulation outcomes but may
not choose a deployed candidate.

P0 must show at least three indoor cases where center/nearest-free selection is
executable-poor or task-inappropriate while a field-selected pose is objectively
executable and task-satisfying. If this requires fabricated annotations or
privileged deployment input, stop.
