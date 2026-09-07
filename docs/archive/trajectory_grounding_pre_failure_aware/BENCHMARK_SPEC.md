# Trajectory Grounding P1 Benchmark Specification

## Data protocol

Use existing verified indoor Habitat-GS scenes only, split by scene into train,
validation and unseen. Each deterministic ImageNav episode has a goal image,
current RGB-D, current pose and K=8 local candidate trajectories. Candidate
generation cannot use final-goal direction/pose or NavMesh; Habitat/NavMesh is
used only after proposals for labels: reachability, collision-free, clearance,
geodesic progress, outcome and failure reason.

Build 20k--50k candidate trajectories if existing scenes support it. Require
at least 100 fixed episodes for validation and 100 for unseen evaluation;
expand economically without protocol changes. All VLM outputs are offline
cached. Train/val/unseen scenes never overlap.

## Comparisons

All methods use identical episodes, trajectories, executor and termination:

- Planner-only;
- VLM-only;
- DINO-only;
- Planner + VLM;
- Planner + DINO;
- fixed score fusion;
- learned trajectory ranker;
- privileged oracle upper bound.

Report candidate ranking accuracy/regret plus SR, SPL, Final DTG, path length,
collision-invalid rate and bootstrap episode confidence intervals. Report seen
validation and unseen results separately.

## Required qualitative evidence

Use indoor corridor branch, doorway/room entrance and repeated hallway cases.
Each case shows goal image; current RGB with eight numbered projected
trajectories; VLM, DINO and ranker scores; final choice; top-down rollout; and
success/failure outcome.
