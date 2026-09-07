# ExecField P1 Benchmark Specification — Static Indoor ImageNav

## Scope

P1 is a scene-disjoint static indoor Image-Goal Navigation benchmark. P0 is
retained solely as oracle-gap motivation and is neither regenerated nor changed.
Dynamic avatars, recovery, real robot and planner innovation are out of scope.

## Data and split

Use only existing verified Habitat-GS indoor assets. Select and record
scene-disjoint train, validation and unseen splits after a minimal visual
availability check. Generate fixed deterministic ImageNav episodes and K fixed
polar candidates per decision state. Candidate generation uses robot-centric
geometry only: it must not read the hidden goal direction or NavMesh. NavMesh
is allowed only after sampling to create privileged labels and to execute the
fixed evaluator.

Target at least 10,000 valid candidate-level samples and at least 100 episodes
per validation and unseen split (expand only when generation throughput makes
it economical). Store RGB, aligned depth, goal image, intrinsics, robot pose,
candidate metric poses, cached VLM scores and teacher labels. All VLM calls are
offline/cache-only and never expose goal pose, NavMesh, or oracle labels.

## Teacher labels

For every candidate store reachability, collision-free state, clearance,
normalized geodesic progress, binary failure, eventual outcome and failure
reason. These values are training/evaluation labels only and cannot be used by
M0--M3 inference.

## Methods and fairness

Every method shares the exact episodes, candidate set, VLM cache, action
budget, executor and termination criterion:

- M0: geometry-only candidate heuristic;
- M1: cached VLM semantic prior only;
- M2: VLM plus non-privileged handcrafted clearance/free-space rules;
- M3: VLM plus learned frozen-DINOv3 ExecField;
- M4: privileged oracle executability upper bound only.

P1 ablations use the same learned fusion architecture: geometry only;
DINOv3+geometry; VLM+geometry; DINOv3+geometry+VLM; and full plus failure
history.

## Metrics

Report SR, SPL, Final DTG, path length, invalid-subgoal rate, wrong-goal
arrival, collision, timeout, mean decisions and ExecField latency for seen
validation and unseen scenes. Also report reachability accuracy/F1, progress
MAE, failure AUROC/F1, training time and parameter count. Use episode-level
bootstrap confidence intervals for formal navigation metrics.

## Assets

Raw artifacts live under `outputs/formal/ExecField/P1/`; report-ready tables
and figures live in `paper_assets/`. Formal qualitative cases must be indoor:
corridor branch, doorway/room entrance and repeated hallway/ambiguous area.
