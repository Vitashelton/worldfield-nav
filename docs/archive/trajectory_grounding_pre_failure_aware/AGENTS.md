# AGENTS.md — Trajectory Grounding Research Harness

The active paper is **Trajectory Grounding for VLM-Guided Image-Goal
Navigation**. The project assembles a testable hierarchical navigation system:
a conventional local planner proposes physically executable trajectories; a
frozen VLM supplies low-frequency semantic preference; frozen DINOv3 supplies
trajectory-to-goal visual matching; and a lightweight learned ranker selects a
trajectory for a fixed executor.

The source of truth is `docs/RESEARCH_CONTRACT.md`, `docs/MODEL_SPEC.md`,
`docs/BENCHMARK_SPEC.md`, followed by exactly one plan in
`docs/exec-plans/active/`. Do not change the task identity, retrain DINOv3/VLM,
alter the executor, add a new planner, or start real-robot work without a new
active plan.

P0 ExecField and all WorldFlow/GeoAnchor assets are archived preliminary
evidence. Do not rerun P0 or include its outdoor-style scenes in the formal
indoor benchmark. Use only existing verified Habitat-GS assets; do not download
new scenes or rebuild the environment. Store raw P1 artifacts under
`outputs/formal/TrajectoryGrounding/P1/` and report-ready assets under
`paper_assets/`.

Candidate trajectories must not use hidden final-goal pose/direction or NavMesh
to select proposals. Privileged Habitat/NavMesh outcomes may form offline
training/evaluation labels and the fixed executor is shared by every method.
VLM calls are cache-only; it receives goal/current imagery and a numbered
trajectory overlay, never geometry oracle values or coordinates.
