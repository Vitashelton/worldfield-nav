# AGENTS.md — ExecField Research Harness

The active paper is **ExecField: Privileged-Simulation-Trained Executability
Field for VLM-Guided Image-Goal Navigation**. Read the research contract, model
specification, benchmark specification and the single active plan before work.

P0 is completed and passed its oracle-gap gate. The only authorized work is
P1: static indoor DINOv3-based ExecField. Do not rerun or alter P0, start
dynamic recovery, real-robot work, API serving, planner changes, or a new
research phase.

VLM provides cached semantic priors only; DINOv3-S/16 is frozen; Habitat-GS
privileged outcomes supervise ExecField only during training. Candidate
generation may not use hidden goal direction or NavMesh. NavMesh may create
labels and evaluate fixed execution only. Never feed oracle labels or hidden
goal coordinates into online candidate selection.

Preserve WorldFlow and GeoAnchor assets as archived preliminary work. Place P1
raw data in `outputs/formal/ExecField/P1/` and publication-ready assets only in
`paper_assets/`. A task is complete only with reproducible configs, metrics,
paper assets, a results note and registry update.
