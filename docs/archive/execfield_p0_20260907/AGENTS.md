# AGENTS.md — ExecField Research Harness

The active paper is **ExecField: Privileged-Simulation-Trained Executability
Field for VLM-Guided Image-Goal Navigation**. Read the research contract, model
specification, benchmark specification and the one active plan before work.

VLM supplies semantic priors; it is not the learned contribution and may be
cached. Habitat-GS privileged outcomes supervise ExecField during training only.
The conventional navigation stack is a fixed executor. Never feed simulator
oracle labels or goal coordinates to online inference.

Respect the P0 oracle kill gate before dataset scale-up, field training,
recovery, dynamic-avatar benchmark, real robot, or API integration. Preserve
WorldFlow and GeoAnchor branches/assets as archived preliminary work. Put raw
outputs in `outputs/formal/ExecField/` and only report-ready assets in
`paper_assets/`.
