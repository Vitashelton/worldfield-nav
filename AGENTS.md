# AGENTS.md — Failure-Aware Rolling Subgoal Selection

The active paper is **Failure-Aware Rolling Subgoal Selection for VLM-Guided
Indoor Image-Goal Navigation**. Its learned component is a lightweight
candidate scorer that uses frozen DINOv3 visual features, local geometry,
cached low-frequency VLM semantic priors, candidate pose and execution history
to predict short-horizon execution value.

The one active plan governs work. Preserve P0, the 2,520 indoor episodes and
20,160 candidate trajectories. Do not develop BEV fields, U-Nets,
trajectory-image grounding, manifold/flow methods or a new planner. The K=8
generator and fixed executor are shared by all methods. Candidate generation
may not access final-goal direction or NavMesh; Habitat privileged outcomes
are labels/evaluation only.

VLM API calls are offline/cache-only and deliberately budgeted. Dynamic
ImageNav recovery follows static and unseen results; Ranger logging records
only recoverable abort/costmap cases and does not authorize robot control.
