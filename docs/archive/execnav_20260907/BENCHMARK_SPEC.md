# ExecNav Benchmark Specification

Reuse the existing 10-scene split and C1 RGB-D/pose assets: train
scene01/02/03/09/interior_0405_840145, validation scene04/05, unseen
scene56/57/58. Do not expand the dataset before the pilot passes.

Curated image-goal episodes cover ordinary goals, large viewpoint changes,
occlusion/revisit, repeated structures, and blocked/invalid semantic targets.
Habitat ImageNav/NavMesh supplies the common executor. Dynamic-avatar cases are
an explicitly labelled stress subset.

Report semantic target validity, executable/reachable/collision-free goal rate,
image-goal success, path length, SPL, wrong-goal arrival, critic calibration,
recovery success and recovery count, broken out by static/occlusion/repeated/
dynamic and unseen subsets.

The goal image's hidden pose and privileged simulator outcomes are
evaluation/training supervision only. VLM inputs, candidate budgets, starts,
NavMesh and executor are shared across methods. Final tables/figures/videos go
to `paper_assets/`; raw data goes to `outputs/formal/ExecNav/`.
