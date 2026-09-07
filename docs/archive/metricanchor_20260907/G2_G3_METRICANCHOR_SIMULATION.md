# G2+G3 — MetricAnchor Simulation Evidence Sprint

## Authority

This is the sole active plan. It authorizes only completed-C1 feature caching,
geometry correspondence mining, M1/M2/M3 residual-adapter training, G2
representation evaluation, and G3 Habitat reference-place retrieval/navigation
evaluation. It explicitly excludes real robot work, ROS2, deployment packages,
LLM/API, Nav2 real-robot execution, new backbones, and new model families.

## Fixed protocol

Use frozen DINOv3-S/16 and a one-time FP16 cache over 4,500 C1 frames. Train
scenes are scene01/02/03/09/interior_0405_840145; validation scenes are
scene04/05; unseen scenes are scene56/57/58. Positives are geometry-derived;
sim-LiDAR-like points validate geometry only. M0--M3 use the same candidate set
and G1 protocol. M2/M3 receive three seeds when the initial result is sane.

## Required artifacts

Generate all named tables/figures/video in the current benchmark spec and a
single `docs/results/METRICANCHOR_SIMULATION_RESULTS.md` suitable as a paper
experiment-section basis. G3 has at least 50 validation and 50 unseen episodes
and compares Frozen vs MetricAnchor-Full with an unchanged Habitat planner.

## Stop condition

After results, assets, registry update, and plan archival, stop. Do not create
the next plan.
