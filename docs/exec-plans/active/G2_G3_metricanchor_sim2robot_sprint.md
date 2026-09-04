# G2+G3 — MetricAnchor Simulation-to-Robot Sprint

## Authority and stop condition

This is the sole active plan. It authorizes the frozen-DINOv3 cache, metric
correspondence manifests, M1/M2/M3 adapter training, Habitat place-retrieval
and unchanged-executor navigation evaluation, paper assets, and a
Habitat-independent deployment package. It does not authorize ROS2, real robot
control, a new backbone, or a new research direction.

## Required execution

1. Cache every C1 RGB frame once as FP16 16×16×384 frozen DINOv3 tokens.
2. Build train/val/unseen geometry correspondence manifests and ≥3-view tracks.
3. Train M1/M2/M3 with the fixed adapter; retain M3 as `adapter_best.pt`.
4. Evaluate M0--M3 using the one physical retrieval protocol and produce G2
   figures/tables.
5. Select validation-best method, run G3 ≥50 validation and ≥50 unseen
   image-goal place-retrieval/navigation episodes using the same executor.
6. Build and smoke-test generic retrieval tools and `deployment/metricanchor`.
7. Write scientific result notes, a meeting package, real-robot handoff, and
   ROS2 interface contract; archive this plan and stop.

## Acceptance

The sprint passes only if scientific evidence, group-meeting assets, and a
non-Habitat RGB descriptor/index/query deployment path all exist. If M3 does
not improve over M0, report NO-GO rather than changing the model family.
