# Research Contract — MetricAnchor

## Working title

**MetricAnchor: Geometry-Anchored Foundation Visual Features for
Viewpoint-Robust Indoor Robot Navigation**

## Scientific question

The physical world remains stable while robot observations change under
viewpoint change, occlusion, revisit, and repeated indoor structure. Can metric
geometry provide physical-correspondence supervision for a lightweight adapter
over frozen foundation visual features, improving metric place retrieval and
image-goal navigation?

## Evidence already accepted

G1 established frozen DINOv3 physical-identity degradation using
geometry-validated correspondence: small R@1 0.461, medium 0.194, large 0.056,
revisit 0.034, and unseen 0.344. G1 is motivation, not the MetricAnchor result.

## G2 and G3 claim boundary

G2 evaluates a frozen-DINOv3 residual adapter learned from metric positives,
physically distinct hard negatives, multi-view tracks, and feature
preservation. G3 evaluates whether the representation improvement transfers to
a generic reference-place retrieval interface and the same Habitat shortest
path executor.

This is not a new SLAM, planner, end-to-end controller, VLM, or Livox
simulation paper. Navigation is a downstream validation using an unchanged
executor.

## Sim-to-robot contract

Simulation uses RGB-D, absolute pose, and sim-LiDAR-like geometry. The future
robot uses Ranger Mini 2.0, D435i RGB-D, and Mid-360S/LIO metric pose. The
adapter consumes RGB only; geometry is supervision, database metric pose, and
evaluation—not an RGB inference input.
