# MetricAnchor Benchmark Specification

## Frozen split

| Role | Scenes |
| --- | --- |
| Train | scene01, scene02, scene03, scene09, interior_0405_840145 |
| Validation | scene04, scene05 |
| Unseen test | scene56, scene57, scene58 |

Only completed C1 trajectories and their deterministic RGB-D/pose records are
used. Metric positives require bounds, depth agreement ≤0.12m, occlusion
validity, and reconstructed world residual ≤0.08m. Sim-LiDAR-like distance is
recorded as supplementary validation only.

## Representation evaluation

Report positive/hard-negative cosine, margin, R@1, R@5, and retrieved
world-position error for small, medium, large, revisit, validation, and unseen
sets. Retrieval candidates are dense patches observing valid metric surfaces.

## G3 place and navigation evaluation

Reference records use the portable keyframe schema: RGB path, timestamp,
world pose, descriptor path, and keyframe ID. Querying returns ranked IDs,
similarity, world pose, and metric error. G3 contains at least 50 validation
and 50 unseen image-goal episodes over small/large, doorway/corner, repeated
structure, and revisit cases. M0 and M3 share the same reference index and
Habitat shortest-path executor. Report R@1/R@5/error/success@0.5m/@1m and
navigation SR/SPL/wrong-place arrival/path length.

## Required outputs

`metricanchor_main_results.csv`, `metricanchor_ablation.csv`,
`habitat_navigation_results.csv`, the specified seven figures, a 1–2 minute
Habitat demo, and `docs/results/METRICANCHOR_SIMULATION_RESULTS.md` are
mandatory.
