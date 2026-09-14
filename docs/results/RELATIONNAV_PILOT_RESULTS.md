# RelationNav Pilot Results

This is an internal pilot diagnostic from the existing four-scene, 400-episode manifest. It is not the final benchmark and does not claim a new planner. Habitat-GS supplies deterministic RGB-D, pose, NavMesh and privileged relation predicates.

## Main observation

Arrival-only execution reports nominal completion even when the relation predicate is false. Relation-preserving recovery keeps the same entity-relation contract and selects a deeper admissible realization after a failed predicate.

## Assets

- `paper_assets/tables/relationnav_pilot_summary.csv`
- `paper_assets/figures/relationnav_pilot_results.png`
- `paper_assets/figures/relationnav_episode_completion_matrix.png`
- `paper_assets/videos/relationnav_habitat_pilot.mp4`

The shallow boundary realization used by this pilot is a diagnostic stress protocol; it must not be presented as the final paper benchmark without scene-scale automatic generation and an executor protocol review.
