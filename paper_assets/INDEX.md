# Paper Asset Index — Active Paper: Failure-Aware Rolling Subgoal Selection

## Current-paper evidence status

The active claim is that an **Execution Bridge** trained from matched
planner-branch outcomes improves *final rolling ImageNav success* over
handcrafted geometric rules.  An asset is paper-ready only after it is tied to
that scene-disjoint rolling evaluation.

| Status | Asset | Correct use |
|---|---|---|
| Motivation only | `figures/execfield_p0_oracle_gap_overview.png` | P0 oracle-gap: semantic preference can differ from physical executability. Not a final benchmark result. |
| Motivation only | `figures/execfield_p0_why_semantic_ranking_fails.png` | Three illustrative P0 examples. Do not use as indoor main evidence. |
| Internal diagnostic | `tables/failure_aware_p1_branch_value.csv` | One-step branch selection against real outcomes. It is **not** final ImageNav SR/SPL. |
| Internal sanity | `figures/trajectory_grounding_p1_projection_sanity.png` | Checks candidate camera projection only; not a paper figure. |
| Internal sanity | `figures/trajectory_grounding_p1_scene_availability.png` | Confirms available indoor scenes; not a paper figure. |

## Required P1 publication package — not yet generated

| Output | Required evidence |
|---|---|
| `tables/failure_aware_p1_main.csv` | Val rolling ImageNav: Planner Geometry vs Learned Execution Bridge vs Oracle; SR, SPL, Final DTG, collision/invalid rate, path length. |
| `tables/failure_aware_p1_unseen.csv` | Same protocol on scene-disjoint unseen scenes. |
| `figures/failure_aware_p1_main_results.png` | Main SR/SPL comparison with paired episode outcomes. |
| `figures/failure_aware_p1_qualitative_cases.png` | Corridor, doorway and repeated-hallway: candidates, chosen branch, failure history and rollout. |
| `figures/failure_aware_p1_method.png` | Goal image / optional VLM intent / planner branches / Execution Bridge / fixed executor / failure memory. |
| `videos/failure_aware_p1_rolling_recovery.mp4` | Short fixed-start comparison of rules versus bridge, including an abort/reselection case. |

## Archived assets — preserve, do not cite for the active paper

- `b1_p0_*`, `causal_vs_oracle.png`, `persistent_field_rollout.mp4`, and
  `occlusion_revisit_example.mp4`: WorldFlow preliminary work.
- `g1_*`, `metricanchor_*`, and `metricanchor_image_goal_localization.csv`:
  GeoAnchor preliminary work.
- `execfield_p0_*`: retained motivation only, not a formal indoor result.
