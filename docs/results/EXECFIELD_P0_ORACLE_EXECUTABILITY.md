# ExecField P0 — Oracle Executability Kill Test

## Protocol

Four existing Habitat-GS scenes (scene01, scene02, scene03, and
interior_0405_840145) produced six deterministic ImageNav episodes each. Each
state used eight polar candidates generated without NavMesh or hidden-goal
direction. Habitat/NavMesh was queried only afterward for oracle labels. A
local DeepSeek Vision cache ranked the identical candidate views for M1 and the
oracle upper bound.

## Results

| Method | SR | SPL | Final DTG (m) | Invalid subgoal | Wrong goal |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 geometry-only | 0.292 | 0.283 | 10.030 | 0.458 | 0.250 |
| M1 VLM-only | 0.250 | 0.238 | 12.005 | 0.500 | 0.250 |
| M1 + Oracle executability | 0.875 | 0.821 | 10.261 | 0.125 | 0.000 |

The same VLM rankings leave a large oracle gap: selecting the first executable
semantic candidate improved SR by 62.5 percentage points and SPL by 58.3
points, while removing wrong-goal selections. P0 therefore passes its kill
gate: learning an online executability field is justified.

## Limits

This is a 24-episode oracle pilot with NavMesh-based termination, not an online
learned critic, dynamic-avatar recovery result, or final paper benchmark.

## Assets

- `paper_assets/tables/execfield_p0_oracle_gap.csv`
- `paper_assets/figures/execfield_p0_oracle_gap.png`
- `paper_assets/figures/execfield_p0_vlm_oracle_cases.png`
