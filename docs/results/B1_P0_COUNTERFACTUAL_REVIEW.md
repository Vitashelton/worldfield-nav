# B1 P0 Counterfactual Informativeness Review

This review reuses the existing 253 unique starts and 1012 branch rows; no
Habitat-GS probe was rerun.

## Coordinate and control audits

The P0 code converts both initial and future depth observations to global X/Z
world-coordinate cells before subtracting `future - initial`. It does not
compare tensor indices from different crop origins. The coordinate audit thus
passes, with no before/after metric difference. P0 did not persist the actual
per-branch cell masks or start poses, so spatial pairwise IoU/Jaccard and mask
examples cannot be reconstructed from the saved artifacts.

The saved continuous 15-step `(v, omega, dt)` sequences are one-to-one with
the discrete proxy: each tick executes the corresponding turn (if any) and a
forward step. `v=0.35 m/s`, `dt=0.2 s`, `omega=0` for straight and
`±0.261799 rad/s` for turns; mixed-turn switches sign after seven ticks.

## Magnitude review

| context | R_max P25 / median / P75 (m²) | delta_R P25 / median / P75 (m²) | C_mask |
|---|---:|---:|---:|
| open_space | 46.536 / 52.814 / 61.343 | 9.084 / 12.701 / 17.200 | unavailable |
| turn_corner | 17.047 / 33.954 / 40.076 | 10.028 / 15.900 / 22.034 | unavailable |
| doorway_occlusion | 51.366 / 62.732 / 70.653 | 11.015 / 13.885 / 17.578 | unavailable |

The provisional magnitude-only screen is `R_max >= positive-area R_max P25 =
36.621 m²` and `delta_R >= delta_R P75 = 19.037 m²`. It retains 51 starts,
but it is not a final spatial-contrast eligibility rule because `C_mask` is
not available.

Final spatial rule (not evaluable from P0 artifacts):
`R_max >= 36.621 m² AND delta_R >= 19.037 m² AND C_mask >= a distribution-derived threshold`.

Old global-threshold eligibility retained 188 starts. Final spatially revised
count is intentionally left undefined pending mask persistence; no starts are
silently relabeled.

The baseline name in `configs/benchmark/b1_pilot.yaml` is `M0-Transport`.

This review does not authorize P1.
