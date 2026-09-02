# Preliminary Results: S0--S2

These are accepted infrastructure and mechanism checks. They are preserved for
provenance and are not the formal paper benchmark.

## S0: Persistent Local World Field

- Output: `outputs/exp001/`
- Field: `[320, 4, 128, 128]`, 10m x 10m, 0.078125m/cell.
- Accepted checks: consecutive IoU approximately 0.99498, static occupancy
  drift approximately 0.00295, camera/world projection verified.

## S1: Counterfactual Ego Rollout

- Output: `outputs/exp002_counterfactual_ego_flow/`
- Accepted property: identical initial field and robot pose yield different
  future world fields under different action sequences.

## S2: Transport--Revelation Decomposition

- Output: `outputs/exp003_transport_revelation/`
- Deterministic transport preserved nearly all known static occupancy on the
  current short trajectories.
- In C-right at 3.0s, the two newly observed cells had 758.23x the mean
  residual energy of outside cells (point-biserial r=0.803).
- Limitation: revelation was sparse; A and B had no newly observed cells by
  3.0s and C's newly observed region was only 0.012207 m2. This validates a
  mechanism but does not justify a complete learned revelation model alone.
