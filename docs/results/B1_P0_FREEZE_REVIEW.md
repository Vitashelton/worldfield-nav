# B1 P0 Freeze Review

The review is derived from `outputs/formal/B1/p0/metrics.json` and the
four-branch candidate table. Counts below are unique eligible starts, not
branch rows (each start has four action-family rows).

| scene | open | corner | doorway | total |
|---|---:|---:|---:|---:|
| scene01 | 10 | 0 | 10 | 20 |
| scene02 | 6 | 3 | 8 | 17 |
| scene03 | 10 | 1 | 10 | 21 |
| scene09 | 10 | 0 | 10 | 20 |
| interior_0405_840145 | 10 | 0 | 0 | 10 |
| scene04 | 10 | 0 | 10 | 20 |
| scene05 | 10 | 2 | 9 | 21 |
| scene56 | 10 | 0 | 10 | 20 |
| scene57 | 10 | 0 | 9 | 19 |
| scene58 | 10 | 0 | 10 | 20 |
| **total** | **96** | **6** | **86** | **188** |

## Interpretation

- Raw corner availability is scene-dependent: the probe selected 0–10 corner
  starts per scene (scene58 had none), so part of the reduction is geometric
  context scarcity rather than thresholding.
- The single global threshold is also unfavorable to corners: corner branch
  revelation has median 19.464 m² and P25 8.521 m², below the frozen 27.138 m²
  threshold. Of 42 raw corner starts, only 6 remain eligible.
- The current freeze has no per-context threshold and no post-threshold
  context-quota mechanism. It uses a global threshold with `open: valid` and
  `corner/doorway: every matched family >= threshold`; the probe only targets
  up to 10 starts per context when available.

This is a P0 freeze review only. It does not authorize P1 or a new sampling
policy.
