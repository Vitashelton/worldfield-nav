# B1 P1 — Deterministic Episode Manifest

## Scope

P1 was executed offline from the frozen B1 P0 candidate table. No Habitat-GS
simulation, scene download, or probe rerun was performed.

## Contract check

| Check | Result |
|---|---:|
| Selected scenes / starts | 10 / 10 |
| Matched branches | 4 per start (40 episodes) |
| Byte-equivalent second build | PASS |
| Continuous controls | 15 `(v, omega, dt)` tuples, `dt=0.2s`, fixed 3.0s |
| GT sampling clock | 10Hz schema; exact 0.5/1/2/3s frame slots |
| Kinematic relative-pose integration | PASS |
| Absolute start pose available | BLOCKED |
| O/H/V/A snapshots available | BLOCKED |

The manifest records explicit source references and `available: false` for the
two missing artifact classes. P0 persisted revelation scalars and controls,
but not the absolute start poses or field arrays required to feed a model.
The P0 scalar table used its legacy 5Hz frame map; this is retained as
provenance and is not relabeled as the P1 exact-horizon GT.

## Outputs

- `outputs/formal/B1/p1_manifest/manifest.json`
- `outputs/formal/B1/p1_manifest_repeat/manifest.json`
- `outputs/formal/B1/p1_manifest/metrics.json`
- `experiments/runs/B1/20260902T100418Z/run.json`

P1 therefore has a deterministic, reproducible manifest contract, but its full
acceptance criteria do not pass because the inherited P0 artifacts are
incomplete. P2 is not entered; no data or model training is started.
