# TopoNav P1 Static Habitat-GS Results

## Experimental Setup

The paired benchmark contains 80 tasks across three indoor Habitat-GS scenes (20 S1 semantic, 30 S2 relational, and 30 S3 route-constrained tasks). All methods share the same tasks, starts, frozen Qwen3-VL-8B backend, candidate topology, and metric executor. The VLM receives no metric coordinates, NavMesh path, or hidden goal pose.

## Main Results

TopoNav-Harness achieves **73.8% SR**, versus 10.0% Direct-VLM, 35.0% FullTopo-VLM, and 57.5% History-Agent. Paired analysis shows 53 Direct-VLM failures and 18 History-Agent failures converted to successes.

| Method | SR | 95% bootstrap CI | Final DTG | Tool validity |
|---|---:|---:|---:|---:|
| Direct-VLM | 10.0% | [3.8, 16.2] | 7.69 m | 94.8% |
| FullTopo-VLM | 35.0% | [25.0, 45.0] | 5.84 m | 96.5% |
| FullTopo+Validator | 35.0% | [25.0, 45.0] | 5.84 m | 96.5% |
| History-Agent | 57.5% | [46.2, 67.5] | 4.79 m | 90.6% |
| TopoNav-Harness | 73.8% | [63.7, 82.5] | 3.83 m | 88.1% |

## Task-Type Analysis

- S1: Direct 40.0% vs Harness 90.0%.
- S2: Direct 0.0% vs Harness 90.0%.
- S3: Direct 0.0% vs Harness 46.7%.

S3 is the principal limitation: structured grounding does not yet solve strict route constraints.

## Metric Integrity

`Relation_Completion=100%` is an internal state-machine marker, not independent navigation success, and is excluded from the headline claim. `SPL_proxy` is not standard Habitat SPL and is also excluded pending explicit shortest-path logging.

## Evidence Boundary

All qualitative RGB and paths replay recorded Habitat-GS logs. Dynamic-avatar recovery is not claimed because licensed SMPL-X assets were unavailable; no blockage or recovery event is fabricated.
