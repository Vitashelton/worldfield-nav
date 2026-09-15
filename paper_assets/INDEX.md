# Paper Asset Index — Active Paper: TopoNav Harness

## Active claim

The active paper studies semantic-topology-grounded, closed-loop VLM planning
for indoor mobile robots. A frozen VLM selects named topology transitions;
the harness compiles task-relevant context, validates tool calls, executes the
selected transition in Habitat-GS, and returns typed feedback.

## Paper and group-meeting assets

| Asset | Purpose |
|---|---|
| `figures/toponav_harness_architecture.html` | Editable/self-contained method architecture. |
| `figures/toponav_method_and_contribution.png` | Paper Fig. 1: reused components versus the five observable harness stages. |
| `figures/toponav_p1_main_results.png` | Main SR comparison and paired 80-episode outcome matrix. |
| `figures/toponav_p1_paired_qualitative.png` | Three same-task Direct-VLM failure / Harness success replays. |
| `tables/toponav_p1_main.csv` | Overall static benchmark with bootstrap confidence intervals. |
| `tables/toponav_p1_tasktype.csv` | S1 semantic, S2 relational, and S3 route-constrained breakdown. |
| `tables/toponav_p1_efficiency.csv` | Tool validity, VLM calls, and VLM latency. |
| `videos/toponav_p1_habitat_demo.mp4` | Actual Habitat-GS RGB/path/tool-trace replay for the group meeting. |

## Interpretation boundaries

- `Relation_Completion=100%` is an internal state-machine marker and is not a
  headline navigation metric.
- `SPL_proxy` is not standard Habitat SPL and is not included in paper assets.
- Dynamic-avatar recovery is not claimed because the licensed SMPL-X model was
  unavailable.
- Raw caches, manifests, images, simulator logs, and episode results remain in
  `outputs/formal/TopoNav/P1/`.

Historical preliminary results remain recoverable from Git history and their
raw experiment directories, but obsolete figures/tables/videos are deliberately
excluded from this active-paper directory.
