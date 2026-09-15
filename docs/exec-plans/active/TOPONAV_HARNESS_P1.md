# TopoNav Harness P1 — Static Semantic-Topology Navigation

## Objective

Run the fixed 80-task indoor Habitat-GS development benchmark with a resident
frozen Qwen VLM, cached tool calls and actual metric execution.  Produce the
first paper-grade static table, tool traces and a true execution video.

## Protocol

Task types: S1 semantic goal (20), S2 relation task (30), and S3
route-constrained task (30).  The VLM may invoke only `NAVIGATE`, `OBSERVE`,
`BACKTRACK`, or `STOP`; it receives no metric coordinates, NavMesh or oracle
route.  Metric execution is fixed and logs typed feedback.

## Methods

- Direct-VLM: candidates but no topology;
- FullTopo-VLM: full graph context;
- FullTopo+Validator;
- History-Agent: full raw history;
- Ours: task-relevant subgraph, selective images, validator, typed feedback,
  and relation-preserving recovery.

## Outputs

`outputs/formal/TopoNav/P1/` must contain manifest, VLM cache, per-episode
results and summary.  Paper assets must include an architecture/context figure,
a static result table, an agent-trace figure and an actual Habitat execution
video.  State all static-only limitations; do not claim dynamic recovery until
official avatar executions are available.

## Acceptance

All 80 paired tasks execute under each method with cache provenance and no
oracle leakage.  Results, tool-validity/error metrics, and a readable real-log
video are saved.  Then stop; dynamic or real-robot work needs its own plan.
