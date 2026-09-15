# Benchmark Specification — TopoNav Harness

## Static semantic-topology benchmark

Use a fixed 80-task development manifest over installed multi-route indoor
Habitat-GS scenes.  Task counts are S1 semantic goal=20, S2 relational=30,
S3 route-constrained=30.  Each method shares task, start, topology, fixed
executor, seed and termination protocol.

Compare Direct-VLM, FullTopo-VLM, FullTopo+Validator, History-Agent, and the
full harness.  Report SR, SPL, DTG, path length, tool validity, hallucinated
entity rate, relation completion, repeated failures, recovery success, VLM
calls, images, tokens and latency.

## Dynamic benchmark

Dynamic recovery is a separate benchmark and may be reported only from actual
official avatar trajectories with logged interaction outcomes.  Synthetic
`edge.blocked` flags are interface smoke tests, not evidence.
