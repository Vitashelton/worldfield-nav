# ExecField P0 — Oracle Executability Kill Test

## Purpose

Before training an executability field, measure whether privileged candidate
executability can improve frozen-VLM image-goal navigation. This is a kill test,
not a training benchmark.

## Authorized work

Use 3–5 existing Habitat-GS scenes and create 15–30 deterministic ImageNav
episodes. At each decision state, sample a fixed set of local candidate points.
Build a cached/curated frozen-VLM semantic ranking when a live backend is not
available. For each candidate, obtain teacher labels from Habitat/NavMesh:
reachability, collision-free path, clearance, geodesic progress, target
visibility, and eventual outcome.

Compare M0 semantic-free geometry, M1 VLM-only ranking, and M1+Oracle using
identical candidates, starts, goals, budget, and executor. The oracle is an
upper bound only and never a deployed method.

## Metrics and assets

Save task manifest, candidate/label table and metrics for executable-goal rate,
ImageNav SR, SPL/path length, wrong-goal arrival and failure reason. Generate
one paper-readable VLM-only versus oracle comparison figure under
`paper_assets/figures/`.

## Decision

GO only when oracle selection produces clear, repeatable SR/SPL improvement and
fewer invalid/wrong-goal failures. Otherwise archive ExecField without training
the field. Stop at the decision; no large dataset, field training, recovery,
dynamic avatars, real robot, or API work in P0.
