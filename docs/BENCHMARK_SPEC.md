# Benchmark Specification

## Scope

The formal benchmark evaluates counterfactual future world-field prediction in
Habitat-GS. It is multi-scene, split-fixed, reproducible, and horizon-resolved.

## Pilot B1 split contract

The B1 pilot targets five train scenes, two validation scenes, and three
unseen-test scenes. Resolve local assets first. If fewer than ten valid distinct
scenes are installed, acquire only the final missing pilot scenes from official
Habitat-GS assets; do not download a complete scene collection. Commit exact
identifiers, provenance, version/checksum, and local paths in the B1 config.

Before freezing starts, run deterministic probe trajectories to measure the
unknown-to-observed-area distribution by context and action family. Freeze a
quantitative revelation threshold, its rationale, and eligible-start list in
the config. Formal corner/doorway episodes must exceed that threshold; nearly
zero-revelation candidates are rejected.

## Episodes and actions

For each scene, sample deterministic starts across open areas, turns, and
doorway/occlusion transitions. Each start has matched action branches sharing
the same initial `Phi_t`, start pose, seed, fixed 3.0s duration, and sampling
clock. `straight`, `left`, `right`, and `mixed_turn` are action-family labels;
every branch stores a continuous sequence of `(v, omega, dt)` controls, its
kinematically integrated poses, translational distance, and mismatch against
the matched translational-motion budget.

## Horizons and metrics

Sample at a fixed rate of at least 5Hz and store exact GT at 0.5, 1.0, 2.0, and
3.0 seconds. Formal benchmark episodes missing 0.5s are invalid; another frame
may never be relabeled as 0.5s.

Partition every metric into persistent-known and newly-revealed regions.
Required reporting includes Known-IoU, height MAE on jointly visible known
cells, visibility IoU, Reveal-IoU, Reveal-F1, revelation latent similarity when
`Z` exists, and Persistent Drift. Report split, scene, action family, and
horizon—not only an aggregate average.

## Baselines and outputs

Required model families are M0-Transport (deterministic transport-only),
M1-Direct (direct neural field predictor), M2-RSSM (RSSM-style latent
state-space predictor), and M3-WorldFlow. Formal outputs include matched
GT/M0/M1/M2/M3 figures, long-horizon rollouts, unseen-scene comparison, metrics
tables, and a run record. Paper assets belong under `paper_assets/`.
