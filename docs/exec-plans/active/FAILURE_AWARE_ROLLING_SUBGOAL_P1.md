# P1 — Failure-Aware Rolling Subgoal Selection

1. Reuse existing 2,520 episodes/20,160 K=8 candidate proposals.
2. Execute every candidate for a fixed short horizon with the fixed Habitat
   executor and save reached/collision/stuck/progress/path-length labels.
3. Reuse frozen DINO features; do not issue VLM requests for P1 decision
   states. The goal image is the standard task condition.
4. Train the lightweight history-aware candidate scorer from visual,
   geometric, candidate-pose and failure-history inputs.
5. Implement the shared rolling evaluator.  It must repeatedly generate the
   same K goal-independent branches, score, execute one branch, update failure
   history after collision/stuck/no-progress, and terminate only on evaluator-
   side target-distance success, timeout or budget exhaustion.
6. Run shared static and unseen ImageNav comparisons: Planner Geometry,
   Learned Execution Bridge, and Outcome Oracle upper bound.  Report both
   local branch diagnostics and final navigation SR / SPL / Final DTG.
7. Produce the following publication assets only after final navigation
   evaluation: a main table, an unseen table, a paired episode-success plot,
   three indoor failure-to-recovery cases, and one short rolling-navigation
   video.  Intermediate/sanity assets must be marked as such in
   `paper_assets/INDEX.md`.
8. Then proceed to dynamic ImageNav recovery and stop. An optional VLM task
   parser is evaluated only as a separate system-interface demonstration after
   the core benchmark; it must not affect the P1 result tables.
