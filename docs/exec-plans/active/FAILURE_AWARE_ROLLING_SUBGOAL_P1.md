# P1 — Failure-Aware Rolling Subgoal Selection

1. Reuse existing 2,520 episodes/20,160 K=8 candidate proposals.
2. Execute every candidate for a fixed short horizon with the fixed Habitat
   executor and save reached/collision/stuck/progress/path-length labels.
3. Reuse frozen DINO features; do not issue VLM requests for P1 decision
   states. The goal image is the standard task condition.
4. Train the lightweight history-aware candidate scorer from visual,
   geometric, candidate-pose and failure-history inputs.
5. Run shared static and unseen rolling ImageNav comparisons.
6. Produce main/unseen tables, recovery statistics and three indoor cases;
then proceed to dynamic ImageNav recovery and stop. An optional VLM task
parser is evaluated only as a separate system-interface demonstration after
the core benchmark; it must not affect the P1 result tables.
