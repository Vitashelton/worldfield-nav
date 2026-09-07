# P1 — Failure-Aware Rolling Subgoal Selection

1. Reuse existing 2,520 episodes/20,160 K=8 candidate proposals.
2. Execute every candidate for a fixed short horizon with the fixed Habitat
   executor and save reached/collision/stuck/progress/path-length labels.
3. Cache frozen DINO features; issue a budgeted offline VLM request package.
4. Train the lightweight history-aware candidate scorer.
5. Run shared static and unseen rolling ImageNav comparisons.
6. Produce main/unseen tables, recovery statistics and three indoor cases;
   then proceed to dynamic ImageNav recovery and stop.
