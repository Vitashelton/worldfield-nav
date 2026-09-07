# Trajectory Grounding P1 — Static Indoor Habitat-GS Benchmark

1. Confirm an existing scene-disjoint indoor split without downloading assets.
2. Generate deterministic K=8 goal-independent local planner trajectories and
   privileged labels; target 20k--50k trajectories.
3. Render and save a trajectory-to-RGB projection sanity figure.
4. Cache one frozen VLM response per numbered trajectory-overlay decision
   image; cache frozen DINOv3 current/goal/corridor scores.
5. Train only the lightweight trajectory ranker.
6. Run all specified baselines on >=100 validation and >=100 unseen episodes.
7. Generate main/unseen/ablation tables and the required three indoor
   qualitative figures. Record results and stop.

Acceptance requires no train/test scene overlap, no goal/NavMesh leakage in
candidate generation or online scores, shared executor/episodes, and all named
metrics/assets saved. No dynamic/recovery/outdoor/robot work is authorized.
