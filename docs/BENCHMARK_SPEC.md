# Failure-Aware Rolling Subgoal Selection Benchmark

Reuse the fixed 2,520 indoor episodes and K=8 candidates. Add a short-horizon
rollout record for every candidate without changing proposal generation. Train,
validation and unseen remain scene-disjoint.

Static and unseen comparisons use the same episodes, candidates, executor,
action budget and termination: Geometry/Nav2, VLM-only, VLM+rules,
VLM+learned scorer and privileged Oracle. Report SR, SPL, Final DTG, path
length, collision/invalid rate, stuck rate, candidate ranking accuracy and
regret. Dynamic ImageNav follows only after those tables are saved.

VLM cache is capped to representative training states plus all formal
evaluation states; it is low-frequency and no API key is placed on the cloud.
