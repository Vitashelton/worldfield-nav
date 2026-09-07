# ExecField P1 — DINOv3 Static Indoor Benchmark

Train and evaluate ExecField on static indoor Habitat-GS ImageNav only. Use
frozen DINOv3-S/16 dense features, depth-conditioned robot-centric metric
projection, local geometry, cached VLM semantic priors and simple failure
history. A lightweight spatial fusion encoder-decoder predicts reachability,
progress and failure fields at candidate locations.

Use scene-disjoint train/validation/unseen splits. Candidate generation is
fixed polar geometry and cannot use hidden goal direction or NavMesh. NavMesh
only supplies privileged labels. Generate >=10k valid candidate samples, then
scale only if throughput allows. Cache every VLM query.

Compare M0 geometry-only, M1 VLM-only, M2 VLM plus non-privileged handcrafted
rules, M3 VLM plus DINOv3 ExecField, and M4 oracle. Run specified modality and
history ablations. Report static seen/unseen SR, SPL, Final DTG, path length,
invalid/wrong goal, collision/timeout, decisions and latency with episode-level
statistics. Produce the named P1 raw artifacts, paper tables/figures and result
note. Do not start dynamic recovery or real-robot work.
