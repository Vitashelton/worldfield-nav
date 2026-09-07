# ExecNav — Image-Goal Executability Sprint

Build the Habitat-GS feasibility evidence for VLM semantic subgoals and a
privileged-outcome-trained executability critic. Reuse C1 scenes and existing
GeoAnchor features; do not expand data first.

1. Curate 15–30 ImageNav tasks over 3–5 scenes, including blocked semantic
   targets, occlusion/revisit, repeated structures, and dynamic-avatar stress.
2. Define a common candidate-subgoal schema and teacher labels: reachable,
   collision-free, clear, visible, success, and recovery outcome.
3. Implement M0 semantic-free, M1 VLM-only, M2 handcrafted feasibility, and
   Ours critic + recovery through the same Habitat executor.
4. Train only the lightweight critic; keep VLM frozen/cached.
5. Report target validity, executable/reachable/collision-free rates, success,
   SPL/path length, wrong-goal arrival, critic calibration, recovery success.
6. Generate paper tables, figures and a concise demo under `paper_assets/` and
   a result note. Stop after this sprint; do not start real-robot work.
