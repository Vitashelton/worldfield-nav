# ExecField Model Specification

## Inputs

`M_t [H,W,C]` is local BEV state from depth-projected occupancy, free/unknown
state, obstacle clearance, candidate mask and failed-region history.
`S_sem [H,W]` rasterizes frozen-VLM semantic relevance over candidate subgoals.

## Outputs

A lightweight BEV CNN/U-Net produces three spatial maps:

- `F_reach(x,y)`: predicted reachability;
- `F_prog(x,y)`: predicted geodesic goal progress;
- `F_fail(x,y)`: predicted collision, blockage, dead-end or no-progress risk.

`F_exec = F_reach + lambda_p F_prog - lambda_f F_fail` is queried only at
candidate positions. The selected subgoal maximizes VLM relevance plus the
ExecField score. After execution failure, the BEV/history is refreshed and the
remaining candidates are re-scored.

## Supervision

For each candidate, Habitat-GS/NavMesh supplies privileged labels:
reachable, collision-free, clearance, geodesic progress, target visibility,
eventual success and recovery result. Labels train the field only; they are
never inference inputs.

## Comparisons

- M0: semantic-free geometric candidate;
- M1: VLM semantic prior only;
- M2: VLM plus handcrafted feasibility;
- M3: VLM plus learned ExecField without recovery;
- Ours: VLM plus ExecField and recovery.
