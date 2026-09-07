# ExecField Benchmark Specification

## Phased scale

P0 is a 15–30 episode oracle kill test: VLM-only candidates versus identical
candidates selected with privileged executability. No model training or large
generation occurs until P0 passes.

After P0, the formal scale target is 50–100 Habitat-GS scenes and 10^4–10^5
ImageNav episodes/candidate states. Each sample stores goal image, current
RGB-D, local BEV/candidates, cached VLM semantic scores, and privileged labels.

## Evaluation

Use held-out scenes and fixed starts/goals/candidate budgets. Report semantic
target validity, executable/reachable/collision-free rate, ImageNav SR, SPL,
path length, wrong-goal arrival, calibration/AUROC, and recovery success.
Report static, occlusion/revisit, repeated structure, dynamic-avatar and unseen
subsets separately.

## Fairness

VLM receives identical goal/current imagery for all VLM methods and cannot
provide world coordinates. VLM cache is produced offline and is not privileged
geometry. Teacher labels are simulation training/evaluation only. Every method
shares the same Habitat executor and candidate set.

## Artifacts

Raw data/checkpoints: `outputs/formal/ExecField/`.
Publication assets: `paper_assets/tables/`, `paper_assets/figures/`, and
`paper_assets/videos/`.
