# G1 — Cross-View Foundation Feature Audit

## Authority

This is the sole active plan. It validates whether frozen DINOv3 features are
physically consistent under robot motion before any GeoAnchor adapter is
considered. It does not authorize training, full-C1 extraction, navigation,
Nav2, real-robot work, LLM planning, or any modification to C1.

## Execution gate — GPU required

G1 is **blocked until an accessible CUDA GPU is available**. Before any model
command, verify `torch.cuda.is_available()` and record the device and free
memory. While this gate is closed, only contracts, configs, interfaces, and
static code review may change.

Do not download DINOv3 weights, run a CPU fallback, process C1 frames, execute
the two-image smoke, or generate audit metrics while the GPU is unavailable.
The absence of GPU is an expected wait state, not evidence for GO or NO-GO.

## Prerequisite

Complete exactly one two-image GPU smoke of
`timm/vit_small_patch16_dinov3.lvd1689m`. Record the actual
`forward_features()` structure, prefix/register-token count, patch-token count,
dense shape, time, and peak GPU memory. Do not use a checkpoint other than the
public timm DINOv3-S/16 target and do not download DINOv3 beyond this smoke.

## Audit set and miner

Use only representative C1 trajectories spanning train, validation, and unseen
scenes. Include small/medium/large viewpoint change, translation, rotation,
doorway/corner views, and the known revisit sequence
`interior_0405_840145_traj00` frame 8 -> 15 -> 23.

Build positives with RGB patch + depth + intrinsics + absolute pose:
back-project to world, reproject, require image bounds, depth agreement,
visibility/occlusion validity, and recorded world residual. Use sim-LiDAR-like
geometry as an additional check when available. Mine hard negatives from nearby
but physically different world surfaces. Never use appearance to create a
positive pair.

## Required outputs

- frozen DINOv3 metrics by small/medium/large/revisit and train/val/unseen
- geometry correspondence validation error
- `paper_assets/figures/g1_crossview_correspondence.png`
- `paper_assets/figures/g1_dinov3_viewpoint_degradation.png`
- `paper_assets/figures/g1_revisit_failure.png`
- `paper_assets/tables/g1_dinov3_baseline.csv`

## Decision

GO to a future adapter plan only if a reproducible multi-scene large-view or
revisit failure occurs while geometry labels remain valid: >=10 percentage-point
R@1 degradation from small pairs, >=0.10 positive-cosine drop, or a substantial
margin collapse. Otherwise declare NO-GO and stop.

## Stop condition

After the G1 result note, registry update, plan archival, and explicit GO/NO-GO
decision, stop. Do not create the next plan.
