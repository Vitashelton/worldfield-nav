# GeoAnchor Research Harness

## Mission and authority

This repository currently studies **GeoAnchor: LiDAR-Anchored Adaptation of
Foundation Visual Features for Embodied Navigation**.

The current question is whether frozen foundation visual features remain
physically consistent for a moving robot under viewpoint change, occlusion,
and revisit—and, only if they do not, whether metric geometry can supply
reliable physical-correspondence supervision for lightweight adaptation.

Authority order:

1. `docs/RESEARCH_CONTRACT.md`
2. `docs/MODEL_SPEC.md`
3. `docs/BENCHMARK_SPEC.md`
4. Exactly one plan in `docs/exec-plans/active/`

Do not independently alter this direction. If documents conflict, stop and
report the conflict.

## Archived infrastructure

WorldFlow S0--S2, B1, C1, E0, C2, their outputs, paper assets, Habitat-GS
assets, simulation locks, and model environment are retained preliminary
infrastructure. Do not delete, regenerate, or reinterpret them as GeoAnchor
paper evidence. In particular, do not continue archived E0/C2, download
DINOv3 beyond the authorized public smoke, or train a WorldFlow model.

## G1 boundary

G1 uses completed C1 RGB, depth, sim-LiDAR-like geometry, absolute pose, and
known occlusion/revisit case only. It authorizes the DINOv3 two-image smoke and
a small frozen-feature physical-correspondence audit. It does not authorize
adapter training, full-dataset extraction, navigation, Nav2, real robot work,
or LLM planning.

Positive pairs are created exclusively by metric correspondence: back-project
RGB patches with depth and pose, reproject into another frame, then validate
in-frame bounds, depth/occlusion agreement, and world-coordinate residual.
Never use image appearance to define a positive pair.

## Runtime policy

- Primary root: `/root/autodl-tmp/worldfield_nav`.
- Preserve the verified simulator and C1 output. Do not reinstall PyTorch/CUDA,
  recreate environments, broadly audit, or rebuild Habitat-GS without a real
  error.
- DINOv3 is a frozen audit baseline in G1, not a claimed contribution.
- Use `python scripts/run_experiment.py` for formal execution and register
  outputs under `paper_assets/` and `experiments/registry.yaml`.

## Done and reporting

Complete only the active plan's acceptance criteria, save its results, update
the registry, archive the plan, and stop. Reports state numerical evidence,
paths, limitations, and the explicit GO/NO-GO decision.
