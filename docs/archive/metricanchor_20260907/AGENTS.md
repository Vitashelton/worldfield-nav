# MetricAnchor Research Harness

## Mission and authority

This repository studies **MetricAnchor: Geometry-Anchored Foundation Visual
Features for Viewpoint-Robust Indoor Robot Navigation**.

The central question is whether frozen foundation visual descriptors preserve
physical identity under robot motion, occlusion, revisit, and repeated indoor
structure, and whether metric geometry can supervise a small adapter that
improves place localization and image-goal navigation.

Authority order:

1. `docs/RESEARCH_CONTRACT.md`
2. `docs/MODEL_SPEC.md`
3. `docs/BENCHMARK_SPEC.md`
4. Exactly one plan in `docs/exec-plans/active/`

Never change the paper identity or create an adjacent research phase without
an explicit active plan.

## Retained infrastructure

WorldFlow S0--S2, B1, C1, E0, C2, G1, their outputs, Habitat-GS assets, and
the verified Python environments are retained. Never delete/regenerate C1 or
reinterpret WorldFlow as MetricAnchor evidence. C1 is reusable simulation
infrastructure only.

## MetricAnchor boundary

The frozen backbone is `timm/vit_small_patch16_dinov3.lvd1689m`. Geometry,
not appearance, creates positive pairs: depth back-projection, intrinsics,
absolute pose, reprojection, depth/occlusion validation, and world residual.
Sim-LiDAR-like points are an additional geometry validation signal; they are
not a simulation claim for a physical Livox sensor.

The only authorized learned component is the lightweight residual adapter
specified in `docs/MODEL_SPEC.md`. Do not add a transformer, attention, LoRA,
second backbone, backbone fine-tuning, ConvGRU, diffusion, ROS2, Nav2 real
robot control, or an LLM/API.

## Runtime and assets

- Primary root: `/root/autodl-tmp/worldfield_nav`.
- Do not reinstall CUDA/PyTorch or rebuild Habitat-GS without a real binary
  failure. Use the existing `worldfield_model` environment.
- Formal runs use `python scripts/run_experiment.py`, append-only outputs, and
  `experiments/registry.yaml` run records.
- Feature extraction is performed once and cached in FP16; no adapter script
  may re-forward DINOv3.

## Done and reporting

A task is complete only when code, numeric evidence, paper assets, a result
note, and registry entry are present.
Archive the active plan and stop. Final reporting is concise: result, numbers,
paths, limits, and GO/NO-GO.
