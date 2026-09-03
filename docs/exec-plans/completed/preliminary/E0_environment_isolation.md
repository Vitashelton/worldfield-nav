# E0 — Simulation/Model Environment Isolation

## Authority

C2 is paused at `docs/exec-plans/paused/C2_multimodal_latent_worldfield.md`.
This is the sole active plan. It authorizes only environment recovery,
freezing, and a two-image frozen-backbone smoke. It does not authorize C2
feature extraction, WorldFlow implementation or training, navigation, or C1
data changes.

## Goal

Freeze the working Habitat-GS/C1 simulation environment and establish an
isolated model environment at `/root/autodl-tmp/worldfield_nav/envs/worldfield_model`. The model
environment will test public timm DINOv3-S/16 weights without Transformers or
the gated Meta repository.

## Simulation checks

Record the verified Python, PyTorch/CUDA, NumPy, Pillow, and Habitat-Sim
versions in `envs/worldfield_sim.lock.txt`. Run only the default sanity check,
a 20-frame Habitat-GS RGB-D smoke on a frozen C1 scene, and one C1
Dataset/DataLoader batch. Do not install, upgrade, or otherwise modify
`/root/miniconda3`.

## Isolated model environment

Create `/root/autodl-tmp/worldfield_nav/envs/worldfield_model` without altering the simulation
environment. Freeze its explicit dependency versions in
`envs/worldfield_model.lock.txt`. Install `timm` with an exact version and
without dependency resolution after confirming that torch, numpy, and Pillow
will not be changed.

Use only `hf_hub:timm/vit_small_patch16_dinov3.lvd1689m` with
`timm.create_model(..., pretrained=True)`. Read exactly two C1 RGB images,
apply the model's resolved pretrained transform, execute `model.eval()` and
`torch.no_grad()` on GPU, and print the actual `forward_features()` structure.
Extract dense patch tokens only after inspecting actual shapes, prefix tokens,
and patch-grid dimensions. If the public timm mirror cannot be downloaded,
test public DINOv2-S/14 solely as a temporary pipeline-development backbone;
do not attempt the gated Meta repository or install Transformers.

## Acceptance

- Simulation sanity, Habitat 20-frame, and C1 one-batch checks pass.
- Simulation and model locks exist and the immutable policy is present.
- The model environment imports NumPy and PyTorch and reports exact versions.
- A frozen public DINOv3-S/16 (or documented DINOv2-S/14 fallback) completes a
  two-image GPU forward with actual dense-token shape, timing, and peak-memory
  evidence.
- No C1 trajectory or result artifact is modified.
