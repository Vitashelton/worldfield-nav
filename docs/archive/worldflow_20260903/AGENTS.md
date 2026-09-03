# WorldFlow Research Harness

## Mission and authority

WorldFlow studies persistent multimodal world fields for indoor mobile-robot
perception and navigation. Its central object is a world-aligned internal state
updated from RGB-D, LiDAR geometry, pose, and (when authorized) frozen visual
features. Action-conditioned rollout is optional evidence, not the project
identity. It is not autonomous driving, manipulation, human forecasting,
generic occupancy prediction, video generation, or a Habitat-GS benchmark
paper.

Research authority is deliberately separated from implementation:

1. `docs/RESEARCH_CONTRACT.md` defines the scientific question and exclusions.
2. `docs/MODEL_SPEC.md` defines the authorized model target.
3. `docs/BENCHMARK_SPEC.md` defines data, splits, metrics, and baselines.
4. Exactly one plan in `docs/exec-plans/active/` authorizes current work.

If these conflict, stop and report the conflict. Do not silently redesign the
research or begin an adjacent phase.

## Required task workflow

Before substantive work, read this file, the research contract, and the single
active plan. Inspect relevant code, then take the smallest step that advances
that plan. Run only `python scripts/sanity_check.py` as the default preflight.

When an active plan meets its acceptance criteria: save results, register the
run, write its result note, move the plan to `completed/`, and stop. Never
invent the next research phase.

## Runtime and safety constraints

- Primary root: `/root/autodl-tmp/worldfield_nav`.
- Habitat-GS, PyTorch, CUDA, and the current GPU setup are already verified.
- Do not reinstall PyTorch/CUDA, recreate environments, broadly audit the
  system, rebuild Habitat-GS without a concrete CUDA binary error, or rerun
  S0--S2 unless explicitly requested.
- Use error-driven debugging. Keep the 128x128 field and mixed precision by
  default; a larger model is not evidence of better research.
- Habitat-GS is the controllable data generator. Do not add Habitat-Lab,
  dynamic avatars, GAMMA, unrelated datasets, or a new scene without an active
  plan explicitly authorizing it.

## Immutable simulation environment

The verified simulation environment is immutable. Do not install
foundation-model or training dependencies into it. In particular, do not run
`pip install transformers`, `pip install -U numpy`, `pip install -U pillow`, or
`pip install timm` in `/root/miniconda3` unless an active plan explicitly
authorizes an environment migration. Foundation-model extraction, training, and
evaluation belong in the isolated model environment on the data disk. Do not
change C1 data or its completed records during environment work.

## Representation and model boundary

Physical field channels are `O` occupancy, `H` height, `V` visibility, and
`A` information age; a future `Z` visual latent is permitted only when the
model spec and active plan authorize learned-model work. LiDAR is a first-class
metric-geometry stream, RGB-D is complementary dense geometry, and pose aligns
all streams in the world frame. DINOv3 is frozen and never the contribution.

WorldFlow maintains `Phi_t` through deterministic pose transport and a learned
multimodal observation update. It must not collapse into a scalar risk score or
trajectory ranker. Future action rollout is an optional planning query only.

## Formal experiment contract

Formal experiments require fixed splits, sequential trajectories, quantitative
and qualitative comparisons, immutable run records, and reproducible config.
Required baseline families are M0 Frame-Only, M1 Geometric Memory, M2 ConvGRU
Memory, and M3 WorldFlow.

Use `python scripts/run_experiment.py` for every formal run. It records an
immutable run manifest and updates `experiments/registry.yaml`; do not manually
overwrite a formal result. Store paper-quality figures/videos/tables under
`paper_assets/`, not just debug plots under `outputs/`.

## Definition of done and reporting

A task is done only when code runs, required metrics and paper assets exist,
the exact config and run status are registered, and limitations are written.
Report concisely: execution, numbers, output paths, limitations, and whether
acceptance criteria passed. Do not narrate installation or propose new research
unless explicitly asked.
