# Research Contract

## Working title

**GeoAnchor: LiDAR-Anchored Adaptation of Foundation Visual Features for
Embodied Navigation**

## Scientific question

Are frozen foundation visual features physically consistent enough for a moving
robot under viewpoint change, occlusion, and revisit? If not, can metric
geometry provide reliable physical-correspondence supervision for lightweight
adaptation?

## Modalities

Simulation uses Habitat-GS RGB, depth, sim-LiDAR-like geometry, and absolute
pose. The later robot platform is Ranger Mini 2.0 with D435i RGB-D, Mid-360S
geometry, LIO pose, and Nav2. LLM/API task planning is a future system
demonstration only, not a G1 component or core contribution.

## Current evidence rule

G1 is problem validation, not an adapter claim. A cross-view positive pair
means two image patches observe the same physical world surface, proven by
depth, intrinsics, absolute pose, reprojection, visibility/occlusion tests,
and bounded world-coordinate residual. Image appearance never creates a
positive pair.

## Go/no-go rule

Adapter development is justified only when frozen DINOv3 has reproducible,
multi-scene degradation on large-viewpoint or revisit pairs while geometric
correspondence remains valid. Evidence includes at least one of: a >=10 point
R@1 drop versus small-view pairs, >=0.10 positive-cosine drop, or substantial
positive-negative margin collapse. Otherwise stop and report NO-GO.

## Not the paper

- Persistent-world-field modeling, traditional mapping, or a DINOv3 paper.
- End-to-end visual control, a new planner, Nav2 redesign, or LLM task
  planning.
- An adapter claimed before the frozen-feature audit demonstrates need.
