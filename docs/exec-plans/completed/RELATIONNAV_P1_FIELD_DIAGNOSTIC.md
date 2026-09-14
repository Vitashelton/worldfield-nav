# RelationNav P1 — Spatial Transition Dataset and Field Benchmark

## Objective

Build the first complete RelationNav benchmark and learned relation-conditioned
spatial goal field using only the four audited strong InteriorGS scenes.

## Execution

1. Create an auditable portal, area and landmark annotation package. Store
   scene, entity type, visual anchor, portal plane or area polygon, sides and
   provenance.
2. Curate three to six entities per scene. Do not use automatic VLM output as
   ground truth.
3. Generate deterministic three-to-four-stage episodes and offline relation
   masks. Save starts, graph states, masks, candidates and evaluation labels.
4. Implement frozen-DINO plus RGB-D metric lifting and lightweight
   relation-conditioned goal field.
5. Train on three scenes, evaluate on held-out scene, and run B0-B3, Ours and
   Oracle with the same executor.
6. Generate main/heldout/ablation tables, transition/recovery figures and a
   readable multistage Habitat-GS video.

## Constraints

Do not download scenes, train a VLM/DINO backbone, change the low-level
planner, add a backbone or claim indoor dynamic-avatar evaluation. Do not start
Ranger work in this plan.

## Acceptance

The annotation contract, deterministic manifest, learned field, scene-disjoint
main table, transition/recovery metrics and required visual assets must exist.
Then write results, archive this plan and stop.
