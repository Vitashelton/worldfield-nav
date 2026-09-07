# Research Contract — Trajectory Grounding

## Working title

**Trajectory Grounding: Hierarchical VLM and Foundation-Feature Fusion for
Image-Goal Navigation**

## Scientific question

How can an indoor robot ground an image-goal's high-level semantic relevance
into a choice among locally executable planner trajectories without asking a
VLM to control the robot directly?

## Fixed system boundary

`Goal image + current RGB -> frozen DINOv3 and frozen VLM; local planner -> K
candidate trajectories; DINOv3 corridor matching + VLM semantic scores +
planner geometry -> lightweight trajectory ranker -> fixed executor.`

The contribution is a reproducible hierarchical integration and its measured
benefit, not a new VLM, foundation backbone, local planner, SLAM algorithm, or
end-to-end controller. Habitat-GS supplies controlled indoor data and
privileged outcome labels during training/evaluation only.

## Scope

P1 evaluates static indoor Habitat-GS scene-disjoint train/validation/unseen
splits. Dynamic interference, recovery, outdoor stress tests and Ranger Mini
are subsequent, explicitly separate work.
