# Research Contract — ExecField

## Working title

**ExecField: Privileged-Simulation-Trained Executability Field for VLM-Guided
Image-Goal Navigation**

## Scientific question

VLMs can express semantic navigation intent but cannot reliably determine
whether a proposed local target is currently reachable, safe, useful for goal
progress, or likely to fail. Can privileged outcomes from Habitat-GS train a
robot-centric spatial executability field that calibrates frozen-VLM semantic
priors and enables recovery when navigability changes?

## Method identity

VLM provides a semantic prior only. Habitat-GS provides training-only privileged
labels: reachability, geodesic progress, collision/blockage, clearance,
visibility and recovery outcomes. A lightweight ExecField maps local BEV
geometry, semantic prior and execution history to spatial reachability,
progress and failure fields. A conventional navigator executes the selected
candidate; it is not the contribution.

## Deployment boundary

At test time: goal image -> frozen/cached VLM semantic prior; current RGB-D ->
local geometry/candidates; ExecField -> selected subgoal; navigation stack ->
execution. No simulator oracle is available at test time. API calls are cached
offline for simulation training; real-time API use is a later system demo.

## Evidence required

1. VLM semantic prior beats a semantic-free candidate baseline.
2. Learned ExecField beats handcrafted feasibility scoring.
3. ExecField recovery beats no recovery under changing navigability.

Static ImageNav is the core benchmark. Dynamic Gaussian-avatar blockage is a
separately reported recovery benchmark. GeoAnchor is archived preliminary work,
not evidence for this paper.
