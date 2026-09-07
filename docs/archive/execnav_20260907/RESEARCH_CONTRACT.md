# Research Contract — ExecNav

## Working title

**ExecNav: Learning Executable Semantic Subgoals for Vision-Language Image Navigation**

## Scientific question

Can a robot calibrate semantic navigation intent proposed by a frozen VLM into physically executable subgoals, using privileged simulator outcomes for training and a geometry-aware critic for failure detection and recovery?

## Task and claim boundary

The primary task is image-goal navigation. A user-provided goal image and optional instruction are parsed by a VLM into structured semantic/spatial intent. The system selects an executable 2-D subgoal, then delegates motion to the existing Habitat/Nav2 executor. The VLM does not output low-level actions or world coordinates. The learned component is a lightweight robot-centric executability critic trained from privileged Habitat outcomes; it scores reachability, clearance, visibility and likely success, and triggers recovery after failure.

This is not a new SLAM system, planner, VLM, low-level controller, or claim of being the first VLM navigation method. GeoAnchor/G1/G2 and WorldFlow S0–S2/C1 remain preliminary reusable assets.

## Required evidence

1. VLM semantic intent improves over a semantic-free baseline.
2. A learned executability critic improves over handcrafted feasibility rules.
3. Recovery improves over no-recovery under occlusion, dynamic obstacles, and blocked/invalid subgoals.

All comparisons share scenes, episodes, candidate budgets, and executor. Simulator teacher labels are never online oracle inputs.
