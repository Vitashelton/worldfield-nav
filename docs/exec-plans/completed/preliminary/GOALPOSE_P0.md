# Goal-Pose Field P0 — Problem Signal and Contract Validation

## Objective

Validate that semantic target location and executable task goal pose differ in
existing Habitat-GS indoor scenes.

## Scope

Reuse currently verified indoor scenes only. Curate target anchors from rendered
views and record scene, target anchor, goal image, approach template and
curation rationale. Do not download scenes, train DINO/VLM/a policy, or use
privileged geometry at deployment.

Required templates include at least three of `observe_doorway`,
`approach_doorway`, `inspect_target`, `room_entrance`.

## Method

Generate a fixed `(x,y,theta)` lattice around each target. M0–M3 share it. M3
uses observed geometry, clearance, explicit visibility and the fixed template.
NavMesh is evaluation-only except M4.

## Outputs

- `outputs/formal/GoalPose/P0/manifest.json`
- candidate-pose records and checks
- `paper_assets/figures/goalpose_p0_problem_cases.png`
- `paper_assets/figures/goalpose_p0_field_examples.png`
- `paper_assets/tables/goalpose_p0_summary.csv`
- `docs/results/GOALPOSE_P0_RESULTS.md`

## Acceptance

1. At least 12 curated indoor cases across four existing scenes.
2. Every case has target center, goal image, lattice and finite geometry/
   visibility checks.
3. Three qualitative examples visibly show why a semantic center is not a task
   goal.
4. M3 improves task-relation satisfaction or target visibility without reducing
   executable-goal rate versus M1/M2.
5. Curation and all privileged quantities are disclosed.

Then write results, update registry, archive the plan and stop.
