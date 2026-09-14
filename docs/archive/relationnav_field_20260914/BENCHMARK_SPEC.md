# RelationNav Benchmark Specification

## Assets and split

Use interior_0405_840145, interior_0135_840032, interior_0121_840013 and
interior_0093_839966. Curate three to six portal, area or landmark entities
per scene with visual anchors and geometric annotations. Use three scenes for
development and one held out. This is a custom benchmark, not a public one.

## Episodes

Each deterministic episode contains three or four phases: APPROACH portal,
CROSS portal, ENTER target area, and optionally OBSERVE landmark. Generate
multiple source sides and starts per entity; target at least 300 closed-loop
episodes. Candidate generation is independent of hidden final goals.

## Methods

B0 fixed geometric portal offset; B1 nearest navigable entity-neighborhood
point; B2 VLM direct candidate choice; B3 geometry-only relation grounding;
Ours learned relation-conditioned field with transition verification and
relation-preserving recovery; Oracle is evaluation only.

## Metrics

Field: region IoU, region recall and selected-cell validity.

Relation: relation satisfaction, portal side-transition correctness, completion
guard precision/recall, wrong-phase transition and recovery success.

Task: full task success, phase completion, SPL, final DTG, path length,
collision, timeout, repeated failure and decision count. Report bootstrap
confidence intervals and held-out scene results.
