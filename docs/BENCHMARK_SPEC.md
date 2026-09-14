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

B0 Arrival-only advances on executor goal arrival. B1 Same-goal retry retries
the failed realization without semantic verification. B2 Relation-verified
advances only after the declared predicate is true. Ours Relation-verified plus
relation-preserving recovery reselects another admissible realization while
retaining the same relation. Oracle is evaluation only.

## Metrics

Relation: relation satisfaction, portal side-transition correctness, completion
guard precision/recall, false-completion rate, wrong-phase transition and
recovery success.

Task: full task success, phase completion, SPL, final DTG, path length,
collision, timeout, repeated failure and decision count. Report bootstrap
confidence intervals and held-out scene results.
