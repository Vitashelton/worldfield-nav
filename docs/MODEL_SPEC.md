# Model Specification — Relation Execution Contract

## Contract

For a stage ``(entity, relation)``, the system stores an ordinary admissible
goal region, a relation-specific completion predicate, failure predicates and
a relation-preserving recovery rule. Relations are APPROACH, CROSS, ENTER and
OBSERVE. Entity evidence is an upstream visual hypothesis or curated simulator
anchor, never hidden world coordinates at inference.

Inputs to verification are robot pose, observed RGB-D/geometry, the declared
entity geometry and the latest navigation event. Frozen DINOv3 may support an
upstream entity observation interface, but it is not trained or used to
hallucinate a relation field.

## Completion predicates

``APPROACH`` validates source-side portal-neighborhood condition. ``CROSS``
validates a source-to-destination signed portal-side transition. ``ENTER``
validates containment in the target area. ``OBSERVE`` validates target
visibility in current camera geometry. A distance threshold alone cannot
advance a task phase.

## Transition and recovery

Failure retains entity/relation and records the failed candidate/route. Recovery
selects another permissible goal realization but cannot advance, replace or
reinterpret the high-level relation. The controller never emits low-level
controls.
