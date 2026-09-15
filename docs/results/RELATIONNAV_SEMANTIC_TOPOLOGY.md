# Semantic Topology Graph (Habitat-GS)

The RelationNav benchmark now has a deterministic, scene-local semantic
topology representation built from the curated entity contracts.  It is an
interface for task execution, not a new planner or a learned map.

Each scene graph contains portal, area and landmark nodes.  Directed edges
encode the executable relations `APPROACH`, `CROSS`, `ENTER` and `OBSERVE`.
The graph records world coordinates/polygons and the provenance of every
entity; graph construction does not query hidden goals or NavMesh outcomes.

At run time `RelationState` keeps `(entity, relation)` active until its
relation-specific predicate is satisfied.  A failed realization is
blacklisted while the same relation is preserved, allowing a fixed executor
to select another admissible realization.  The state log records poses,
completion events and recovery events for episode-level auditing.

Current four-scene graph: 12 nodes (one portal, area and landmark per scene)
and 16 directed relation edges.  The graph is deliberately conservative:
unannotated room connectivity is not fabricated.  Additional curated
entities can be added through `scripts/relationnav_annotation.py` without
changing the schema.

Artifacts:

- `outputs/formal/RelationNav/topology/semantic_topology.json`
- `paper_assets/figures/relationnav_semantic_topology.png`
- `scripts/semantic_topology.py`
- `scripts/build_semantic_topology.py`

