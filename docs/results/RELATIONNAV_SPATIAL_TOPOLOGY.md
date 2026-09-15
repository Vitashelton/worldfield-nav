# Spatial Semantic Topology (Habitat-GS)

The RelationNav representation now separates two layers:

1. `E_spatial`: room/area, corridor and portal connectivity extracted from
   InteriorGS `structure.json` (`CONNECTS`, `SPATIAL_ADJACENCY`).
2. `E_relation`: task execution contracts attached to the same entities
   (`APPROACH`, `CROSS`, `ENTER`, `OBSERVE`).

This prevents the task-relation graph from being mistaken for a global map.
The graph is heterogeneous and scene-local.  Door/hole profiles are assigned
to their two nearest room polygons; when a scene has no structure file, its
audited portal/area/landmark contracts are retained as a conservative
curated-only fallback.  No hidden goal or execution outcome is used to create
an edge.

The current four-scene asset yields 25 nodes and 95 edges.  The structurally
rich scenes contain 4 areas/5 portals (0135), 1 area/3 portals (0121), and
2 areas/3 portals (0093), while 0405 uses the audited fallback because its
downloaded semantic package has no `structure.json`.  These graphs already
provide junction-like alternatives in the 0135 scene; route feasibility is
checked separately against Habitat's NavMesh by the evaluator.

Machine-readable artifact:

`outputs/formal/RelationNav/topology/spatial_semantic_topology.json`

Overview figure:

`paper_assets/figures/relationnav_spatial_topology.png`

