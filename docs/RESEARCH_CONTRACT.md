# Research Contract — TopoNav Harness

## Working title

**Semantic-Topology-Grounded Closed-Loop VLM Planning for Indoor Mobile Robot Navigation**

## Scientific question

How can a frozen vision-language model act as a high-level indoor-navigation
agent without exposing it to metric controls or allowing open-ended reasoning
to bypass robot execution constraints?

## Closed-loop contract

`task, topology state, selected images, memory -> compiled context -> VLM tool
call -> validator -> fixed executor -> typed feedback -> updated memory`.

VLM actions are named topology tools: `NAVIGATE`, `OBSERVE`, `BACKTRACK`, and
`STOP`.  The harness alone maps a valid named transition to a metric goal.

## Method boundary

The method consists of task-relevant context compilation, topology-grounded
tools, typed execution feedback, and relation-preserving recovery.  VLM,
Habitat/NavMesh and the low-level planner are frozen external components.

## Online/offline separation

Topology anchors, NavMesh, relation predicates and hidden target pose may make
episodes and evaluate outcomes offline.  The online VLM sees only task text,
semantic node/transition descriptions, selected images, and prior typed
feedback.  It does not see oracle routes, NavMesh, metric coordinates, or
hidden goal pose.
