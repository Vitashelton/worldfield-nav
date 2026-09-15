# Model Specification — TopoNav Harness

No navigation network is trained.  The learned/frozen component is a
replaceable VLM backend.  The robot-facing model is a strict protocol:

- **Context compiler:** extracts a budgeted local topology subgraph and at
  most two event-relevant RGB references.
- **VLM agent:** returns one JSON tool call only.
- **Validator:** checks existence, adjacency, relation preconditions, STOP
  validity and retry budget before any execution.
- **Metric executor:** converts a valid symbolic transition to the existing
  Habitat/Nav2 metric realization.
- **Relation state:** retains `(entity, relation, completion)` on recoverable
  failure and suppresses the failed realization rather than changing intent.

The deployment interface is deliberately backend-agnostic: `RGB evidence +
semantic topology + typed feedback -> named tool call`.
