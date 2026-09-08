# Model Specification — Task-Conditioned Goal-Pose Field

For a metric target hypothesis `T`, form a fixed lattice `Q(T)` of candidate
robot poses. Select `q* = argmax(q in Q(T)) E(q | T,G,A)`, where `G` contains
observation-derived free space/clearance and `A` is the frozen-VLM approach
template.

`E = w_free E_free + w_clear E_clear + w_reach E_reach + w_vis E_visible + w_task E_task - w_unc E_uncertainty`.

- `E_free`: candidate footprint is unoccupied in current geometry.
- `E_clear`: obstacle clearance.
- `E_reach`: available costmap/planner check; NavMesh only evaluates it.
- `E_visible`: target visibility where the template requires it.
- `E_task`: heading/standoff consistency with the approach template.
- `E_uncertainty`: target-hypothesis uncertainty.

The first formal version is an explicit reproducible field evaluation, not a
neural policy. A learned residual is unauthorized unless an active plan records
a documented failure of this explicit field.
