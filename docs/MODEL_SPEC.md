# Failure-Aware Rolling Subgoal Scorer

For candidate `c_i`, frozen inputs are DINOv3 current/goal descriptors,
geometry features, relative pose and cached VLM semantic score. Rolling state
contains prior selection, failed/stuck region and recent visitation. A
lightweight MLP scorer (target <2M parameters) predicts executable probability,
normalized short-horizon progress, failure risk and expected path cost. Its
combined score selects one candidate.

Teacher labels are obtained by actual short-horizon fixed-executor rollouts in
Habitat-GS: reached, collision, stuck/timeout, geodesic progress and executed
path length. At inference none of those labels or goal coordinates are inputs.
After abort/stuck, history changes and the same frozen proposal/executor stack
is re-evaluated.
