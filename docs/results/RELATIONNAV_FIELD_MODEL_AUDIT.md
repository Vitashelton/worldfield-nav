# RelationNav learned-field audit

## Status

This is a correctness audit, not a positive result. The existing `relationnav_field.pt` was evaluated on the held-out scene using its predicted spatial field to select one of the eight online candidate endpoints. The evaluator then executed the selected endpoint in Habitat-GS and checked the actual relation predicate.

## Results

- Held-out episodes: 100
- Held-out stages: 308
- Trainable parameters: 793,921
- Best held-out dense-field IoU during training: 19.1% (epoch 10)
- Best held-out dense-field recall: 43.3%
- Final dense-field IoU: 14.8%
- Candidate execution episode success: 0.0%
- Candidate execution relation-stage success: 24.7%

## Interpretation

The dense field is not a reliable executable-goal selector. The earlier `relationnav_execution_*` videos visualize a fixed/privileged execution replay and must not be cited as learned-model evidence. They remain debugging/demo assets only.

This audit rules out the current formulation as a completed paper method. It does not prove that relation-verified execution is useless; it proves that the present DINO-projected dense field does not yet solve relation-conditioned candidate selection.

## Reproducibility

- Checkpoint: `artifacts/relationnav/relationnav_field.pt`
- Training metrics: `outputs/formal/RelationNav/P1/training_metrics.json`
- Held-out candidate evaluation: `outputs/formal/RelationNav/P1/field_eval_heldout.json`
- Evaluator: `scripts/eval_relation_field.py`

No privileged target coordinate is used for the learned candidate choice; privileged geometry is used only by the post-execution relation predicate.
