# RelationNav candidate-set feasibility audit

The current learned field cannot be interpreted without checking whether the online candidate set contains a valid realization of the requested relation.

## Audit protocol

For every one of the 1,408 P1 decision states, each of the eight goal-independent candidate endpoints was executed in Habitat-GS. A candidate was marked valid only when the fixed executor arrived and the relation predicate (`APPROACH`, `CROSS`, `ENTER`, or `OBSERVE`) was true. No learned score was used.

## Results

| relation | states | states with ≥1 valid candidate | mean valid candidates/state |
|---|---:|---:|---:|
| APPROACH | 400 | 58.75% | 1.36 |
| CROSS | 400 | 88.00% | 1.35 |
| ENTER | 400 | 25.50% | 0.39 |
| OBSERVE | 208 | 69.71% | 0.94 |
| **all** | **1,408** | **59.23%** | **1.02** |

## Consequence

The current candidate protocol is not a valid test of candidate selection: 40.77% of decision states have no successful candidate at all, and `ENTER` is infeasible in 74.5% of states. A model receiving such a candidate set cannot succeed regardless of its score. The dense field's 0% episode success is therefore a mixture of representation failure and candidate-set impossibility.

## Required correction before a paper experiment

Keep the online generator goal-independent, but add a fixed local-horizon/branch-completion rule so every formal state has at least one feasible candidate, or mark no-valid-candidate states as a separate abstention category. Then train/evaluate a candidate-level scorer against actual branch outcomes. Do not continue tuning the current dense-field model before this protocol is corrected.
