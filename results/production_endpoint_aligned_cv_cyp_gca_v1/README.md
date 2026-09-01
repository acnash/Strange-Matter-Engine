# EA-CV-CYP-GCA v1 production campaign

Endpoint-Aligned Cross-Validated CYP-Specialist Graph Cellular Automata
(EA-CV-CYP-GCA) aligns recurrent backpropagation, the differentiable ridge
support/query solve, early stopping, and validation with one active CYP
endpoint. The campaign screened all ten transition rules for every endpoint,
confirmed three candidates per endpoint across five scaffold folds and two
seeds, and selected sparse rule combinations from out-of-fold predictions.

## Sealed validation

| Metric | Result |
|---|---:|
| Point MA-ST-RAE | **0.754503** |
| Bootstrap mean MA-ST-RAE | **0.755073** |
| 95% bootstrap interval | 0.717656 to 0.797014 |
| RMSE | **0.852280 pIC50** |
| Bootstrap macro MAE | 0.621833 pIC50 |
| Bootstrap macro R-squared | 0.286429 |
| Bootstrap macro Spearman rho | 0.532744 |
| Bootstrap macro Kendall tau | 0.381552 |

The preceding CV-CYP-GCA system achieved point MA-ST-RAE 0.768985 and RMSE
0.862535 pIC50 on the same sealed holdout.

## Selected transition rules

| Endpoint | Selected rules |
|---|---|
| CYP1A2 | FitzHugh-Nagumo, Gray-Scott, conservative graph flux |
| CYP2C9 | Gray-Scott, damped symplectic |
| CYP2D6 | Delayed memory, FitzHugh-Nagumo |
| CYP3A4 | Damped symplectic, FitzHugh-Nagumo, delayed memory |

## Submission artifacts

- `endpoint_aligned_cv_cyp_gca_submission.csv` contains the challenge-ready
  regression predictions.
- `blind_predictions_long.csv` retains endpoint and selected-rule details.
- `reserved_holdout_predictions.csv` contains the sealed validation records.
- `study_summary.json` contains configurations, selected ridge states, and the
  complete 1,000-resample validation report.
- `inference_manifest.json` confirms 750 unique molecules, 3,000 finite
  predictions, official column order, successful schema validation, and
  `labels_loaded: false`.

The blinded test labels were unavailable and played no role in training,
hyperparameter selection, checkpoint promotion, or rule selection.
