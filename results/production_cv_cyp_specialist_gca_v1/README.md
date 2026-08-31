# CV-CYP-GCA v1 production campaign

This directory records the Cross-Validated CYP-Specialist Graph Cellular
Automata campaign. It is an experimental successor to CFT-DS-GCAE and does not
replace the submitted method until its sealed-holdout evidence is complete.

Four independent nonlinear Graph-CA systems are trained, one for CYP1A2,
CYP2C9, CYP2D6, and CYP3A4. Backpropagation therefore specialises the cellular
transition dynamics themselves rather than applying CYP specificity only in a
final stacking layer. Every model retains the differentiable closed-form ridge
readout over multiscale trajectory fingerprints.

The resumable production protocol:

1. screens all ten transition rules for every CYP;
2. compares the established configuration with a CYP-directed chemical and
   temporal configuration;
3. confirms the three strongest rule/configuration pairs per CYP across five
   scaffold folds and two training seeds;
4. uses cross-validated MA-ST-RAE to select a single rule or a sparse subset for
   each CYP;
5. evaluates the frozen selection once on the reserved scaffold holdout; and
6. retains checkpoints and trajectory-compatible states for blinded inference
   and subsequent dynamical analysis.

`progress.json` is updated throughout execution. `screening_summary.json`,
`study_summary.json`, and `reserved_holdout_predictions.csv` are created as the
campaign advances. Intermediate checkpoints remain local during execution;
curated models, predictions, metrics, and reports are committed when the study
completes.

## Completed validation outcome

The campaign completed 160 screening fits, 120 five-fold and two-seed
confirmation fits, and 120 recovered reserved-holdout checkpoint evaluations
without a training failure. The sealed holdout contained 1,309 molecule-CYP
observations.

| Metric | Result |
|---|---:|
| Point MA-ST-RAE | **0.768985** |
| Bootstrap mean MA-ST-RAE | **0.769697** |
| 95% bootstrap interval | 0.731973 to 0.812732 |
| RMSE | 0.862535 pIC50 |
| Macro MAE | 0.629648 pIC50 |
| Macro R-squared | 0.269356 |
| Macro Spearman rho | 0.521821 |
| Macro Kendall tau | 0.371549 |

The selected endpoint systems are:

| Endpoint | Transition rules | Point ST-RAE |
|---|---|---:|
| CYP1A2 | Conservative graph flux, delayed memory, Gray-Scott | 0.820068 |
| CYP2C9 | Damped symplectic, activator-inhibitor | 0.751381 |
| CYP2D6 | Delayed memory, FitzHugh-Nagumo | 0.947044 |
| CYP3A4 | Gray-Scott, damped symplectic, coupled map | 0.557449 |

## Blind inference

Frozen inference generated four pIC50 values for each of the 750 blinded
molecules. The challenge-ready file is
[`cv_cyp_gca_submission.csv`](cv_cyp_gca_submission.csv). It contains 750 rows,
the six required columns in their official order, 750 unique molecule names,
and 3,000 finite predictions. Molecule names and SMILES match the supplied
blinded test file row for row. [`inference_manifest.json`](inference_manifest.json)
records the completed schema checks and confirms that labels were not loaded.

The executable protocol is
[`scripts/run_cv_cyp_specialist_gca.py`](../../scripts/run_cv_cyp_specialist_gca.py).
