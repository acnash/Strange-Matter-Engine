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

The executable protocol is
[`scripts/run_cv_cyp_specialist_gca.py`](../../scripts/run_cv_cyp_specialist_gca.py).
