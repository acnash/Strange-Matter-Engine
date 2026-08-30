# Strange Matter Engine method report

## Status and document policy

This is the permanent, version-independent method report for the Strange Matter Engine entry in the OpenADMET CYP Inhibition Blind Challenge. Its path remains stable. Whenever a new leading model replaces the submitted model, this document is updated on `main` to identify and describe that model, while versioned result directories preserve the historical artifacts.

**Current production candidate:** Cross-Fitted Target-Specific Dual-Scale Graph Cellular Automata Ensemble (CFT-DS-GCAE v1)

**Previously submitted model:** Dual-Scale Graph Cellular Automata Ensemble (DS-GCAE v1)

**Challenge track:** Direct Inhibition, regression

**Predicted endpoints:** CYP1A2, CYP2C9, CYP2D6, and CYP3A4 direct-inhibition pIC50

**Primary validation metric:** Macro-Averaged Soft-Threshold Relative Absolute Error (MA-ST-RAE), lower is better

## Scientific objective

Strange Matter Engine tests whether learned cellular-automata dynamics on molecular graphs can provide a useful and interpretable representation for CYP inhibition prediction. Atoms are cells, chemical bonds define local neighbourhoods, and the same transition function is applied recurrently across generations. The full dynamical trajectory contributes to the molecular representation rather than serving only as an intermediate calculation.

The model predicts one continuous pIC50 value for each molecule and CYP context. Time-dependent inhibition classification is a separate challenge track and is not included in this submission.

## Molecular representation

Each SMILES string is converted to a bonded molecular graph. Atom inputs contain chemical descriptors selected during training-only hyperparameter searches, including combinations of periodic, valence, electronic, ring, and local-neighbour properties. Bond descriptors carry the chemical relationship between adjacent atoms. CYP identity is supplied as task context so that the same molecular encoder can support the four direct-inhibition endpoints.

No blinded-test activity labels were loaded. The blinded molecules were used only after model selection, when the frozen checkpoints generated the final predictions.

## Graph cellular-automata members

Every ensemble member follows the same general computation:

1. Chemical atom features initialise a fixed number of dynamical channels.
2. Bond-conditioned messages pass information between directly bonded atoms.
3. A shared local transition rule updates all atom states recurrently for the selected number of generations.
4. Statistics from several points in the trajectory form a molecular dynamical fingerprint.
5. A CYP-conditioned differentiable ridge readout maps the fingerprint to predicted pIC50.

The current ensemble contains five complementary transition rules:

- gated residual;
- delayed memory;
- inertial reaction diffusion;
- Kuramoto-Sakaguchi; and
- FitzHugh-Nagumo.

The repeated local update and bonded neighbourhood remain present in every member. Ensemble averaging occurs after each member has completed its own graph-CA trajectory and ridge prediction.

## Learning and ridge readout

Backpropagation through time optimises the parameters of each graph cellular automaton. Adam updates those nonlinear CA parameters. The final linear readout is genuine ridge regression: its coefficients are obtained through a regularised differentiable linear solve rather than by treating an ordinary Adam-trained linear layer as ridge regression. Gradients through that solve allow the prediction loss to shape the preceding CA dynamics.

Trajectory pooling includes final-state means and variances, time-averaged states, temporal variance, and state-change energy. Multiscale pooling extends this representation with information collected over several temporal windows. The resulting representation retains information about both terminal behaviour and transient dynamics.

## Model selection and validation

Hyperparameter searches were performed using scaffold-aware partitions of the labelled training data. Search dimensions included atom-feature profiles, generation count, dynamical-channel count, learning rate, ridge penalty, CA regularisation, update scale, support fraction, batch size, bond-message temperature, initial-state scaling and noise, gradient clipping, pooling choices, and transition-rule-specific parameters.

MA-ST-RAE was the primary promotion and selection metric. RMSE, MAE, R-squared, Spearman correlation, and Kendall correlation were retained as secondary diagnostics. Final metric uncertainty was estimated with 1,000 bootstrap resamples. The challenge-blinded test set did not participate in hyperparameter tuning, early stopping, seed selection, rule weighting, ensemble blending, or dynamical-regime selection.

## CFT-DS-GCAE v1 ensemble construction

CFT-DS-GCAE v1 retains the two five-rule expert families developed for DS-GCAE:

- **Original experts:** one trained seed from each of the five transition rules.
- **Multiscale experts:** seeds 1701, 2909, and 4211 averaged within each transition rule.

This produces ten prediction signals for each molecule and endpoint. A separate standardised ridge stack is fitted for CYP1A2, CYP2C9, CYP2D6, and CYP3A4. Ridge strength is selected inside nested scaffold-grouped folds using endpoint ST-RAE. An affine slope and offset calibration is then evaluated using predictions that were themselves generated out of fold. Calibration is retained only when it improves held-out ST-RAE.

The final target-specific settings are:

| Endpoint | Ridge penalty | Calibration slope | Calibration offset |
|---|---:|---:|---:|
| CYP1A2 | 1000 | 1.0 | 0.0 |
| CYP2C9 | 100 | 1.0 | 0.0 |
| CYP2D6 | 1000 | 1.0 | 0.0 |
| CYP3A4 | 100 | 1.0 | 0.0 |

Cross-validation selected identity calibration for all four final stacks. This is a valid calibrated-model outcome: the optional correction was rejected because it did not generalise within the development folds. The final system still contains the 20 frozen checkpoint evaluations used to construct its ten seed-averaged signals.

## Validation results

| Metric | Result |
|---|---:|
| Point MA-ST-RAE | **0.773895** |
| Bootstrap MA-ST-RAE mean | **0.774878** |
| MA-ST-RAE 95% bootstrap interval | 0.735589 to 0.819301 |
| RMSE | 0.858630 pIC50 |
| Macro MAE, bootstrap mean | 0.632268 pIC50 |
| Macro R-squared, bootstrap mean | 0.275448 |
| Macro Spearman rho, bootstrap mean | 0.518365 |
| Macro Kendall tau, bootstrap mean | 0.369915 |

Endpoint point ST-RAE values were:

| Endpoint | ST-RAE |
|---|---:|
| CYP1A2 | 0.817214 |
| CYP2C9 | 0.761089 |
| CYP2D6 | 0.940906 |
| CYP3A4 | 0.576373 |

DS-GCAE v1 achieved point MA-ST-RAE 0.784156 and RMSE 0.867775 on the same sealed validation set. CFT-DS-GCAE improves those values by approximately 0.0103 and 0.0091 pIC50 respectively. The bootstrap intervals overlap, so the available uncertainty does not establish a statistically resolved separation.

The first DS-GCAE blind submission ranked 80th of 89. The organiser reported MA-ST-RAE 1.0132, MA 1.0893, macro R-squared -0.0827, macro Spearman rho 0.4751, and macro Kendall tau 0.3323. This divergence from local validation motivated target-specific stacking and stronger cross-fitting. The blind result is reported as evidence about DS-GCAE and is not used as a label-level training signal for CFT-DS-GCAE.

## Blinded inference and submission

Frozen inference ran on all 750 blinded challenge molecules and produced 3,000 finite CFT-DS-GCAE predictions, one for every molecule and CYP endpoint. The inference manifest records `labels_loaded: false`, successful schema validation, the feature order, target-specific ridge states, and calibration decisions.

The current regression submission contains exactly 750 rows and the six required columns in the official order:

```text
SMILES
Molecule_Name
CYP1A2_pIC50_direct_inhibition
CYP2C9_pIC50_direct_inhibition
CYP2D6_pIC50_direct_inhibition
CYP3A4_pIC50_direct_inhibition
```

The submission contains no missing or non-finite predictions, no duplicate molecule identifiers, and preserves the blinded test-set molecule and SMILES alignment.

## Dynamical analysis

The project stores trajectory-derived diagnostics and, for selected runs, richer trajectory archives to support downstream study of information flow, convergence, oscillation, recurrence, transient structure, perturbation sensitivity, and candidate attractor regimes. These diagnostics did not influence challenge model selection.

Finite-time recurrence or spectral structure is treated as a candidate dynamical regime rather than proof of chaos or a strange attractor. Stronger claims require long-horizon propagation, perturbation-pair divergence, Lyapunov-style estimates, recurrence analysis across seeds, and checks against timestep and numerical-precision artifacts.

## Reproducibility and artifacts

The current candidate artifacts are retained in [`results/production_cross_fitted_target_calibrated_gcae_v1`](results/production_cross_fitted_target_calibrated_gcae_v1). Important files include:

- [`cft_ds_gcae_submission.csv`](results/production_cross_fitted_target_calibrated_gcae_v1/cft_ds_gcae_submission.csv), the challenge-ready regression file;
- [`cft_ds_gcae_blinded_predictions_long.csv`](results/production_cross_fitted_target_calibrated_gcae_v1/cft_ds_gcae_blinded_predictions_long.csv), the auditable expert and stacked predictions;
- [`study_summary.json`](results/production_cross_fitted_target_calibrated_gcae_v1/study_summary.json), the nested cross-fitting parameters and validation report;
- [`inference_manifest.json`](results/production_cross_fitted_target_calibrated_gcae_v1/inference_manifest.json), the frozen inference specification; and
- [`scripts/run_cross_fitted_target_calibrated_ensemble.py`](scripts/run_cross_fitted_target_calibrated_ensemble.py), the validation and fitting runner.

Production training and inference used an NVIDIA GeForce RTX 5070 Ti through CUDA. The shared implementation also supports CPU execution for inspection and forward-only analysis. Environment files, training scripts, saved checkpoints, validation tables, figures, and PDF reports are committed in this repository.

## Limitations

The validation estimates arise from one challenge dataset and its scaffold-aware partitions. CYP2D6 remains the weakest endpoint by ST-RAE. The improvement from target-specific stacking is modest, and the preceding blind result demonstrates that local validation can substantially overestimate performance on the challenge distribution. The current candidate provides calibrated point-model selection rather than predictive uncertainty intervals.

## Update history

| Date | Leading model or candidate | Change |
|---|---|---|
| 29 August 2026 | DS-GCAE v1 | Created the permanent method report and documented the first frozen regression submission. |
| 29 August 2026 | CFT-DS-GCAE v1 candidate | Added nested target-specific ridge stacking after the first blind result; generated a new label-free submission candidate. |
