# Strange Matter Engine method report

## Status and document policy

This is the permanent, version-independent method report for the Strange Matter Engine entry in the OpenADMET CYP Inhibition Blind Challenge. Its path remains stable. Whenever a new leading model replaces the submitted model, this document is updated on `main` to identify and describe that model, while versioned result directories preserve the historical artifacts.

**Current submitted model:** Dual-Scale Graph Cellular Automata Ensemble (DS-GCAE v1)  
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

## DS-GCAE v1 ensemble construction

DS-GCAE v1 combines two five-rule ensembles:

- **Original component, weight 0.425:** one trained seed per rule with equal rule weights of 0.2.
- **Multiscale component, weight 0.575:** seeds 1701, 2909, and 4211 are averaged within each rule, followed by validation-selected rule weights.

The multiscale rule weights, in the transition-rule order listed above, are:

| Transition rule | Weight |
|---|---:|
| Gated residual | 0.103615 |
| Delayed memory | 0.199877 |
| Inertial reaction diffusion | 0.196589 |
| Kuramoto-Sakaguchi | 0.244697 |
| FitzHugh-Nagumo | 0.255223 |

The two-scale mixing coefficient was selected from cross-fitted validation predictions using MA-ST-RAE. The final ensemble contains 20 frozen checkpoint evaluations: five original members and fifteen multiscale seed members.

## Validation results

| Metric | Result |
|---|---:|
| Point MA-ST-RAE | **0.784156** |
| Bootstrap MA-ST-RAE mean | **0.784989** |
| MA-ST-RAE 95% bootstrap interval | 0.746381 to 0.827386 |
| RMSE | 0.867775 pIC50 |
| Macro MAE, bootstrap mean | 0.638716 pIC50 |
| Macro R-squared, bootstrap mean | 0.270601 |
| Macro Spearman rho, bootstrap mean | 0.509842 |
| Macro Kendall tau, bootstrap mean | 0.362550 |

Endpoint point ST-RAE values were:

| Endpoint | ST-RAE |
|---|---:|
| CYP1A2 | 0.822787 |
| CYP2C9 | 0.746083 |
| CYP2D6 | 0.955599 |
| CYP3A4 | 0.612157 |

The original equal-weight five-rule ensemble achieved point MA-ST-RAE 0.786477 and RMSE 0.868771. DS-GCAE improved the point primary metric by approximately 0.0023. This is a modest improvement, and the available uncertainty does not establish a statistically resolved separation between these closely related ensembles.

## Blinded inference and submission

Frozen inference ran on all 750 blinded challenge molecules and produced 3,000 finite predictions, one for every molecule and CYP endpoint. The inference manifest records `labels_loaded: false`, successful schema validation, the CUDA device, blend weights, seeds, and all checkpoint paths.

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

The current model artifacts are retained in [`results/production_dual_scale_graph_ca_ensemble_v1`](results/production_dual_scale_graph_ca_ensemble_v1). Important files include:

- [`ds_gcae_submission.csv`](results/production_dual_scale_graph_ca_ensemble_v1/ds_gcae_submission.csv), the challenge-ready regression file;
- [`ds_gcae_blinded_predictions_long.csv`](results/production_dual_scale_graph_ca_ensemble_v1/ds_gcae_blinded_predictions_long.csv), the auditable component and blended predictions;
- [`inference_manifest.json`](results/production_dual_scale_graph_ca_ensemble_v1/inference_manifest.json), the frozen inference specification;
- [`README.md`](results/production_dual_scale_graph_ca_ensemble_v1/README.md), the version-specific result summary; and
- [`scripts/run_ds_gcae_blind_inference.py`](scripts/run_ds_gcae_blind_inference.py), the frozen ensemble inference runner.

Production training and inference used an NVIDIA GeForce RTX 5070 Ti through CUDA. The shared implementation also supports CPU execution for inspection and forward-only analysis. Environment files, training scripts, saved checkpoints, validation tables, figures, and PDF reports are committed in this repository.

## Limitations

The validation estimates arise from one challenge dataset and its scaffold-aware partitions. CYP2D6 remains the weakest endpoint by ST-RAE. The improvement from dual-scale blending is small, and public leaderboard performance may differ from local validation because the blinded compounds form a distinct analog-expansion set. The current submission provides point predictions rather than calibrated predictive intervals.

## Update history

| Date | Leading submitted model | Change |
|---|---|---|
| 29 August 2026 | DS-GCAE v1 | Created the permanent method report and documented the first frozen regression submission. |
