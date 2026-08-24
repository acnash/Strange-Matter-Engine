# Coupled-map atom-property production study

## Result

The final restored grouped-validation RMSE is **0.8839 pIC50**. The selected configuration confirmed at **0.8806 ± 0.0021 RMSE** across seeds 1701, 2909, and 4211. The confirmation mean and variability are strong, although the restored final seed is 0.0051 pIC50 worse than the preceding coupled-map final result of 0.8788. The earlier model therefore remains the best coupled-map checkpoint by single-seed final validation RMSE.

The new model is saved at `runs/final_model/model.pt`. It uses 250 cellular-automata generations, 24 dynamical channels, and the `valence_electronic` profile. Its 35 atom properties combine the original 25 descriptors with total valence, implicit valence, heavy-atom degree, radical-electron count, absolute formal charge, Pauling electronegativity, approximate polarizability, heteroatom and halogen indicators, and conjugated-bond fraction.

## Search protocol

The search evaluated 72 broad candidates spanning nine atom-property profiles together with coupled-map trajectory length, channel count, optimization, regularization, initialization, graph-support, bond-conditioning, and transition-dynamics controls. Sixteen candidates advanced to refinement. Four finalists were repeated over three confirmation seeds and ranked by mean validation RMSE plus 0.25 times its population standard deviation. The selected configuration was then trained for up to 18 epochs on the complete permitted fitting split with early stopping.

The search took 6,316 seconds, or 105.3 minutes. Final-model training took 742.0 seconds on an NVIDIA GeForce RTX 5070 Ti and used approximately 1.13 GiB of peak allocated GPU memory. The readout remained a differentiable closed-form ridge solve, while Adam updated the bonded-graph cellular-automata parameters.

## Selected configuration

| Parameter | Value |
|---|---:|
| Atom-feature profile | `valence_electronic` |
| Atom-feature count | 35 |
| Generations | 250 |
| Dynamical channels | 24 |
| CA learning rate | 0.003 |
| Ridge penalty | 1.0 |
| CA L2 | 1e-6 |
| Gradient clip | 2.0 |
| Batch molecules | 64 |
| Update scale | 0.15 |
| Initial-state scale | 1.0 |
| Initial noise | 0.005 |
| Support fraction | 0.85 |
| Bond temperature | 2.0 |
| Coupled-map A/B/C/D | 0.2 / 0.2 / 0.5 / 0.05 |

## Dynamical screening

All 1,309 grouped-validation trajectories were screened. The median recurrence ratio was zero, median late motion was 3.78e-6, and median spectral entropy was effectively zero. The upper tail reached recurrence 0.0111, late motion 2.13e-5, and spectral entropy 0.4577. These results indicate a predominantly strongly contracting finite-time regime together with a small tail of more complex transients. Longer renormalized trajectories and perturbation-pair tests are required before identifying point attractors, strange attractors, or chaos. Perturbation confirmation was deferred after a native CUDA failure and did not influence model selection.

## Data separation

The blinded challenge set was excluded from cache preparation, tuning, confirmation, final fitting, prediction, and dynamical screening. `study_summary.json` records `blind_data_used: false`.
