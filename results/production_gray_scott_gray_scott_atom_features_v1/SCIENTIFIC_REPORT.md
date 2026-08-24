# Gray-Scott atom-property production study

## Result

The final restored grouped-validation RMSE is **0.8651 pIC50**. The selected configuration produced a confirmation RMSE of **0.8749 ± 0.0043** across seeds 1701, 2909, and 4211. This improves the previous Gray-Scott final validation RMSE of 0.8674 by 0.0023 pIC50.

The winning model is saved at `runs/final_model/model.pt`. It uses 32 cellular-automata generations, 24 dynamical channels, and the `periodic_electronic` atom-feature profile. Its 35 atom properties combine the original 25 descriptors with atomic number, atomic mass, covalent and van der Waals radii, outer-electron count, Pauling electronegativity, approximate polarizability, heteroatom and halogen indicators, and conjugated-bond fraction.

## Search protocol

The search evaluated 72 broad candidates spanning nine atom-property profiles together with Gray-Scott trajectory length, channel count, optimization, regularization, initialization, graph-support, bond-conditioning, and transition-dynamics controls. Sixteen candidates advanced to a larger refinement stage. Four finalists were repeated over three confirmation seeds, ranked by mean validation RMSE plus 0.25 times its population standard deviation. The selected configuration was then trained for up to 18 epochs on the complete permitted fitting split with early stopping.

The search took 6,948 seconds, or 115.8 minutes. Final-model training took 94.4 seconds on an NVIDIA GeForce RTX 5070 Ti and used approximately 272 MiB of peak allocated GPU memory. The readout remained a differentiable closed-form ridge solve; Adam updated the bonded-graph cellular-automata parameters.

## Selected configuration

| Parameter | Value |
|---|---:|
| Atom-feature profile | `periodic_electronic` |
| Atom-feature count | 35 |
| Generations | 32 |
| Dynamical channels | 24 |
| CA learning rate | 0.003 |
| Ridge penalty | 0.01 |
| CA L2 | 1e-5 |
| Gradient clip | 0.5 |
| Batch molecules | 128 |
| Update scale | 0.4 |
| Initial-state scale | 1.5 |
| Initial noise | 0 |
| Support fraction | 0.75 |
| Bond temperature | 0.5 |
| Gray-Scott A/B/C/D | 0.5 / 0.2 / 0.2 / 0.3 |

## Dynamical screening

All 1,309 grouped-validation trajectories were screened for finite-time recurrence, late motion, and spectral entropy. The trajectories form a continuous, tightly correlated high-recurrence regime with measurable persistent motion and spectral complexity. These measurements identify transient and recurrent candidates for longer investigation, while they do not establish point attractors, strange attractors, or chaos. Perturbation-pair confirmation was deferred after a native CUDA failure and did not influence model selection.

## Data separation

The blinded challenge set was excluded from cache preparation, tuning, confirmation, final fitting, prediction, and dynamical screening. `study_summary.json` records `blind_data_used: false`.
