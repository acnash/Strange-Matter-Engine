# Latest production summary

Updated: 24 August 2026  
Latest search version: `coupled_map_atom_features_v1`  
Hardware: NVIDIA GeForce RTX 5070 Ti, PyTorch 2.11.0 with CUDA 12.8

## Challenge-metric alignment

Production code now uses the official direct-inhibition **MA-ST-RAE** for checkpoint selection, early stopping, hyperparameter promotion, and three-seed finalist selection. The implementation matches the OpenADMET tutorial reference function exactly and weights CYP1A2, CYP2C9, CYP2D6, and CYP3A4 equally. Differentiable MSE remains the optimization loss, while RMSE remains a secondary diagnostic. Existing production results below were selected under the earlier RMSE protocol and remain historical comparisons; subsequent studies use the `ma_st_rae` search versions and regenerated caches containing the assay credible intervals.

## Latest coupled-map chemistry search

The chemistry-augmented coupled-map study completed the same 72 broad trials, 16 refinements, four three-seed confirmations, final training, and dynamical screening protocol used for Gray-Scott. The final restored grouped-validation RMSE is **0.8839 pIC50**. The selected configuration confirmed at **0.8806 ± 0.0021 RMSE** across seeds 1701, 2909, and 4211. Its confirmation variability is lower than the preceding coupled-map study, while its final seed is 0.0051 pIC50 worse than the preceding final RMSE of 0.8788.

The winning `valence_electronic` profile contains 35 atom properties: the original 25 descriptors plus total and implicit valence, heavy-atom degree, radical-electron count, absolute formal charge, Pauling electronegativity, approximate polarizability, heteroatom and halogen indicators, and incident conjugated-bond fraction. The selected graph cellular automaton uses 250 generations and 24 dynamical channels. The complete search took 105.3 minutes, and final-model training took 742.0 seconds on CUDA.

The blinded set remained sealed. The saved model is `results/production_coupled_map_coupled_map_atom_features_v1/runs/final_model/model.pt`, and the report is `output/pdf/coupled_map_coupled_map_atom_features_v1_production_report.pdf`. Among 1,309 screened validation trajectories, median recurrence was zero, median late motion was 3.78e-6, and median spectral entropy was effectively zero. A small tail reached recurrence 0.0111 and spectral entropy 0.4577. This pattern is consistent with a largely strongly contracting regime plus a small set of more complex transients; it is not proof of point attractors, strange attractors, or chaos.

## Latest Gray-Scott chemistry search

The chemistry-augmented Gray-Scott study completed 72 broad trials, promoted 16 configurations to refinement, confirmed four finalists across three seeds, and trained the selected configuration on the full permitted training split. The restored final validation RMSE is **0.8651 pIC50**, improving the preceding Gray-Scott result of 0.8674 by 0.0023 pIC50. Across confirmation seeds 1701, 2909, and 4211, the selected configuration achieved **0.8749 ± 0.0043 RMSE**.

The winning `periodic_electronic` profile contains the original 25 atom descriptors plus atomic number, atomic mass, covalent radius, van der Waals radius, outer-electron count, Pauling electronegativity, approximate polarizability, heteroatom identity, halogen identity, and incident conjugated-bond fraction, giving 35 atom properties in total. The selected cellular automaton uses 32 generations and 24 dynamical channels, with a 0.003 CA learning rate and 0.01 differentiable ridge penalty. The complete search took 115.8 minutes, while final-model training took 94.4 seconds on CUDA.

The blinded set remained sealed throughout. The saved model is `results/production_gray_scott_gray_scott_atom_features_v1/runs/final_model/model.pt`, the complete study is in `results/production_gray_scott_gray_scott_atom_features_v1`, and the report is `output/pdf/gray_scott_gray_scott_atom_features_v1_production_report.pdf`. All 1,309 validation trajectories were screened for finite-time recurrence, late motion, and spectral entropy. Their continuous high-recurrence regime is scientifically interesting, although the current diagnostics do not establish a point attractor, strange attractor, or chaos.

## Outcome

Five graph cellular-automata transition rules were searched under the same production protocol. Model selection used grouped-validation pIC50 RMSE only. The blinded challenge set was neither loaded nor predicted during tuning, confirmation, final fitting, or dynamical screening.

The coupled-map rule is the predictive leader at 0.8788 pIC50 validation RMSE. Damped-symplectic is effectively tied at 0.8792 and has the lowest confirmation-seed standard deviation. Damped-symplectic also presents the broadest finite-time dynamical range, while coupled-map presents four sharply separated recurrence-motion bands.

| Rank | Transition rule | Final validation RMSE | Confirmation RMSE, mean ± SD | Generations | Channels | Runtime | Saved final model |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | coupled map | **0.8788** | **0.8848 ± 0.0044** | 16 | 16 | about 36 min\* | `results/production_coupled_map_enhanced_v3/runs/final_model/model.pt` |
| 2 | damped symplectic | **0.8792** | **0.8857 ± 0.0022** | 125 | 16 | 73.5 min | `results/production_damped_symplectic_enhanced_v3/runs/final_model/model.pt` |
| 3 | gated residual | 0.8943 | 0.8973 ± 0.0025 | 64 | 8 | 29.1 min | `results/production_gated_residual_enhanced_v3/runs/final_model/model.pt` |
| 4 | inertial reaction diffusion | 0.8998 | 0.8856 ± 0.0049 | 125 | 16 | 67.9 min | `results/production_inertial_reaction_diffusion_enhanced_v3/runs/final_model/model.pt` |
| 5 | activator inhibitor | 0.9229 | 0.9310 ± 0.0058 | 64 | 8 | 43.2 min | `results/production_activator_inhibitor_enhanced_v3/runs/final_model/model.pt` |

\*The coupled-map controller resumed after one numerically singular broad-search candidate. Its JSON records 25.2 minutes after the resume; approximate end-to-end elapsed time was 36 minutes. The failed candidate was assigned an infinite selection score, and its log remains available for audit.

The final RMSE is the restored seed-1701 model evaluated on all 1,309 grouped-validation observations. The confirmation statistic is the mean and population standard deviation across seeds 1701, 2909, and 4211, and is the stronger measure for comparing reproducibility.

## Selected hyperparameters

| Rule | CA learning rate | Ridge | CA L2 | Clip | Batch | Update scale | Initial scale/noise | Support fraction | Bond temperature | Rule dynamics A/B/C/D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gated residual | 0.003 | 0.01 | 1e-4 | 1.0 | 128 | 0.40 | 1.50 / 0 | 0.75 | 0.5 | 0.2 / 0.8 / 0.5 / 0.05 |
| inertial reaction diffusion | 0.003 | 0.1 | 1e-5 | 0.5 | 128 | 0.25 | 1.50 / 0.005 | 0.60 | 1.0 | 0.5 / 0.2 / 0.8 / 0.15 |
| activator inhibitor | 0.0003 | 0.01 | 1e-6 | 2.0 | 128 | 0.40 | 0.50 / 0.01 | 0.85 | 2.0 | 0.5 / 0.2 / 0.5 / 0.3 |
| coupled map | 0.001 | 0.1 | 1e-5 | 1.0 | 64 | 0.40 | 0.50 / 0 | 0.85 | 2.0 | 0.2 / 0.5 / 0.8 / 0.3 |
| damped symplectic | 0.003 | 0.1 | 1e-5 | 0.5 | 128 | 0.25 | 1.50 / 0.005 | 0.60 | 1.0 | 0.5 / 0.2 / 0.8 / 0.15 |

All models retain bonded-graph cellular-automata message passing. Bond-conditioned messages, stochastic support selection, multitime trajectory pooling, cosine learning-rate decay, and a genuine differentiable closed-form ridge readout were shared across the five studies. Adam updates the graph-CA parameters only; ridge coefficients and the unpenalized intercept are solved from the permitted support fingerprints during each differentiable training step.

## Dynamical screening

Every final model screened 1,309 validation trajectories. The following ranges describe finite-time diagnostics and are useful for prioritising follow-up experiments.

| Rule | Recurrence-ratio range | Late-motion range | Spectral-entropy range | Current interpretation |
|---|---:|---:|---:|---|
| gated residual | 0.6974 to 0.8857 | 3.36e-4 to 1.59e-3 | 0.3939 to 0.6488 | Moderately recurrent, low persistent motion, with a continuous family of transient regimes. |
| inertial reaction diffusion | 0.8928 to 0.9918 | 1.75e-3 to 8.11e-3 | 0.3768 to 0.5135 | Highest recurrence and strongest late motion, consistent with slowly evolving or recurrent transients. |
| activator inhibitor | 0.5375 to 0.6739 | 1.11e-4 to 7.73e-3 | 0.4348 to 0.7171 | Several distinct dynamical bands, including a compact low-motion population and higher-motion branches. |
| coupled map | 0.1923 to 0.2159 | 2.72e-3 to 4.22e-3 | 0.6167 to 0.6948 | Four sharply separated recurrence-motion bands with consistently high spectral complexity. |
| damped symplectic | 0.0571 to 0.6506 | 1.37e-6 to 1.78e-3 | 0.2865 to 0.7846 | Broadest regime span, ranging from nearly stationary trajectories to branched higher-motion populations. |

These screens identify transient structure and candidate recurrent regimes. They do not establish point attractors, strange attractors, or chaos. Such claims require longer renormalised trajectories, perturbation-pair divergence, Lyapunov-style estimates, recurrence tests across multiple seeds, and checks that the behaviour survives numerical precision and timestep changes. Perturbation confirmation is recorded as `deferred_after_native_cuda_failure` in the reports; no dynamical-interest score influenced model selection.

## Reports

- `output/pdf/coupled_map_coupled_map_atom_features_v1_production_report.pdf`
- `output/pdf/gray_scott_gray_scott_atom_features_v1_production_report.pdf`
- `output/pdf/gated_residual_enhanced_v3_production_report.pdf`
- `output/pdf/inertial_reaction_diffusion_enhanced_v3_production_report.pdf`
- `output/pdf/activator_inhibitor_enhanced_v3_production_report.pdf`
- `output/pdf/coupled_map_enhanced_v3_production_report.pdf`
- `output/pdf/damped_symplectic_enhanced_v3_production_report.pdf`

Each study directory contains all broad-search, refinement, confirmation, and final-model metrics, saved weights, prediction tables, trajectory diagnostics, logs, and report figures. Historical conditioned-v2 studies remain intact for comparison.

## Recommended next production work

The next efficient predictive experiment is a narrow, seed-confirmed search around coupled-map and damped-symplectic, concentrating on generation counts near 16 for coupled-map and 64 to 250 for damped-symplectic, while varying ridge strength, support fraction, initial-state scaling, and trajectory pooling normalisation. Feature pruning should be assessed by ablation rather than selected on the same validation scores. The highest scientific priority is a separate CPU-safe perturbation and Lyapunov screening pass on the distinctive coupled-map bands and the broad damped-symplectic branches, keeping the blinded set sealed until a final frozen model is chosen.
