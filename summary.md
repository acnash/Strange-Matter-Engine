# Latest production summary

## Current production candidate: CFT-DS-GCAE v1

Updated: 29 August 2026  
Hardware: NVIDIA GeForce RTX 5070 Ti, CUDA  
Primary selection metric: MA-ST-RAE, lower is better

The current candidate is the **Cross-Fitted Target-Specific Dual-Scale Graph Cellular Automata Ensemble (CFT-DS-GCAE v1)**. It preserves all recurrent bonded-graph cellular-automata members and uses a separate nested cross-fitted ridge stack for each CYP endpoint. The preceding DS-GCAE submission ranked 80th of 89 on the blind leaderboard, with MA-ST-RAE 1.0132, revealing a substantial local-to-blind generalisation gap.

| Validation result | Value |
|---|---:|
| Point MA-ST-RAE | **0.7739** |
| Bootstrap MA-ST-RAE mean | **0.7749** |
| 95% bootstrap interval | 0.7356 to 0.8193 |
| RMSE | 0.8586 pIC50 |
| CYP1A2 ST-RAE | 0.8172 |
| CYP2C9 ST-RAE | 0.7611 |
| CYP2D6 ST-RAE | 0.9409 |
| CYP3A4 ST-RAE | 0.5764 |

The final meta-model has ten inputs per endpoint: five original graph-CA predictions and five multiscale predictions, with three seeds averaged inside every multiscale rule. Ridge penalties are 1000 for CYP1A2 and CYP2D6 and 100 for CYP2C9 and CYP3A4. Optional affine calibration was evaluated leakage-safely and identity calibration was selected for all four endpoints.

Blind inference completed on all 750 challenge molecules using the 20 previously frozen checkpoint predictions, producing 3,000 finite pIC50 predictions. Test labels were not loaded. The candidate submission is `results/production_cross_fitted_target_calibrated_gcae_v1/cft_ds_gcae_submission.csv`; it contains exactly 750 rows and six columns in the official order and has no missing or non-finite predictions.

CFT-DS-GCAE improves over DS-GCAE by 0.0103 point MA-ST-RAE and 0.0091 pIC50 RMSE on the sealed validation set. The bootstrap intervals overlap, so this remains a modest candidate improvement rather than a guaranteed blind-set gain.

## Challenge-aligned v5 production campaign complete

The ten-rule rerun now uses scaffold cross-validation inside the fitting pool, a sealed scaffold holdout, point MA-ST-RAE for efficient screening, and the official 1,000-resample bootstrap for finalists. The common search includes expanded chemical feature groups, rule-specific dynamics, surrogate-guided refinement, successive halving, five-fold and three-seed confirmation, secondary challenge metrics, and visually verified PDF pagination. No challenge-blinded molecules enter model development.

All ten transition-rule studies completed on the NVIDIA GeForce RTX 5070 Ti. Every six-page PDF was rendered and visually inspected, including the dedicated perturbation page. Lower MA-ST-RAE is better.

| Rank | Transition rule | MA-ST-RAE | 95% bootstrap CI | RMSE | Generations | Channels | Atom feature profile |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | gated residual | **0.8073** | 0.7610 to 0.8531 | 0.8774 | 64 | 16 | local environment |
| 2 | delayed memory | **0.8075** | 0.7643 to 0.8506 | 0.8748 | 125 | 16 | electronic local |
| 3 | inertial reaction diffusion | **0.8186** | 0.7743 to 0.8641 | 0.8865 | 32 | 16 | electronic |
| 4 | Kuramoto-Sakaguchi | **0.8211** | 0.7788 to 0.8677 | 0.8844 | 32 | 16 | periodic |
| 5 | FitzHugh-Nagumo | **0.8272** | 0.7797 to 0.8757 | 0.8847 | 16 | 16 | electronic |
| 6 | activator inhibitor | **0.8366** | 0.7936 to 0.8817 | 0.8987 | 64 | 16 | valence electronic |
| 7 | coupled map | **0.8387** | 0.7936 to 0.8820 | 0.9019 | 32 | 16 | electronic local |
| 8 | damped symplectic | **0.8396** | 0.7969 to 0.8816 | 0.9011 | 16 | 16 | local environment |
| 9 | Gray-Scott | **0.8475** | 0.8037 to 0.8925 | 0.9135 | 125 | 16 | periodic electronic |
| 10 | conservative graph flux | **0.9090** | 0.8588 to 0.9611 | 0.9502 | 250 | 16 | comprehensive |

Gated residual and delayed memory are statistically indistinguishable on this holdout: their point estimates differ by only 0.0002 and their bootstrap intervals overlap almost completely. Delayed memory has the lowest RMSE, while gated residual has the marginally lower primary metric. The blind challenge set remained sealed for every study.

The dynamical screens contain structured transient and recurrent regimes, including separated branches for delayed memory, Kuramoto-Sakaguchi, damped symplectic, activator inhibitor, and coupled map. These finite-time patterns are candidates for longer dynamical analysis rather than evidence of strange attractors or sustained chaos. Perturbation confirmation remained deferred because the Windows CUDA perturbation path was guarded after a native runtime failure, and dynamical-interest measures never influenced model selection.

### 1 of 10: gated residual complete

Gated residual completed in 61.8 minutes. Its sealed-holdout bootstrapped MA-ST-RAE is **0.8073** (95% CI 0.7610 to 0.8531), with point MA-ST-RAE 0.8030 and RMSE 0.8774. Endpoint ST-RAE values are CYP1A2 0.8257, CYP2C9 0.8028, CYP2D6 0.9767, and CYP3A4 0.6239. The winner uses 64 generations, 16 dynamical channels, and the local-environment atom profile. All 1,309 holdout trajectories were screened; the recurrence-motion distribution forms several continuous bands, while Windows CUDA perturbation propagation remains guarded against a native runtime failure and no attractor or chaos claim is made.

### 2 of 10: inertial reaction diffusion complete

Inertial reaction diffusion completed in 103.4 minutes including resumed work. Its sealed-holdout bootstrapped MA-ST-RAE is **0.8186** (95% CI 0.7743 to 0.8641), with RMSE 0.8865. The winner uses 32 generations, 16 dynamical channels, the electronic atom profile, CA learning rate 0.003, and ridge penalty 1.0. Confirmation across five scaffold folds and three seeds gave MA-ST-RAE 0.8533. All 1,309 holdout trajectories were screened; recurrence is approximately 0.992 to 0.997 with persistent low-amplitude late motion and spectral entropy approximately 0.575 to 0.633. These are finite-time recurrent-transient candidates rather than confirmed attractors or chaos. The blinded set remained sealed.

### 3 of 10: activator inhibitor complete

Activator inhibitor completed in 93.0 minutes. Its sealed-holdout bootstrapped MA-ST-RAE is **0.8366** (95% CI 0.7936 to 0.8817), with RMSE 0.8987. Endpoint behaviour remains weakest for CYP2D6 and strongest for CYP3A4. The winner uses 64 generations, 16 dynamical channels, the valence-electronic atom profile, CA learning rate 0.001, and ridge penalty 1.0. Confirmation across five scaffold folds and three seeds gave MA-ST-RAE 0.8807 with seed SD 0.0023. Dynamical screening shows several recurrence-motion branches, including a lower-motion branch around recurrence 0.9 to 0.95, but no attractor or chaos claim is made. The blinded set remained sealed. PDF verification also identified and corrected an infinity/NaN confirmation-report aggregation defect before publication.

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
