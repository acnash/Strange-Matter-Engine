# Cross-Fitted Target-Specific Dual-Scale Graph Cellular Automata Ensemble v1

This production candidate replaces fixed averaging with four target-specific ridge stacks. Each stack receives ten predictions produced by the existing graph cellular automata: one original and one three-seed multiscale prediction from each of gated residual, delayed memory, inertial reaction diffusion, Kuramoto-Sakaguchi, and FitzHugh-Nagumo.

Model construction used five nested scaffold-grouped folds. For every outer-fold prediction, both the ridge penalty and the optional affine calibration were selected without access to that fold. The sealed validation holdout remained outside model fitting and selection. The challenge-blinded table was loaded only after the four final stacks were frozen, and it contained no activity or credible-interval columns.

## Validation results

| Metric | CFT-DS-GCAE v1 | Previous DS-GCAE v1 |
|---|---:|---:|
| Point MA-ST-RAE | **0.7739** | 0.7842 |
| Bootstrap MA-ST-RAE mean | **0.7749** | 0.7850 |
| 95% bootstrap interval | 0.7356 to 0.8193 | 0.7464 to 0.8274 |
| RMSE, pIC50 | **0.8586** | 0.8678 |
| Bootstrap macro MAE, pIC50 | **0.6323** | 0.6387 |
| Bootstrap macro R-squared | **0.2754** | 0.2706 |
| Bootstrap macro Spearman rho | **0.5184** | 0.5098 |
| Bootstrap macro Kendall tau | **0.3699** | 0.3626 |

| Endpoint | Point ST-RAE | Ridge penalty | Selected calibration |
|---|---:|---:|---|
| CYP1A2 | 0.8172 | 1000 | Identity |
| CYP2C9 | 0.7611 | 100 | Identity |
| CYP2D6 | 0.9409 | 1000 | Identity |
| CYP3A4 | 0.5764 | 100 | Identity |

Calibration was searched rather than assumed. Cross-validation selected identity calibration for all four final stacks, indicating that an additional scale or offset correction was unsupported once the target-specific ridge stacks were fitted.

## Files

- `cft_ds_gcae_submission.csv`: challenge-formatted regression predictions for 750 blinded molecules.
- `cft_ds_gcae_blinded_predictions_long.csv`: 3,000 auditable endpoint predictions with all ten expert inputs.
- `study_summary.json`: nested-fold settings, fitted ridge states, validation results, and 1,000-resample bootstrap metrics.
- `validation_predictions.csv`: sealed validation predictions and member signals.
- `inference_manifest.json`: frozen inference specification and leakage safeguards.

The validation improvement is modest and its bootstrap interval overlaps that of DS-GCAE v1. The challenge result for DS-GCAE showed a substantial local-to-blind generalisation gap, so CFT-DS-GCAE should be regarded as a better validated candidate rather than a guaranteed leaderboard improvement.
