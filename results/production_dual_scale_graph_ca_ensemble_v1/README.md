# DS-GCAE v1 production result

This directory contains the frozen blinded-test inference output for the **Dual-Scale Graph Cellular Automata Ensemble (DS-GCAE v1)**, the current Strange Matter Engine direct-inhibition submission model.

## Model composition

DS-GCAE combines five graph cellular-automata transition rules: gated residual, delayed memory, inertial reaction diffusion, Kuramoto-Sakaguchi, and FitzHugh-Nagumo. Every member independently evolves atom states across the bonded molecular graph, pools its trajectory, and produces four CYP-specific pIC50 values through its fitted ridge readout.

The final prediction is a global blend of:

- 42.5% original ensemble, using one seed per rule and equal 0.2 rule weights;
- 57.5% multiscale ensemble, averaging seeds 1701, 2909, and 4211 within each rule and using rule weights 0.1036, 0.1999, 0.1966, 0.2447, and 0.2552.

## Validation

| Metric | Result |
|---|---:|
| Point MA-ST-RAE | 0.784156 |
| Bootstrap MA-ST-RAE mean | 0.784989 |
| 95% bootstrap interval | 0.746381 to 0.827386 |
| RMSE | 0.867775 pIC50 |

## Files

- `ds_gcae_submission.csv`: challenge-ready 750-row regression submission in the official six-column order.
- `ds_gcae_blinded_predictions_long.csv`: auditable long-form predictions showing the original, multiscale, and final blended value for each molecule and CYP.
- `inference_manifest.json`: model identity, blend weights, seeds, checkpoint paths, device, validation flags, and elapsed inference time.
- `progress.json`: completed-run status and a copy of the final inference manifest.
- `inference_members/`: per-checkpoint blind predictions and console logs used to reconstruct the final ensemble.

The run generated 3,000 finite predictions for 750 molecules and four CYP isoforms. It did not load blinded labels. Model training, hyperparameter selection, ensemble weighting, and validation were completed before blind inference.
