# Graph-CA visual prototype results

This is one fixed-design scientific prototype, trained with seed 1701. It is an exploratory run rather than the final nested-cross-validation estimate.

## Result snapshot

- Fit observations: 5216
- Grouped-validation observations: 1309
- Restored fit RMSE: 0.808 pIC50
- Restored grouped-validation RMSE: 0.869 pIC50
- Epochs run: 65
- Blinded predictions: 750 molecules × four CYPs
- Visual trajectories: five molecules × four CYPs = 20 objects

## PyMOL

Open PyMOL, choose **File → Run Script**, and select `load_20_trajectories.pml` from this directory. All 20 objects appear in the right-hand object panel; the first is enabled. Enable one desired object and disable the previous one, then use the **movie playback controls** at the bottom of PyMOL.

Use the movie Play, Previous-frame, and Next-frame buttons. The ordinary object-state selector changes coordinates but does not execute the per-frame recolouring commands.

States 1–17 are graph-CA generations 0–16. State 18 is the labelled visual coda: display-only hydrogens become lime using the final heavy-atom activity. The model never received 3D coordinates or hydrogen nodes.

The PDB B-factor column stores scaled eight-channel atom-state magnitude. It is unrelated to the learned ridge coefficient beta. Lossless eight-channel values are in the matching NPZ files.

## Figures

- `01_learning_curve.png`: fit and grouped-validation error over training.
- `02_prediction_scatter.png`: predictions against measurements; the diagonal is perfect agreement.
- `03_residual_distributions.png`: signed error by CYP.
- `04_atom_activity_heatmap.png`: atom-by-generation activity for one example.
