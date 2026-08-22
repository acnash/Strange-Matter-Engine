# Graph-CA visual prototype results

This is one fixed-design scientific prototype, trained with seed 1701. It is an exploratory run rather than the final nested-cross-validation estimate.

## Result snapshot

- Fit observations: 5216
- Grouped-validation observations: 1309
- Restored fit RMSE: 0.906 pIC50
- Restored grouped-validation RMSE: 0.965 pIC50
- Epochs run: 5
- Blinded predictions: 750 molecules × four CYPs
- Visual trajectories: 20 molecule–CYP cases selected from dynamical screening

## PyMOL

Open PyMOL, choose **File → Run Script**, and select `load_20_trajectories.pml` from this directory. All 20 objects appear in the right-hand object panel; the first is enabled. Enable one desired object and disable the previous one.

The supplied controller does not use PyMOL's movie subsystem. Enter `gca_next`, `gca_previous`, `gca_state 251`, or `gca_play` in the PyMOL command line. `gca_play 0.25, 2` uses a 0.25-second delay and plays two cycles; `gca_stop` stops playback.

Playback runs in the background so PyMOL can repaint between states. Only the currently enabled trajectory object is recoloured, which keeps display-memory use modest.

Before recolouring, the controller explicitly installs the selected generation's activity values into the enabled object's B-factor field. This preserves the true 501-state gradient even in PyMOL versions that treat a multi-model PDB's B-factor as one shared atom property.

States 1–501 are graph-CA generations 0–500. State 502 is the labelled visual coda: display-only hydrogens become lime using the final heavy-atom activity. The model never received 3D coordinates or hydrogen nodes.

The PDB B-factor column stores scaled eight-channel atom-state magnitude. It is unrelated to the learned ridge coefficient beta. Lossless eight-channel values are in the matching NPZ files.

## Figures

- `01_learning_curve.png`: fit and grouped-validation error over training.
- `02_prediction_scatter.png`: predictions against measurements; the diagonal is perfect agreement.
- `03_residual_distributions.png`: signed error by CYP.
- `04_atom_activity_heatmap.png`: atom-by-generation activity for one example.
- `05_hyperparameter_screen.png`: validation error for the six tested configurations.
- `06_prototype_rmse_comparison.png`: grouped-validation RMSE across Prototypes 1–4.
- `07_per_cyp_rmse.png`: grouped-validation RMSE for each CYP target.
- `08_blinded_prediction_comparison.png`: Prototype 1 and Prototype 4 predictions for the blinded challenge set.
- `09_dynamical_screening.png`: recurrence and late-motion screening across 3,000 molecule–CYP cases.
- `10_persistent_transients.png`: late step sizes for the five most persistent selected trajectories.
- `11_finite_time_lyapunov.png`: perturbation-response slopes for the selected trajectories.
- `12_state_space_projections.png`: two-dimensional projections of four persistent trajectories.

## Tables and report

- `SCIENTIFIC_REPORT.md`: interpretation of predictive performance and trajectory dynamics.
- `validation_set_predictions.csv`: the 1,309 validation observations, with experimental pIC50, predicted pIC50, and residual.
- `validation_predictions.csv`: all 6,525 labelled observations, marked as either fitting or validation observations.
- `blinded_test_predictions.csv`: 3,000 predictions for 750 challenge molecules and four CYP targets. Experimental values are not available for this blinded set.
