# Graph-CA visual prototype results

This is one fixed-design scientific prototype, trained with seed 1701. It is an exploratory run rather than the final nested-cross-validation estimate.

## Result snapshot

- Fit observations: 5216
- Grouped-validation observations: 1309
- Restored fit RMSE: 0.835 pIC50
- Restored grouped-validation RMSE: 0.893 pIC50
- Epochs run: 49
- Blinded predictions: 750 molecules × four CYPs
- Visual trajectories: 20 molecule–CYP cases selected from dynamical screening

## PyMOL

Open PyMOL, choose **File → Run Script**, and select `load_20_trajectories.pml` from this directory. All 20 objects appear in the right-hand object panel; the first is enabled. Enable one desired object and disable the previous one.

The supplied controller does not use PyMOL's movie subsystem. Enter `gca_next`, `gca_previous`, `gca_state 51`, or `gca_play` in the PyMOL command line. `gca_play 0.25, 2` uses a 0.25-second delay and plays two cycles; `gca_stop` stops playback.

Playback runs in the background so PyMOL can repaint between states. Only the currently enabled trajectory object is recoloured, which keeps display-memory use modest.

Before recolouring, the controller explicitly installs the selected generation's activity values into the enabled object's B-factor field. This preserves the true 101-state gradient even in PyMOL versions that treat a multi-model PDB's B-factor as one shared atom property.

States 1–101 are graph-CA generations 0–100. State 102 is the labelled visual coda: display-only hydrogens become lime using the final heavy-atom activity. The model never received 3D coordinates or hydrogen nodes.

The PDB B-factor column stores scaled eight-channel atom-state magnitude. It is unrelated to the learned ridge coefficient beta. Lossless eight-channel values are in the matching NPZ files.

## Figures

- `01_learning_curve.png`: fit and grouped-validation error over training.
- `02_prediction_scatter.png`: predictions against measurements; the diagonal is perfect agreement.
- `03_residual_distributions.png`: signed error by CYP.
- `04_atom_activity_heatmap.png`: atom-by-generation activity for one example.
- `05_dynamical_screening.png`: all 3,000 trajectories in recurrence–late-motion space.
- `06_persistent_transients.png`: step-size decay for the five most persistent selected cases.

See [SCIENTIFIC_REPORT.md](SCIENTIFIC_REPORT.md) for the predictive and dynamical interpretation.
