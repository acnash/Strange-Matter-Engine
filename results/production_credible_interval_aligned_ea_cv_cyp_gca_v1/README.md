# CIA-EA-CV-CYP-GCA v1 production campaign

This campaign refines Endpoint-Aligned Cross-Validated CYP-Specialist Graph
Cellular Automata by tuning the backpropagation objective against the challenge
credible intervals. It retains the bonded molecular graph, recurrent local
cellular-automata transitions, complete dynamical trajectories, and genuine
differentiable ridge readout.

For every transition-rule specialist selected by EA-CV-CYP-GCA, the screening
stage compares interval-loss weights 0.00, 0.25, 0.50, and 0.75. A weight of
zero reproduces endpoint-aligned MSE training. Positive weights combine MSE
with a smooth penalty for predictions lying outside the experimental credible
interval. Selection uses endpoint ST-RAE across two scaffold folds. The best
loss weight for every retained rule advances to five-fold, two-seed
confirmation and leakage-safe sparse ridge selection.

The reserved scaffold holdout remains sealed until configuration and rule
selection finish. The comparison baseline is EA-CV-CYP-GCA point MA-ST-RAE
0.754503 and RMSE 0.852280 pIC50. Blinded challenge labels remain unavailable
and play no role in this campaign.

## Completed result

The campaign completed on 2 September 2026. Reserved-holdout point MA-ST-RAE
was 0.748490 and RMSE was 0.847738 pIC50, improving both internal diagnostics
over EA-CV-CYP-GCA. The 1,000-resample bootstrap MA-ST-RAE was 0.749064 with a
95% interval from 0.710791 to 0.789424. Endpoint ST-RAE values were 0.816136
for CYP1A2, 0.713018 for CYP2C9, 0.936774 for CYP2D6, and 0.528032 for CYP3A4.
Blind inference completed without loading blind labels, and the validated
submission is `credible_interval_aligned_ea_cv_cyp_gca_submission.csv`.
