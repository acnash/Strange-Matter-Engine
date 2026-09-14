# Evolutionary optimizer tuning with fixed Graph-CA

This campaign tunes the derivative-free optimizer while holding the molecular
Graph-CA configuration fixed to the canonical CYP3A4 damped-symplectic expert.
The fixed model uses 16 hidden channels, 32 recurrent generations, multiscale
trajectory pooling, periodic-electronic atom features, typed bond messages,
and an analytic ridge readout with penalty 0.1.

The screen evaluates 32 optimizer configurations on scaffold folds 0 and 1.
The search covers population sizes 32 to 128, perturbation scales 0.005 to
0.08, evolutionary learning rates 0.0005 to 0.01, batches of 256 to 1,600
molecules, random and activity-stratified support/query composition, and four
stopping policies. The five strongest configurations advance to five-fold,
two-seed confirmation. Five CUDA workers execute concurrently.

Selection uses development-fold MA-ST-RAE, with RMSE retained as a secondary
diagnostic. The sealed holdout and blind labels are excluded throughout. The
backpropagation-trained EA-CV-CYP-GCA remains canonical during the campaign.

## Completed result

The campaign completed all 114 planned runs on 13 September 2026: 64
two-fold screening runs across 32 optimizer configurations and 50 five-fold,
two-seed confirmation runs across five finalists. No run used the sealed
holdout or blind labels.

Configuration 24 was the confirmation winner. It used a population of 128,
perturbation scale 0.02, evolutionary learning rate 0.01, batches of 256
molecules with activity-stratified composition, and the patient stopping
policy. Across its ten confirmation runs, mean MA-ST-RAE was **0.659505**
with sample standard deviation **0.043028** and range **0.598289 to
0.722891**. Mean RMSE was **0.817975 pIC50**, with sample standard deviation
**0.042879**.

For the matched two-fold screen, evolutionary configuration 04 produced the
best mean MA-ST-RAE, **0.638768**, and mean RMSE **0.803659 pIC50**. The fixed
backpropagation-trained CYP3A4 damped-symplectic Graph-CA had recorded
MA-ST-RAE **0.626897** on the same two screening folds. Evolutionary training
was therefore 0.011871 MA-ST-RAE, or 1.9%, higher on the matched comparison.
The broader confirmation ranking selected configuration 24 and reinforced
the decision to retain backpropagation-trained EA-CV-CYP-GCA as the canonical
model. The evolutionary model was not advanced to sealed or blind evaluation.

Compact campaign records are stored in `optimizer_design.json`,
`screen_ranking.csv`, `screen_ranking.json`, `confirmation_ranking.csv`,
`confirmation_ranking.json`, and `progress.json`. The implementation and GPU
launcher are in `scripts/run_graph_ca_visual_prototype.py`,
`scripts/run_evolutionary_optimizer_campaign.py`, and
`scripts/start_evolutionary_optimizer_campaign.ps1`.
