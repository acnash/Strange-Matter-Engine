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
