# TA-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests whether the predictive fingerprint benefits from learned
temporal weighting of the cellular-automata trajectory. It begins from the
CIA-EA-CV-CYP-GCA endpoint specialists and compares the existing fixed
multiscale checkpoint concatenation with a differentiable temporal-attention
summary. The attention variant learns five softmax weights across 12.5%, 25%,
50%, 75%, and 100% of the trajectory and retains both the weighted mean and
weighted temporal variance.

The weights are trained through the genuine differentiable ridge objective and
backpropagate through every retained Graph-CA generation. Scaffold-safe model
selection, the sealed holdout, complete trajectory retention, and blind-label
exclusion remain unchanged. The current leader is CIA-EA-CV-CYP-GCA, with
point MA-ST-RAE 0.748490 and RMSE 0.847738 pIC50.
