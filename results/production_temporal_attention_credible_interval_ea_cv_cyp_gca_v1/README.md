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

## Completed result

The campaign completed sealed evaluation on 3 September 2026. Point MA-ST-RAE
was 0.755021 and RMSE was 0.850082 pIC50. Learned temporal attention was
selected for four of the twelve confirmed rule specialists, although the final
combination remained weaker than CIA-EA-CV-CYP-GCA. Endpoint point ST-RAE was
0.821358 for CYP1A2, 0.718054 for CYP2C9, 0.943648 for CYP2D6, and 0.537022
for CYP3A4. The model was retained as an experimental result and was not
promoted as the submission leader.
