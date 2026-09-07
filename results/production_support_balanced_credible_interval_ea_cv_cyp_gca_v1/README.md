# SBR-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tunes the support/query allocation used by the differentiable
ridge readout during recurrent Graph-CA training. It compares support fractions
0.60, 0.75, and 0.85, allowing each CYP specialist to balance a stable ridge
fit against the number of held-out query observations supplying gradients.

Atoms remain cells with chemical state, typed bonds define local interactions,
and every candidate retains explicit recurrent generations, nonlinear trajectory
fingerprints, backpropagation, genuine differentiable ridge regression,
scaffold-safe selection, sealed evaluation, and blind-label exclusion.

CIA-EA-CV-CYP-GCA remains the comparison leader at sealed point MA-ST-RAE
0.748490 and RMSE 0.847738 pIC50.

## Completed result

The sealed scaffold holdout produced point MA-ST-RAE **0.755412** and RMSE
**0.852409 pIC50**. This trails the leader by 0.006922 MA-ST-RAE and 0.004671
pIC50 RMSE. The 0.75 support fraction remained strongest for several leading
candidates, so support/query rebalancing was not promoted.
