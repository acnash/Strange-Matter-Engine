# HWR-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests whether additional cellular-state capacity improves
generalisation. Each leading candidate is evaluated at its established hidden
width and at 1.5 times that width, while retaining its rule, generation count,
chemical atom encoding, typed-bond interactions, and trajectory readout.

The campaign preserves explicit recurrent generations, retained nonlinear
trajectories, backpropagation through the Graph-CA, genuine differentiable
ridge regression, scaffold-safe selection, sealed holdout isolation, and blind
label exclusion. CIA-EA-CV-CYP-GCA remains the comparison leader at point
MA-ST-RAE 0.748490 and RMSE 0.847738 pIC50.

## Completed result

The sealed scaffold holdout produced point MA-ST-RAE **0.751954** and RMSE
**0.851544 pIC50**. This trails the leader by 0.003464 MA-ST-RAE and 0.003806
pIC50 RMSE. Expanded cellular-state width was retained as an experimental
result and was not promoted. Blind-label exclusion remained intact.
