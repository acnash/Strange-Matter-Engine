# PC-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests initial-state perturbation consistency during recurrent
Graph-CA training. Each selected CYP specialist compares its established
objective with an additional penalty that encourages pIC50 predictions to
remain consistent after a small random perturbation of the initial atom states.
The cellular trajectory itself remains free to evolve nonlinearly.

The experiment compares consistency weights 0.00 and 0.05 at perturbation
magnitude 0.001. Atom cells, typed bonds, explicit generations, differentiable
ridge regression, scaffold-safe selection, sealed evaluation, and blind-label
exclusion remain unchanged. CIA-EA-CV-CYP-GCA remains the comparison leader at
point MA-ST-RAE 0.748490 and RMSE 0.847738 pIC50.

## Completed result

The campaign completed sealed evaluation on 4 September 2026. Point
MA-ST-RAE was 0.759449 and RMSE was 0.852318 pIC50. Endpoint point ST-RAE was
0.832041 for CYP1A2, 0.722042 for CYP2C9, 0.943665 for CYP2D6, and 0.540046
for CYP3A4. Screening selected a consistency weight of 0.00 for every retained
candidate, so the tested perturbation penalty provided no validation benefit.
The result was retained as an experimental record. CIA-EA-CV-CYP-GCA remains
the internal leader.
