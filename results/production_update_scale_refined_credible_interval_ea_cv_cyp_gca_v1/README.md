# USR-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tunes the magnitude of each recurrent cellular state transition.
It tests 0.75, 1.00, and 1.25 times every selected rule's established update
scale, changing the speed of nonlinear molecular evolution while retaining the
same explicit generation sequence and trajectory representation.

Atoms remain chemically encoded cells and typed bonds govern local graph
interactions. The campaign preserves backpropagation through recurrent states,
genuine differentiable ridge regression, scaffold-safe selection, sealed
holdout isolation, and blind-label exclusion. CIA-EA-CV-CYP-GCA remains the
comparison leader at point MA-ST-RAE 0.748490 and RMSE 0.847738 pIC50.

## Completed result

The sealed scaffold holdout produced point MA-ST-RAE **0.748039** and RMSE
**0.845606 pIC50**. This improves the preceding CIA-EA-CV-CYP-GCA leader by
0.000451 MA-ST-RAE and 0.002132 pIC50 RMSE. Across 1,000 bootstrap resamples,
mean MA-ST-RAE was 0.748534 with a 95% interval from 0.709822 to 0.789700.
The complementary bootstrap macro metrics were MAE 0.617964 pIC50, R-squared
0.294774, Spearman rho 0.534683, and Kendall tau 0.383997.

The selected endpoint systems used update-scale multipliers of 0.75, 1.00,
or 1.25 relative to their established recurrent rules. The final sparse
ensembles produced sealed ST-RAE values of 0.818155 for CYP1A2, 0.709644 for
CYP2C9, 0.943724 for CYP2D6, and 0.520633 for CYP3A4. Blind inference completed
from the frozen selected systems without loading blind labels. The method is
promoted as the internal leader under the abbreviation
USR-CIA-EA-CV-CYP-GCA.
