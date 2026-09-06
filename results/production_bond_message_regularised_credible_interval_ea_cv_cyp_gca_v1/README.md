# BMR-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests low-rate bond-message regularisation during recurrent
training. Five percent of directed bond messages are stochastically masked in
the training trajectory and surviving messages are rescaled. Validation and
inference use the complete molecular graph. The objective is to prevent a
specialist from depending too heavily on individual message paths while
retaining chemically typed local interactions.

Screening compares dropout rates 0.00 and 0.05 for the leading
credible-interval-aligned candidates. Atoms remain cells, typed bonds define
the molecular neighbourhood, and all candidates retain explicit generations,
trajectory fingerprints, backpropagation, differentiable ridge regression,
scaffold-safe selection, sealed evaluation, and blind-label exclusion.
CIA-EA-CV-CYP-GCA is the comparison leader at point MA-ST-RAE 0.748490 and
RMSE 0.847738 pIC50.

## Completed result

The campaign completed sealed evaluation on 6 September 2026. Point
MA-ST-RAE was 0.755994 and RMSE was 0.852721 pIC50. Endpoint point ST-RAE was
0.816727 for CYP1A2, 0.725517 for CYP2C9, 0.942684 for CYP2D6, and 0.539048
for CYP3A4. The regularised candidates did not improve the common sealed
result, so CIA-EA-CV-CYP-GCA remains the internal leader.
