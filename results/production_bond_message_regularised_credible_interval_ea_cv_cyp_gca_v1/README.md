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
