# TES-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests a differentiable temporal-extrema signature derived from
the retained Graph-CA trajectory. Smooth channel-wise maxima and minima expose
transient peaks and troughs in molecular cellular states across the recurrent
horizon. The log-sum-exp construction preserves gradient flow through every
generation while avoiding a discontinuous hard-extrema operator.

The bounded screen compares the established fingerprint with and without the
temporal-extrema signature for every selected endpoint-rule candidate across
two scaffold folds. Successful candidates then undergo the established
five-fold, two-seed confirmation and sealed-holdout protocol.

Atoms remain chemically encoded cellular-automaton cells, typed bonds govern
local graph interactions, and nonlinear recurrent rules produce explicit
successive generations. The campaign retains trajectories, backpropagates
through every recurrent step and the genuine differentiable ridge solve, uses
scaffold-safe selection, keeps the sealed holdout isolated, and excludes blind
labels. USR-CIA-EA-CV-CYP-GCA remains the sealed-validation leader at point
MA-ST-RAE 0.748039 and RMSE 0.845606 pIC50.

## Result

The sealed scaffold holdout produced point MA-ST-RAE **0.749176** and RMSE
**0.846845 pIC50**. The bootstrap mean MA-ST-RAE was **0.749744**, with a
95% interval of **0.709536 to 0.792656** across 1,000 resamples. The point
score was 0.001137 above the USR sealed-validation leader, so the temporal
extrema signature was retained as an evaluated model without promotion.

The selected ensemble used the temporal-extrema signature for both CYP1A2
members, one CYP2C9 member, both CYP3A4 members, and one CYP2D6 member. This
shows that smooth recurrent peaks and troughs carried useful endpoint-specific
signal, while their aggregate sealed effect remained slightly weaker than the
leader. The reserved holdout remained excluded from selection and blind labels
were never loaded.
