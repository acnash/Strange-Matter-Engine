# CAT-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests learned channel-specific update rates within each recurrent
Graph-CA rule. A bounded sigmoid timescale is learned for every cellular-state
channel and applied to the proposed state change at each generation. The
initial value is close to the established full update, allowing training to
slow individual chemical-dynamical channels when a different recurrent pace
improves scaffold-safe prediction.

The bounded screen compares the established shared update timing with the
channel-adaptive mechanism for every selected endpoint-rule candidate across
two scaffold folds. Successful candidates then undergo the established
five-fold, two-seed confirmation and sealed-holdout protocol.

Atoms remain chemically encoded cellular-automaton cells, typed bonds govern
local graph interactions, and nonlinear recurrent rules produce explicit
successive generations. The campaign retains trajectories, backpropagates
through every recurrent step and the genuine differentiable ridge solve, uses
scaffold-safe selection, keeps the sealed holdout isolated, and excludes blind
labels. USR-CIA-EA-CV-CYP-GCA remains the sealed-validation leader at point
MA-ST-RAE 0.748039 and RMSE 0.845606 pIC50.
