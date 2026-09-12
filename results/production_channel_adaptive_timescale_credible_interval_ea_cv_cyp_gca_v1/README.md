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

## Completed result

The campaign completed on 12 September 2026 with sealed point MA-ST-RAE
**0.749949** and RMSE **0.847707 pIC50**. This was 0.001910 MA-ST-RAE and
0.002101 pIC50 RMSE above the USR sealed leader, so CAT remains an evaluated
experimental model and the leader is unchanged. The 1,000-resample bootstrap
mean MA-ST-RAE was 0.750416 with a 95% interval of 0.712614 to 0.790865.

Adaptive channel timing was selected for Gray-Scott and FitzHugh-Nagumo on
CYP1A2, conservative graph flux on CYP2C9, and damped-symplectic dynamics on
CYP3A4. The CYP2D6 candidates all retained shared timing. The mixed selection
supports rule-specific recurrent pacing as a useful representation choice,
while the assembled sealed score remained slightly above the leader.

The sealed holdout remained excluded from model and hyperparameter selection,
and blind labels were unavailable throughout training and evaluation.
