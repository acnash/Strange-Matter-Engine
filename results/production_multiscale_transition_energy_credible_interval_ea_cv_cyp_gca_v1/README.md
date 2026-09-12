# MTE-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests a temporally localized nonlinear observable derived from
the retained Graph-CA trajectory. For the five established recurrent
checkpoints, it augments the molecular fingerprint with the channel-wise
squared change between each pair of consecutive checkpoints. These four
multiscale transition-energy blocks expose when cellular-state motion occurs,
rather than reducing all state-change energy to one trajectory-wide average.

The bounded screen compares the established fingerprint with and without the
new transition-energy blocks for every selected endpoint-rule candidate across
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
**0.756595** and RMSE **0.846761 pIC50**. This was 0.008556 MA-ST-RAE and
0.001155 pIC50 RMSE above the USR sealed leader, so MTE remains an evaluated
experimental model and the leader is unchanged. The 1,000-resample bootstrap
mean MA-ST-RAE was 0.757211 with a 95% interval of 0.714863 to 0.800546.

The transition-energy features were selected for FitzHugh-Nagumo and
conservative graph-flux specialists on CYP1A2, FitzHugh-Nagumo on CYP2D6, and
damped-symplectic and FitzHugh-Nagumo specialists on CYP3A4. The CYP2C9 screen
retained the established fingerprint for all three candidates. This mixed
selection shows that temporally localized motion carried useful endpoint-rule
specific signal, although the assembled sealed model did not improve the
primary metric.

The sealed holdout remained excluded from model and hyperparameter selection,
and blind labels were unavailable throughout training and evaluation.
