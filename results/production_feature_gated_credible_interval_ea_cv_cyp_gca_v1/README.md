# FG-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tests endpoint-specific learnable gates over the atomic chemical
descriptor vector. Each CYP specialist compares the existing direct descriptor
input with a gated representation in which differentiable sigmoid weights
control how strongly every atomic property enters Graph-CA initialization and
each recurrent local update.

The gates are trained jointly with the nonlinear transition rule through the
differentiable ridge objective. Atom cells, typed-bond neighbourhoods, explicit
generations, retained trajectories, scaffold-safe selection, the sealed
holdout, and blind-label exclusion remain unchanged. The comparison leader is
CIA-EA-CV-CYP-GCA at point MA-ST-RAE 0.748490 and RMSE 0.847738 pIC50.
