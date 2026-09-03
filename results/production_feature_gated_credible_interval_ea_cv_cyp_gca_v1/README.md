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

## Completed result

The campaign completed sealed evaluation on 3 September 2026. Point MA-ST-RAE
was 0.751384 and RMSE was 0.848686 pIC50. Endpoint point ST-RAE was 0.818467
for CYP1A2, 0.718155 for CYP2C9, 0.943620 for CYP2D6, and 0.525295 for
CYP3A4. The result remained close to the leader and improved CYP3A4 ST-RAE,
while the macro score remained weaker than CIA-EA-CV-CYP-GCA. The model was
retained as an experimental result and was not promoted.
