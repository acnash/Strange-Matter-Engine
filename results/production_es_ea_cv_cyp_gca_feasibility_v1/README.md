# ES-EA-CV-CYP-GCA feasibility campaign

Status: configured for GPU execution.

This bounded comparison changes the nonlinear optimizer while preserving the
EA-CV-CYP-GCA representation and validation design. It uses chemically encoded
atoms as cells, typed bonded interactions, 32 explicit recurrent generations,
retained multiscale trajectories, endpoint-specific training, an analytic
ridge readout for every candidate, and five scaffold-separated development
folds executed as concurrent CUDA workers.

The nonlinear Graph-CA parameters are optimized with 64 mirrored evolutionary
candidates per generation. Candidate evaluations run on CUDA with up to 1,024
molecules in the evolutionary batch. The initial comparison uses the canonical
CYP3A4 damped-symplectic expert configuration because it was the strongest
single endpoint expert in the submitted ensemble.

The sealed holdout and blind labels are excluded. EA-CV-CYP-GCA trained by
backpropagation through time remains the canonical model unless the complete
scaffold comparison establishes a defensible improvement.
