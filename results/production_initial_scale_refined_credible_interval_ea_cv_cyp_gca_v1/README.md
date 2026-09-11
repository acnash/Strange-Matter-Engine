# ISR-CIA-EA-CV-CYP-GCA v1 production campaign

This campaign tunes the amplitude with which chemical atom descriptors seed the
initial cellular state. It tests 0.75, 1.00, and 1.25 times every selected
rule's established initialization scale, changing the starting magnitude of
the encoded molecular state before recurrent evolution begins.

Atoms remain chemically encoded cells and typed bonds govern local graph
interactions. The campaign preserves explicit recurrent generations, retained
nonlinear trajectories, backpropagation, genuine differentiable ridge
regression, scaffold-safe selection, sealed-holdout isolation, and blind-label
exclusion. USR-CIA-EA-CV-CYP-GCA is the sealed-validation leader at point
MA-ST-RAE 0.748039 and RMSE 0.845606 pIC50. CIA-EA-CV-CYP-GCA remains the
external reference model for this refinement lineage.

## Completed result

The campaign completed on 11 September 2026 with sealed point MA-ST-RAE
**0.752851** and RMSE **0.851661 pIC50**. This was 0.004813 MA-ST-RAE and
0.006055 pIC50 RMSE above the USR sealed leader, so ISR remains an evaluated
experimental model and the leader is unchanged. The 1,000-resample bootstrap
mean MA-ST-RAE was 0.753324 with a 95% interval of 0.715505 to 0.793792.

The selected initial-state multipliers were rule and endpoint dependent:

- CYP1A2: Gray-Scott 0.75x, FitzHugh-Nagumo 1.00x, conservative graph flux 1.00x.
- CYP2C9: Gray-Scott 1.25x, damped symplectic 1.00x, conservative graph flux 0.75x.
- CYP2D6: delayed memory 1.25x, inertial reaction-diffusion 0.75x, FitzHugh-Nagumo 1.25x.
- CYP3A4: delayed memory 0.75x, damped symplectic 1.25x, FitzHugh-Nagumo 0.75x.

The sealed holdout remained excluded from model and hyperparameter selection,
and blind labels were unavailable throughout training and evaluation.
