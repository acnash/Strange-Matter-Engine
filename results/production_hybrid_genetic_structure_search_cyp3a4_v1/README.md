# Hybrid genetic Graph-CA structure search for CYP3A4

This campaign uses a genetic algorithm to evolve discrete structures for the
CYP3A4 encoded graph cellular automaton. Atoms remain chemically encoded cells,
typed bonds remain local neighbourhoods, and each candidate retains explicit
recurrent generations and its nonlinear trajectory. Every candidate's
continuous weights are trained through the same bounded backpropagation budget,
and the readout is the genuine differentiable closed-form ridge solve.

The genetic genome controls the transition-rule family, recurrent depth,
hidden-state width, atom-feature profile, trajectory pooling, degree
normalization, chemical-feature gating, initial-state anchoring,
channel-adaptive timescales, and optional trajectory observables. Numerical
settings for damped-symplectic, FitzHugh-Nagumo, and delayed-memory rules follow
their established CYP3A4 profiles, isolating the structural search from an
additional continuous-optimizer search.

The full campaign contained four genetic generations of 24 candidates. All 96
structures were screened on two scaffold folds under a 35-epoch maximum inner
training budget. Four leading candidates and the canonical CYP3A4
damped-symplectic reference advanced to five-fold, two-seed confirmation under
an 80-epoch maximum budget. The campaign therefore completed 242 training runs
with five concurrent CUDA workers.

Development MA-ST-RAE is the genetic fitness and RMSE is the secondary
diagnostic. The sealed holdout and blind labels are excluded from genetic
fitness, parent selection, crossover, mutation, checkpoint selection, and
confirmation. The protected holdout was opened once after the confirmed
development result justified final evaluation and the winner was frozen.

## Completed result

All 242 planned runs completed successfully. The selected genome was `g01_c00`,
which used the FitzHugh-Nagumo rule, 128 recurrent generations, 24 hidden
channels, comprehensive atom features, multiscale trajectory pooling, unit
degree normalization, dynamic observables, and multiscale transition-energy
summaries. Across five folds and two seeds it achieved mean development
MA-ST-RAE **0.596981 ± 0.042987** and mean RMSE **0.781116 ± 0.035890 pIC50**.
The canonical confirmation reference achieved MA-ST-RAE 0.638936.

The frozen ten-checkpoint CYP3A4 mean was combined with the established
CIA-EA-CV-CYP-GCA predictions for CYP1A2, CYP2C9, and CYP2D6. On the protected
holdout this HGS-CIA-EA-CV-CYP-GCA system achieved point MA-ST-RAE **0.743723**
and RMSE **0.841572 pIC50**. The CIA reference values were 0.748490 and
0.847738 pIC50. CYP3A4 point ST-RAE improved from 0.528032 to 0.508964.

The blinded submission contains 750 rows and 3,000 finite endpoint
predictions. Blind labels were never loaded. The next distinct proposed
methodology is genetic programming of the cellular update equations.
