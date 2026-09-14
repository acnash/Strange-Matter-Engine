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

The full campaign contains four genetic generations of 24 candidates. All 96
structures are screened on two scaffold folds under a 35-epoch maximum inner
training budget. Four leading candidates and the canonical CYP3A4
damped-symplectic reference advance to five-fold, two-seed confirmation under
an 80-epoch maximum budget. The campaign therefore plans 242 training runs and
uses five concurrent CUDA workers.

Development MA-ST-RAE is the genetic fitness and RMSE is the secondary
diagnostic. The sealed holdout and blind labels are excluded from genetic
fitness, parent selection, crossover, mutation, checkpoint selection, and
confirmation. A sealed evaluation will require a confirmed development result
that justifies opening the protected holdout.

After this campaign is completed, the next distinct proposed methodology is
genetic programming of the cellular update equations.
