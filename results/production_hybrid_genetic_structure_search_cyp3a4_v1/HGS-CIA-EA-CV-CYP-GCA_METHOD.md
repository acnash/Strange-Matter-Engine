# HGS-CIA-EA-CV-CYP-GCA method report

## Submission identity

- Method: HGS-CIA-EA-CV-CYP-GCA
- Task: CYP1A2, CYP2C9, CYP2D6, and CYP3A4 direct-inhibition pIC50 prediction
- Submission rows: 750 blinded molecules
- Predictions: 3,000 finite values
- Blind labels loaded: no

## Molecular Graph-CA

Each molecule is an encoded graph cellular automaton in which atoms are cells
and typed covalent bonds define local neighbourhoods. Learned nonlinear local
updates generate explicit successive cellular generations. The complete
trajectory is retained and pooled into a molecular fingerprint. A genuine
differentiable closed-form ridge solve maps these fingerprints to pIC50 during
support-query training, so prediction loss propagates through the ridge
solution and every recurrent generation.

The underlying cellular propagation equations and transition-rule
implementations are unchanged. The hybrid genetic procedure selects among
existing, validated Graph-CA components.

## Hybrid genetic structure search

The genetic search was restricted to CYP3A4. A genome encoded the transition
rule, recurrent depth, hidden width, atom-feature profile, trajectory pooling,
degree normalization, chemical-feature gating, initial-state anchoring,
channel-adaptive timescales, and optional retained-trajectory summaries.
Continuous weights for every candidate were trained by backpropagation through
time, which makes this a genetic architecture search rather than a
derivative-free replacement for weight optimization.

The campaign used four generations of 24 unique structures. Tournament
selection used groups of three, crossover was uniform, and every gene had
mutation probability 0.25. Each of the 96 structures was screened on two
scaffold folds with a maximum of 35 epochs. The four leading evolved genomes
and the canonical reference were confirmed over five scaffold folds and two
seeds with a maximum of 80 epochs. The complete design comprised 192 screening
runs and 50 confirmation runs, for 242 successful CUDA runs.

MA-ST-RAE was the genetic fitness and RMSE was the secondary diagnostic. The
reserved scaffold holdout and blind labels were excluded from fitness, parent
selection, crossover, mutation, early stopping, and confirmation.

## Selected CYP3A4 genome

| Component | Selected value |
|---|---|
| Candidate | `g01_c00` |
| Transition rule | FitzHugh-Nagumo |
| Recurrent generations | 128 |
| Hidden channels | 24 |
| Atom features | Comprehensive chemical profile |
| Trajectory pooling | Multiscale |
| Degree normalization power | 1.0 |
| Dynamic observables | Enabled |
| Multiscale transition energy | Enabled |
| Chemical-feature gating | Disabled |
| Initial-state anchoring | Disabled |
| Channel-adaptive timescale | Disabled |
| Multi-lag recurrence signature | Disabled |
| Temporal-extrema signature | Disabled |
| Directional-flux signature | Disabled |
| Graph-CA learning rate | 0.003 |
| Ridge penalty | 0.01 |
| Update scale | 0.08 |
| Initial-state scale | 0.5 |
| Initial noise | 0.005 |
| Support fraction | 0.75 |
| Batch molecules | 64 |
| Gradient clipping | 0.5 |
| Graph-CA L2 penalty | 0.0001 |

## Development confirmation

Across five scaffold folds and two seeds, the selected genome achieved mean
MA-ST-RAE 0.596981 with sample standard deviation 0.042987 and range 0.519109
to 0.651933. Mean RMSE was 0.781116 pIC50 with sample standard deviation
0.035890 and range 0.730928 to 0.824233 pIC50. The canonical genome achieved
mean MA-ST-RAE 0.638936 under the same confirmation design. The genetic winner
therefore reduced development MA-ST-RAE by 0.041955, or 6.6%.

## Frozen submission composition

The final submission retains the frozen CIA-EA-CV-CYP-GCA predictions for
CYP1A2, CYP2C9, and CYP2D6. CYP3A4 is predicted by averaging the ten frozen
`g01_c00` checkpoints from five scaffold folds and two seeds. No leaderboard
labels or other blind outcomes enter this combination.

## Sealed holdout result

| Metric | CIA-EA-CV-CYP-GCA | HGS-CIA-EA-CV-CYP-GCA |
|---|---:|---:|
| Point MA-ST-RAE | 0.748490 | **0.743723** |
| RMSE, pIC50 | 0.847738 | **0.841572** |
| CYP1A2 point ST-RAE | 0.816136 | 0.816136 |
| CYP2C9 point ST-RAE | 0.713018 | 0.713018 |
| CYP2D6 point ST-RAE | 0.936774 | 0.936774 |
| CYP3A4 point ST-RAE | 0.528032 | **0.508964** |

For HGS-CIA-EA-CV-CYP-GCA, 1,000 bootstrap resamples gave mean MA-ST-RAE
0.744400 with a 95% interval from 0.705821 to 0.785104. Bootstrap macro MAE
was 0.612397 pIC50, macro R-squared was 0.299458, macro Spearman rho was
0.542811, and macro Kendall tau was 0.390396.

The sealed holdout was opened once after the genetic winner and its checkpoints
had been frozen. The prepared blind file has passed row count, column schema,
unique molecule, and finite-prediction checks. External leaderboard metrics
remain pending.
