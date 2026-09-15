# GP-CIA-EA-CV-CYP-GCA method report

## Submission identity

- Method: GP-CIA-EA-CV-CYP-GCA
- Task: CYP1A2, CYP2C9, CYP2D6, and CYP3A4 direct-inhibition pIC50 prediction
- Submission rows: 750 blinded molecules
- Predictions: 3,000 finite values
- Blind labels loaded: no

## Molecular Graph-CA

Each molecule is an encoded graph cellular automaton in which chemically
encoded atoms are cells and typed covalent bonds define local neighbourhoods.
Learned nonlinear local updates produce 128 explicit successive cellular
generations. Retained trajectory states are pooled into a molecular
fingerprint using multiscale pooling, dynamic observables, and multiscale
transition-energy summaries.

A genuine differentiable closed-form ridge solve maps the fingerprints to
pIC50 during support-query training. Prediction loss is backpropagated through
the ridge solution and the full recurrent cellular trajectory, providing
backpropagation through time for the continuous Graph-CA parameters.

## Genetic programming search

The genetic-programming campaign searched the symbolic form of the CYP3A4
cellular update drive while holding the remaining Graph-CA configuration
fixed. Program terminals comprised the learned reaction signal, current cell
state, typed-bond neighbour-state difference, and delayed-state difference.
The bounded differentiable primitive set comprised addition, subtraction,
multiplication, averaging, negation, hyperbolic tangent, sine, and fixed
scaling.

The search used four generations of 24 unique symbolic programs. Tournament
selection used groups of three, crossover exchanged subtrees, and subtree
mutation probability was 0.35. Tree depth was limited to four, and fitness
included a parsimony penalty of 0.0005 per node. Each of the 96 programs was
screened on two scaffold folds with a maximum of 35 epochs. Five finalists
were confirmed over five scaffold folds and two seeds with a maximum of 80
epochs. The complete campaign comprised 192 screening runs and 50
confirmation runs, for 242 successful CUDA runs.

MA-ST-RAE was the primary genetic fitness and RMSE was the secondary
diagnostic. The reserved scaffold holdout and blinded challenge labels were
excluded from fitness, parent selection, crossover, mutation, early stopping,
and confirmation.

## Fixed Graph-CA configuration

| Component | Fixed value |
|---|---|
| CYP specialist | CYP3A4 endpoint-only objective |
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
| Graph-CA learning rate | 0.003 |
| Ridge penalty | 0.01 |
| Update scale | 0.08 |
| Initial-state scale | 0.5 |
| Initial noise | 0.005 |
| Support fraction | 0.75 |
| Batch molecules | 64 |
| Gradient clipping | 0.5 |
| Graph-CA L2 penalty | 0.0001 |

## Selected symbolic program

Candidate `g00_c02`, introduced in genetic generation zero, was the confirmed
winner. Its update drive is:

```text
mean(reaction, neighbour_delta)
```

This program averages the learned intracellular reaction signal with the
typed-bond neighbourhood difference before applying the fixed recurrent update
scale. It has three nodes and depth one.

## Development confirmation

Across five scaffold folds and two seeds, the selected program achieved mean
MA-ST-RAE 0.598075 and mean RMSE 0.772706 pIC50. The HGS development reference
under the same fold-and-seed design achieved MA-ST-RAE 0.596981 and RMSE
0.781116 pIC50. The GP program was 0.001094 higher on MA-ST-RAE and 0.008410
pIC50 lower on RMSE.

## Frozen submission composition

The final submission retains the frozen CIA-EA-CV-CYP-GCA predictions for
CYP1A2, CYP2C9, and CYP2D6. CYP3A4 is predicted by averaging the ten frozen
`g00_c02` checkpoints from five scaffold folds and two seeds. No leaderboard
labels or other blind outcomes enter this combination.

## Sealed holdout result

| Metric | CIA-EA-CV-CYP-GCA | HGS-CIA-EA-CV-CYP-GCA | GP-CIA-EA-CV-CYP-GCA |
|---|---:|---:|---:|
| Point MA-ST-RAE | 0.748490 | **0.743723** | 0.745824 |
| RMSE, pIC50 | 0.847738 | **0.841572** | 0.848265 |
| CYP1A2 point ST-RAE | 0.816136 | 0.816136 | 0.816136 |
| CYP2C9 point ST-RAE | 0.713018 | 0.713018 | 0.713018 |
| CYP2D6 point ST-RAE | 0.936774 | 0.936774 | 0.936774 |
| CYP3A4 point ST-RAE | 0.528032 | **0.508964** | 0.517368 |

For GP-CIA-EA-CV-CYP-GCA, 1,000 bootstrap resamples gave mean MA-ST-RAE
0.746602 with a 95% interval from 0.708890 to 0.788464. Bootstrap macro MAE
was 0.614381 pIC50, macro R-squared was 0.292879, macro Spearman rho was
0.542557, and macro Kendall tau was 0.390626.

The sealed holdout was opened once after program selection and checkpoint
freezing. The resulting point MA-ST-RAE improves on CIA-EA-CV-CYP-GCA by
0.002666 and remains 0.002101 above HGS-CIA-EA-CV-CYP-GCA. The prepared blind
file passed row-count, column-schema, unique-molecule, and finite-prediction
checks.
