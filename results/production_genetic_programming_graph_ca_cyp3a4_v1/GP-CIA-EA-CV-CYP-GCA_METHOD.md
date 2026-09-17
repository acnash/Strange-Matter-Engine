# GP-CIA-EA-CV-CYP-GCA method report

## Submission identity

- Method: GP-CIA-EA-CV-CYP-GCA
- Task: CYP1A2, CYP2C9, CYP2D6, and CYP3A4 direct-inhibition pIC50 prediction
- Submission rows: 750 blinded molecules
- Predictions: 3,000 finite values
- Blind labels loaded: no

## Molecular Graph-CA

Each molecule is treated as a cellular automaton (CA) defined on its molecular
graph. An atom is one cell, a typed covalent bond is a permitted local
communication edge, and the full set of bonded atoms is the CA lattice. This
replaces the regular square or cubic lattice of a conventional CA with the
irregular topology of a molecule. The cell state is a learned 24-channel
continuous vector rather than a single discrete state. Its initial value is
constructed from the atom's chemical feature vector, while the CYP endpoint is
supplied as a conditioning context.

At cellular generation \(t\), atom \(i\) receives messages only from atoms
joined to it by covalent bonds. Bond features pass through a learned gate and
bond embedding, so the message depends upon the chemical type of the local
interaction. The incoming messages are degree-normalised and combined with
the atom's current state, its original chemical features, and the CYP context
to form a bounded reaction signal. In compact notation,

```text
m_i(t) = degree-normalised sum of typed-bond messages from bonded neighbours
r_i(t) = tanh(self(h_i(t)) + m_i(t) + chem(x_i) + context(CYP) + bias)
h_i(t+1) = tanh(h_i(t) + update_scale * F_i(t))
```

Here, \(h_i(t)\) is the latent state of atom \(i\), \(r_i(t)\) is the learned
reaction signal, and \(F_i(t)\) is the local transition drive. The genetic
program specifies the symbolic form of \(F_i(t)\). Every atom is updated for
128 explicit recurrent generations. The same selected transition rule is
applied across the molecular dataset, while each molecule develops its own
atom-by-generation trajectory from its topology, chemistry, CYP context, and
learned continuous parameters.

The complete trajectory is retained as an atom-by-generation state field. The
readout therefore uses the path taken through latent molecular state space,
as well as its final state. It contains the final atom-state mean and variance,
the time-averaged atom state, temporal variance, accumulated update energy,
final displacement from the initial state, lagged temporal correlation, graph
summaries at 12.5%, 25%, 50%, 75%, and 100% of the trajectory, and squared
transition energies between those scales. These pooled quantities form the
molecular trajectory fingerprint.

A genuine differentiable closed-form ridge solve maps the fingerprints to
pIC50 during support-query training. Prediction loss is backpropagated through
the ridge solution, every pooled trajectory quantity, and all 128 recurrent
cellular generations. This provides backpropagation through time for the
continuous Graph-CA parameters inside every candidate symbolic rule.

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

## Molecular space-time

Molecular space-time is the atom-by-generation state field generated by the
Graph-CA. Molecular space is represented by the bonded graph rather than by a
Cartesian simulation box. Time is represented by the ordered cellular
generations. A single molecule therefore produces a tensor in which each atom
has a multichannel state at each of 128 successive generations. Local changes
propagate along typed bonds, creating spatially distributed temporal patterns
that can converge, oscillate, recur, or remain dynamically complex. The
prediction model learns from summaries of these patterns.

The genetic program changes the equation that generates this trajectory. Its
terminal quantities were:

- `reaction`: the learned local nonlinear response \(r_i(t)\);
- `state`: the current cell state \(h_i(t)\);
- `neighbour_delta`: the difference between the mean bonded-neighbour state
  and the current atom state;
- `memory_delta`: the difference between a delayed state and the current atom
  state.

Candidate equations combined these quantities through addition, subtraction,
multiplication, averaging, negation, `tanh`, sine, and fixed scaling. For
example, the initial population included direct reaction updates, additive
reaction-plus-neighbour updates, bounded nonlinear combinations, and rules
that incorporated delayed-state differences. Subtree crossover copied a
mathematical subexpression from one parent rule into another. Subtree mutation
replaced a selected expression with a newly generated expression. Tournament
selection favoured low scaffold-validation MA-ST-RAE, with RMSE as a secondary
diagnostic and a small node-count penalty to control expression growth.

Each candidate rule was held fixed while its continuous Graph-CA parameters
were trained by backpropagation through time and differentiable ridge
regression. Its resulting scaffold-validation score became the genetic fitness
used to select parents for the next population. Genetic programming therefore
operated between complete training runs. It did not alter the transition rule
partway through an individual ligand trajectory.

The selected program was:

```text
F_i(t) = mean(reaction_i(t), neighbour_delta_i(t))
h_i(t+1) = tanh(h_i(t) + 0.08 * F_i(t))
```

This rule gives equal weight to a learned atom-local reaction term and a
bonded-neighbour diffusion term. Its trajectories consequently reflect both
chemical response at each atom and state differences propagating through the
molecular graph.

### Trajectory videos

#### Fixed-rule and genetic-programmed molecular space-time

https://github.com/user-attachments/assets/c0ad14b7-4cd1-471e-92d6-8e78310d4b5e

[Open the molecular space-time trajectory video](../../output/videos/molecular_spacetime_trajectory_cascade_gp_vs_backprop_linkedin.mp4).

This paired visualisation shows one illustrative molecular trajectory under a
fixed inertial reaction-diffusion rule and under the selected genetic program.
Vertical progression represents cellular generation, colour represents latent
atom state, and the retained trace represents propagation through molecular
space-time. It is a scientific visual explanation of the two update schemes,
rather than a direct export of the final competition checkpoints.

#### Four 1,000-generation molecular trajectories

[![Four Graph-CA molecular trajectory cascades](../../assets/readme/graph-ca-four-trajectory-cascade.gif)](../ds_gcae_1000_generation_pymol/trajectories_05_06_07_08_four_column_atom_cascade.mp4)

[Open the four-trajectory propagation video](../ds_gcae_1000_generation_pymol/trajectories_05_06_07_08_four_column_atom_cascade.mp4).

This video shows four recorded 1,000-generation Graph-CA atom-state cascades.
Each column is a separate molecule-rule trajectory. The molecular graph moves
down the frame as generations advance, leaving a colour-coded record of the
latent information state carried by each atom.

#### Four candidate dynamical regimes

https://github.com/user-attachments/assets/e30b0641-6eca-4ded-a3a2-dd6f9d149b2c

[Open the four-regime dynamics video](../long_horizon_attractor_campaign_v1/videos/four_graph_ca_dynamical_regimes_2x2.mp4).

This long-horizon comparison presents a point-attractor convergence case, a
hyperchaotic strange-attractor candidate, a persistent or complex candidate,
and a period-two oscillator candidate. The classifications are dynamical
screening labels supported by the associated long-horizon analyses. They are
included to show the range of spatial and temporal regimes that encoded
molecular Graph-CAs can generate.
