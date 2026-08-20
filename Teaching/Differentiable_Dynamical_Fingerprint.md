# The Differentiable Dynamical Fingerprint

## Learning objective

A graph cellular automaton produces a trajectory whose size depends on the number of atoms in the molecule. The pIC50 readout requires a fixed-length vector.

The **dynamical fingerprint** is the mathematical bridge:

```math
\text{variable-sized atom trajectory}
\longrightarrow
\text{fixed-length molecular vector}.
```

For the initial production prototype, the fingerprint will contain five differentiable summaries for each of eight evolving channels, giving 40 components.

## The accepted production decision

For each dynamical channel, we will calculate:

1. final mean across atoms;
2. final variance across atoms;
3. mean activity across the complete trajectory;
4. temporal variance of the molecule-level mean; and
5. mean squared step-to-step change.

With eight channels,

```math
p=5\times8=40.
```

The resulting fingerprint is

```math
z\in\mathbb R^{40}.
```

These components are differentiable, invariant to atom numbering, normalised for molecule and trajectory size, and scientifically interpretable.

## 1. Starting from the trajectory

For a molecule with $n$ atoms and eight dynamical channels,

```math
H^{(t)}\in\mathbb R^{n\times8},
\qquad
t=0,1,\ldots,T.
```

The entry

```math
h_{ik}^{(t)}
```

is the state of atom $i$, dynamical channel $k$, at generation $t$.

Our prototype uses $T=16$, so the stored trajectory contains 17 states:

```math
H^{(0)},H^{(1)},\ldots,H^{(16)}.
```

Generation 0 is included. Because the accepted initialisation is $H^{(0)}=0$, it provides a common neutral reference from which learned activity develops.

## 2. Molecule-level channel mean

Several fingerprint components use the atom mean for channel $k$ at generation $t$:

```math
\mu_k^{(t)}
=
\frac1n\sum_{i=1}^{n}h_{ik}^{(t)}.
```

This converts one channel across $n$ atoms into one molecule-level value.

Dividing by $n$ matters. A sum would tend to increase with molecule size even if the typical atom state were unchanged. The mean makes values more comparable across molecules with different atom counts.

## 3. Component 1: final mean across atoms

The first component for channel $k$ is

```math
z_{k,\rm final\ mean}
=
\mu_k^{(T)}
=
\frac1n\sum_{i=1}^{n}h_{ik}^{(T)}.
```

It asks:

> What is the molecule's average channel activity after the final CA generation?

A positive value indicates positive final activity on average; a negative value indicates negative activity; and a value near zero may indicate weak activity or cancellation between positive and negative atoms.

Because averaging can hide spatial differences, it is paired with final variance.

## 4. Component 2: final variance across atoms

The second component is

```math
z_{k,\rm final\ variance}
=
\frac1n
\sum_{i=1}^{n}
\left(
h_{ik}^{(T)}-\mu_k^{(T)}
\right)^2.
```

It asks:

> How different are the atoms from one another at the final generation?

- a value near zero means the channel is nearly homogeneous across atoms;
- a larger value means the molecule retains spatially differentiated atom states.

We use variance rather than a hard range or maximum–minimum difference. Variance is smooth, uses every atom, and has a straightforward derivative.

## 5. Component 3: mean activity across the trajectory

First average across atoms at every generation, then average those values through time:

```math
z_{k,\rm trajectory\ mean}
=
\frac1{T+1}
\sum_{t=0}^{T}
\mu_k^{(t)}.
```

It asks:

> What was the channel's average molecule-level activity over its complete history?

Two trajectories can have the same final mean but different trajectory means. One may rise early and remain active; another may remain quiet until the final generations.

Including $t=0$ anchors the summary to the accepted neutral initial state. Excluding it can later be tested as an ablation.

## 6. Component 4: temporal variance

Let the trajectory mean from the previous section be

```math
\bar\mu_k
=
\frac1{T+1}
\sum_{t=0}^{T}\mu_k^{(t)}.
```

The temporal variance is

```math
z_{k,\rm temporal\ variance}
=
\frac1{T+1}
\sum_{t=0}^{T}
\left(
\mu_k^{(t)}-\bar\mu_k
\right)^2.
```

It asks:

> How much did the molecule-level channel mean vary over the trajectory?

A nearly constant trajectory has low temporal variance. A trajectory that changes substantially—whether through a transient, drift, or oscillation—has greater temporal variance.

Temporal variance measures activity magnitude but does not identify its temporal order. A smooth rise and an oscillation can sometimes have similar variance. Later fingerprint extensions can distinguish them.

## 7. Component 5: mean squared step-to-step change

For every adjacent pair of generations, measure how much every atom changes:

```math
z_{k,\rm step}
=
\frac1{Tn}
\sum_{t=0}^{T-1}
\sum_{i=1}^{n}
\left(
h_{ik}^{(t+1)}-h_{ik}^{(t)}
\right)^2.
```

It asks:

> How much dynamical movement occurred from one generation to the next?

The divisions by $T$ and $n$ make the value an average per transition and per atom.

- a small value indicates little stepwise movement on average;
- a larger value indicates more active evolution;
- a converged trajectory contributes almost no step change after convergence; and
- sustained oscillation can retain a non-zero value.

Unlike a hard convergence time, this quantity changes smoothly when the states change.

## 8. Constructing the 40-component vector

For each channel $k\in\{1,\ldots,8\}$, define

```math
z_k=
\begin{bmatrix}
z_{k,\rm final\ mean}\\
z_{k,\rm final\ variance}\\
z_{k,\rm trajectory\ mean}\\
z_{k,\rm temporal\ variance}\\
z_{k,\rm step}
\end{bmatrix}.
```

The complete fingerprint concatenates all eight channel summaries:

```math
z=
\begin{bmatrix}
z_1\\z_2\\\vdots\\z_8
\end{bmatrix}
\in\mathbb R^{40}.
```

The component order will be fixed, named, and stored with every model so that coefficient $\beta_j$ always maps to an identifiable summary and channel.

## 9. A worked one-channel example

Consider two atoms, one dynamical channel, and two update generations:

```math
H^{(0)}=
\begin{bmatrix}0\\0\end{bmatrix},
\qquad
H^{(1)}=
\begin{bmatrix}0.2\\0.4\end{bmatrix},
\qquad
H^{(2)}=
\begin{bmatrix}0.4\\0.6\end{bmatrix}.
```

The molecule-level means are

```math
\mu^{(0)}=0,
\qquad
\mu^{(1)}=0.3,
\qquad
\mu^{(2)}=0.5.
```

### Final mean

```math
z_{\rm final\ mean}=0.5.
```

### Final variance

```math
\begin{aligned}
z_{\rm final\ variance}
&=
\frac{(0.4-0.5)^2+(0.6-0.5)^2}{2}\\
&=0.01.
\end{aligned}
```

### Trajectory mean

```math
z_{\rm trajectory\ mean}
=
\frac{0+0.3+0.5}{3}
\approx0.2667.
```

### Temporal variance

```math
\begin{aligned}
z_{\rm temporal\ variance}
&=
\frac{
(0-0.2667)^2
+(0.3-0.2667)^2
+(0.5-0.2667)^2
}{3}\\
&\approx0.0422.
\end{aligned}
```

### Mean squared step change

The four atom-level changes are $0.2$, $0.4$, $0.2$, and $0.2$:

```math
\begin{aligned}
z_{\rm step}
&=
\frac{
0.2^2+0.4^2+0.2^2+0.2^2
}{2\times2}\\
&=0.07.
\end{aligned}
```

The one-channel fingerprint is therefore

```math
z=
\begin{bmatrix}
0.5&0.01&0.2667&0.0422&0.07
\end{bmatrix}^{\mathsf T}.
```

The production fingerprint repeats this structure for eight channels.

## 10. Why the fingerprint is invariant to atom numbering

Renumbering atoms applies a permutation $\pi$ to the rows of $H^{(t)}$. Sums over all atoms are unchanged:

```math
\sum_{i=1}^{n}h_{\pi(i)k}^{(t)}
=
\sum_{i=1}^{n}h_{ik}^{(t)}.
```

The same is true of sums of squared deviations and squared step changes. Therefore the five summaries do not depend on arbitrary atom indices.

This invariance is essential: changing the SMILES traversal must not change the molecular prediction.

## 11. Why the fingerprint is differentiable

Each component is constructed from:

- addition;
- subtraction;
- multiplication;
- squaring; and
- division by fixed non-zero counts.

These operations have defined derivatives. Therefore

```math
\frac{\partial z}{\partial H^{(t)}}
```

can be calculated, allowing pIC50 error to pass through the fingerprint into all CA generations.

No hard threshold, integer first-event time, sorting operation, or discrete family assignment occurs in the training fingerprint.

## 12. Fingerprint standardisation

The 40 components have different numerical scales. Before the linear readout, component $j$ is standardised using statistics fitted from the current training fold:

```math
z'_{mj}
=
\frac{z_{mj}-\mu_{j,\rm train}}
{s_{j,\rm train}+\varepsilon}.
```

The small positive constant $\varepsilon$ prevents division by zero when a component has almost no variation.

Validation and test examples use the training-fold values $\mu_{j,\rm train}$ and $s_{j,\rm train}$. Their own distributions never determine their scaling.

## 13. Relationship to ridge coefficients

The readout is

```math
\widehat y
=
\beta_0+\sum_{j=1}^{40}\beta_jz'_j.
```

Every coefficient can be traced to:

- one dynamical channel; and
- one of the five declared summaries.

Coefficient magnitude and sign help us inspect which fingerprint directions the model uses, but correlated components can share or exchange weight. Coefficients are predictive associations, not automatic causal mechanisms.

## 14. Measurements postponed from training

The initial differentiable fingerprint excludes:

- hard convergence time;
- explicitly detected period;
- hard attractor-family assignment;
- entropy requiring arbitrary binning;
- autocorrelation;
- Fourier or spectral features;
- Lyapunov estimates; and
- perturbation-response measurements.

These quantities remain valuable for post-training scientific analysis. Their exclusion from the first training fingerprint means only that we will not initially use them to transmit gradients.

Later versions can add smooth autocorrelation, spectral power, or differentiable perturbation summaries one family at a time. Each addition must be evaluated through grouped nested cross-validation.

## 15. Scientific checks

For every trained model, we will inspect:

- distribution of all 40 components;
- near-constant or duplicate components;
- correlations among components;
- differences among CYP contexts;
- sensitivity to molecule size;
- sensitivity to trajectory length;
- channel-specific trajectories;
- ridge coefficient stability across folds;
- gradients flowing through each component; and
- performance when each summary family is removed.

The fingerprint succeeds only if it is numerically stable, chemically useful, and predictive on held-out scaffold groups.

## 16. Prototype specification

The accepted fingerprint is fixed as follows:

- eight evolving channels;
- generations $0$ through $16$ retained;
- five summaries per channel;
- 40 named components;
- atom-count and trajectory-length normalisation;
- training-fold-only standardisation;
- differentiable calculations during end-to-end learning; and
- richer hard dynamical measurements retained for post-training interpretation.

## Connection to the course

- [Dynamics](Dynamics.md) defines the behaviours summarised by the fingerprint.
- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) defines the evolving matrix $H^{(t)}$.
- [End-to-End Joint Training](End_to_End_Joint_Training.md) explains why differentiability is required.
- [Ridge Regression](Ridge_Regression.md) maps the 40 components to predicted pIC50.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) controls feature selection and assessment.

