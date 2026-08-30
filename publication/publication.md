# Molecule Space-Time: Predicting Cytochrome P450 Inhibition with a Nonlinear Graph Cellular Automaton

**Anthony Nash**

## Abstract

Cytochrome P450 (CYP) inhibition is a major consideration in drug discovery because it can alter drug metabolism and contribute to clinically significant drug–drug interactions. The OpenADMET CYP Inhibition Challenge provides a blinded setting in which to evaluate computational prediction of direct-inhibition pIC50 across four CYP isoforms. Here, we introduce a molecular graph cellular automaton that represents atoms as cells, chemical bonds as local neighbourhoods, and molecular computation as the repeated evolution of a shared, learned transition rule. Unlike molecular models that reduce a structure directly to a fixed representation, our approach retains the complete sequence of atom states and uses its transient and terminal properties to predict CYP inhibition. We call this evolving representation **molecule space-time**: the joint description of molecular structure and its learned progression through computational time. Predictive performance across the CYP targets was [PREDICTION RESULTS]. As a secondary objective, we investigated the nonlinear dynamics contained within molecule space-time by extending selected trajectories over thousands of generations and examining convergence, recurrence, periodicity, perturbation sensitivity, strange-attractor candidates, and possible chaotic behaviour. This analysis identified [CAPTURED DYNAMICS]. The framework therefore treats prediction and dynamical exploration as complementary views of the same learned molecular process, offering a route toward CYP inhibition models whose internal evolution can be measured, visualised, and studied as a nonlinear system.

## Introduction

## Materials and Methods

### Dataset and Prediction Task

The study used the primary direct-inhibition dataset released for the 2026 OpenADMET CYP Inhibition Blind Challenge. It comprised 4,905 unique compounds represented by a molecule identifier and a SMILES string. Experimental direct-inhibition pIC50 values were provided for four major drug-metabolising cytochrome P450 isoforms: CYP1A2, CYP2C9, CYP2D6, and CYP3A4. Here, pIC50 denotes the negative base-10 logarithm of the half-maximal inhibitory concentration expressed in molar units. Each reported measurement was accompanied by lower and upper uncertainty bounds and an estimated standard deviation from the fitted concentration–response experiment.

The response matrix was incomplete because compounds were not necessarily measured against every CYP isoform. The dataset contained 6,525 observed compound–CYP pairs, distributed as follows:

| CYP isoform | Compounds with an observed direct-inhibition pIC50 |
|---|---:|
| CYP1A2 | 1,412 |
| CYP2C9 | 1,285 |
| CYP2D6 | 1,493 |
| CYP3A4 | 2,335 |
| **Total** | **6,525** |

Each observed compound–CYP pair constituted one supervised regression example. The prediction task was to learn a single CYP-conditioned mapping from molecular structure to direct-inhibition pIC50, rather than fitting an independent model for each isoform. Missing endpoint values were retained as missing and contributed neither targets nor loss terms. The single-concentration, time-dependent-inhibition, and Emax datasets distributed with the challenge were outside the scope of this direct-inhibition study.

To assess generalisation beyond closely related chemistry, compounds were grouped by their standardised Bemis–Murcko scaffold before data partitioning. A fixed 20% subset of scaffold groups was reserved as a sealed holdout, leaving 5,216 observed compound–CYP pairs for model fitting and internal selection and 1,309 pairs for final validation. No scaffold group occurred in both partitions. Hyperparameter selection was conducted within the fitting pool, while the reserved scaffold holdout remained unused until final evaluation.

The challenge test set contained 750 additional compounds for which all direct-inhibition labels were withheld. The trained system was required to generate four finite pIC50 predictions for each compound, giving 3,000 blinded compound–CYP predictions in total. Blinded compounds and their unreleased outcomes were excluded from parameter estimation, hyperparameter selection, early stopping, and dynamical candidate selection. Predictive evaluation followed the challenge formulation: the primary measure was the macro-averaged soft-threshold relative absolute error across the four CYP isoforms, with each isoform weighted equally and predictions falling within the reported experimental uncertainty interval assigned zero error. Conventional regression statistics were retained as complementary measures of predictive agreement.

### Molecular Graph Representation

### Nonlinear Graph Cellular Automaton

### Machine-Learning Training and Validation

### Long-Horizon Dynamical Analysis

#### Targeted discovery of candidate attractor regimes

Dynamical analysis was performed after predictive training, using frozen model parameters and fixed molecular graphs. Candidate selection therefore changed neither the regression model nor its validation score. The objective was to concentrate long-horizon computation on trajectories showing sustained motion, recurrent geometry, broad frequency content, and sensitivity to small perturbations.

An initial screen was calculated from the complete atom-by-channel validation trajectories. For trajectory \(H_t\), the molecular mean state was calculated at each generation, and late-time motion was measured from the mean Euclidean step length over the second half of the observed sequence. Recurrence was assessed over lags from 2 to 64 generations by comparing the mean lagged state distance with the distance expected from the accumulated mean step length. The smallest normalized lagged distance defined the recurrence ratio. Spectral entropy was calculated from the non-zero-frequency Fourier power of each state channel and averaged across channels. A screening score combined these quantities:

$$
S = \frac{v_{\mathrm{late}}\left(1 + H_{\mathrm{spectral}}\right)}{R_{\mathrm{recurrence}}},
$$

where \(v_{\mathrm{late}}\) is late-time motion, \(H_{\mathrm{spectral}}\) is normalized spectral entropy, and \(R_{\mathrm{recurrence}}\) is the recurrence ratio. This score acted as a computational targeting device rather than a definition of chaos. Candidates were drawn from gated residual, delayed memory, inertial reaction–diffusion, Kuramoto–Sakaguchi, and FitzHugh–Nagumo transition-rule families so that the long-horizon analysis included contracting, oscillatory, recurrent, and expanding regimes.

Twenty complete molecular states were propagated through 5,000 frozen cellular-automata generations. Every atom and every dynamical channel was retained at every generation. The first 1,000 generations were treated as burn-in for late-time diagnostics. Ten representative cases were subjected to detailed phase-space analysis, including principal-component projections, recurrence plots, frequency spectra, correlation-dimension estimates, and nearest-neighbour divergence curves. Principal-component coordinates were used exclusively for visualization; all perturbation and Lyapunov calculations operated in the full atom-by-channel state space.

#### Direct perturbation screen

Each detailed case was paired with eight independently oriented full-state perturbations of Euclidean magnitude \(10^{-5}\). Reference and companion states were advanced under the same frozen transition rule, molecular graph, bond features, atom features, and CYP context. Their separation was recorded across the trajectory. Replicated early separation across perturbation directions was used to prioritize candidates for renormalized analysis. Trajectories 7 and 8, corresponding to `OCNT-0494110` conditioned on CYP2C9 and `OCNT-2328784` conditioned on CYP1A2, showed the clearest replicated expanding response and were advanced to the confirmatory tests below.

#### Circular state-space distance

The Kuramoto–Sakaguchi state is phase-like and wrapped to the interval \([-1,1]\). Differences were consequently measured on the circular state space. For reference state \(h\) and companion state \(h'\), the elementwise circular difference was

$$
\Delta(h',h) = \frac{1}{\pi}\mathrm{atan2}\!\left[
\sin\!\left(\pi(h'-h)\right),
\cos\!\left(\pi(h'-h)\right)
\right],
$$

and full-state separation was \(d=\lVert\Delta(h',h)\rVert_2\). This prevented an apparent jump across the phase boundary from being interpreted as physical divergence.

#### Confirmatory protocols

The numerical settings for the confirmatory and population experiments are summarized below. A repeat denotes an independently oriented initial perturbation or orthogonal perturbation basis, as appropriate to the calculation.

| Analysis | Burn-in generations | Measured generations | Renormalization or sampling interval | Perturbation scale | Repeats |
|---|---:|---:|---|---|---:|
| Largest Lyapunov exponent | 1,000 | 4,000 | 10 | \(10^{-4}\), \(10^{-5}\), and \(10^{-6}\) | 8 per scale and molecule |
| Float64 Lyapunov spectrum | 1,000 | 4,000 | 5, 10, and 20 | \(10^{-7}\) | 2 per interval and molecule |
| Attraction basin | 1,000 | 6,000 | Sampled every 10 | Displacement radii 0.1, 0.5, 1.0, and 2.0 | 8 per radius and molecule |
| Population and structural interventions | 1,000 | 2,000 | 10 | \(10^{-7}\) | 2 per system |

#### Renormalized largest Lyapunov exponent

Persistent local instability was tested with a Benettin-style repeated-renormalization calculation. A companion state was placed at distance \(\varepsilon\) from the post-burn-in reference state, and both states were advanced for each interval \(\tau\). The circular separation \(d_k\) was measured, its logarithmic expansion was recorded, and the companion was returned to distance \(\varepsilon\) along the observed separation direction. The largest Lyapunov exponent was estimated as

$$
\lambda_1 = \frac{1}{K\tau}
\sum_{k=1}^{K}\log\!\left(\frac{d_k}{\varepsilon}\right).
$$

The complete design produced 48 estimates. Repeated renormalization tested whether divergence was continually regenerated after local separations had been returned to the same small scale.

#### Float64 Lyapunov spectrum

The leading Lyapunov spectrum was calculated in double precision using eight simultaneous orthogonal perturbation vectors. After each propagation block, the full-state circular difference vectors were assembled into a matrix and subjected to reduced QR decomposition. The logarithms of the absolute diagonal elements of the resulting upper-triangular matrix supplied the local expansion rates, while the orthonormal basis supplied the renormalized companion directions. Stability across the intervals and repeats listed above was used to distinguish persistent multidirectional expansion from numerical precision effects or a single unstable direction.

#### Boundedness and attraction-basin test

Attraction towards a common invariant set was tested by initiating float64 trajectories at the full-state displacement radii listed above. The complete design contained 64 displaced trajectories.

Boundedness was monitored directly from the maximum absolute state. Convergence at the distributional level was evaluated after circular embedding of every state as concatenated sine and cosine coordinates. Distances between early and late trajectory distributions and the reference invariant distribution were estimated over 32 random one-dimensional projections, providing a sliced distribution distance suitable for the high-dimensional state space. A late-to-early distance ratio below one indicated movement towards the reference distribution. Nearest-reference-cloud distance was retained as a complementary finite-sampling diagnostic.

#### Population-level structure–dynamics analysis

The candidate analysis was followed by a broader test of whether molecular structure was associated with the strength of the learned instability. A scaffold-held-out population of 256 molecule–CYP cases was sampled evenly across CYP1A2, CYP2C9, CYP2D6, and CYP3A4, with 64 cases per endpoint. Sampling was stratified across quartiles of the initial dynamical screening score. The two established leading candidates were added explicitly, producing 258 evaluated cases under the population protocol above.

Molecular descriptors comprised molecular weight, calculated logP, topological polar surface area, hydrogen-bond donors and acceptors, rotatable bonds, ring composition, aromatic and heteroatom fractions, formal charge, fraction sp3, and bond-type counts. Graph descriptors comprised atom and bond counts, cyclomatic number, density, degree moments, adjacency spectral radius, algebraic connectivity, largest Laplacian eigenvalue, graph diameter, and mean shortest-path length. Reproducible ETKDG conformers were generated with seed 260830 and MMFF relaxation to calculate radius of gyration, asphericity, eccentricity, inertial shape factor, and spherocity. Cartesian coordinates were absent from the Graph-CA input, so three-dimensional descriptors were interpreted as structural correlates rather than direct dynamical inputs.

Univariate associations with the largest Lyapunov exponent were measured using Spearman correlation and Benjamini–Hochberg correction within each analysis scope. Confidence intervals were calculated from 2,000 bootstrap resamples of complete Bemis–Murcko scaffold clusters. Multivariate reproducibility was assessed with five-fold scaffold-grouped cross-validation using Elastic Net and Extra Trees regression. Held-out permutation importance quantified the contribution of each descriptor while preserving scaffold separation between training and evaluation folds.

#### Frozen-model structural interventions

Causal computational tests were performed on trajectories 7 and 8 while retaining all learned weights, the transition rule, the CYP context, and all parameters of the Lyapunov calculation. Each undirected molecular bond was represented by two directed message-passing edges. Bond deletion removed both directions of a selected connection. Bond-identity interventions replaced its single, double, triple, or aromatic edge encoding with each alternative identity. Ring-opening interventions removed a selected ring connection and suppressed the ring-membership atom encoding. Atom-feature interventions ablated one chemically interpretable group at a time: elemental identity, charge and aromaticity, hybridization, local valence, donor–acceptor state, chirality, or neighbouring-atom chemistry.

The intervention campaign comprised 187 modified and baseline systems evaluated with the population protocol. Effect size was defined as the change in the mean largest Lyapunov exponent relative to the corresponding unmodified molecule–CYP baseline. Positive values indicated faster exponential divergence, while negative values indicated slower divergence. The complete workflow, retained numerical tables, and publication figures are available in the [structure–dynamics campaign archive](../results/structure_dynamics_publication_v1/README.md).

## Results and Discussion

### CYP pIC50 Predictions

### Nonlinear Dynamics in Molecule Space-Time

## Conclusions
