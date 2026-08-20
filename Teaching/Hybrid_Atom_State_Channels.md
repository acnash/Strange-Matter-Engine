# Hybrid Atom States: Fixed Chemistry and Evolving Dynamics

## Learning objective

The atom state in Strange Matter Engine will have two complementary parts:

1. **fixed chemical channels** record what an atom and its bonds chemically are; and
2. **evolving dynamical channels** record what the learned graph cellular automaton causes that atom to do.

This hybrid design preserves chemical identity while creating a separate state space in which propagation, convergence, oscillation, transients, and perturbation sensitivity can emerge.

## The accepted production decision

For a molecule with $n$ atoms, let

```math
C\in\mathbb R^{n\times d_c}
```

be the fixed chemical matrix and

```math
H^{(t)}\in\mathbb R^{n\times d_h}
```

be the evolving dynamical matrix at generation $t$. The complete atom representation available to the update rule is

```math
X^{(t)}=
\left[C\,\middle\|\,H^{(t)}\right],
```

where $\|$ means concatenation of channels. The central constraint is

```math
C^{(t+1)}=C^{(t)}=C,
```

while

```math
H^{(t+1)}=F_\theta\!\left(H^{(t)},C,E,c_{\rm CYP}\right).
```

Here $E$ contains fixed bond information, $c_{\rm CYP}$ represents CYP identity, and $\theta$ contains the learned parameters of the shared local rule.

## 1. What a channel is

A channel is one numerical component attached to every atom. If atom $i$ has a $d$-dimensional state, then

```math
x_i=
\begin{bmatrix}
x_{i1}&x_{i2}&\cdots&x_{id}
\end{bmatrix}^{\mathsf T}.
```

Each component has a declared meaning. Examples include whether the atom is nitrogen, its formal charge, whether it is aromatic, or a continuous dynamical value. A channel is not another atom; it is one coordinate in an atom's state vector.

## 2. Fixed chemical channels

Fixed chemical channels are calculated from molecular structure before the CA begins. For atom $i$, candidate channels include:

- element;
- formal charge;
- aromaticity;
- hybridisation;
- degree;
- attached-hydrogen count;
- ring information; and
- chirality information.

The exact list remains a later design decision. Every property must have a chemical definition, an encoding rule, and an ablation test.

### Why chemistry remains fixed

During our abstract CA trajectory, carbon does not become nitrogen, and a single bond does not spontaneously become aromatic. Holding chemical channels fixed gives every generation a stable reference.

This distinction is essential:

- **fixed channel values** do not change across generations;
- **model parameters acting on those values** can still be learned; and
- **dynamical responses to those values** can change at every generation.

Thus an aromaticity indicator can remain constant while its influence on a message is learned through $\theta$.

## 3. Fixed bond channels

For a bond connecting atoms $i$ and $j$, let

```math
e_{ij}\in\mathbb R^{d_e}
```

contain candidate bond properties such as bond type, bond order, aromaticity, conjugation, ring membership, and stereochemical direction where defined.

These edge channels remain fixed throughout the trajectory. They help determine how information travels between bonded atoms. The same neighbouring atom can influence atom $i$ differently through a single, double, or aromatic bond.

## 4. Evolving dynamical channels

The dynamical state

```math
h_i^{(t)}\in\mathbb R^{d_h}
```

is updated at every generation. Its components are degrees of freedom in which the local rule constructs a molecule- and CYP-dependent trajectory.

Across generations, these channels may propagate local information, integrate evidence from distant graph regions, contract towards a stable state, retain atom-level heterogeneity, oscillate, display long transients, or respond to perturbations.

These behaviours are measured and compressed into the dynamical fingerprint used by the pIC50 readout.

## 5. Initialising the dynamical state

The first prototype will use a neutral initial state:

```math
H^{(0)}=0.
```

Chemical information enters through $C$, connectivity and bond chemistry through $E$, and target context through $c_{\rm CYP}$. Any non-zero dynamical structure must therefore be created by the update rule.

A later alternative is a learned chemical projection:

```math
H^{(0)}=\phi_\omega(C).
```

This may add expressive power, but it makes the origin of early dynamics less transparent. We will treat it as an ablation rather than the initial design.

## 6. One local update

For atom $i$, a message from neighbour $j$ can be written

```math
m_{ij}^{(t)}
=
M_\theta\!\left(
h_i^{(t)},h_j^{(t)},c_i,c_j,e_{ij},c_{\rm CYP}
\right).
```

Messages from neighbourhood $\mathcal N(i)$ are combined with an order-independent operation. A neighbour mean is

```math
\bar m_i^{(t)}
=
\frac{1}{d_i}
\sum_{j\in\mathcal N(i)}m_{ij}^{(t)}.
```

The dynamical state is then updated:

```math
h_i^{(t+1)}
=
U_\theta\!\left(
h_i^{(t)},\bar m_i^{(t)},c_i,c_{\rm CYP}
\right).
```

The functions $M_\theta$ and $U_\theta$ are shared by every atom, molecule, and generation.

## 7. A transparent scalar rule

Suppose each atom has one evolving channel:

```math
h_i^{(t+1)}
=
\tanh\!\left(
\theta_{\rm self}h_i^{(t)}
+\theta_{\rm nbr}\bar h_i^{(t)}
+w^{\mathsf T}c_i
+v^{\mathsf T}c_{\rm CYP}
+b
\right).
```

The terms have distinct roles:

- $\theta_{\rm self}h_i^{(t)}$ carries the current dynamical state forward;
- $\theta_{\rm nbr}\bar h_i^{(t)}$ supplies neighbour influence;
- $w^{\mathsf T}c_i$ lets fixed atom chemistry drive the dynamics;
- $v^{\mathsf T}c_{\rm CYP}$ gives the same molecule a CYP-dependent trajectory; and
- $b$ provides a learned offset.

The chemical vector $c_i$ is consulted at every generation but never overwritten.

## 8. A two-atom example

Consider two bonded atoms with one chemical channel and one dynamical channel:

```math
c_1=1,\qquad c_2=0,\qquad
h_1^{(0)}=h_2^{(0)}=0.
```

Use

```math
h_i^{(t+1)}
=
\tanh\!\left(
0.4h_i^{(t)}
+0.6h_j^{(t)}
+0.5c_i
\right).
```

At generation 1,

```math
h_1^{(1)}=\tanh(0.5)\approx0.462,
\qquad
h_2^{(1)}=0.
```

At generation 2,

```math
\begin{aligned}
h_1^{(2)}
&=\tanh\!\left(0.4(0.462)+0.6(0)+0.5\right)
\approx0.595,\\
h_2^{(2)}
&=\tanh\!\left(0.4(0)+0.6(0.462)\right)
\approx0.270.
\end{aligned}
```

The fixed chemical signal at atom 1 activates its own dynamics and then propagates through the bond to atom 2. The chemistry remains $c_1=1,c_2=0$; only $h_1$ and $h_2$ evolve.

## 9. Receptive field and generations

After one generation, an atom can receive information from immediate neighbours. After two generations, information can arrive from atoms two bonds away. After $T$ generations, the theoretical receptive field extends up to $T$ graph steps:

```math
\text{maximum graph distance reached}\le T.
```

The number of generations controls both how far chemical information can propagate and how long the dynamical system evolves. It will be selected inside grouped nested cross-validation.

## 10. CYP context

The same molecular graph can have different potencies for different CYP enzymes. Initially, CYP identity will be represented by an explicit one-hot vector supplied to every atom in that molecule–CYP example.

A learned CYP embedding is a later alternative. One-hot context is preferable for the first prototype because its meaning is explicit and it introduces no hidden similarity assumption between enzymes.

## 11. From trajectory to prediction

The CA produces

```math
H^{(0)},H^{(1)},\ldots,H^{(T)}.
```

Permutation-invariant summaries convert the variable-sized atom trajectory into a fixed-length molecular fingerprint $z$. Candidate summaries include convergence, step distance, atom and temporal variation, oscillation amplitude, autocorrelation, spectral power, transient length, and perturbation response.

The ridge readout predicts

```math
\widehat y=\beta_0+\beta^{\mathsf T}z.
```

Backpropagation carries prediction error through the readout, fingerprint, trajectory, and local rule to learn $\theta$.

## 12. Why not evolve chemical channels?

Allowing all channels to change would blur two meanings: a change in internal activity and a change in declared chemical identity. If an aromaticity channel drifted from 1 to 0.3, it would no longer be a chemical label; it would be an unnamed latent value with a misleading name.

The hybrid design prevents semantic drift. It also makes interventions clearer: we can perturb a dynamical channel without pretending that an element or bond order changed.

## 13. Leakage boundaries

Only information available for an unseen molecule may enter $C$, $E$, $H^{(0)}$, or $c_{\rm CYP}$. Experimental pIC50, confidence intervals, assay outcomes, and fold membership are never atom or bond channels.

If a continuous chemical channel requires learned scaling, its mean and standard deviation must be fitted inside the current training fold.

## 14. Scientific ablations

The hybrid design supports controlled comparisons:

1. fixed chemistry with neutral dynamics;
2. fixed chemistry with learned initial dynamics;
3. dynamics without CYP context;
4. atom chemistry without bond attributes;
5. individual chemical channels removed one at a time;
6. different dynamical widths;
7. different generation counts; and
8. final-state summaries versus whole-trajectory fingerprints.

Each ablation must preserve the same grouped folds and evaluation metrics.

## 15. Prototype specification

The first prototype will use:

- a fixed two-dimensional molecular graph;
- fixed atom-property channels $C$;
- fixed bond-property channels $E$;
- separate evolving channels $H^{(t)}$;
- neutral initial dynamics $H^{(0)}=0$;
- one-hot CYP context;
- synchronous local updates;
- shared transition parameters $\theta$; and
- trajectory-derived, permutation-invariant fingerprints.

The precise atom properties, bond properties, dynamical width, aggregation, update equation, generation count, learning rate, and regularisation remain explicit design decisions.

## Connection to the course

- [Chemistry](Chemistry.md) defines atom and bond properties.
- [Graph Theory](Graph_Theory.md) defines nodes, edges, neighbourhoods, and invariant aggregation.
- [Emergence](Emergence.md) explains repeated local rules and collective behaviour.
- [Dynamics](Dynamics.md) explains trajectories produced by $H^{(t)}$.
- [Backpropagation](Backpropagation.md) explains how $\theta$ is learned.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) explains how channel choices are selected honestly.

