# Mini-Batching Molecular Graphs and CYP Contexts

## Learning objective

A mini-batch is the set of training examples used to calculate one parameter update. In Strange Matter Engine, batches must accommodate variable molecular graphs, four CYP-specific responses per molecule, and missing experimental labels.

The accepted batch contains 16 unique molecules and all four CYP contexts for each molecule.

## The accepted production decision

```math
B_{\rm mol}=16
```

and, with four CYP contexts,

```math
B_{\rm example}
=
16\times4
=64.
```

One ordinary batch therefore contains up to 64 molecule–CYP examples. Missing labels are masked from the loss rather than invented.

Here $B_{\rm mol}$ is the number of distinct molecules selected for a batch, and $B_{\rm example}$ is the maximum number of molecule–CYP examples produced after expanding those molecules across the four CYP contexts.

## 1. Molecules and supervised examples

Let molecule $m$ have standardised graph

```math
G_m=(V_m,E_m)
```

where $V_m$ is molecule $m$'s atom set and $E_m$ is its bond set. Let $c$ identify a CYP isoform and let $y_{mc}$ be the measured pIC50 for molecule $m$ against CYP $c$. A prediction example is

```math
(G_m,c)\longrightarrow y_{mc}.
```

The graph remains the same across the four CYP examples, but CYP context differs:

```math
\begin{aligned}
(G_m,\mathrm{CYP1A2})&\longrightarrow y_{m,1A2},\\
(G_m,\mathrm{CYP2C9})&\longrightarrow y_{m,2C9},\\
(G_m,\mathrm{CYP2D6})&\longrightarrow y_{m,2D6},\\
(G_m,\mathrm{CYP3A4})&\longrightarrow y_{m,3A4}.
\end{aligned}
```

Because CYP context enters the update rule, one molecular graph may produce four different trajectories.

## 2. Why select molecules first

Selecting 16 molecules and then expanding each into all CYP contexts gives every complete batch equal target representation. It supports:

- balanced contributions from the four CYPs;
- direct comparison of one molecule under four enzyme contexts;
- learning of shared chemical relationships;
- molecule-level shuffling; and
- transparent missing-label handling.

Sampling arbitrary molecule–CYP rows could produce a batch dominated by one CYP or repeatedly include a structure while omitting its other contexts.

## 3. The batch index set

Let

```math
\mathcal M_s=
\{m_1,m_2,\ldots,m_{16}\}
```

be the molecules selected at optimisation step $s$, and let

```math
\mathcal C=
\{\mathrm{1A2},\mathrm{2C9},\mathrm{2D6},\mathrm{3A4}\}.
```

The potential example set is

```math
\mathcal B_s=\mathcal M_s\times\mathcal C.
```

With complete labels,

```math
|\mathcal B_s|
=
|\mathcal M_s||\mathcal C|
=16\times4
=64.
```

## 4. Missing-label masks

Define the binary observation mask

```math
M_{mc}=
\begin{cases}
1,&\text{if }y_{mc}\text{ is observed},\\
0,&\text{if }y_{mc}\text{ is missing}.
\end{cases}
```

The masked batch mean squared error is

```math
\mathcal L_{\rm data}^{(s)}
=
\frac{
\sum_{m\in\mathcal M_s}
\sum_{c\in\mathcal C}
M_{mc}(\widehat y_{mc}-y_{mc})^2
}{
\sum_{m\in\mathcal M_s}
\sum_{c\in\mathcal C}
M_{mc}
}.
```

Only observed labels contribute. Missing pIC50 values are never set to zero, replaced with a mean, or interpreted as weak inhibition.

## 5. The support/query batch loss

```math
\mathcal L_s
=
\mathcal L_{\rm query}^{(s)}
+\lambda_\theta\lVert\theta\rVert_2^2.
```

Molecules are split into support and query subsets. The support fingerprints and labels determine $\beta$ through $(Z^{\mathsf T}Z+\lambda_\beta I)\beta=Z^{\mathsf T}y$. Query observations determine $\mathcal L_{\rm query}^{(s)}$.

Backpropagation differentiates through the support ridge solve and query predictions. Adam updates $\theta$ only; $\beta$ is recomputed by the solve.

## 6. Why mini-batches

**Full-batch training** uses every training observation for every update. Its gradient is stable, but storing all 16-generation trajectories can exceed memory.

**Single-example training** uses little memory but produces highly variable gradients and poor CYP balance.

**Mini-batch training** balances memory, computational efficiency, gradient variability, and target representation.

Our 16-molecule batch is a chemically structured mini-batch, not merely 64 arbitrary table rows.

## 7. Variable-sized graph batches

Molecule $m$ may have $n_m$ atoms and $b_m$ bonds. A batch can form a disjoint graph union:

```math
G_{\rm batch}
=
G_{m_1}\sqcup G_{m_2}\sqcup\cdots\sqcup G_{m_{16}}.
```

There are no bonds between different molecules. A membership map

```math
a(i)=\text{molecule containing atom }i
```

ensures that messages remain within genuine molecular graphs and fingerprint pooling returns one vector per molecule.

## 8. Four trajectories per molecule

For CYP context $c$,

```math
H_{mc}^{(t+1)}
=
F_\theta
\left(
H_{mc}^{(t)},C_m,E_m,c
\right).
```

Here $H_{mc}^{(t)}$ is the dynamical atom-state matrix for molecule $m$ under CYP context $c$ at generation $t$; $C_m$ is its fixed atom-chemistry matrix; $E_m$ is its fixed bond-feature data; and $F_\theta$ is the shared update rule. Fixed graph data $C_m,E_m$ can be reused, but four separate dynamical states are required. In general,

```math
H_{m,\mathrm{1A2}}^{(t)}
\ne
H_{m,\mathrm{3A4}}^{(t)}.
```

## 9. CYP balance

With complete labels, a batch contains 16 examples for each CYP. Equal counts do not guarantee equal difficulty or noise, so we will still report loss and residuals separately by CYP.

If missingness differs by enzyme, masks change the effective counts. Observed examples per CYP will be recorded for every fold and epoch.

## 10. Shuffling and epochs

At the start of each epoch:

1. shuffle training molecules using a recorded random seed;
2. divide the molecule list into groups of 16;
3. expand every selected molecule into its available CYP contexts; and
4. process every batch once.

An epoch is one pass through all molecules in the current training fold. With $N_{\rm mol}$ molecules,

```math
U_{\rm epoch}
=
\left\lceil\frac{N_{\rm mol}}{16}\right\rceil
```

updates occur approximately. The final batch may be smaller and will be retained.

## 11. Batch gradient variability

If $g_m$ is the gradient contribution from molecule $m$, then

```math
g_{\mathcal M_s}
=
\frac1{|\mathcal M_s|}
\sum_{m\in\mathcal M_s}g_m.
```

Larger molecule batches average over more contributions and generally reduce sampling variation. Smaller batches require less memory but give noisier updates.

The effective independent count is 16 molecules, not 64 unrelated examples, because each group of four shares one molecular structure.

## 12. Graph size and memory

Two 16-molecule batches may have very different total sizes:

```math
N_{\rm atom}^{(s)}
=
\sum_{m\in\mathcal M_s}n_m.
```

Memory also grows with four CYP-conditioned trajectories, 17 stored generations, eight channels, messages, gates, and values retained for backpropagation.

We will record atoms and bonds per batch. A deterministic atom-budget policy can be introduced later if exceptionally large graphs cause memory failures.

## 13. Loss weighting remains separate

The initial loss gives every observed molecule–CYP pair equal weight. Alternative objectives could weight CYPs, molecules, or measurements by experimental uncertainty.

Those choices change the scientific objective and will be decided explicitly. Batch construction does not settle them.

The accepted unweighted objective and the role of reported assay uncertainty are explained in [Loss Functions and Assay Uncertainty](Loss_Functions_and_Assay_Uncertainty.md).

## 14. Validation boundaries

Mini-batching occurs only within the current training fold:

- outer-test molecules never enter outer-training batches;
- inner-validation molecules never enter inner-training batches;
- molecule and scaffold groups remain intact across fold boundaries; and
- shuffling changes order, not membership.

A batch is an optimisation unit, not a validation split.

## 15. Reproducibility and monitoring

We will record:

- ordered molecule identifiers;
- random seed and generator state;
- epoch and batch number;
- CYP expansion order;
- missing-label masks;
- final incomplete-batch handling;
- atoms and bonds per batch;
- examples per CYP;
- batch loss and gradient norms;
- processing time; and
- memory peaks.

This reveals omitted, duplicated, imbalanced, or unusually expensive batches.

## 16. Prototype specification

The accepted design is:

- 16 unique molecules per ordinary batch;
- all four CYP contexts generated together;
- up to 64 molecule–CYP examples;
- explicit missing-label masks;
- molecule-level shuffling;
- no cross-molecule graph edges;
- loss divided by the actual observed-label count;
- incomplete final batches retained; and
- complete batch provenance.

## Connection to the course

- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) defines the graph inputs.
- [End-to-End Joint Training](End_to_End_Joint_Training.md) defines the batch forward and backward passes.
- [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md) explains how batch gradients become updates.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) defines which molecules are eligible for each batch.
