"""Generate the code-hidden visual teaching notebooks."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def hidden_code(source):
    cell = nbf.v4.new_code_cell(source.strip())
    cell.metadata["tags"] = ["hide-input"]
    cell.metadata["jupyter"] = {"source_hidden": True}
    return cell


SETUP = """
from _shared import *
display(neon_css())
"""


def build(filename, title, cells):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python (KCC2)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "title": title,
        }
    )
    notebook.cells = [hidden_code(SETUP)] + cells
    nbf.write(notebook, NOTEBOOKS / filename)


build(
    "00_Molecular_Gallery.ipynb",
    "Molecular Gallery",
    [
        markdown(r"""
# Molecular Gallery

## Aim

Meet the 12-molecule teaching set and learn what information belongs to a **molecule**, what belongs to an **assay label**, and what will later belong to an **atom or bond**.

## Approach

Select a molecule below. The left view shows the atom-numbered chemical structure; the right view shows the same object as a graph. The graph has

$$G=(V,E),$$

where $V$ is the set of atoms and $E$ is the set of chemical bonds. Its size is described by $|V|$, the number of atoms, and $|E|$, the number of bonds.

The CYP identity and experimental $\mathrm{pIC}_{50}$ are **molecule-level labels**. They are retained beside the graph but are not intrinsic properties of an individual atom.
"""),
        markdown(r"""
## Interactive laboratory

Use the selector to compare size, ring structure, aromaticity, charge, flexibility, stereochemistry, CYP context, and measured inhibition. The chemical structures use clean RDKit rendering: carbon skeletons are off-white, nitrogen cyan, oxygen hot pink, sulfur acid yellow, and halogens green or orange. Cyberpunk accent colours remain consistent throughout the course.
"""),
        hidden_code("display(gallery_explorer())"),
        markdown(r"""
## Scientific checkpoint

For any selected molecule, identify which displayed quantities describe its chemical structure and which came from the inhibition experiment.
"""),
    ],
)


build(
    "01_From_Molecule_to_Graph.ipynb",
    "From Molecule to Graph",
    [
        markdown(r"""
# From Molecule to Graph

## Aim

Understand the molecular-informatics transformation from a chemical structure into a graph without treating the transformation as a software problem.

## Approach

An atom becomes a node or cell. A chemical bond becomes an edge. For atom $i$, the neighbourhood

$$N(i)=\{j\in V:(i,j)\in E\}$$

is the set of atoms directly bonded to it. The degree is

$$d_i=|N(i)|.$$

Atom numbering is an indexing convenience. Renumbering the same atoms must not change the underlying molecule or its eventual prediction.
"""),
        markdown(r"""
## Interactive atom–node mapping

Select a molecule and an atom. The highlighted atom on the structure must correspond exactly to the highlighted graph node. Inspect its neighbour list and verify that the heavy-atom degree equals the number of listed neighbours.
"""),
        hidden_code("display(atom_explorer('graph'))"),
        markdown(r"""
## Mathematical object produced

At this stage the output is an **attributed graph**:

$$G=(V,E,\{a_i\}_{i\in V},\{e_{ij}\}_{(i,j)\in E}),$$

where $a_i$ contains atom attributes and $e_{ij}$ contains bond attributes. We have not yet chosen their numerical encodings.
"""),
    ],
)


build(
    "02_Atom_Properties.ipynb",
    "Atom Properties",
    [
        markdown(r"""
# Atom Properties

## Aim

Learn the chemical meaning of the candidate information attached to each atom before converting it into numbers.

## Approach

For a selected atom we inspect element, atomic number, formal charge, aromaticity, hybridisation, heavy-atom degree, attached hydrogens, hydrogen-bond donor and acceptor status, ring membership, aromatic-ring membership, mass, and neighbours.

Several distinctions are essential:

- **Formal charge** is integer electron bookkeeping; it is not a measured partial charge.
- **Aromaticity** identifies participation in a special stabilised cyclic conjugated system.
- **Hybridisation** describes local orbital geometry, not the whole molecule.
- **Ring membership** and **aromatic-ring membership** are different questions.
- Donor and acceptor behaviour depends on chemical context, not merely on the element symbol.
"""),
        markdown(r"""
## Interactive chemical inspection

Move between atoms and ask why each property has its displayed value. Compare a carbonyl oxygen, aromatic nitrogen, saturated carbon, formally charged nitrogen, and ring atom.
"""),
        hidden_code("display(atom_explorer('properties'))"),
        markdown(r"""
## Scientific checkpoint

The property record is a chemical description. It becomes a machine-learning input only after we define an encoding, ordering, scale, and treatment of unknown categories.
"""),
    ],
)


build(
    "03_Encoding_Initial_Cell_State.ipynb",
    "Encoding the Initial Cell State",
    [
        markdown(r"""
# Encoding the Initial Cell State

## Aim

Transform chemically meaningful atom properties into a transparent numerical vector while preserving the meaning of every component.

## One-hot encoding

For a categorical variable with $K$ allowed categories, one-hot encoding maps category $k$ to the basis vector

$$\operatorname{onehot}(k)=\mathbf e_k\in\{0,1\}^{K}.$$

Exactly one component is $1$ and the others are $0$. Element and hybridisation use this representation. Binary properties use $0$ or $1$, while small counts and formal charge initially remain explicit numerical values.

## Initial cell state

Let $c_i\in\mathbb R^{d_c}$ be the chemical feature vector for atom $i$. If we reserve $d_f$ free dynamical channels, the initial cellular state is

$$x_i^{(0)}=[c_i,\underbrace{0,\ldots,0}_{d_f\text{ free channels}}].$$

Stacking all $n$ atoms gives

$$X^{(0)}\in\mathbb R^{n\times(d_c+d_f)}.$$

Rows are atoms; columns are state channels. Molecules can have different $n$, but the channel definition must be identical.
"""),
        markdown(r"""
## Interactive encoding laboratory

Select an atom to see the chemical record, its vector, and its row in the complete feature matrix. Ask whether changing atom numbering would merely permute rows rather than change their chemical contents.
"""),
        hidden_code("display(atom_explorer('encoding'))"),
        markdown(r"""
## Scaling postponed deliberately

Atomic mass, electronegativity, van der Waals radius, and polarizability have different units and numerical ranges. Before adding them we will study scaling, dimensional consistency, missing values, and whether they add distinct information.
"""),
    ],
)


build(
    "04_Bonds_and_Local_Neighbourhoods.ipynb",
    "Bonds and Local Neighbourhoods",
    [
        markdown(r"""
# Bonds and Local Neighbourhoods

## Aim

Understand how bond properties define the chemically informed communication pathways of a graph cellular automaton.

## Edge information

For bonded atoms $i$ and $j$, the edge vector $e_{ij}$ may describe bond type, bond order, aromaticity, conjugation, ring membership, and stereochemistry. A provisional local message is

$$m_{ij}^{(t)}=M_{\theta}\!\left(x_i^{(t)},x_j^{(t)},e_{ij}\right).$$

Atom $i$ combines messages from its neighbourhood using an order-independent sum:

$$m_i^{(t)}=\sum_{j\in N(i)}m_{ij}^{(t)}.$$

The sum does not depend on the arbitrary order in which neighbours are listed.
"""),
        markdown(r"""
## Interactive bond inspection

Select an edge and compare single, double, aromatic, conjugated, ring, and non-ring bonds. Aromaticity and conjugation remain separate: every aromatic system is conjugated, while conjugation can also occur outside an aromatic ring.
"""),
        hidden_code("display(bond_explorer())"),
        markdown(r"""
## Scientific checkpoint

The graph topology states **which atoms communicate**. Edge features state **through what chemical relationship they communicate**.
"""),
    ],
)


build(
    "05_First_Transparent_Graph_CA.ipynb",
    "First Transparent Graph Cellular Automaton",
    [
        markdown(r"""
# First Transparent Graph Cellular Automaton

## Aim

Observe a complete graph-CA trajectory using a deliberately simple, non-learned rule whose mathematics can be inspected directly. This is a teaching automaton, not the predictive model.

## Initial scalar state

For this demonstration only, each atom starts with a scalar equal to its Pauling electronegativity:

$$x_i^{(0)}=\chi_i.$$

## Shared local rule

At every synchronous generation, atom $i$ retains a fraction $1-\alpha$ of its own state and receives a fraction $\alpha$ of the mean state of its neighbours:

$$x_i^{(t+1)}=(1-\alpha)x_i^{(t)}+\alpha\frac{1}{d_i}\sum_{j\in N(i)}x_j^{(t)},$$

where $0\leq\alpha\leq1$ and $d_i=|N(i)|$. The same rule acts on every atom at every generation. All new values are calculated from generation $t$ before any are replaced.

In matrix form, with adjacency matrix $A$ and degree matrix $D$, the rule is

$$X^{(t+1)}=\left[(1-\alpha)I+\alpha D^{-1}A\right]X^{(t)}.$$

This rule diffuses differences through the graph and commonly approaches a consensus-like fixed regime. It is useful because convergence can be seen and measured without claiming that this rule is suitable for inhibition prediction.
"""),
        markdown(r"""
## Interactive dynamical laboratory

Change the molecule, coupling strength $\alpha$, and number of generations. The molecular-spacetime image preserves every state. The second panel measures successive global-state movement with

$$d_t=\left\|X^{(t)}-X^{(t-1)}\right\|_2
=\sqrt{\sum_i\left(x_i^{(t)}-x_i^{(t-1)}\right)^2}.$$

A decreasing $d_t$ indicates that successive states are becoming more similar. It is evidence of slowing evolution under this rule, with convergence defined only after choosing a numerical tolerance and persistence requirement.
"""),
        hidden_code("display(ca_explorer())"),
        markdown(r"""
## Connection to the future learned rule

The eventual model will replace this hand-designed averaging operation with a small parameterised message rule $M_\theta$ and update rule $U_\theta$. Its defining CA character will remain: one shared local rule, applied synchronously to every atom and repeatedly through time, with the complete trajectory preserved.
"""),
    ],
)


build(
    "06_Dynamical_Fingerprints.ipynb",
    "Dynamical Fingerprints",
    [
        markdown(r"""
# Dynamical Fingerprints

## Aim

Convert a complete molecular trajectory into a small vector of explicitly defined measurements. This vector is the **dynamical fingerprint** $z$.

## From trajectory to measurement

For the global state $X^{(t)}$, the step-to-step distance is

$$d_t=\left\|X^{(t)}-X^{(t-1)}\right\|_2.$$

Given tolerance $\varepsilon$ and persistence $p$, the convergence time is the first generation $\tau$ for which

$$d_\tau,d_{\tau+1},\ldots,d_{\tau+p-1}<\varepsilon.$$

This is an operational definition: changing the tolerance, persistence, or trajectory length can change the measured convergence time.

If convergence is not observed within $T$ generations, the laboratory records convergence time as $T+1$ and transient length as $T$. This is a numerical sentinel meaning **not observed in the available window**, not a claim that convergence occurs at $T+1$.

Variation across atoms at generation $t$ is

$$\sigma_{\mathrm{atoms}}^2(t)=\frac{1}{n}\sum_i\left(x_i^{(t)}-\bar{x}^{(t)}\right)^2.$$

Temporal variance, oscillation amplitude, lag-one autocorrelation, spectral concentration, and transient length describe different aspects of the same preserved trajectory. They are candidate measurements, not automatically useful predictors.
"""),
        markdown(r"""
## Interactive measurement laboratory

Change the molecule, coupling, trajectory length, and convergence tolerance. The table is the fingerprint vector $z$. Observe which components respond to the numerical definition and which reflect the molecular graph.
"""),
        hidden_code("display(fingerprint_explorer())"),
        markdown(r"""
## Scientific checkpoint

A fingerprint compresses a trajectory and therefore discards information. Every adopted component must be reproducible, permutation-invariant at molecule level, numerically stable, and shown to add useful information under held-out validation.
"""),
    ],
)


build(
    "07_Perturbations_and_Stability.ipynb",
    "Perturbations and Stability",
    [
        markdown(r"""
# Perturbations and Stability

## Aim

Measure how a small, controlled change to the initial molecular state affects the subsequent trajectory.

## Nearby trajectories

Let the reference initial state be $X^{(0)}$ and perturb one atom by a vector of size $\delta_0$:

$$\widetilde X^{(0)}=X^{(0)}+\Delta X^{(0)},\qquad \|\Delta X^{(0)}\|_2=\delta_0.$$

At generation $t$, their separation is

$$\delta_t=\left\|\widetilde X^{(t)}-X^{(t)}\right\|_2.$$

A finite-time divergence rate is

$$\lambda_t=\frac{1}{t}\log\left(\frac{\delta_t}{\delta_0}\right).$$

Positive $\lambda_t$ over a short interval means local finite-time separation. It is not, by itself, evidence of a chaotic attractor. A defensible Lyapunov analysis must state the perturbation size, norm, rescaling protocol, discarded transient, trajectory length, numerical precision, and results across multiple initial conditions.
"""),
        markdown(r"""
## Interactive perturbation laboratory

Choose the perturbed atom and perturbation magnitude. Compare the reference and perturbed trajectories, then inspect whether separation grows, contracts, or remains neutral under the transparent diffusion rule.
"""),
        hidden_code("display(perturbation_explorer())"),
        markdown(r"""
## Scientific checkpoint

The present linear averaging rule is expected to contract many perturbations. Later, the same protocol will test whether a learned rule produces stable sinks, oscillatory regimes, long transients, or genuine sustained sensitivity.
"""),
    ],
)


build(
    "08_Parameterised_Local_Rule.ipynb",
    "A Parameterised Local Rule",
    [
        markdown(r"""
# A Parameterised Local Rule

## Aim

Replace the fixed averaging coefficient with a small shared rule whose parameters can eventually be learned.

## Shared message and update

For this first parameterised scalar rule, atom $i$ receives the mean neighbour state

$$m_i^{(t)}=\frac{1}{d_i}\sum_{j\in N(i)}x_j^{(t)}.$$

Its next state is

$$x_i^{(t+1)}=\tanh\!\left(\theta_{\mathrm{self}}x_i^{(t)}+\theta_{\mathrm{neighbour}}m_i^{(t)}+\theta_{\mathrm{bias}}\right).$$

The three values $\theta=(\theta_{\mathrm{self}},\theta_{\mathrm{neighbour}},\theta_{\mathrm{bias}})$ are shared by every atom and every generation. This sharing is central to the cellular-automaton design. The hyperbolic tangent keeps the scalar state between $-1$ and $1$ and introduces nonlinearity.

This is a teaching rule. The eventual multichannel rule will use atom state, neighbouring state, and bond attributes, but the same local-and-shared logic will remain.
"""),
        markdown(r"""
## Interactive rule laboratory

Change each parameter separately. Identify which settings preserve an atom's own state, amplify neighbour influence, force saturation, produce sign alternation, or approach a fixed regime.
"""),
        hidden_code("display(parameterised_rule_explorer())"),
        markdown(r"""
## Scientific checkpoint

Parameters determine a family of possible dynamical systems. Learning is the process of selecting parameter values using an explicitly defined loss—not evidence that every learned dynamical regime is scientifically useful.
"""),
    ],
)


build(
    "09_Backpropagation_Through_Time.ipynb",
    "Backpropagation Through Time",
    [
        markdown(r"""
# Backpropagation Through Time

## Aim

Understand Learning 1: how prediction error at the end of a trajectory changes the shared local-rule parameters used at every preceding generation.

## Forward computation and loss

Unrolling $T$ generations creates the composition

$$X^{(T)}=F_\theta\!\left(F_\theta\!\left(\cdots F_\theta(X^{(0)})\right)\right).$$

For the teaching example, the molecular prediction is the mean final state

$$\hat y=\frac{1}{n}\sum_i x_i^{(T)},$$

and the squared-error loss is

$$\mathcal L=\frac{1}{2}(\hat y-y)^2.$$

The target $y$ is a standardised version of experimental $\mathrm{pIC}_{50}$ so that it is numerically comparable with the bounded teaching state. Standardisation changes units, not the experimental ordering.

## Reverse-mode chain rule

The gradient receives a contribution from every use of the shared parameter:

$$\frac{\partial\mathcal L}{\partial\theta}
=\sum_{t=0}^{T-1}
\frac{\partial\mathcal L}{\partial X^{(t+1)}}
\frac{\partial X^{(t+1)}}{\partial\theta}.$$

The backward signal may shrink, remain stable, or grow as it passes through many repeated Jacobians. These are the vanishing- and exploding-gradient phenomena.
"""),
        markdown(r"""
## Interactive gradient laboratory

Change the rule parameters and unrolling length. The left panel is the forward trajectory; the right panel shows the norm of the backward adjoint at every generation. The table gives exact analytical derivatives of the loss.
"""),
        hidden_code("display(bptt_explorer())"),
        markdown(r"""
## Scientific checkpoint

Backpropagation computes a gradient; an optimiser uses that gradient to propose a parameter update. Learning rate, optimiser, regularisation, stopping rule, and data splitting remain separate experimental choices.
"""),
    ],
)


build(
    "10_Ridge_Regression_Readout.ipynb",
    "Ridge Regression Readout",
    [
        markdown(r"""
# Ridge Regression Readout

## Aim

Understand Learning 2: map a dynamical fingerprint $z$ to a continuous inhibition measurement while controlling coefficient magnitude.

## Linear readout

For fingerprint $z\in\mathbb R^p$, the prediction is

$$\widehat{\mathrm{pIC}}_{50}=\beta_0+z^\mathsf T\beta.$$

Ridge regression chooses coefficients by minimising

$$\sum_{k=1}^{N}\left(y_k-\beta_0-z_k^\mathsf T\beta\right)^2
+\lambda\|\beta\|_2^2.$$

After centring the target and standardising each fingerprint component, the closed-form coefficient estimate is

$$\hat\beta=(Z^\mathsf T Z+\lambda I)^{-1}Z^\mathsf T y.$$

The penalty $\lambda\geq0$ trades training fit against coefficient size. The intercept is not penalised. Standardisation is learned from the training data only whenever held-out predictions are made.
"""),
        markdown(r"""
## Interactive regularisation laboratory

Change $\lambda$ across several orders of magnitude. Observe the training fit and coefficient norm. This panel deliberately shows a teaching-set fit; it does not estimate generalisation performance.
"""),
        hidden_code("display(ridge_explorer())"),
        markdown(r"""
## Scientific checkpoint

Small training error can coexist with poor prediction of new molecules. Ridge regularisation constrains the readout, but only a held-out evaluation can measure generalisation.
"""),
    ],
)


build(
    "11_Scientific_Validation.ipynb",
    "Scientific Validation",
    [
        markdown(r"""
# Scientific Validation

## Aim

Test whether trajectory-derived information predicts held-out inhibition better than a transparent molecular-descriptor baseline.

## Held-out prediction

For observations $y_k$ and predictions $\hat y_k$, the root-mean-square error is

$$\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{k=1}^{N}(y_k-\hat y_k)^2},$$

and the mean absolute error is

$$\operatorname{MAE}=\frac{1}{N}\sum_{k=1}^{N}|y_k-\hat y_k|.$$

The descriptor baseline uses molecular weight, calculated logP, topological polar surface area, ring count, and rotatable-bond count. The dynamical model uses the fingerprint from notebook 06. Both use the same ridge readout and penalty so that the representation—not readout complexity—is being compared.

## Two illustrative split schemes

- **Leave one molecule out:** fit on 11 molecules and predict the twelfth, repeated 12 times.
- **Leave one CYP out:** fit on three CYP groups and predict all three molecules in the unseen CYP group, repeated four times.

All scaling parameters and ridge coefficients are fitted inside each training fold. The held-out labels never participate in fitting that fold.
"""),
        markdown(r"""
## Interactive validation laboratory

Compare split schemes and regularisation. The 12-molecule smoke-test set is intentionally tiny and deliberately selected, so these values teach the procedure rather than establish model performance.
"""),
        hidden_code("display(validation_explorer())"),
        markdown(r"""
## Decision standard for the future dataset

The trajectory representation earns adoption only if it improves appropriately repeated, chemically defensible held-out evaluation; remains stable across seeds and reasonable hyperparameters; survives comparison with simple baselines; and provides interpretable dynamical measurements rather than accidental leakage.
"""),
    ],
)

print("Built 12 teaching notebooks in", NOTEBOOKS)
