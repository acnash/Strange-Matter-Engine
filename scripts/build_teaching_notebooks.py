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

print("Built 6 teaching notebooks in", NOTEBOOKS)
