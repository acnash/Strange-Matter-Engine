# Strange Matter Engine Visual Laboratory

These notebooks are the interactive teaching environment for the project. They present chemistry, molecular informatics, graph mathematics, and dynamical-systems concepts while keeping implementation cells hidden by default.

## Experiences

1. [Molecular Gallery](00_Molecular_Gallery.ipynb) — explore the 12-molecule development set and distinguish molecular structure from assay labels.
2. [From Molecule to Graph](01_From_Molecule_to_Graph.ipynb) — map atoms to nodes, bonds to edges, and inspect local neighbourhoods.
3. [Atom Properties](02_Atom_Properties.ipynb) — examine the chemical meaning of every candidate atom property.
4. [Encoding the Initial Cell State](03_Encoding_Initial_Cell_State.ipynb) — study one-hot encoding, feature vectors, and the initial state matrix.
5. [Bonds and Local Neighbourhoods](04_Bonds_and_Local_Neighbourhoods.ipynb) — inspect edge properties and the mathematics of local message aggregation.
6. [First Transparent Graph Cellular Automaton](05_First_Transparent_Graph_CA.ipynb) — explore a non-learned diffusion rule, complete trajectories, molecular spacetime, and convergence.

## Opening the laboratory locally

From the repository root:

```text
conda env create -f environment.yml
conda activate strange-matter-engine
jupyter lab notebooks/00_Molecular_Gallery.ipynb
```

In JupyterLab, choose **Run → Run All Cells** to activate the molecule selectors, atom and bond controls, and CA sliders. The implementation cells carry hidden-source metadata; the intended surface is the scientific narrative and its outputs.

## Read-only rendered lessons

Code-hidden HTML snapshots are stored in [`docs/teaching_lab`](../docs/teaching_lab/). They preserve the explanations, equations, and representative neon graphics for reading without executing Jupyter. Interactive controls require the live notebooks because they recalculate scientific results in a Python kernel.

## Visual language

All scientific graphics use a consistent black-background neon system:

- cyan: carbon, primary structures, and principal headings;
- magenta: oxygen, aromatic edges, and secondary quantities;
- violet: nitrogen and intermediate state values;
- lime: selected atoms, high state values, and checkpoints; and
- orange or green: halogens and selected categorical accents.

Colours encode information consistently; they are not decorative replacements for labels.
