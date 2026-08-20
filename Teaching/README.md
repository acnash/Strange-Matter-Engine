# Teaching

This directory is the evolving course for Strange Matter Engine. Every equation, algorithm, chemical representation, and scientific claim used by the project should be explainable from the material recorded here.

The notes are divided into subjects as they might be taught at a university:

1. [Chemistry](Chemistry.md) — molecular structure, atomic and bond properties, and SMILES.
2. [Molecular Standardisation](Molecular_Standardisation.md) — conservative parsing, fragment handling, charge, stereochemistry, canonical identity, and provenance.
3. [Pharmacology](Pharmacology.md) — CYP inhibition, concentration–response measurements, `IC50`, and `pIC50`.
4. [Graph Theory](Graph_Theory.md) — molecules as graphs, neighbourhoods, and graph representations.
5. [Emergence](Emergence.md) — cellular automata, local rules, and collective behaviour.
6. [Dynamics](Dynamics.md) — trajectories, convergence, attractors, oscillations, and perturbations.
7. [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) — fixed chemical and bond information alongside evolving dynamical channels.
8. [Machine Learning](Machine_Learning.md) — overview of loss, the two learned components, and model training.
9. [Backpropagation](Backpropagation.md) — derivatives, the chain rule, backpropagation through the graph CA, and learning the shared rule `θ`.
10. [Ridge Regression](Ridge_Regression.md) — regularised linear regression, dynamical fingerprints, and learning the pIC50 readout `β`.
11. [End-to-End Joint Training](End_to_End_Joint_Training.md) — how one pIC50 loss jointly teaches the CA and ridge-regularised readout.
12. [Validation and Statistics](Validation_and_Statistics.md) — generalisation, data splitting, baselines, and honest evaluation.
13. [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) — molecular and scaffold grouping, nested model selection, leakage prevention, and production evaluation.
14. [Scientific Visualisation](Scientific_Visualisation.md) — making the model's real dynamics visible and interpretable.

## Learning rule

Nothing enters the final model merely because it is conventional or available in a software library. Before a component is adopted, we will understand:

- what problem it solves;
- what every input, output, parameter, and mathematical term means;
- how it transforms information;
- what assumptions it makes;
- how it can fail; and
- how we will test whether it helps.

These documents will grow alongside the implementation. They are teaching notes, design records, and a defence against accidental black-box modelling.

## Visual laboratory

The [interactive notebook course](../notebooks/README.md) turns these subjects into code-hidden visual experiments using the 12-molecule development set. Static, code-hidden exports are retained in the [rendered teaching laboratory](../docs/teaching_lab/README.md).
