# Teaching

This directory is the evolving course for Strange Matter Engine. Every equation, algorithm, chemical representation, and scientific claim used by the project should be explainable from the material recorded here.

The notes are divided into subjects as they might be taught at a university:

1. [Chemistry](Chemistry.md) — molecular structure, atomic and bond properties, and SMILES.
2. [Pharmacology](Pharmacology.md) — CYP inhibition, concentration–response measurements, `IC50`, and `pIC50`.
3. [Graph Theory](Graph_Theory.md) — molecules as graphs, neighbourhoods, and graph representations.
4. [Emergence](Emergence.md) — cellular automata, local rules, and collective behaviour.
5. [Dynamics](Dynamics.md) — trajectories, convergence, attractors, oscillations, and perturbations.
6. [Machine Learning](Machine_Learning.md) — loss, backpropagation, ridge regression, and model training.
7. [Validation and Statistics](Validation_and_Statistics.md) — generalisation, data splitting, baselines, and honest evaluation.
8. [Scientific Visualisation](Scientific_Visualisation.md) — making the model's real dynamics visible and interpretable.

## Learning rule

Nothing enters the final model merely because it is conventional or available in a software library. Before a component is adopted, we will understand:

- what problem it solves;
- what every input, output, parameter, and mathematical term means;
- how it transforms information;
- what assumptions it makes;
- how it can fail; and
- how we will test whether it helps.

These documents will grow alongside the implementation. They are teaching notes, design records, and a defence against accidental black-box modelling.
