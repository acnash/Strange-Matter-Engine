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
8. [Differentiable Dynamical Fingerprint](Differentiable_Dynamical_Fingerprint.md) — the accepted 40-component trajectory representation used during training.
9. [Machine Learning](Machine_Learning.md) — overview of loss, the two learned components, and model training.
10. [Backpropagation](Backpropagation.md) — derivatives, the chain rule, backpropagation through the graph CA, and learning the shared rule `θ`.
11. [Ridge Regression](Ridge_Regression.md) — regularised linear regression, dynamical fingerprints, and learning the pIC50 readout `β`.
12. [End-to-End Joint Training](End_to_End_Joint_Training.md) — how one pIC50 loss jointly teaches the CA and ridge-regularised readout.
13. [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md) — gradients, adaptive moments, separate parameter-group rates, and explicit regularisation.
14. [Regularisation and Parameter Shrinkage](Regularisation_and_Parameter_Shrinkage.md) — separate L2 control of the readout and graph CA.
15. [Mini-Batching Molecular Graphs](Mini_Batching_Molecular_Graphs.md) — molecule-centred batches, four CYP contexts, variable graph sizes, and missing-label masks.
16. [Gradient Clipping](Gradient_Clipping.md) — exploding gradients, global norm clipping, Adam, and recurrent graph-CA stability diagnostics.
17. [Loss Functions and Assay Uncertainty](Loss_Functions_and_Assay_Uncertainty.md) — unweighted MSE, pIC50 residuals, missing labels, reported uncertainty, and robust alternatives.
18. [Hyperparameter Search](Hyperparameter_Search.md) — reproducible random search, promotion, multi-seed confirmation, and selection rules.
19. [Early Stopping and Refitting](Early_Stopping_and_Refitting.md) — checkpoint selection, patience, protected outer tests, and fixed-duration refitting.
20. [Validation and Statistics](Validation_and_Statistics.md) — generalisation, data splitting, baselines, and honest evaluation.
21. [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) — molecular and scaffold grouping, nested model selection, leakage prevention, and production evaluation.
22. [Scientific Visualisation](Scientific_Visualisation.md) — making the model's real dynamics visible and interpretable.
23. [Trajectory Visualisation and PyMOL](Trajectory_Visualisation_and_PyMOL.md) — multi-state molecular animation, display B values, 3D viewing conformers, and the hydrogen visual coda.
24. [Inertial Reaction–Diffusion Graph CA](Inertial_Reaction_Diffusion_Graph_CA.md) — momentum, graph diffusion, restoring forces, nonlinear reactions, and the dynamics observed in prototype two.
25. [Graph CA Transition Rules](Graph_CA_Transition_Rules.md) — five candidate update rules, the corrected prototype lineage, and the mandatory fair-tuning policy for future comparisons.
26. [Rule 1: Gated Residual CA](Transition_Rule_1_Gated_Residual_CA.md) — messages, candidate states, gates, residual updates, and worked calculations.
27. [Rule 2: Inertial Reaction–Diffusion CA](Transition_Rule_2_Inertial_Reaction_Diffusion_CA.md) — velocity, diffusion, restoration, damping, and worked calculations.
28. [Rule 3: Activator–Inhibitor CA](Transition_Rule_3_Activator_Inhibitor_CA.md) — excitation, inhibition, nullclines, waves, and limit cycles.
29. [Rule 4: Coupled-Map CA](Transition_Rule_4_Coupled_Map_CA.md) — nonlinear maps, coupling, bifurcations, synchronisation, and Lyapunov analysis.
30. [Rule 5: Damped Symplectic CA](Transition_Rule_5_Damped_Symplectic_CA.md) — latent coordinates, momentum, potentials, energy, and oscillatory modes.

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
