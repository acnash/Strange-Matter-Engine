# Structure–dynamics publication campaign

## Design

The frozen Kuramoto–Sakaguchi Graph-CA was evaluated on 258 held-out molecule–CYP cases, balanced across the four CYP endpoints and stratified across the earlier dynamical screen. Each largest Lyapunov exponent used a 1000-generation burn-in, repeated perturbations, circular phase distance, and repeated Benettin renormalisation. The chemical analysis contains 2D descriptors, graph topology, bond composition, and reproducible ETKDG 3D conformer descriptors.

Scaffold-cluster bootstrap intervals preserve dependence among molecules sharing a Bemis–Murcko scaffold. Predictive tests use scaffold-grouped cross-validation. The intervention experiment freezes every learned parameter and changes individual message-passing bonds, bond identities, ring status, or chemically meaningful atom-feature groups in trajectories 7 and 8.

## Principal results

- Strongest univariate structural association: **algebraic connectivity**, Spearman rho -0.235, scaffold-bootstrap 95% interval [-0.352, -0.118].
- Extra Trees scaffold-held-out performance: R² 0.051, Spearman rho 0.253.
- Largest intervention effect: **OCNT-2328784**, bond type change at 1-2:SINGLE->double, changing the largest exponent by +0.00367 per generation.

## Interpretation boundary

Descriptor associations identify structural correlates of the learned dynamics. The frozen-model interventions provide direct computational evidence that particular graph connections and encoded chemical features control the measured instability. The 3D descriptors are correlates of molecular constitution because the Graph-CA receives atom and bond features rather than Cartesian coordinates.

## Reproducible outputs

- `structure_dynamics_population.csv`: cohort descriptors and repeated Lyapunov estimates.
- `descriptor_correlations.csv` and `scaffold_bootstrap_correlations.csv`: univariate statistics.
- `scaffold_grouped_model_performance.csv`: scaffold-held-out multivariate tests.
- `causal_interventions_with_effects.csv`: every frozen-model intervention and effect size.
- `figures/structure_dynamics_publication_composite.png`: white-background four-panel publication plate.
- `figures/structure_dynamics_publication_composite.pdf`: vector publication plate.
- `figures/`: individual publication-resolution panels and exploratory dark-background figures.
