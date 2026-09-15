# Genetic programming Graph-CA campaign for CYP3A4

This campaign evolves symbolic cellular update programs for the CYP3A4 encoded
Graph-CA. The molecular representation remains atom-as-cell with typed bonds,
explicit recurrent generations, retained nonlinear trajectories,
backpropagation through time, and a genuine differentiable ridge readout.

The program terminals are the learned reaction signal, current cellular state,
bond-neighbour state difference, and delayed-state difference. Bounded
differentiable primitives comprise addition, subtraction, multiplication,
averaging, negation, hyperbolic tangent, sine, and fixed scaling. Subtree
crossover and mutation evolve trees with maximum depth four. A small parsimony
penalty discourages expression growth without overriding predictive fitness.

The remaining Graph-CA configuration is fixed to the successful hybrid-search
setting: 128 recurrent generations, 24 hidden channels, comprehensive atom
features, multiscale trajectory pooling, dynamic observables, and multiscale
transition-energy summaries. Every candidate's continuous weights are trained
by backpropagation.

The full campaign comprises four genetic generations of 24 programs, each
screened on two scaffold folds, followed by five-fold, two-seed confirmation
of five finalists. This produces 242 planned CUDA runs using five concurrent
workers. The sealed holdout and blind labels are excluded from the complete
search and confirmation process.

## Completed result

All 242 planned CUDA runs completed. Candidate `g00_c02` was selected by the
development confirmation campaign. Its symbolic update drive is
`mean(reaction, neighbour_delta)`, a three-node tree of depth one. Across five
scaffold folds and two seeds, it achieved mean MA-ST-RAE 0.598075 and mean
RMSE 0.772706 pIC50.

After the winner and all ten checkpoints were frozen, the sealed scaffold
holdout was opened once. The resulting GP-CIA-EA-CV-CYP-GCA ensemble retained
CIA-EA-CV-CYP-GCA for CYP1A2, CYP2C9, and CYP2D6 and used the ten-checkpoint
genetic-programming ensemble for CYP3A4.

| Metric | CIA-EA-CV-CYP-GCA | HGS-CIA-EA-CV-CYP-GCA | GP-CIA-EA-CV-CYP-GCA |
|---|---:|---:|---:|
| Point MA-ST-RAE | 0.748490 | **0.743723** | 0.745824 |
| RMSE, pIC50 | 0.847738 | **0.841572** | 0.848265 |
| CYP3A4 point ST-RAE | 0.528032 | **0.508964** | 0.517368 |

The 1,000-resample bootstrap estimate for GP-CIA-EA-CV-CYP-GCA was mean
MA-ST-RAE 0.746602 with a 95% interval from 0.708890 to 0.788464. Bootstrap
macro MAE was 0.614381 pIC50, macro R-squared was 0.292879, macro Spearman rho
was 0.542557, and macro Kendall tau was 0.390626.

The blinded submission contains 750 unique molecules and 3,000 finite
predictions in the required six-column schema. Blind labels were not loaded.
See [GP-CIA-EA-CV-CYP-GCA_METHOD.md](GP-CIA-EA-CV-CYP-GCA_METHOD.md) for the
complete method record.
