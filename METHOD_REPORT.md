# Strange Matter Engine method report

## Canonical submission method

The production method is **Endpoint-Aligned Cross-Validated CYP-Specialist
Graph Cellular Automata, EA-CV-CYP-GCA v1**. This permanent report describes
the exact method represented by the current challenge submission. The selected
model is fixed from here onward unless an explicit future decision replaces it.

**Challenge track:** Direct inhibition regression

**Predicted endpoints:** CYP1A2, CYP2C9, CYP2D6, and CYP3A4 direct-inhibition
pIC50

**Official blind result:** MA-ST-RAE 1.0071, macro MAE 1.0778, macro R-squared
-0.0715, macro Spearman rho 0.5345, and macro Kendall tau 0.3750

## Scientific objective

EA-CV-CYP-GCA tests whether learned cellular-automata dynamics on molecular
graphs can provide a useful and interpretable representation for CYP inhibition
prediction. Atoms are cells, chemical bonds define local neighbourhoods, and a
shared transition function is applied recurrently across explicit generations.
The retained trajectory contributes to the molecular representation.

The method predicts one continuous pIC50 value for each molecule and CYP
context. Time-dependent inhibition classification is a separate challenge
track and is outside this submission.

## Molecular representation

Each SMILES string is standardized and converted to a bonded molecular graph.
Atom inputs contain chemically organized descriptors selected using labelled
development data, including combinations of periodic, valence, electronic,
ring, and local-neighbour properties. Bond vectors encode single, double,
triple, and aromatic identity together with conjugation, ring membership, and
stereochemical indicators. CYP identity is supplied as task context.

The challenge-blinded molecules enter the workflow only after every model,
hyperparameter, ensemble member, and ridge coefficient has been fixed. Blind
activity labels are unavailable and are never loaded.

## Graph cellular automata

Each ensemble member follows the same computation:

1. Chemical atom features initialize a fixed number of cellular-state channels.
2. Typed bond-conditioned messages pass information between directly bonded atoms.
3. A shared nonlinear local rule updates every atom recurrently for the selected number of generations.
4. Final and intermediate trajectory statistics form a molecular fingerprint.
5. A genuine differentiable ridge readout maps the fingerprint to endpoint pIC50.

The repeated local update and bonded neighbourhood remain present in every
member. The retained rule families are FitzHugh-Nagumo, Gray-Scott,
conservative graph flux, damped symplectic, and delayed memory.

Trajectory pooling contains final atom-state means and variances, time-averaged
states, temporal variance, state-change energy, and molecular mean states at
12.5%, 25%, 50%, 75%, and 100% of the recurrent horizon.

## Endpoint-aligned learning

Four independent nonlinear systems were trained, one for each CYP isoform.
Support targets, query targets, backpropagation loss, early stopping, checkpoint
promotion, and the differentiable ridge solve contained observations from the
active CYP only. This endpoint alignment allows each isoform to learn its own
molecular cellular dynamics.

All ten implemented transition rules were screened for every endpoint using
two scaffold folds. For each rule, the screen compared an established
configuration with a CYP-directed alternative varying chemical features,
trajectory length, and ridge regularization. The three leading
rule-configuration pairs per endpoint advanced to five-fold scaffold
confirmation with two training seeds.

The final systems are:

| Endpoint | Selected transition rules | Final ridge penalty |
|---|---|---:|
| CYP1A2 | FitzHugh-Nagumo, Gray-Scott, conservative graph flux | 1000 |
| CYP2C9 | Gray-Scott, damped symplectic | 100 |
| CYP2D6 | Delayed memory, FitzHugh-Nagumo | 1000 |
| CYP3A4 | Damped symplectic, FitzHugh-Nagumo, delayed memory | 100 |

Each selected rule contributes ten frozen predictions per blind molecule,
comprising five scaffold folds and two seeds. Predictions are averaged within
each rule. A saved endpoint-specific sparse ridge combination then generates
the final pIC50 prediction.

## Differentiable ridge readout

Training batches are divided by molecule into support and query subsets. The
Graph-CA generates support fingerprint matrix $F_s$, standardized column-wise
to $Z_s$. With centered targets $y_s-\bar y_s$, the ridge coefficients are
obtained by the differentiable closed-form solve

```math
\beta=\left(Z_s^{\mathsf T}Z_s+\lambda I\right)^{-1}
Z_s^{\mathsf T}(y_s-\bar y_s).
```

For query fingerprint $f_q$, prediction is

```math
\widehat y_q=\bar y_s+
\left(\frac{f_q-\bar F_s}{s_F}\right)^{\mathsf T}\beta.
```

The intercept is excluded from the ridge penalty. The solve remains connected
to the Graph-CA computation graph, so query-loss gradients pass through the
ridge coefficients and every recurrent generation. Adam updates the nonlinear
initialization, message, reaction, and transition-rule parameters using a
cosine learning-rate schedule, gradient clipping, and cellular-automata L2
regularization.

## Model selection and validation

Compounds were grouped by standardized Bemis-Murcko scaffold. A fixed 20% set
of scaffold groups was reserved as a sealed holdout, producing 5,216 fitting
observations and 1,309 sealed observations. No scaffold group occurred in both
partitions.

MA-ST-RAE was the primary selection metric. RMSE, MAE, R-squared, Spearman rho,
and Kendall tau were retained as complementary diagnostics. Sparse rule-subset
selection and ridge-penalty selection used out-of-fold predictions from the
fitting pool. Final uncertainty was calculated from 1,000 bootstrap resamples.
The sealed holdout did not influence hyperparameters, early stopping, rule
selection, or ensemble weights.

## Sealed validation result

| Metric | EA-CV-CYP-GCA result |
|---|---:|
| Point MA-ST-RAE | 0.754503 |
| Bootstrap mean MA-ST-RAE | 0.755073 |
| MA-ST-RAE 95% bootstrap interval | 0.717656 to 0.797014 |
| RMSE | 0.852280 pIC50 |
| Bootstrap macro MAE | 0.621833 pIC50 |
| Bootstrap macro R-squared | 0.286429 |
| Bootstrap macro Spearman rho | 0.532744 |
| Bootstrap macro Kendall tau | 0.381552 |

Endpoint point ST-RAE values were 0.820810 for CYP1A2, 0.724620 for CYP2C9,
0.938153 for CYP2D6, and 0.534428 for CYP3A4.

## Official blind evaluation

The challenge organisers evaluated the frozen EA-CV-CYP-GCA submission against
labels unavailable during development. The recorded result was:

| Official metric | Result |
|---|---:|
| MA-ST-RAE | **1.0071** |
| Macro MAE | **1.0778** |
| Macro R-squared | **-0.0715** |
| Macro Spearman rho | **0.5345** |
| Macro Kendall tau | **0.3750** |

The submission was recorded at rank 99 of 111 on 1 September 2026. Rank is a
time-specific snapshot because leaderboard membership changes, while the
metric values provide the stable official evaluation record.

## Blinded inference and submission

Frozen inference generated four predictions for each of the 750 blinded
molecules, giving 3,000 finite endpoint values. The inference manifest records
`labels_loaded: false`, 750 unique molecule names, complete finite predictions,
and successful schema validation.

The regression submission contains exactly these six columns:

```text
SMILES
Molecule_Name
CYP1A2_pIC50_direct_inhibition
CYP2C9_pIC50_direct_inhibition
CYP2D6_pIC50_direct_inhibition
CYP3A4_pIC50_direct_inhibition
```

## Reproducibility and artifacts

The canonical artifacts are stored in
[`results/production_endpoint_aligned_cv_cyp_gca_v1`](results/production_endpoint_aligned_cv_cyp_gca_v1):

- [`endpoint_aligned_cv_cyp_gca_submission.csv`](results/production_endpoint_aligned_cv_cyp_gca_v1/endpoint_aligned_cv_cyp_gca_submission.csv), the challenge-ready local submission;
- [`study_summary.json`](results/production_endpoint_aligned_cv_cyp_gca_v1/study_summary.json), the selected configurations and sealed evaluation;
- [`screening_summary.json`](results/production_endpoint_aligned_cv_cyp_gca_v1/screening_summary.json), the endpoint-specific screen;
- [`inference_manifest.json`](results/production_endpoint_aligned_cv_cyp_gca_v1/inference_manifest.json), the frozen inference and schema record; and
- [`scripts/run_cv_cyp_specialist_gca.py`](scripts/run_cv_cyp_specialist_gca.py), the canonical training, validation, and inference runner.

The production runner is pinned to EA-CV-CYP-GCA. Historical refinement code
and artifacts remain available for provenance and do not define the production
submission method.

## Limitations

The internal estimate comes from one dataset and one scaffold-aware partition.
CYP2D6 remains the weakest endpoint by sealed ST-RAE. The difference between
sealed MA-ST-RAE 0.754503 and official blind MA-ST-RAE 1.0071 indicates a
substantial distribution shift or calibration limitation. The official blind
result therefore provides the principal evidence for choosing this method as
the canonical submission.
