# Grouped Nested Cross-Validation for Molecular Models

## Learning objective

Grouped nested cross-validation answers two separate questions without allowing one answer to contaminate the other:

1. **Model selection:** which hyperparameters and design choices work best using training data alone?
2. **Model assessment:** how well does that complete selection procedure generalise to chemically unfamiliar molecules?

For Strange Matter Engine, the unit we ultimately want to generalise to is a new molecule. Our validation design must therefore keep the same molecule—and deliberately similar chemical scaffolds—out of both sides of a training/validation boundary.

## The decision for Strange Matter Engine

We will use **chemically grouped nested cross-validation**:

- every measurement belonging to the same molecule stays in one group;
- all CYP records for that molecule remain in the same fold;
- molecules sharing a Bemis–Murcko scaffold are grouped together where practical;
- outer folds estimate generalisation;
- inner folds choose hyperparameters;
- preprocessing is fitted again inside every training fold; and
- the official blinded test set remains untouched until the entire pipeline is frozen.

This design is a scientific control against memorisation and chemical leakage.

## 1. Why ordinary row splitting is unsafe

Our direct-inhibition table contains a molecular structure and potentially several CYP-specific pIC50 values. If each molecule–CYP row were split independently, one molecule could appear as:

- a CYP1A2 training record;
- a CYP2C9 training record; and
- a CYP3A4 validation record.

The validation molecule would then already be known to the model. CYP identity differs, but its atoms, bonds, scaffold, descriptors, and much of its chemical information are identical.

Let molecule $m$ have observations

```math
\mathcal O_m=
\{(m,c,y_{mc}):c\in\mathcal C_m\},
```

where:

- $m$ identifies one molecule;
- $c$ identifies one CYP isoform;
- $y_{mc}$ is the measured pIC50 for molecule $m$ against CYP $c$;
- $\mathcal C_m$ is the set of CYP isoforms for which molecule $m$ has an observed pIC50; and
- $\mathcal O_m$ is therefore the set of all observed molecule–CYP records belonging to molecule $m$.

Molecular grouping imposes

```math
\mathcal O_m\subseteq\mathcal D_k
```

for exactly one fold $\mathcal D_k$. The observations of molecule $m$ are never divided across folds.

## 2. Chemical analogue leakage

Keeping exact molecules together prevents identity leakage, but close analogues can still make validation unrealistically easy. Consider a medicinal-chemistry series in which compounds share the same central ring system and differ by one substituent. If members of that series appear in both training and validation, the model may interpolate within a familiar family rather than extrapolate to new chemistry.

The resulting score is useful for measuring interpolation, but it is not a strong estimate of performance on an unfamiliar scaffold.

We therefore introduce a chemical grouping function

```math
g(m)=\text{scaffold assigned to molecule }m.
```

Let $q(m)$ denote the fold index assigned to molecule $m$. The split must satisfy

```math
g(m_i)=g(m_j)
\quad\Longrightarrow\quad
q(m_i)=q(m_j).
```

Every molecule with the same grouping scaffold enters the same fold.

## 3. Bemis–Murcko scaffolds

A Bemis–Murcko scaffold retains the main ring systems and the linkers connecting them while removing most terminal side chains. It provides a reproducible approximation to a molecule's structural core.

For example, several compounds may have:

- the same fused bicyclic core;
- different halogen substitutions;
- different terminal alkyl groups; and
- different measured CYP inhibition.

Their complete molecular graphs are different, but their Bemis–Murcko scaffold can be the same. Placing them together prevents a validation compound from being surrounded by near-identical training analogues.

Scaffold grouping is a useful operational definition, not a perfect statement of chemical similarity. Important limitations include:

- acyclic molecules may collapse to an empty scaffold;
- scaffold definitions can group some compounds too broadly or too narrowly;
- stereochemistry and substituent effects may be lost;
- different scaffolds can still be highly similar; and
- one very common scaffold can create an unusually large group.

We will therefore record the grouping algorithm and inspect group sizes, CYP coverage, and chemical diversity before accepting any split.

## 4. Cross-validation

In $K$-fold cross-validation, the available development data are partitioned into $K$ disjoint folds:

```math
\mathcal D=
\mathcal D_1\cup\mathcal D_2\cup\cdots\cup\mathcal D_K,
\qquad
\mathcal D_i\cap\mathcal D_j=\varnothing
\quad(i\ne j).
```

Here $\mathcal D$ is the complete development dataset, $\mathcal D_k$ is fold $k$, and $K$ is the total number of folds. Indices $i$ and $j$ label folds in the disjointness statement.

For fold $k$, the test portion is $\mathcal D_k$ and the fitting portion is

```math
\mathcal D_{-k}=
\bigcup_{j\ne k}\mathcal D_j.
```

Each observation is tested once. If $\widehat y_i^{(-k)}$ denotes the prediction for observation $i$ from a model that did not train on fold $k$, the pooled cross-validated RMSE is

```math
\mathrm{RMSE}_{\rm CV}
=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{K}
\sum_{i\in\mathcal D_k}
\left(y_i-\widehat y_i^{(-k)}\right)^2
}.
```

Here $y_i$ is observation $i$'s measured pIC50, $N$ is the total number of out-of-fold observations, and the superscript $(-k)$ means “fitted without fold $k$.”

These are **out-of-fold predictions**: every prediction is made by a model that excluded the corresponding molecule group from fitting.

## 5. Why cross-validation must be nested

Suppose we try 100 combinations of:

- CA state dimension;
- atom and bond channels;
- number of generations;
- learning rate;
- transition-rule architecture;
- regularisation strength; and
- ridge penalty $\lambda$.

If we choose the best combination using a set of folds and quote its score on those same folds, the score benefits from repeated selection. Even if many designs are equally uninformative, one can look best by chance.

Nested cross-validation separates selection from assessment:

```math
\text{outer loop: estimate generalisation},
```

```math
\text{inner loop: select hyperparameters}.
```

The outer test fold is invisible to every decision made in the corresponding outer iteration.

The accepted search budget, random sampling, promotion, and multi-seed confirmation are defined in [Reproducible Hyperparameter Search](Hyperparameter_Search.md).

## 6. The nested procedure

Let the outer grouped folds be

```math
\mathcal D^{\rm outer}_1,\ldots,\mathcal D^{\rm outer}_{K_{\rm out}}.
```

Here $K_{\rm out}$ is the number of outer folds. Within each outer iteration, $K_{\rm in}$ is the number of inner folds, $\ell$ indexes an inner fold, $\mathcal H$ is the declared set of candidate hyperparameter configurations, and $h$ denotes one candidate from that set.

For outer iteration $k$:

1. reserve $\mathcal D^{\rm outer}_k$ as the outer test fold;
2. use the remaining groups as the outer training set;
3. divide only that outer training set into grouped inner folds;
4. evaluate each candidate configuration $h$ across the inner folds;
5. choose $h_k^*$ using the prespecified inner metric;
6. refit the complete pipeline on all outer-training groups using $h_k^*$; and
7. predict the untouched outer test fold once.

Mathematically, the selected configuration is

```math
h_k^*
=
\underset{h\in\mathcal H}{\arg\min}
\;
\frac{1}{K_{\rm in}}
\sum_{\ell=1}^{K_{\rm in}}
\mathrm{RMSE}
\left(h;\mathcal D_{k,\ell}^{\rm inner}\right).
```

The outer score is then

```math
s_k=
\mathrm{RMSE}
\left(
y_k^{\rm outer},
\widehat y_k^{\rm outer}(h_k^*)
\right).
```

In this expression, $y_k^{\rm outer}$ is the vector of measured pIC50 values in outer fold $k$, $\widehat y_k^{\rm outer}(h_k^*)$ is the corresponding prediction vector from the refitted model using the selected configuration, and $s_k$ is that fold's RMSE. The symbol $\bar s$ below is the arithmetic mean of the $K_{\rm out}$ outer-fold scores.

The final nested estimate is summarised across outer folds:

```math
\bar s=\frac{1}{K_{\rm out}}\sum_{k=1}^{K_{\rm out}}s_k.
```

We will also report variability rather than presenting $\bar s$ as an exact constant.

## 7. A worked CYP example

Imagine 1,000 molecules, with multiple CYP measurements per molecule, grouped into 250 scaffold families. For a five-fold outer split:

- approximately 200 molecules' scaffold groups form each outer test fold;
- the remaining approximately 800 molecules enter inner model selection;
- a four-fold inner split compares candidate hyperparameters;
- the winning inner configuration is refitted on those 800 molecules; and
- it predicts the approximately 200 molecules in the untouched outer fold.

This is repeated five times. Every molecule eventually receives an outer prediction, but no molecule or same-scaffold group is used to select the model that predicts it.

Because group sizes and CYP measurements are unequal, exact fold sizes will differ. We will balance folds at the group level while preserving the grouping constraint.

## 8. What belongs inside each fold

Every operation that learns from data must be fitted using the current training portion only. This includes:

- molecular-feature scaling;
- imputation rules;
- removal of zero-variance features;
- feature selection;
- learned atom embeddings;
- graph-CA parameters $\theta$;
- early stopping;
- dynamical-fingerprint scaling;
- ridge coefficients $\beta$;
- ridge penalty selection;
- generation-count selection; and
- any calibration transformation.

For feature $j$, training-fold standardisation is

```math
z'_{ij}
=
\frac{z_{ij}-\mu_{j,\rm train}}{s_{j,\rm train}}.
```

Here $z_{ij}$ is the unstandardised value of feature $j$ for observation $i$, $z'_{ij}$ is its standardised value, and $\mu_{j,\rm train}$ and $s_{j,\rm train}$ are respectively the mean and standard deviation of feature $j$ calculated from the current training fold only.

The validation or outer-test observations use the same $\mu_{j,\rm train}$ and $s_{j,\rm train}$. Their own values must not influence the transformation.

## 9. Shared CYP model and grouping

Our accepted architecture uses a shared model across CYP1A2, CYP2C9, CYP2D6, and CYP3A4. CYP identity enters as context, but molecular grouping remains the higher-level split constraint.

If molecule $m$ has labels for several enzymes, all those labels stay together. This lets the training fold exploit cross-CYP information while ensuring the outer model confronts a genuinely unseen molecular structure.

We will calculate:

- an overall regression score across available molecule–CYP observations;
- a separate score for each CYP;
- the number of molecules and observations per CYP in every fold; and
- uncertainty or variability across outer folds.

A strong overall score must not hide failure on one enzyme.

## 10. Metrics for pIC50 regression

For residual $r_i=\widehat y_i-y_i$, root mean squared error is

```math
\mathrm{RMSE}
=
\sqrt{\frac1N\sum_{i=1}^{N}r_i^2}.
```

Mean absolute error is

```math
\mathrm{MAE}
=
\frac1N\sum_{i=1}^{N}|r_i|.
```

RMSE gives more influence to large errors. MAE describes the typical absolute error more robustly. Both are in pIC50 units.

An absolute pIC50 error $e$ corresponds to an IC50 concentration factor

```math
10^e.
```

Thus $e=0.3$ is approximately twofold, $e=0.5$ is approximately 3.16-fold, and $e=1$ is tenfold.

Correlation can be informative, but it does not replace error magnitude. A prediction can correlate with the observations while being systematically shifted or badly calibrated.

## 11. Choosing the final production configuration

Nested cross-validation produces several selected configurations $h_1^*,\ldots,h_{K_{\rm out}}^*$, because each outer training set is slightly different. After the validation protocol and search space are fixed, we need a declared rule for choosing one final configuration.

A defensible procedure is:

1. inspect which configurations are consistently selected or perform robustly;
2. use a prespecified aggregation rule or repeat inner cross-validation on all released training data;
3. choose the final hyperparameters without using blinded test labels;
4. refit every learned component on the complete released training set;
5. freeze preprocessing, $\theta$, fingerprint definitions, scaling, $\beta$, and all hyperparameters; and
6. generate predictions for the official blinded test molecules.

The blinded test set is for final inference, not another round of model development.

## 12. Repeated nested cross-validation

One assignment of scaffold groups to folds can be unusually favourable or difficult. Repeating the grouped nested procedure with several deterministic seeds can measure split sensitivity.

If repetition $r$ gives score $\bar s_r$, we can report

```math
\bar s_{\rm repeated}
=
\frac1R\sum_{r=1}^{R}\bar s_r
```

Here $r$ indexes a complete repetition of nested cross-validation, $R$ is the declared number of repetitions, and $\bar s_r$ is the mean outer-fold score from repetition $r$. We report this average along with its dispersion and the complete distribution of outer-fold results.

Repetition increases computational cost substantially because every outer fold contains an inner search. We will choose the number of folds, repetitions, and search budget before examining final results.

## 13. Failure modes and diagnostic checks

We will explicitly test for:

- the same canonical molecule appearing in multiple folds;
- salts, tautomers, stereoisomers, or duplicated records evading identity grouping;
- the same scaffold appearing in multiple folds;
- feature scaling performed before splitting;
- early stopping based on an outer test fold;
- hyperparameters chosen from outer-fold performance;
- CYP imbalance producing folds with inadequate target coverage;
- very large scaffold groups dominating a fold;
- random seeds changed selectively after seeing results; and
- the blinded test set influencing any modelling decision.

We will save fold-assignment tables so every prediction can be traced to the model that produced it.

## 14. What this validation can establish

Grouped nested cross-validation can support the claim:

> Under the declared chemical grouping and training procedure, the model generalises to held-out molecule groups with the reported error distribution.

It does not prove universal generalisation to every possible chemical series, assay, laboratory, or biological context. The claim is conditional on the dataset, endpoint definitions, grouping rule, and applicability domain.

## 15. Our production data flow

```math
\begin{aligned}
\text{released labelled molecules}
&\longrightarrow \text{canonical molecular groups}\\
&\longrightarrow \text{scaffold groups}\\
&\longrightarrow \text{outer grouped folds}\\
&\longrightarrow \text{inner grouped hyperparameter search}\\
&\longrightarrow \text{outer out-of-fold predictions}\\
&\longrightarrow \text{locked final design}\\
&\longrightarrow \text{refit on all released training data}\\
&\longrightarrow \text{frozen prediction on blinded molecules}.
\end{aligned}
```

This process makes the validation boundary part of the scientific model rather than an administrative afterthought.

## Connection to the course

- [Validation and Statistics](Validation_and_Statistics.md) introduces generalisation, leakage, and baselines.
- [Chemistry](Chemistry.md) supplies the structural meaning behind chemical grouping.
- [Backpropagation](Backpropagation.md) explains how graph-CA parameters are fitted inside each fold.
- [Ridge Regression](Ridge_Regression.md) explains the readout and inner selection of $\lambda$.
