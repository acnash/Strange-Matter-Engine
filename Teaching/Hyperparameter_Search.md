# Reproducible Hyperparameter Search

## Learning objective

Model parameters are learned by backpropagation. **Hyperparameters** define how the model and its training process are constructed.

Examples include dynamical width, generation count, learning rates, and regularisation strengths. Hyperparameter search is the controlled experiment used to select them without exploiting outer-test or blinded-test outcomes.

## The accepted production decision

The initial search will:

1. run the accepted default configuration as a verified baseline;
2. sample 48 configurations from the declared search space using a recorded seed;
3. evaluate every configuration on identical grouped inner folds;
4. promote the five best stable configurations;
5. repeat those five across three initialisation seeds; and
6. select by mean inner-validation RMSE, using stability and lower complexity as prespecified tie-breakers.

This is a **staged reproducible random search**.

## 1. Parameters and hyperparameters

Learned parameters include:

- graph-CA weights and biases $\theta$;
- gate parameters;
- linear readout coefficients $\beta$; and
- the readout intercept $\beta_0$.

They are updated from gradients within a training run.

Hyperparameters include:

- dynamical width $d_h$;
- generation count $T$;
- learning rates $\eta_\theta$ and $\eta_\beta$;
- penalties $\lambda_\theta$ and $\lambda_\beta$;
- training budget;
- early-stopping settings; and
- random-search design.

Hyperparameters are selected by comparing complete training runs.

## 2. The declared search space

The current structural candidates include

```math
d_h\in\{4,8,16,32\}
```

and

```math
T\in\{8,16,32,64\}.
```

Learning-rate pairs come from

```math
\eta_\theta\in
\{3\times10^{-4},10^{-3},3\times10^{-3}\}
```

and

```math
\eta_\beta\in
\{10^{-3},3\times10^{-3},10^{-2}\}.
```

Regularisation pairs come from

```math
\lambda_\theta\in
\{10^{-6},10^{-5},10^{-4}\}
```

and

```math
\lambda_\beta\in
\{10^{-4},10^{-3},10^{-2}\}.
```

## 3. Why the full grid is large

There are

```math
4\times4\times3\times3\times3\times3
=1296
```

unique configurations.

If one inner evaluation uses $K_{\rm in}$ folds, the number of fitted models is

```math
1296K_{\rm in}.
```

Here $K_{\rm in}$ is the number of grouped inner-validation folds. One separate model fit is required for every configuration–fold combination.

With four inner folds,

```math
1296\times4=5184
```

training runs are required for one outer-training partition before repeated seeds or outer folds are considered.

Because each run unrolls a graph CA through multiple generations, exhaustive enumeration would be expensive.

## 4. Grid search

Grid search evaluates every Cartesian-product combination.

Its advantages are:

- complete coverage of the declared discrete grid;
- simple reproducibility; and
- straightforward comparison.

Its limitations are:

- cost grows multiplicatively;
- many trials vary unimportant dimensions;
- boundary choices remain arbitrary; and
- adding one candidate to each dimension can increase cost sharply.

Grid search is appropriate for a small space but inefficient for our combined design.

## 5. Random search

Random search samples combinations from the declared space.

Let the complete set be $\mathcal H$ with

```math
|\mathcal H|=1296.
```

Using a pseudo-random generator with recorded seed $s_{\rm search}$, we select

```math
\mathcal H_{48}
\subset\mathcal H,
\qquad
|\mathcal H_{48}|=48.
```

Here $s_{\rm search}$ is the random-search seed, $\mathcal H$ is the complete candidate set, and $\mathcal H_{48}$ is the 48-configuration subset selected for first-stage evaluation. Vertical bars around a set denote its number of members.

Sampling is performed without replacement, so a configuration is not accidentally evaluated twice during the first stage.

Random search explores combinations across all dimensions and gives each sampled configuration the same evaluation protocol.

## 6. Why the default is run separately

The accepted default is:

```math
\begin{aligned}
d_h&=8,\\
T&=16,\\
\eta_\theta&=10^{-3},\\
\eta_\beta&=3\times10^{-3},\\
\lambda_\theta&=10^{-5},\\
\lambda_\beta&=10^{-3}.
\end{aligned}
```

It is run even if random sampling would not select it. This provides:

- a known reference;
- an end-to-end pipeline check;
- a benchmark for search gains; and
- a stable comparison across outer folds.

The default does not receive privileged access to validation data.

## 7. Identical inner folds

Every candidate is evaluated on the same grouped inner folds.

If candidate $h$ produces score $s_{h\ell}$ on inner fold $\ell$, its mean is

```math
\bar s_h
=
\frac1{K_{\rm in}}
\sum_{\ell=1}^{K_{\rm in}}s_{h\ell}.
```

Here $h$ identifies one hyperparameter configuration, $\ell$ indexes inner folds, and $s_{h\ell}$ is that configuration's RMSE on inner fold $\ell$.

Using identical folds creates a paired comparison: differences arise from the configuration and training randomness rather than different validation molecules.

## 8. First-stage ranking

The primary quantity is mean inner-validation RMSE:

```math
\bar s_h^{\rm RMSE}.
```

We also retain:

- RMSE for every inner fold;
- per-CYP RMSE and MAE;
- training failures;
- gradient and state stability;
- runtime;
- memory use; and
- parameter count.

A configuration producing non-finite values or failing declared numerical checks is not promoted merely because its surviving folds look favourable.

## 9. Promoting five configurations

After the 48-config stage, the five best stable configurations are promoted:

```math
\mathcal P=
\{h_{(1)},h_{(2)},\ldots,h_{(5)}\}.
```

The subscript $(j)$ denotes rank after applying the prespecified selection and failure rules.

Promotion is based only on inner validation. Outer-test outcomes remain unavailable.

## 10. Why repeat across seeds

Neural training depends on random initialisation, molecule order, and mini-batch composition.

For promoted configuration $h$ and seed $r\in\{1,2,3\}$, let

```math
s_{h\ell r}
```

be its inner-fold score.

The confirmation mean is

```math
\bar s_h
=
\frac1{3K_{\rm in}}
\sum_{r=1}^{3}
\sum_{\ell=1}^{K_{\rm in}}
s_{h\ell r}.
```

We also measure dispersion:

```math
\sigma_h
=
\sqrt{
\frac1{3K_{\rm in}-1}
\sum_{r,\ell}
\left(s_{h\ell r}-\bar s_h\right)^2
}.
```

Here $r$ indexes the three training seeds and $\sigma_h$ is the sample standard deviation of configuration $h$'s fold-and-seed scores.

A small apparent gain accompanied by severe seed instability is weak evidence.

## 11. Tie-breaking

The primary rule is lower mean inner-validation RMSE.

If two configurations are practically tied within a declared tolerance, we prefer:

1. fewer failures;
2. lower fold-and-seed variability;
3. fewer dynamical channels;
4. fewer generations;
5. lower runtime or memory use; and
6. the simpler documented configuration.

This is a parsimony rule: additional complexity must earn its place through meaningful held-out improvement.

The tolerance must be declared before final ranking.

## 12. Search inside nested validation

For outer fold $k$:

1. hold out the outer-test scaffold groups;
2. create inner grouped folds from outer-training molecules only;
3. run the default and 48 sampled configurations;
4. promote and confirm five configurations;
5. select $h_k^*$;
6. refit $h_k^*$ on all outer-training molecules; and
7. predict outer fold $k$ once.

The outer score estimates the performance of the complete search-and-fit procedure, not merely one hand-selected model.

## 13. Randomness must be separated

We distinguish:

- **search seed:** selects the 48 configurations;
- **fold seed:** assigns eligible scaffold groups to folds;
- **initialisation seed:** initialises trainable parameters;
- **shuffle seed:** orders training molecules; and
- **analysis seed:** controls any resampling used for uncertainty summaries.

Reusing one unexplained seed for every purpose makes experiments harder to audit.

## 14. Computational accounting

With $K_{\rm in}$ inner folds, stage 1 requires approximately

```math
(1+48)K_{\rm in}
```

runs, including the default.

Confirmation requires

```math
5\times3\times K_{\rm in}
```

runs if the three-seed evaluation is performed afresh.

For four inner folds, that is

```math
49(4)+5(3)(4)=196+60=256
```

runs per outer-training partition, before any deliberately reused results are accounted for.

The exact accounting will be logged so search cost is visible.

## 15. Search failures

A failed configuration receives a recorded failure status and reason, such as:

- non-finite loss;
- memory exhaustion;
- non-finite states or predictions;
- persistent saturation;
- invalid fingerprint scaling; or
- training-time limit exceeded.

Failures are data about the design space. They are not silently removed and resampled until a favourable set remains.

## 16. Search-space revisions

After a complete declared experiment, evidence may justify a new search space. Examples include:

- the best values repeatedly lying on a boundary;
- one region producing universal instability;
- width or generation count having negligible effect; or
- compute cost overwhelming potential benefit.

A revised search is a new experimental phase with a new configuration version. It must not be presented as the original untouched evaluation.

## 17. What random search can and cannot establish

The procedure can identify a strong configuration among the sampled candidates under the declared folds and budget.

It does not prove:

- that the global optimum was found;
- that unsampled configurations are inferior;
- that performance transfers outside the dataset's applicability domain; or
- that a hyperparameter has a causal chemical interpretation.

The objective is robust model selection, not a claim of mathematical optimality.

## 18. Reproducibility record

We will save:

- complete search-space definition;
- ordered list of all 48 sampled configurations;
- search seed;
- fold definitions;
- initialisation and shuffle seeds;
- training budget and stopping rule;
- scores and diagnostics for every run;
- failures and reasons;
- promoted configurations;
- confirmation results;
- tie-breaking decisions; and
- selected configuration for every outer fold.

## 19. Prototype specification

The accepted search design is:

- one mandatory default baseline;
- 48 randomly sampled configurations without replacement;
- fixed search seed;
- identical grouped inner folds;
- primary ranking by mean inner RMSE;
- stability and complexity diagnostics;
- five promoted configurations;
- three-seed confirmation;
- prespecified parsimony tie-breakers;
- complete failure accounting; and
- no use of outer-test or blinded-test outcomes during selection.

## Connection to the course

- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) defines the inner and outer loops.
- [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md) defines the learning-rate candidates.
- [Regularisation and Parameter Shrinkage](Regularisation_and_Parameter_Shrinkage.md) defines the lambda candidates.
- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) defines dynamical width.
- [Dynamics](Dynamics.md) explains generation count and stability.
