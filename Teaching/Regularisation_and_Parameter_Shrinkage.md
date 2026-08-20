# Regularisation and Parameter Shrinkage

## Learning objective

Regularisation discourages a model from relying on unnecessarily large parameter values. It changes the training objective so that predictive fit is balanced against parameter magnitude.

Strange Matter Engine has two different parameter families:

- $\beta$: the 40 fingerprint-to-pIC50 readout coefficients; and
- $\theta$: the parameters governing messages, gates, and repeated graph-CA updates.

They require separate regularisation strengths because they have different roles and numerical behaviour.

## The accepted production decision

The default penalties are

```math
\lambda_\beta=10^{-3}
```

and

```math
\lambda_\theta=10^{-5}.
```

Inner grouped cross-validation will evaluate

```math
\lambda_\beta\in
\left\{
10^{-4},10^{-3},10^{-2}
\right\}
```

and

```math
\lambda_\theta\in
\left\{
10^{-6},10^{-5},10^{-4}
\right\}.
```

Zero-penalty models remain explicit scientific ablations rather than members of every initial tuning run.

## 1. The regularised objective

The complete prototype objective is

```math
\mathcal L(\theta,\beta)
=
\mathcal L_{\rm MSE}
+\lambda_\beta\lVert\beta\rVert_2^2
+\lambda_\theta\lVert\theta\rVert_2^2.
```

The three terms have distinct purposes:

- $\mathcal L_{\rm MSE}$ rewards agreement with experimental pIC50;
- $\lambda_\beta\lVert\beta\rVert_2^2$ constrains the linear readout; and
- $\lambda_\theta\lVert\theta\rVert_2^2$ constrains the graph-CA rule.

The two lambda values are hyperparameters. They are not learned by backpropagation from the same fold on which performance is assessed.

## 2. The L2 norm

For

```math
\beta=
\begin{bmatrix}
\beta_1&\beta_2&\cdots&\beta_p
\end{bmatrix}^{\mathsf T},
```

the squared L2 norm is

```math
\lVert\beta\rVert_2^2
=
\sum_{j=1}^{p}\beta_j^2.
```

For example, if

```math
\beta=
\begin{bmatrix}
2&-1&0.5
\end{bmatrix}^{\mathsf T},
```

then

```math
\lVert\beta\rVert_2^2
=
2^2+(-1)^2+0.5^2
=5.25.
```

Large positive and negative coefficients are penalised equally because both are squared.

## 3. How regularisation changes a gradient

For coefficient $\beta_j$,

```math
\frac{\partial}{\partial\beta_j}
\left(
\lambda_\beta\sum_k\beta_k^2
\right)
=
2\lambda_\beta\beta_j.
```

The full coefficient gradient is

```math
\frac{\partial\mathcal L}{\partial\beta_j}
=
\frac{\partial\mathcal L_{\rm MSE}}{\partial\beta_j}
+2\lambda_\beta\beta_j.
```

If $\beta_j>0$, the penalty gradient is positive and gradient descent pushes it down. If $\beta_j<0$, the penalty gradient is negative and gradient descent pushes it up. In both cases, the force is towards zero.

For the CA parameters,

```math
\nabla_\theta\mathcal L
=
\nabla_\theta\mathcal L_{\rm MSE}
+2\lambda_\theta\theta.
```

## 4. Fit versus simplicity

Training seeks a compromise:

```math
\text{small prediction error}
\quad\text{and}\quad
\text{moderate parameter magnitude}.
```

Increasing a parameter is worthwhile only if the reduction in data loss compensates for the additional penalty.

A model with slightly worse training MSE can generalise better if its parameter values are less tailored to noise in the training molecules.

## 5. Why beta needs ridge shrinkage

The readout uses 40 standardised dynamical fingerprint components:

```math
\widehat y
=
\beta_0+\sum_{j=1}^{40}\beta_jz'_j.
```

Some components can be correlated. For example:

- final channel activity may correlate with trajectory mean;
- temporal variance may correlate with step activity; and
- related dynamical channels may evolve similarly.

Without regularisation, correlated predictors can acquire large coefficients of opposite sign that nearly cancel:

```math
\widehat y
=
8z'_1-7.4z'_2
\qquad
\text{when }z'_1\approx z'_2.
```

The prediction may fit training data, but small changes in the relationship between $z'_1$ and $z'_2$ can cause instability.

Ridge regularisation favours smaller combined coefficient magnitude:

```math
\lVert\beta\rVert_2^2
=
\beta_1^2+\cdots+\beta_{40}^2.
```

It shrinks coefficients continuously rather than deleting features.

## 6. Why theta needs a separate penalty

The graph-CA parameters are reused at every atom and all 16 generations. A weight that is moderately too large can repeatedly amplify its influence.

The $\theta$ penalty can help limit:

- excessive recurrent gains;
- gate saturation;
- highly unstable messages;
- extreme sensitivity to small perturbations; and
- overfitting of the training scaffolds.

Regularisation does not guarantee stable dynamics. We still inspect trajectories, Jacobians, state magnitudes, gates, and gradient norms.

## 7. Why the CA penalty is gentler

The accepted defaults satisfy

```math
\lambda_\theta<\lambda_\beta.
```

The readout directly combines 40 potentially correlated measurements, so meaningful coefficient shrinkage is desirable from the start.

The CA must create useful dynamics through repeated nonlinear updates. An overly strong $\theta$ penalty could collapse the transition rule towards weak, nearly identical trajectories.

The gentler default is a starting hypothesis, not a conclusion. Inner validation tests values an order of magnitude above and below it.

## 8. A simple numerical penalty

Suppose

```math
\lVert\beta\rVert_2^2=6,
\qquad
\lVert\theta\rVert_2^2=20.
```

At the default strengths,

```math
\mathcal L_{\rm readout}
=(10^{-3})(6)
=0.006,
```

and

```math
\mathcal L_{\rm CA}
=(10^{-5})(20)
=0.0002.
```

These values cannot be interpreted without the MSE scale. During training, all three loss components will therefore be recorded separately.

## 9. Feature standardisation and lambda

Ridge regularisation is scale-dependent. If one fingerprint component is numerically tiny, its coefficient may need to be large to have the same predictive effect as another component.

We standardise fingerprint component $j$ inside each training fold:

```math
z'_{ij}
=
\frac{z_{ij}-\mu_{j,\rm train}}
{s_{j,\rm train}+\epsilon}.
```

After standardisation, a unit change represents approximately one training-fold standard deviation. The same penalty then acts more comparably across fingerprint components.

## 10. Parameters normally excluded

The readout intercept $\beta_0$ is not included in the ridge penalty. It sets the mean prediction level rather than sensitivity to a fingerprint component.

Some future model components may also receive special treatment, such as fixed chemical encodings or non-trainable constants. The exact contents of $\theta$ and the penalty mask will be recorded.

## 11. Selecting the two lambdas

Within each outer training fold, the inner grouped folds compare the nine combinations:

```math
3\ \lambda_\beta\text{ values}
\times
3\ \lambda_\theta\text{ values}
=9.
```

Each pair uses the same folds, training budget, seeds, optimiser settings, and evaluation metric.

The outer test fold remains untouched until a pair has been selected and the complete model refitted on the outer training data.

## 12. Why search logarithmically

The candidates differ by powers of ten:

```math
10^{-4}\longrightarrow10^{-3}\longrightarrow10^{-2}.
```

Regularisation strength is naturally explored multiplicatively. The difference between $10^{-6}$ and $10^{-5}$ is a tenfold change even though their decimal difference appears small.

If the best result occurs at a boundary, a later prespecified search can extend the range.

## 13. Zero-penalty ablations

Setting

```math
\lambda_\beta=0
```

removes ridge shrinkage from the readout. Setting

```math
\lambda_\theta=0
```

removes the CA L2 penalty.

These comparisons answer scientific questions:

- Does ridge shrinkage improve held-out prediction and coefficient stability?
- Does CA shrinkage improve training stability or merely suppress useful dynamics?

They will be run as explicit ablations so their interpretation is clear and they do not multiply every initial tuning combination.

## 14. What regularisation does not establish

Small parameter norms do not prove:

- chemical correctness;
- causal interpretation;
- dynamical stability;
- absence of leakage;
- calibration; or
- generalisation beyond the applicability domain.

Regularisation is one control within a larger validation system.

## 15. Monitoring

Every run will record:

- MSE data loss;
- $\beta$ penalty contribution;
- $\theta$ penalty contribution;
- $\lVert\beta\rVert_2$;
- $\lVert\theta\rVert_2$;
- maximum absolute coefficients and CA weights;
- per-fold coefficient stability;
- state and gradient magnitudes; and
- validation RMSE and MAE.

## 16. Prototype specification

The accepted design is:

- default $\lambda_\beta=10^{-3}$;
- default $\lambda_\theta=10^{-5}$;
- a three-value logarithmic range for each;
- explicit penalties in the loss;
- no penalty on the readout intercept;
- training-fold fingerprint standardisation;
- separate reporting of all loss terms; and
- zero-penalty scientific ablations.

## Connection to the course

- [Ridge Regression](Ridge_Regression.md) derives coefficient shrinkage.
- [End-to-End Joint Training](End_to_End_Joint_Training.md) defines the shared objective.
- [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md) explains how penalty gradients are applied.
- [Differentiable Dynamical Fingerprint](Differentiable_Dynamical_Fingerprint.md) defines the 40 readout inputs.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) governs lambda selection.

