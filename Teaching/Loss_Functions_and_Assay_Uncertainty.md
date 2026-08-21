# Loss Functions and Assay Uncertainty

## Learning objective

A loss function converts prediction errors into the numerical objective used to train a model. It therefore defines which errors matter, how strongly they matter, and what the optimiser attempts to reduce.

For the initial Strange Matter Engine prototype, every observed direct-inhibition pIC50 value will contribute equally through unweighted mean squared error.

## The accepted production decision

For observed molecule–CYP pairs,

```math
\mathcal L_{\rm MSE}
=
\frac1{N_{\rm obs}}
\sum_{i=1}^{N_{\rm obs}}
\left(\widehat y_i-y_i\right)^2.
```

Here:

- $y_i$ is experimental pIC50;
- $\widehat y_i$ is predicted pIC50; and
- $N_{\rm obs}$ is the number of observed labels contributing to the loss.

Every observed label receives the same initial weight. Reported confidence intervals and standard deviations are retained for diagnostics and later uncertainty-aware experiments.

## 1. Residuals

The residual for observation $i$ is

```math
r_i=\widehat y_i-y_i.
```

Its sign gives the direction of error:

- $r_i>0$: predicted pIC50 is too high;
- $r_i<0$: predicted pIC50 is too low; and
- $r_i=0$: prediction equals the measured value.

Because larger pIC50 means lower IC50, a positive residual means the model predicts stronger inhibition than measured.

## 2. Squared error

The squared error is

```math
r_i^2=(\widehat y_i-y_i)^2.
```

Squaring has three important consequences:

- positive and negative errors cannot cancel;
- larger errors receive disproportionately greater influence; and
- the function is smooth and differentiable.

For example,

```math
0.2^2=0.04,
\qquad
0.5^2=0.25,
\qquad
1.0^2=1.0.
```

An error five times larger contributes 25 times the squared loss.

## 3. Mean squared error

MSE averages squared residuals:

```math
\mathrm{MSE}
=
\frac1{N_{\rm obs}}\sum_{i=1}^{N_{\rm obs}}r_i^2.
```

Dividing by $N_{\rm obs}$ makes losses more comparable across batches containing different numbers of observed labels. The notation matches the definition at the start of this chapter: $N_{\rm obs}$ is the number of observed labels in the collection being averaged.

MSE is measured in squared pIC50 units. For reporting, root mean squared error returns to pIC50 units:

```math
\mathrm{RMSE}
=
\sqrt{\mathrm{MSE}}.
```

The training objective can use MSE while evaluation reports both RMSE and mean absolute error.

## 4. A numerical example

Suppose three experimental and predicted values are

```math
y=
\begin{bmatrix}
6.0&7.0&5.5
\end{bmatrix}^{\mathsf T},
\qquad
\widehat y=
\begin{bmatrix}
6.2&6.5&6.0
\end{bmatrix}^{\mathsf T}.
```

The residuals are

```math
r=
\begin{bmatrix}
0.2&-0.5&0.5
\end{bmatrix}^{\mathsf T}.
```

Therefore

```math
\begin{aligned}
\mathrm{MSE}
&=
\frac{0.2^2+(-0.5)^2+0.5^2}{3}\\
&=
\frac{0.04+0.25+0.25}{3}\\
&=0.18,
\end{aligned}
```

and

```math
\mathrm{RMSE}
=
\sqrt{0.18}
\approx0.424.
```

## 5. Meaning in IC50 space

pIC50 is logarithmic:

```math
\mathrm{pIC50}
=
-\log_{10}
\left(
\mathrm{IC50\ in\ molar\ units}
\right).
```

An absolute error $e$ in pIC50 corresponds to an IC50 concentration factor

```math
10^e.
```

Examples are:

```math
10^{0.3}\approx2.0,
\qquad
10^{0.5}\approx3.16,
\qquad
10^1=10.
```

Thus an RMSE of 0.5 pIC50 units represents a characteristic error scale of approximately 3.16-fold in IC50, although RMSE itself should not be transformed as if every residual had identical magnitude.

## 6. Missing labels

Let the binary mask $M_{mc}$ equal 1 when molecule $m$ has an observed value for CYP $c$, and 0 otherwise. Here $m$ indexes molecules and $c$ indexes CYP isoforms. The batch data loss is

```math
\mathcal L_{\rm data}
=
\frac{
\sum_m\sum_c
M_{mc}
(\widehat y_{mc}-y_{mc})^2
}{
\sum_m\sum_c M_{mc}
}.
```

Missing labels contribute neither an error nor a count. They are not replaced by zero, a mean, a detection limit, or a guessed potency.

## 7. What unweighted means

The general weighted MSE is

```math
\mathcal L_{\rm weighted}
=
\frac{
\sum_iw_i(\widehat y_i-y_i)^2
}{
\sum_iw_i
}.
```

Unweighted MSE sets

```math
w_i=1
```

for every observed label. “Unweighted” does not mean that large errors are ignored; squaring still gives them greater influence. It means that no observation receives an additional externally assigned importance factor.

## 8. Inverse-variance weighting

If observation $i$ has known standard uncertainty $\sigma_i$, a common proposal is

```math
w_i=\frac1{\sigma_i^2}.
```

Under a Gaussian measurement model with correctly estimated, comparable variances, this can give more influence to precise observations.

For example,

```math
\sigma_1=0.05,\qquad\sigma_2=0.50
```

would produce

```math
\frac{w_1}{w_2}
=
\frac{1/0.05^2}{1/0.50^2}
=100.
```

The first observation would influence the data loss 100 times more strongly.

## 9. Why uncertainty weighting is postponed

In the released direct-inhibition dataset, pIC50 standard deviations are available for only a minority or subset of measurements:

- CYP1A2: 1,412 of 4,905 molecules;
- CYP2C9: 1,285 of 4,905;
- CYP2D6: 1,493 of 4,905; and
- CYP3A4: 2,335 of 4,905.

A weighting scheme must then decide what weight to assign most observations with missing $\sigma_i$. That decision could dominate the apparent statistical sophistication.

Further concerns include:

- very small standard deviations creating extreme weights;
- uncertainty estimates differing in reliability across assays;
- reported uncertainty describing curve fitting but not every source of experimental error;
- weights changing CYP balance; and
- a small precise subgroup dominating representation learning.

Unweighted MSE gives the initial prototype a clear and reproducible objective.

## 10. Retaining uncertainty information

Postponing uncertainty weighting does not mean discarding uncertainty. We will retain:

- reported standard deviations;
- lower and upper confidence limits;
- CYP identity;
- missingness indicators;
- assay provenance; and
- any quality-control annotations.

We can then compare residuals with reported uncertainty after training.

## 11. Uncertainty diagnostics

Useful questions include:

- Are larger residuals associated with larger reported standard deviations?
- Does performance differ between records with and without uncertainty estimates?
- Are confidence intervals systematically wider for one CYP?
- Are extreme errors concentrated in low-confidence measurements?
- Does uncertainty missingness correlate with chemical series or potency?

A standardised residual for records with uncertainty is

```math
u_i=
\frac{\widehat y_i-y_i}{\sigma_i}.
```

This is a diagnostic quantity. It is not automatically the training objective.

## 12. Outliers and MSE

Because errors are squared, MSE is sensitive to extreme residuals. A large residual may arise from:

- genuine unusual chemistry;
- an activity cliff;
- model failure;
- incorrect molecular standardisation;
- assay noise;
- a transcription problem; or
- an applicability-domain violation.

We will investigate large residuals scientifically. We will not remove observations merely because the model predicts them poorly.

## 13. Robust alternatives

Mean absolute error uses

```math
\mathrm{MAE}
=
\frac1N\sum_i|r_i|.
```

It is less dominated by extreme errors but has a non-smooth point at zero.

Huber loss behaves quadratically for small residuals and linearly for large ones. It can reduce outlier influence while retaining smooth optimisation.

These remain controlled alternatives. The first prototype uses MSE so the objective and gradient are simple and directly aligned with RMSE-style assessment.

## 14. CYP-specific reporting

The shared training loss pools observed examples from all four CYPs. We will also calculate

```math
\mathrm{RMSE}_c
=
\sqrt{
\frac1{N_c}
\sum_{i\in c}r_i^2
}
```

for each CYP $c$.

An acceptable overall score must not conceal systematic failure on one enzyme.

## 15. Interaction with regularisation

The full training objective is

```math
\mathcal L
=
\mathcal L_{\rm MSE}
+\lambda_\beta\lVert\beta\rVert_2^2
+\lambda_\theta\lVert\theta\rVert_2^2.
```

The MSE term measures disagreement with experiment. The regularisation terms constrain model parameters. They solve different problems and must be reported separately during training.

## 16. Validation boundaries

Any future uncertainty-weighting rule must be fixed using training data only. Weight floors, caps, missing-uncertainty substitutions, and robust-loss thresholds are hyperparameters.

They belong inside inner grouped cross-validation and must not be selected by examining outer-fold or blinded-test outcomes.

## 17. Prototype specification

The accepted loss design is:

- unweighted MSE over observed pIC50 values;
- equal external weight for every observed molecule–CYP pair;
- missing labels masked;
- explicit division by the observed count;
- separate L2 penalties on $\beta$ and $\theta$;
- uncertainty fields retained for diagnostics;
- overall and per-CYP RMSE and MAE reported; and
- uncertainty-aware objectives postponed to controlled ablations.

## Connection to the course

- [Pharmacology](Pharmacology.md) defines pIC50 and assay dependence.
- [Mini-Batching Molecular Graphs](Mini_Batching_Molecular_Graphs.md) defines masked batch construction.
- [End-to-End Joint Training](End_to_End_Joint_Training.md) connects the loss to $\theta$ and $\beta$.
- [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md) explains how loss gradients become updates.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) controls future loss-function comparisons.
