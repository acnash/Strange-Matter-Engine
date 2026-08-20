# Ridge Regression and the pIC50 Readout

## Learning objective

Ridge regression answers a different question from backpropagation:

> Given a dynamical fingerprint for each molecule, what stable linear combination of its components best predicts experimental pIC50?

It is the proposed transparent regression readout of Strange Matter Engine. Its coefficients show how the fingerprint enters the prediction, while its regularisation limits unstable reliance on any one component.

## Where it sits in our model

The graph CA turns molecular structure into a trajectory,

$$
G_m,X_m^{(0)}\longrightarrow
(X_m^{(0)},X_m^{(1)},\ldots,X_m^{(T)}),
$$

and fingerprint extraction compresses that trajectory into

$$
z_m=
\begin{bmatrix}
z_{m1}&z_{m2}&\cdots&z_{mp}
\end{bmatrix}^{\!\top}.
$$

Candidate components include convergence time, step-to-step distance, atom and temporal variation, oscillation amplitude, autocorrelation, spectral power, transient length, and perturbation response.

Ridge regression maps $z_m$ to predicted pIC50:

$$
\boxed{
\widehat y_m=\beta_0+\sum_{j=1}^{p}\beta_jz_{mj}
=\beta_0+\beta^\top z_m
}.
$$

Here $\beta_0$ is the intercept and $\beta_j$ is the coefficient for fingerprint component $j$.

## 1. The meaning of a linear prediction

Suppose a standardised fingerprint contains three quantities:

$$
z=
\begin{bmatrix}
\text{convergence time}\\
\text{oscillation amplitude}\\
\text{perturbation response}
\end{bmatrix}
=
\begin{bmatrix}
0.5\\-1.0\\0.2
\end{bmatrix},
$$

and the fitted model is

$$
\widehat y=6.2+0.4z_1-0.3z_2+0.5z_3.
$$

Then

$$
\widehat y
=6.2+0.4(0.5)+(-0.3)(-1.0)+(0.5)(0.2)
=6.8.
$$

The predicted pIC50 is $6.8$, corresponding to

$$
IC_{50}=10^{-6.8}\ \mathrm M\approx1.58\times10^{-7}\ \mathrm M
=0.158\ \mu\mathrm M.
$$

A one-unit increase in a standardised $z_1$, holding other components fixed, changes predicted pIC50 by $0.4$. Because pIC50 is logarithmic, this corresponds to an IC50 factor of

$$
10^{0.4}\approx2.51.
$$

Coefficient signs describe fitted associations conditional on the other features. They are not automatically causal chemical statements.

## 2. Ordinary least squares

For $N$ training molecules, collect the fingerprints as rows of a design matrix

$$
Z=
\begin{bmatrix}
z_1^\top\\
z_2^\top\\
\vdots\\
z_N^\top
\end{bmatrix}
\in\mathbb R^{N\times p},
$$

and the measured pIC50 values in

$$
y=
\begin{bmatrix}
y_1&y_2&\cdots&y_N
\end{bmatrix}^{\!\top}.
$$

After centring $y$ and the columns of $Z$, ordinary least squares chooses $\beta$ to minimise the residual sum of squares:

$$
\mathcal L_{\rm OLS}(\beta)
=\sum_{m=1}^{N}(y_m-z_m^\top\beta)^2
=\|y-Z\beta\|_2^2.
$$

Its gradient is

$$
\nabla_\beta\mathcal L_{\rm OLS}
=-2Z^\top(y-Z\beta).
$$

Setting this to zero gives the normal equations

$$
Z^\top Z\beta=Z^\top y.
$$

When $Z^\top Z$ is invertible,

$$
\widehat\beta_{\rm OLS}=(Z^\top Z)^{-1}Z^\top y.
$$

## 3. Why correlated fingerprints create instability

Dynamical summaries can overlap. For example, a long transient may correlate with a large late step-to-step distance; oscillation amplitude may correlate with spectral power at a dominant frequency.

If two columns of $Z$ are nearly redundant, many combinations of their coefficients can produce similar training predictions. A small change in the data may then cause a large change in the individual coefficients. In matrix language, $Z^\top Z$ is poorly conditioned.

This matters especially when $p$ is not small relative to $N$, as in our 12-molecule smoke-test set. That set teaches the process and tests plumbing; it cannot establish a reliable production relationship with many fitted features.

## 4. The ridge objective

Ridge regression adds an L2 penalty:

$$
\boxed{
\mathcal L_{\rm ridge}(\beta)
=\|y-Z\beta\|_2^2+\lambda\|\beta\|_2^2
}
$$

or equivalently

$$
\mathcal L_{\rm ridge}(\beta)
=\sum_{m=1}^{N}(y_m-z_m^\top\beta)^2
+\lambda\sum_{j=1}^{p}\beta_j^2.
$$

The first term rewards agreement with measured pIC50. The second discourages large coefficients. The hyperparameter $\lambda\ge0$ controls the trade-off:

- $\lambda=0$ recovers ordinary least squares when its solution is defined;
- small $\lambda$ applies mild shrinkage;
- large $\lambda$ drives coefficients closer to zero; and
- $\lambda\rightarrow\infty$ leaves predictions approaching the intercept-only model when features and target are centred.

The intercept is conventionally not penalised because it sets the overall response level rather than sensitivity to a feature.

## 5. Deriving the ridge solution

Expand and differentiate:

$$
\nabla_\beta\mathcal L_{\rm ridge}
=-2Z^\top(y-Z\beta)+2\lambda\beta.
$$

At the minimum,

$$
-2Z^\top y+2Z^\top Z\beta+2\lambda\beta=0.
$$

Therefore

$$
(Z^\top Z+\lambda I)\beta=Z^\top y,
$$

and

$$
\boxed{
\widehat\beta_{\rm ridge}
=(Z^\top Z+\lambda I)^{-1}Z^\top y
}.
$$

Adding $\lambda I$ increases the diagonal eigenvalues of $Z^\top Z$, improving numerical conditioning and reducing coefficient variance.

In computation, this linear system should normally be solved directly rather than forming an explicit matrix inverse. The equation above remains the clearest mathematical description.

## 6. A one-feature numerical example

Assume centred data with

$$
Z=
\begin{bmatrix}
-1\\0\\1
\end{bmatrix},
\qquad
y=
\begin{bmatrix}
-2\\0\\2
\end{bmatrix}.
$$

Then

$$
Z^\top Z=2,
\qquad
Z^\top y=4.
$$

Ordinary least squares gives

$$
\widehat\beta_{\rm OLS}=\frac{4}{2}=2.
$$

With $\lambda=2$, ridge gives

$$
\widehat\beta_{\rm ridge}=\frac{4}{2+2}=1.
$$

The fitted slope is deliberately reduced. Predictions move closer to the mean:

$$
\widehat y_{\rm ridge}=
\begin{bmatrix}
-1\\0\\1
\end{bmatrix},
$$

so training fit is worse than OLS, but the model may be less sensitive to sampling noise. This is the bias–variance trade-off.

## 7. A two-feature correlated example

Suppose $z_1$ and $z_2$ are almost identical. OLS might fit

$$
\widehat y=6.0+8.0z_1-7.1z_2.
$$

The net contribution is modest when $z_1\approx z_2$, yet the individual coefficients are large and cancel. A small disturbance that separates the two features can create a large prediction change.

Ridge might instead fit

$$
\widehat y=6.0+0.48z_1+0.42z_2.
$$

The exact numbers depend on the data and $\lambda$, but the principle is that the penalty prefers a smaller-norm solution among similarly predictive alternatives. Ridge shrinks correlated coefficients; it does not select one and delete the other.

## 8. Why standardisation is essential

Suppose convergence time is measured in generations and perturbation response is $10^{-4}$-scale. Because the penalty is

$$
\lambda\sum_j\beta_j^2,
$$

its effect depends on the numerical scale of each feature.

Within each training fold, standardise feature $j$ using

$$
z'_{mj}=\frac{z_{mj}-\mu_j}{s_j},
$$

where

$$
\mu_j=\frac1{N_{\rm train}}\sum_{m\in\mathrm{train}}z_{mj}
$$

and $s_j$ is its training-fold standard deviation. The held-out molecules use those same $\mu_j$ and $s_j$. Recalculating them with held-out data would leak information.

The target can be centred similarly. If

$$
y'_m=y_m-\bar y_{\rm train},
$$

then predictions return to the pIC50 scale by adding $\bar y_{\rm train}$.

## 9. Choosing lambda honestly

$\lambda$ is a hyperparameter, not a coefficient learned from the final held-out labels. It should be selected using an inner validation procedure contained entirely within the available training data.

For candidate values such as

$$
\lambda\in\{10^{-4},10^{-3},\ldots,10^3,10^4\},
$$

the procedure is:

1. split the current training data into inner training and validation folds;
2. fit scaling using the inner training fold only;
3. fit ridge for each candidate $\lambda$;
4. measure validation error;
5. select $\lambda$ by the predeclared rule;
6. refit on the complete outer training fold; and
7. evaluate once on the untouched outer test fold.

This is nested validation when an outer loop estimates generalisation. Every transformation—including fingerprint scaling and feature selection—belongs inside the fitting boundary.

## 10. Prediction error in pIC50 space

For residual

$$
r_m=\widehat y_m-y_m,
$$

common metrics include

$$
\mathrm{MSE}=\frac1N\sum_{m=1}^N r_m^2,
$$

$$
\mathrm{RMSE}=\sqrt{\frac1N\sum_{m=1}^N r_m^2},
$$

and

$$
\mathrm{MAE}=\frac1N\sum_{m=1}^N|r_m|.
$$

RMSE and MAE are expressed in pIC50 units. A pIC50 error of magnitude $|r|$ corresponds to an IC50 factor of

$$
10^{|r|}.
$$

Thus $0.3$ pIC50 units is approximately a twofold concentration error because $10^{0.3}\approx2.0$; one pIC50 unit is tenfold.

## 11. Ridge and backpropagation together

The two learning mechanisms have different immediate jobs:

$$
\theta:\quad \text{learn the shared atom-to-atom dynamical rule},
$$

$$
\beta:\quad \text{learn the regularised fingerprint-to-pIC50 readout}.
$$

For fixed $\theta$, we can generate $Z_\theta$ and solve

$$
\widehat\beta(\theta)
=\arg\min_\beta
\left\|y-Z_\theta\beta\right\|_2^2
+\lambda\|\beta\|_2^2.
$$

Several scientifically legitimate training designs are possible.

### Staged learning

1. choose or pretrain $\theta$;
2. freeze $\theta$;
3. generate fingerprints $Z_\theta$;
4. fit ridge coefficients $\beta$; and
5. validate the complete frozen pipeline.

This is easiest to interpret because the dynamical representation and readout are separated.

### Alternating learning

1. hold $\theta$ fixed and fit $\beta$;
2. hold or temporarily fix $\beta$ and update $\theta$ using backpropagation;
3. repeat under a predeclared stopping rule.

This allows the representation and readout to adapt to one another.

### End-to-end learning

Treat the ridge-style penalty as part of a differentiable objective:

$$
\mathcal L(\theta,\beta)
=\left\|y-Z_\theta\beta\right\|_2^2
+\lambda_\beta\|\beta\|_2^2
+\lambda_\theta R(\theta).
$$

Backpropagation then computes gradients for both $\theta$ and $\beta$. This is mathematically convenient when the fingerprint components are differentiable. It changes the optimisation procedure, while the readout remains linear and L2-regularised.

The training schedule will be treated as an experimental design choice. We will compare it under identical, leakage-free splits rather than assuming that greater coupling is better.

## 12. What the coefficients can and cannot tell us

After standardisation, coefficient magnitude helps show which fingerprint directions the fitted model relies upon. However:

- correlated inputs share and redistribute weight;
- shrinkage deliberately biases coefficients toward zero;
- coefficient signs may vary across resampled training sets;
- a predictive association is not a causal mechanism; and
- the fingerprint itself depends on learned CA dynamics.

We should therefore report coefficient paths over $\lambda$, stability across folds, feature correlations, and prediction performance. Interpretation is strongest when several lines of evidence agree.

## 13. Baselines and scientific controls

The dynamical fingerprint should be compared with a conventional molecular-descriptor matrix under the same ridge readout:

$$
\text{descriptor baseline}+\text{ridge}
\quad\text{versus}\quad
\text{dynamical fingerprint}+\text{ridge}.
$$

Keeping the readout and validation folds the same isolates the representation as the main changed factor. If the dynamical model improves held-out prediction reproducibly, that supports added information in its representation. If it does not, the result identifies where further scientific work is needed.

## 14. Final model flow

For a new molecule after all choices are frozen:

$$
\text{SMILES}
\rightarrow G
\rightarrow X^{(0)}
\xrightarrow{F_{\widehat\theta},\,T}
\text{trajectory}
\rightarrow z
\xrightarrow[\text{training means/scales}]{\text{standardise}}
z'
\rightarrow
\widehat{\mathrm{pIC50}}
=\widehat\beta_0+\widehat\beta^\top z'.
$$

The unseen molecule does not refit the CA, scaling constants, ridge coefficients, or $\lambda$. It passes through the frozen scientific pipeline.

## Connection to the course

- [Dynamics](Dynamics.md) explains the trajectory measurements entering $z$.
- [Backpropagation](Backpropagation.md) explains how prediction error can teach $\theta$.
- [Machine Learning](Machine_Learning.md) gives the overview of both learned components.
- [Validation and Statistics](Validation_and_Statistics.md) governs honest model comparison.

