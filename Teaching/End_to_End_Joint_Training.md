# End-to-End Joint Training

## Learning objective

End-to-end joint training means that one measured pIC50 error teaches both:

- the **linear readout coefficients** $\beta$, which convert a dynamical fingerprint into a prediction; and
- the **graph cellular-automaton parameters** $\theta$, which create the trajectory from which that fingerprint is measured.

The complete prediction pathway is treated as one connected mathematical function. Backpropagation applies the chain rule through that entire pathway.

## The accepted production decision

For the first prototype, we will train the graph CA and the pIC50 readout jointly:

```math
\text{molecular graph}
\longrightarrow
\text{CA trajectory}
\longrightarrow
\text{differentiable fingerprint}
\longrightarrow
\text{linear pIC50 prediction}
\longrightarrow
\text{loss}.
```

The learning signal then travels in the reverse direction:

```math
\text{loss}
\longrightarrow
\beta
\longrightarrow
\text{fingerprint}
\longrightarrow
\text{all CA generations}
\longrightarrow
\theta.
```

This is **Learning 1 and Learning 2 in one optimisation process**:

- backpropagation is the derivative-calculation mechanism;
- the optimiser updates the parameters;
- the CA parameters learn the molecular dynamics; and
- the linear readout learns a ridge-regularised mapping to pIC50.

## 1. The four stages of one prediction

For molecule–CYP example $m$, the model performs four forward stages.

### Stage A: initialise the atom states

Fixed chemical channels $C_m$, bond channels $E_m$, neutral dynamical channels $H_m^{(0)}$, and CYP context $c_m$ enter the model:

```math
H_m^{(0)}=0.
```

### Stage B: evolve the graph CA

The shared update rule is applied for $T=16$ generations:

```math
H_m^{(t+1)}
=
F_\theta\!\left(
H_m^{(t)},C_m,E_m,c_m
\right),
\qquad
t=0,\ldots,15.
```

This produces the complete trajectory

```math
\mathcal T_m=
\left(
H_m^{(0)},H_m^{(1)},\ldots,H_m^{(16)}
\right).
```

### Stage C: calculate the dynamical fingerprint

A differentiable summary function $S$ converts the variable-sized trajectory into a fixed-length vector:

```math
z_m=S(\mathcal T_m).
```

### Stage D: predict pIC50

The linear readout calculates

```math
\widehat y_m
=
\beta_0+\beta^{\mathsf T}z_m.
```

The prediction $\widehat y_m$ is compared with experimental pIC50 $y_m$.

## 2. What “joint” means

The prediction depends on both parameter sets:

```math
\widehat y_m
=
\widehat y_m(\theta,\beta).
```

Changing $\beta$ changes how the existing fingerprint is interpreted. Changing $\theta$ changes the CA trajectory and therefore changes the fingerprint itself.

Joint training repeatedly performs:

1. a forward pass using the current $\theta$ and $\beta$;
2. loss calculation;
3. backpropagation to obtain gradients for both parameter sets; and
4. simultaneous parameter updates.

Neither parameter set is permanently fitted before the other begins.

## 3. The training objective

For $N$ training examples, the proposed objective is

```math
\mathcal L(\theta,\beta)
=
\frac1N
\sum_{m=1}^{N}
\left(\widehat y_m-y_m\right)^2
+\lambda_\beta\lVert\beta\rVert_2^2
+\lambda_\theta\lVert\theta\rVert_2^2.
```

It contains three parts.

### Prediction error

```math
\mathcal L_{\rm data}
=
\frac1N
\sum_{m=1}^{N}
\left(\widehat y_m-y_m\right)^2.
```

This rewards agreement with measured pIC50. Large residuals receive greater weight because they are squared.

### Readout regularisation

```math
\mathcal L_{\rm readout}
=
\lambda_\beta\lVert\beta\rVert_2^2
=
\lambda_\beta\sum_{k=1}^{p}\beta_k^2.
```

This is the ridge penalty. It discourages the linear readout from relying on very large, unstable coefficients.

### CA regularisation

```math
\mathcal L_{\rm CA}
=
\lambda_\theta\lVert\theta\rVert_2^2.
```

This separately controls the magnitude of the transition-rule parameters.

The intercept $\beta_0$ is normally excluded from the ridge penalty.

## 4. Why this is ridge-regularised

Classical ridge regression with a fixed design matrix $Z$ has the objective

```math
\lVert y-Z\beta\rVert_2^2
+\lambda_\beta\lVert\beta\rVert_2^2.
```

If $Z$ is fixed, the coefficients can be obtained from the closed-form expression

```math
\widehat\beta
=
\left(Z^{\mathsf T}Z+\lambda_\beta I\right)^{-1}
Z^{\mathsf T}y.
```

In our joint system, however,

```math
Z=Z_\theta,
```

because the fingerprints depend on the learned CA parameters. Every update to $\theta$ changes the design matrix.

We therefore optimise the linear coefficients and CA parameters together using gradients. The readout is accurately described as a **ridge-regularised linear readout** because it is linear and has the same L2 penalty. It is not a single, one-time closed-form ridge fit while the fingerprints remain fixed.

This distinction is mathematical rather than cosmetic.

## 5. How one error teaches beta

For one example,

```math
\widehat y=\beta_0+\sum_{k=1}^{p}\beta_kz_k
```

and

```math
\mathcal L_{\rm data}
=
\frac12(\widehat y-y)^2.
```

The derivative with respect to readout coefficient $\beta_k$ is

```math
\frac{\partial\mathcal L_{\rm data}}{\partial\beta_k}
=
(\widehat y-y)z_k.
```

Including ridge regularisation gives

```math
\frac{\partial\mathcal L}{\partial\beta_k}
=
(\widehat y-y)z_k
+2\lambda_\beta\beta_k.
```

The first term asks the coefficient to reduce prediction error. The second pulls it towards zero.

## 6. How the same error teaches theta

The loss depends on $\theta$ through a chain:

```math
\theta
\longrightarrow
\mathcal T
\longrightarrow
z
\longrightarrow
\widehat y
\longrightarrow
\mathcal L.
```

The chain rule gives

```math
\frac{\partial\mathcal L}{\partial\theta}
=
\frac{\partial\mathcal L}{\partial\widehat y}
\frac{\partial\widehat y}{\partial z}
\frac{\partial z}{\partial\mathcal T}
\frac{\partial\mathcal T}{\partial\theta}
+2\lambda_\theta\theta.
```

Each factor has a clear meaning:

- $\partial\mathcal L/\partial\widehat y$: how prediction error changes with the prediction;
- $\partial\widehat y/\partial z$: which fingerprint directions the readout uses;
- $\partial z/\partial\mathcal T$: how fingerprint values depend on the trajectory;
- $\partial\mathcal T/\partial\theta$: how all CA states depend on the update parameters; and
- $2\lambda_\theta\theta$: the CA regularisation gradient.

This is how an error measured at the molecular pIC50 level reaches local update rules acting on individual atoms.

## 7. A small numerical example

Consider one CA parameter $\theta$, one fingerprint value $z$, and one readout coefficient $\beta$:

```math
z=2\theta,
\qquad
\widehat y=\beta z.
```

Let

```math
\theta=0.5,
\qquad
\beta=0.8,
\qquad
y=1.2.
```

### Forward pass

The fingerprint is

```math
z=2(0.5)=1.
```

The prediction is

```math
\widehat y=(0.8)(1)=0.8.
```

Using half squared error,

```math
\mathcal L_{\rm data}
=
\frac12(0.8-1.2)^2
=0.08.
```

### Gradient for the readout

Ignoring regularisation temporarily,

```math
\frac{\partial\mathcal L}{\partial\beta}
=
(\widehat y-y)z
=
(-0.4)(1)
=-0.4.
```

The negative gradient indicates that increasing $\beta$ would locally raise the low prediction and reduce the loss.

### Gradient for the CA

The chain rule gives

```math
\frac{\partial\mathcal L}{\partial\theta}
=
\frac{\partial\mathcal L}{\partial\widehat y}
\frac{\partial\widehat y}{\partial z}
\frac{\partial z}{\partial\theta}.
```

The three factors are

```math
\frac{\partial\mathcal L}{\partial\widehat y}=-0.4,
\qquad
\frac{\partial\widehat y}{\partial z}=\beta=0.8,
\qquad
\frac{\partial z}{\partial\theta}=2.
```

Therefore

```math
\frac{\partial\mathcal L}{\partial\theta}
=
(-0.4)(0.8)(2)
=-0.64.
```

The same prediction error has now produced one gradient for $\beta$ and another for $\theta$. One loss teaches both components.

## 8. What makes a fingerprint differentiable

A function is differentiable when a small change in its inputs produces a mathematically traceable local change in its output.

Examples of differentiable trajectory summaries include:

- means of atom states;
- variances with a small numerical stabiliser;
- mean squared step-to-step change;
- soft temporal weighting;
- autocorrelation expressed through sums and products; and
- frequency power derived through differentiable Fourier operations.

For example, mean squared step distance is

```math
z_{\rm step}
=
\frac{1}{T}
\sum_{t=0}^{T-1}
\frac{1}{n}
\sum_{i=1}^{n}
\left\lVert
h_i^{(t+1)}-h_i^{(t)}
\right\rVert_2^2.
```

Every operation in this expression has a derivative, so pIC50 error can flow backwards through it.

## 9. Non-differentiable scientific summaries

A hard convergence time might be defined as the first generation satisfying

```math
\left\lVert H^{(t+1)}-H^{(t)}\right\rVert<\varepsilon.
```

The word “first” and the hard threshold create a discontinuous decision. A tiny state change may abruptly move convergence time from one integer generation to another.

We have three principled options:

1. use a smooth approximation during training;
2. retain the hard measurement for post-training scientific analysis; or
3. compare models with and without a differentiable proxy.

We will not pretend that a non-differentiable quantity passes an ordinary gradient when it does not.

## 10. One training step

For one mini-batch, the training cycle is:

1. construct standardised 2D molecular graphs;
2. attach fixed atom and bond channels;
3. initialise eight dynamical channels to zero;
4. apply the gated residual CA for 16 generations;
5. calculate differentiable trajectory summaries;
6. standardise fingerprints using training-fold statistics;
7. calculate pIC50 predictions;
8. calculate prediction and regularisation losses;
9. backpropagate through the readout and all generations;
10. update $\beta$ and $\theta$; and
11. record loss, gradients, state magnitudes, and validation diagnostics.

This cycle repeats over many mini-batches and epochs.

## 11. Learning rates

The two parameter families may require different learning rates:

```math
\beta_{\rm new}
=
\beta-\eta_\beta\nabla_\beta\mathcal L,
```

```math
\theta_{\rm new}
=
\theta-\eta_\theta\nabla_\theta\mathcal L.
```

Using separate $\eta_\beta$ and $\eta_\theta$ allows the simple linear readout and recurrent CA to learn at different speeds. Whether one shared learning rate is sufficient is a model-selection decision.

The accepted Adam optimiser and learning-rate search are derived in [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md).

## 12. Why joint learning is attractive

Joint learning lets the CA discover trajectories useful to the prediction task. If the readout relies strongly on oscillation or temporal variation, gradients can encourage CA parameters that make those quantities informative.

It also avoids freezing an arbitrary CA before we know whether its dynamics relate to CYP inhibition.

## 13. Risks and controls

### Representation collapse

The CA may produce nearly identical fingerprints for all molecules. We will monitor fingerprint variance, atom-state diversity, and predictive performance.

### Readout domination

The linear readout may exploit a few easy fingerprint components while richer dynamics remain unused. We will inspect coefficient paths, ablations, and comparisons with static descriptors.

### Unstable recurrent gradients

Sixteen repeated updates can produce vanishing or exploding gradients. We will monitor gradient norms, use bounded states and gated residual updates, and test gradient clipping if required.

### Regularisation imbalance

If $\lambda_\beta$ or $\lambda_\theta$ is too large, useful signal may be suppressed. If too small, parameters may become unstable or overfit. They are separate hyperparameters selected inside the inner validation loop.

### Hyperparameter leakage

Learning rates, regularisation, generation count, early stopping, and fingerprint definitions must be selected without using outer-fold or blinded-test outcomes.

## 14. What remains interpretable

End-to-end learning does not remove our obligation to understand the model. We will retain:

- explicit fixed chemical and bond channels;
- a declared local update equation;
- only eight evolving channels initially;
- shared parameters across atoms and generations;
- named fingerprint calculations;
- a linear pIC50 readout;
- ridge coefficient inspection;
- trajectory visualisation;
- dynamical-family and perturbation analyses; and
- controlled comparison against static descriptor baselines.

The purpose is a learned dynamical representation, not an unexplained prediction engine.

## 15. Validation boundary

Within each outer training fold, the inner procedure selects:

- learning rates;
- $\lambda_\beta$ and $\lambda_\theta$;
- early stopping;
- fingerprint choices;
- and any other tunable training decision.

The selected pipeline is then refitted on the full outer training fold and evaluated once on the untouched outer test fold.

After the production design is locked, the model is fitted to all released labelled data. Standardisation rules, feature encodings, CA parameters, readout coefficients, fingerprint scaling, and hyperparameters are frozen before predicting the challenge test set.

## 16. Terminology summary

- **Forward pass:** calculate trajectory, fingerprint, prediction, and loss.
- **Backpropagation:** calculate derivatives of that loss through the connected model.
- **Optimiser:** use gradients and learning rates to update parameters.
- **Joint training:** update $\theta$ and $\beta$ within the same training process.
- **End-to-end:** allow the final pIC50 error to influence every differentiable learned component.
- **Ridge-regularised readout:** a linear readout with an L2 penalty on $\beta$.
- **Closed-form ridge:** solve ridge coefficients algebraically for a fixed fingerprint matrix.

Our accepted prototype uses joint gradient optimisation of a ridge-regularised linear readout and the graph CA.

## Connection to the course

- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) defines $C$, $E$, and $H^{(t)}$.
- [Backpropagation](Backpropagation.md) derives the chain rule through repeated CA updates.
- [Ridge Regression](Ridge_Regression.md) derives the ridge objective and its closed-form fixed-feature solution.
- [Dynamics](Dynamics.md) defines the trajectory measurements.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) controls model selection and assessment.
