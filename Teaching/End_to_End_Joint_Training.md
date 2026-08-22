# End-to-End Joint Training

## Learning objective

End-to-end differentiable ridge training means that query pIC50 error teaches:

- the **graph cellular-automaton parameters** $\theta$, which create the support and query fingerprints; while
- the **ridge coefficients** $\beta$ are solved exactly from support fingerprints and labels rather than updated by Adam.

The complete prediction pathway is treated as one connected mathematical function. Backpropagation applies the chain rule through that entire pathway.

## The accepted production decision

The accepted implementation partitions each molecule-centred optimization batch into support and query molecules:

```math
\text{molecular graph}
\longrightarrow
\text{CA trajectory}
\longrightarrow
\text{differentiable fingerprint}
\longrightarrow
\text{support ridge solve}
\longrightarrow
\text{query pIC50 prediction}
\longrightarrow
\text{loss}.
```

The learning signal then travels in the reverse direction:

```math
\text{query loss}
\longrightarrow
\text{differentiable ridge solve}
\longrightarrow
\text{fingerprint}
\longrightarrow
\text{all CA generations}
\longrightarrow
\theta.
```

This connects representation learning and ridge regression in one differentiable process:

- backpropagation is the derivative-calculation mechanism;
- Adam updates $\theta$ only;
- the CA parameters learn the molecular dynamics; and
- `torch.linalg.solve` obtains a genuine ridge mapping from support fingerprints to pIC50.

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

Training repeatedly performs:

1. generate support and query fingerprints using the current $\theta$;
2. standardize support fingerprints using support-only statistics;
3. solve $\beta$ and the unpenalized intercept from support observations;
4. predict query observations with the frozen support ridge state;
5. backpropagate query error through the solve and both fingerprint paths; and
6. update $\theta$ with Adam.

The ridge state is recomputed as the representation changes. It is never an independently optimized neural parameter.

## 3. The training objective

For query set $\mathcal Q$ and ridge state solved from disjoint support set $\mathcal S$, the optimization objective is

```math
\mathcal L(\theta)
=
\frac1{|\mathcal Q|}
\sum_{m\in\mathcal Q}
\left(\widehat y_m-y_m\right)^2
+\lambda_\theta\lVert\theta\rVert_2^2.
```

It contains query prediction error and explicit CA regularisation. Ridge regularisation is enforced inside the support solve.

Here $m$ indexes query examples, $\lambda_\beta$ is the ridge strength used in the support solve, $\lambda_\theta$ is the CA L2 strength, and $\lVert\cdot\rVert_2^2$ denotes a squared Euclidean norm.

### Prediction error

```math
\mathcal L_{\rm data}
=
\frac1{|\mathcal Q|}
\sum_{m\in\mathcal Q}
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

This ridge penalty is enforced by the support solve itself. It is not added again to the query loss, and the intercept is excluded through support centring.

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

We solve the linear coefficients again whenever the fingerprint matrix changes. PyTorch differentiates the query loss through this solve, giving a genuine differentiable ridge layer rather than an Adam-trained approximation to the ridge objective.

This distinction is mathematical rather than cosmetic.

## 5. How one error differentiates through beta

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
6. divide molecules into support and query subsets;
7. standardise using support-only statistics and solve ridge on support observations;
8. calculate query pIC50 predictions and loss;
9. backpropagate through the solve and all generations;
10. update $\theta$ with Adam; and
11. record loss, gradients, state magnitudes, and validation diagnostics.

This cycle repeats over many mini-batches and epochs.

## 11. Learning rate

Only the CA parameters have an optimizer learning rate:

```math
\theta_{\rm new}
=
\theta-\eta_\theta\nabla_\theta\mathcal L.
```

The ridge coefficients are solved, not stepped. The ridge penalty $\lambda_\beta$ controls coefficient shrinkage and is selected separately from $\eta_\theta$.

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
- **Joint training:** recompute $\beta$ and update $\theta$ within the same differentiable training process.
- **End-to-end:** allow the final pIC50 error to influence every differentiable learned component.
- **Differentiable ridge readout:** solve ridge coefficients algebraically while retaining autograd connections to the fingerprint matrix.
- **Support/query training:** solve ridge on support molecules and teach the CA using disjoint query error.

Our accepted implementation uses a differentiable ridge solve and Adam optimization of the graph CA only.

## Connection to the course

- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) defines $C$, $E$, and $H^{(t)}$.
- [Backpropagation](Backpropagation.md) derives the chain rule through repeated CA updates.
- [Ridge Regression](Ridge_Regression.md) derives the ridge objective and its closed-form fixed-feature solution.
- [Dynamics](Dynamics.md) defines the trajectory measurements.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) controls model selection and assessment.
