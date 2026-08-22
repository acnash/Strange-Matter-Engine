# Optimisation, Adam, and Learning Rates

## Learning objective

Backpropagation calculates how the loss changes with each parameter. An **optimiser** uses those derivatives to change the parameters.

Adam updates only the graph-CA parameters $\theta$. Ridge coefficients $\beta$ are obtained by solving the regularized normal equations and therefore have no optimizer learning rate.

## The accepted production decision

The defaults are

```math
\eta_\theta=10^{-3}.
```

Inner grouped cross-validation will evaluate

```math
\eta_\theta\in
\left\{3\times10^{-4},10^{-3},3\times10^{-3}\right\}
```

The ridge penalty $\lambda_\beta$ enters the matrix $Z^{\mathsf T}Z+\lambda_\beta I$. The CA penalty $\lambda_\theta\lVert\theta\rVert_2^2$ remains in the query loss. We do not add AdamW weight decay.

Here $\eta_\theta$ is the learning rate for graph-CA parameters, $\lambda_\theta$ is their explicit L2 strength, and $\lambda_\beta$ is the genuine ridge penalty.

## 1. Gradients and ordinary gradient descent

For parameter vector $w$, backpropagation produces, at optimisation step $s$,

```math
g_s=\nabla_w\mathcal L_s
```

Here $w$ denotes whichever trainable parameter vector is being updated, $\mathcal L_s$ is the loss calculated from step $s$'s mini-batch, and $g_s$ is its gradient with respect to $w$. Ordinary gradient descent uses

```math
w_{s+1}=w_s-\eta g_s.
```

The negative sign moves against the local direction of increasing loss. The learning rate $\eta>0$ controls the step size.

If

```math
w_s=0.8,\qquad g_s=-0.17,\qquad\eta=0.1,
```

then

```math
w_{s+1}=0.8-(0.1)(-0.17)=0.817.
```

The gradient supplies direction and sensitivity; the learning rate decides how far to move.

## 2. Learning-rate behaviour

If $\eta$ is too small, learning may be stable but very slow. Training may finish before reaching a useful region.

If $\eta$ is too large, updates may overshoot useful regions. Loss can oscillate or diverge, atom states can saturate, and recurrent gradients can become unstable.

An appropriate learning rate makes reliable progress without destabilising the learned dynamics. It is a hyperparameter selected from inner-validation evidence, never from blinded-test outcomes.

## 3. Why only theta has a learning rate

The linear readout is shallow:

```math
\widehat y=\beta_0+\beta^{\mathsf T}z.
```

The CA parameters are reused through 16 generations:

```math
H^{(t+1)}=F_\theta(H^{(t)},C,E,c_{\rm CYP}).
```

Their gradients accumulate through a recurrent computation and can be affected by gates, contraction, expansion, and tanh saturation.

We therefore permit

```math
\theta_{s+1}
=
\theta_s-\eta_\theta u_{\theta,s},
```

Ridge coefficients are instead recomputed on each fitting boundary:

```math
\beta=(Z_{\rm support}^{\mathsf T}Z_{\rm support}+\lambda_\beta I)^{-1}
Z_{\rm support}^{\mathsf T}y_{\rm support}.
```

The implementation uses `torch.linalg.solve` rather than an explicit inverse. Query loss differentiates through this solve, so support and query fingerprints both teach $\theta$.

## 4. Adam's first moment

For coordinate $j$, let

```math
g_{s,j}
=
\frac{\partial\mathcal L_s}{\partial w_j}.
```

Adam keeps an exponentially weighted gradient mean:

```math
m_{s,j}
=
\rho_1m_{s-1,j}
+(1-\rho_1)g_{s,j}.
```

This acts like momentum: gradient directions that persist across steps accumulate, while isolated fluctuations have less influence.

## 5. Adam's second moment

Adam also tracks recent squared gradient magnitude:

```math
v_{s,j}
=
\rho_2v_{s-1,j}
+(1-\rho_2)g_{s,j}^2.
```

A coordinate with repeatedly large gradients develops a larger $v_{s,j}$, causing Adam to reduce its effective step relative to coordinates with smaller gradients.

The initial settings are

```math
\rho_1=0.9,\qquad\rho_2=0.999.
```

Values near 1 give longer memory.

## 6. Bias correction

Adam starts with

```math
m_{0,j}=0,\qquad v_{0,j}=0.
```

The early moving averages are therefore biased towards zero. Adam corrects them:

```math
\widehat m_{s,j}
=
\frac{m_{s,j}}{1-\rho_1^s},
```

```math
\widehat v_{s,j}
=
\frac{v_{s,j}}{1-\rho_2^s}.
```

## 7. The Adam update

The complete coordinate update is

```math
w_{s+1,j}
=
w_{s,j}
-
\eta
\frac{\widehat m_{s,j}}
{\sqrt{\widehat v_{s,j}}+\epsilon}.
```

The numerator estimates recent direction. The denominator scales the update using recent gradient magnitude. The small constant

```math
\epsilon=10^{-8}
```

prevents division by zero.

## 8. A one-step example

Let

```math
w_0=0.8,\quad
g_1=-0.2,\quad
\rho_1=0.9,\quad
\rho_2=0.999.
```

With $m_0=v_0=0$,

```math
m_1=(0.1)(-0.2)=-0.02,
```

```math
v_1=(0.001)(-0.2)^2=0.00004.
```

Bias correction gives

```math
\widehat m_1=-0.2,\qquad
\widehat v_1=0.04.
```

With $\eta=10^{-3}$ and negligible $\epsilon$,

```math
\begin{aligned}
w_1
&=
0.8
-
10^{-3}
\frac{-0.2}{\sqrt{0.04}}\\
&=0.801.
\end{aligned}
```

Later updates depend on the accumulated histories $m_s$ and $v_s$, not only the current gradient.

## 9. Mini-batch gradients

For a molecule-centred mini-batch split into ridge-support set $\mathcal S_s$ and query set $\mathcal Q_s$, the optimization loss is

```math
\mathcal L_s
=
\frac1{|\mathcal Q_s|}
\sum_{m\in\mathcal Q_s}
(\widehat y_m-y_m)^2
+\lambda_\theta\lVert\theta\rVert_2^2.
```

The ridge state used for $\widehat y_m$ is solved from $\mathcal S_s$ with penalty $\lambda_\beta$. Splitting by molecule prevents observations from one molecule appearing on both sides of the same differentiable solve.

Its gradient estimates the complete training-set gradient. Smaller batches introduce more sampling variation; larger batches require more memory and produce fewer updates per epoch. Batch construction is a separate design decision.

The accepted molecule-centred batch is developed in [Mini-Batching Molecular Graphs and CYP Contexts](Mini_Batching_Molecular_Graphs.md).

## 10. CA regularisation and ridge shrinkage

The explicit CA regularisation gradient is

```math
\nabla_\theta
\left(
\lambda_\theta\lVert\theta\rVert_2^2
\right)
=
2\lambda_\theta\theta.
```

It becomes part of the raw gradient supplied to Adam. Ridge shrinkage acts inside the differentiable solve and influences $\theta$ through the derivative of that solve.

## 11. Adam versus AdamW

AdamW adds **decoupled weight decay**, a separate shrinkage operation. This is mathematically distinct from adding an L2 penalty to Adam's loss.

The initial prototype uses Adam plus explicit L2 penalties. It will not silently add AdamW decay on top. AdamW can later be tested as a replacement design under the same validation protocol.

## 12. Selecting the CA learning rate

The proposed values create three CA learning-rate candidates. Every candidate must use:

- the same inner folds;
- the same declared initialisation seeds;
- the same training budget;
- the same early-stopping rule;
- the same metric; and
- the same failure criteria.

Rates are spaced multiplicatively because their useful scale is usually explored logarithmically. If the best value lies at a boundary, that is evidence for a later, prespecified extension of the range.

## 13. What we monitor

Every run will record:

- training prediction loss;
- ridge penalty and CA regularisation diagnostics;
- inner-validation RMSE and MAE;
- gradient norms for $\theta$ and ridge-solve conditioning;
- parameter norms;
- atom-state minima, maxima, means, and variances;
- tanh saturation;
- gate distributions;
- fingerprint distributions;
- learning rates;
- optimisation-step count;
- random seed; and
- stopping reason.

Prediction error alone cannot establish that the recurrent dynamics are numerically healthy.

## 14. Failure criteria

A run is marked failed if it produces non-finite losses, parameters, states, fingerprints, or predictions. We will also diagnose uncontrolled gradient growth, persistent saturation, and complete fingerprint collapse.

Failed configurations remain part of the experimental record rather than disappearing from the hyperparameter comparison.

## 15. Constant rates first

The first prototype will use constant learning rates with early stopping. Schedulers, warm-up, and cyclical rates introduce additional design choices and will be postponed to controlled ablations.

## 16. Reproducible optimiser state

An exact training checkpoint must retain:

- $\theta$ and the support-fitted ridge state;
- Adam first and second moments;
- optimisation-step count;
- CA learning rate and Adam moment constants;
- random-number-generator states;
- data-order state; and
- complete preprocessing and model configuration.

Saving only model weights permits prediction but may not permit an identical continuation of training.

## 17. Prototype specification

The accepted design is:

- Adam;
- $\rho_1=0.9$;
- $\rho_2=0.999$;
- $\epsilon=10^{-8}$;
- default $\eta_\theta=10^{-3}$;
- no readout learning rate;
- a three-candidate CA learning-rate search;
- ridge-solve $\lambda_\beta$ and explicit CA penalty $\lambda_\theta$;
- no additional AdamW decay; and
- constant rates initially.

## Connection to the course

- [Backpropagation](Backpropagation.md) explains gradient calculation.
- [End-to-End Joint Training](End_to_End_Joint_Training.md) defines the shared loss.
- [Ridge Regression](Ridge_Regression.md) explains the penalty on $\beta$.
- [Dynamics](Dynamics.md) explains recurrent-state stability.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) governs learning-rate selection.
