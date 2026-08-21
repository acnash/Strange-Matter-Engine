# Optimisation, Adam, and Learning Rates

## Learning objective

Backpropagation calculates how the loss changes with each parameter. An **optimiser** uses those derivatives to change the parameters.

For the first prototype, Adam will update the graph-CA parameters $\theta$ and ridge-regularised readout parameters $\beta$ with separate learning rates.

## The accepted production decision

The defaults are

```math
\eta_\theta=10^{-3},
\qquad
\eta_\beta=3\times10^{-3}.
```

Inner grouped cross-validation will evaluate

```math
\eta_\theta\in
\left\{3\times10^{-4},10^{-3},3\times10^{-3}\right\}
```

and

```math
\eta_\beta\in
\left\{10^{-3},3\times10^{-3},10^{-2}\right\}.
```

The explicit ridge penalty $\lambda_\beta\lVert\beta\rVert_2^2$ and CA penalty $\lambda_\theta\lVert\theta\rVert_2^2$ remain in the loss. We will not add AdamW weight decay to the initial prototype.

Here $\eta_\theta$ and $\eta_\beta$ are the learning rates for the graph-CA parameters $\theta$ and readout coefficients $\beta$, respectively. Likewise, $\lambda_\theta$ and $\lambda_\beta$ are the corresponding L2-regularisation strengths.

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

## 3. Why theta and beta have separate rates

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

```math
\beta_{s+1}
=
\beta_s-\eta_\beta u_{\beta,s},
```

where $u_{\theta,s}$ and $u_{\beta,s}$ are Adam-adjusted directions. The larger default readout rate allows the simple linear mapping to adapt faster while the recurrent dynamics change more cautiously.

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

For mini-batch $\mathcal B_s$, the joint loss is

```math
\mathcal L_{\mathcal B_s}
=
\frac1{|\mathcal B_s|}
\sum_{m\in\mathcal B_s}
(\widehat y_m-y_m)^2
+\lambda_\beta\lVert\beta\rVert_2^2
+\lambda_\theta\lVert\theta\rVert_2^2.
```

Here $\mathcal B_s$ is the set of examples in optimisation step $s$, and $|\mathcal B_s|$ is the number of examples in that set.

Its gradient estimates the complete training-set gradient. Smaller batches introduce more sampling variation; larger batches require more memory and produce fewer updates per epoch. Batch construction is a separate design decision.

The accepted molecule-centred batch is developed in [Mini-Batching Molecular Graphs and CYP Contexts](Mini_Batching_Molecular_Graphs.md).

## 10. Explicit L2 regularisation

The regularisation gradients are

```math
\nabla_\beta
\left(
\lambda_\beta\lVert\beta\rVert_2^2
\right)
=
2\lambda_\beta\beta
```

and

```math
\nabla_\theta
\left(
\lambda_\theta\lVert\theta\rVert_2^2
\right)
=
2\lambda_\theta\theta.
```

They become part of the raw gradients supplied to Adam. This keeps the ridge penalty explicit and tied to the declared scientific objective.

## 11. Adam versus AdamW

AdamW adds **decoupled weight decay**, a separate shrinkage operation. This is mathematically distinct from adding an L2 penalty to Adam's loss.

The initial prototype uses Adam plus explicit L2 penalties. It will not silently add AdamW decay on top. AdamW can later be tested as a replacement design under the same validation protocol.

## 12. Selecting the rate pair

The proposed values create a $3\times3=9$-combination grid. Every pair must use:

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
- readout and CA regularisation terms;
- inner-validation RMSE and MAE;
- gradient norms for $\theta$ and $\beta$;
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

- $\theta$ and $\beta$;
- Adam first and second moments;
- optimisation-step count;
- learning rates and moment constants;
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
- default $\eta_\beta=3\times10^{-3}$;
- a three-by-three inner-validation rate search;
- explicit $\lambda_\beta$ and $\lambda_\theta$;
- no additional AdamW decay; and
- constant rates initially.

## Connection to the course

- [Backpropagation](Backpropagation.md) explains gradient calculation.
- [End-to-End Joint Training](End_to_End_Joint_Training.md) defines the shared loss.
- [Ridge Regression](Ridge_Regression.md) explains the penalty on $\beta$.
- [Dynamics](Dynamics.md) explains recurrent-state stability.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) governs learning-rate selection.
