# Backpropagation Through the Graph Cellular Automaton

## Learning objective

Backpropagation answers one precise question:

> If the model's prediction is wrong, how much did each trainable parameter contribute to that error?

In Strange Matter Engine, it is the mechanism that can teach the **shared local graph-cellular-automaton rule**. The rule is applied to every atom and reused at every generation, so a parameter can influence the final prediction through many atoms and many points in time.

Backpropagation is an efficient application of the chain rule. It calculates derivatives; an optimiser uses those derivatives to update parameters.

## Where it sits in our model

For molecule $m$, the proposed path is

```math
\text{SMILES}
\longrightarrow G_m=(V_m,E_m)
\longrightarrow X_m^{(0)}
\xrightarrow{F_\theta}\cdots\xrightarrow{F_\theta}X_m^{(T)}
\longrightarrow z_m
\longrightarrow \widehat y_m,
```

where:

- $G_m$ is the molecular graph;
- $X_m^{(0)}\in\mathbb R^{n_m\times d}$ contains the initial states of its $n_m$ atoms;
- $F_\theta$ is the shared CA update rule;
- $X_m^{(t)}$ is the state of all atoms at generation $t$;
- $z_m\in\mathbb R^p$ is the trajectory-derived dynamical fingerprint;
- $\widehat y_m$ is predicted pIC50; and
- $y_m$ is experimental pIC50.

The ridge readout learns coefficients $\beta$. Backpropagation carries the prediction error further backwards through $z_m$ and through all $T$ CA generations to learn $\theta$.

## 1. Derivatives as sensitivity

For a scalar function $y=f(x)$, the derivative

```math
\frac{dy}{dx}
```

measures the local change in $y$ caused by a small change in $x$:

```math
f(x+\Delta x)\approx f(x)+\frac{df}{dx}\Delta x.
```

If $df/dx=3$, increasing $x$ by $0.01$ changes $f$ by approximately $0.03$. A positive derivative means local increase; a negative derivative means local decrease; a large magnitude means high local sensitivity.

For many parameters $\theta=(\theta_1,\ldots,\theta_q)$, the gradient is

```math
\nabla_\theta \mathcal L=
\begin{bmatrix}
\partial\mathcal L/\partial\theta_1\\
\vdots\\
\partial\mathcal L/\partial\theta_q
\end{bmatrix}.
```

It points in the direction of steepest local increase of the loss $\mathcal L$. Therefore $-\nabla_\theta\mathcal L$ is the steepest local descent direction.

## 2. The chain rule

Suppose

```math
u=ax+b,\qquad y=\tanh(u),\qquad \mathcal L=\frac12(y-y^*)^2.
```

The parameter $a$ affects the loss through $u$ and then $y$. The chain rule gives

```math
\frac{\partial\mathcal L}{\partial a}
=
\frac{\partial\mathcal L}{\partial y}
\frac{\partial y}{\partial u}
\frac{\partial u}{\partial a}.
```

The three local derivatives are

```math
\frac{\partial\mathcal L}{\partial y}=y-y^*,\qquad
\frac{\partial y}{\partial u}=1-\tanh^2(u)=1-y^2,\qquad
\frac{\partial u}{\partial a}=x.
```

Hence

```math
\boxed{
\frac{\partial\mathcal L}{\partial a}
=(y-y^*)(1-y^2)x
}.
```

Backpropagation evaluates chains like this from the loss backwards, reusing intermediate results rather than repeatedly expanding every possible path.

## 3. A numerical example

Let

```math
x=0.5,\quad a=0.8,\quad b=0.1,\quad y^*=0.9.
```

The forward calculation is

```math
u=(0.8)(0.5)+0.1=0.5,
```

```math
y=\tanh(0.5)\approx0.4621,
```

```math
\mathcal L=\frac12(0.4621-0.9)^2\approx0.0959.
```

The backward calculation is

```math
\frac{\partial\mathcal L}{\partial y}=0.4621-0.9=-0.4379,
```

```math
\frac{\partial y}{\partial u}=1-0.4621^2\approx0.7864,
```

```math
\frac{\partial\mathcal L}{\partial a}
=(-0.4379)(0.7864)(0.5)\approx-0.1722.
```

With learning rate $\eta=0.1$, gradient descent updates

```math
a_{\text{new}}=a-\eta\frac{\partial\mathcal L}{\partial a}
=0.8-0.1(-0.1722)=0.8172.
```

The negative gradient tells us that increasing $a$ locally reduces this example's loss.

## 4. Our local graph-CA rule

The scalar teaching model uses

```math
m_i^{(t)}=\frac{1}{d_i}\sum_{j\in\mathcal N(i)}x_j^{(t)},
```

```math
u_i^{(t)}=\theta_{\rm self}x_i^{(t)}
+\theta_{\rm neighbour}m_i^{(t)}
+\theta_{\rm bias},
```

```math
x_i^{(t+1)}=\tanh\!\left(u_i^{(t)}\right).
```

Here $\mathcal N(i)$ is atom $i$'s neighbourhood and $d_i=|\mathcal N(i)|$. The three trainable quantities are shared:

- $\theta_{\rm self}$: persistence of the atom's own state;
- $\theta_{\rm neighbour}$: strength and sign of neighbour influence;
- $\theta_{\rm bias}$: a uniform offset before the nonlinearity.

For one update, the direct local derivatives are

```math
\frac{\partial x_i^{(t+1)}}{\partial\theta_{\rm self}}
=\left(1-(x_i^{(t+1)})^2\right)x_i^{(t)},
```

```math
\frac{\partial x_i^{(t+1)}}{\partial\theta_{\rm neighbour}}
=\left(1-(x_i^{(t+1)})^2\right)m_i^{(t)},
```

```math
\frac{\partial x_i^{(t+1)}}{\partial\theta_{\rm bias}}
=1-(x_i^{(t+1)})^2.
```

These are direct effects for one atom at one generation. The total derivative must also include indirect effects through earlier states and neighbouring atoms.

## 5. Why this is backpropagation through time

The repeated rule creates a recurrence:

```math
X^{(t+1)}=F_\theta(X^{(t)},G).
```

After $T$ generations,

```math
X^{(T)}=F_\theta\!\left(F_\theta\!\left(\cdots F_\theta(X^{(0)},G)\right),G\right).
```

Unrolling this recurrence makes a computational graph with $T$ copies of the operation. They are copies of the operation, not separate parameter sets: the same $\theta$ is reused.

Let the adjoint

```math
A^{(t)}=\frac{\partial\mathcal L}{\partial X^{(t)}}
```

mean the sensitivity of the final loss to the state at generation $t$. Backwards propagation obeys

```math
A^{(t)}=A^{(t+1)}
\frac{\partial X^{(t+1)}}{\partial X^{(t)}}.
```

Because $\theta$ appears at every generation, its total gradient is a sum:

```math
\boxed{
\frac{\partial\mathcal L}{\partial\theta}
=\sum_{t=0}^{T-1}
A^{(t+1)}
\frac{\partial X^{(t+1)}}{\partial\theta}
}.
```

It also sums over atoms because the rule is shared spatially. One parameter therefore learns from every molecular neighbourhood, every generation, and every training molecule.

## 6. A two-generation scalar example

Consider the simplified recurrence

```math
x^{(t+1)}=\theta x^{(t)},\qquad x^{(0)}=2,qquad T=2,
```

with target $y=3$, prediction $\widehat y=x^{(2)}$, and

```math
\mathcal L=\frac12(\widehat y-y)^2.
```

Since

```math
x^{(1)}=2\theta,\qquad x^{(2)}=2\theta^2,
```

we can differentiate directly:

```math
\frac{d\mathcal L}{d\theta}
=(2\theta^2-3)(4\theta).
```

At $\theta=1$, $\widehat y=2$, $\mathcal L=0.5$, and

```math
\frac{d\mathcal L}{d\theta}=(2-3)(4)=-4.
```

The backpropagated calculation finds the same answer while exposing both uses of $\theta$:

```math
\frac{d\mathcal L}{d\theta}
=
\frac{\partial\mathcal L}{\partial x^{(2)}}
\left(
\frac{\partial x^{(2)}}{\partial\theta}
+
\frac{\partial x^{(2)}}{\partial x^{(1)}}
\frac{\partial x^{(1)}}{\partial\theta}
\right).
```

At $\theta=1$, the bracket is $2+2=4$, while the final residual is $-1$, giving $-4$. Ignoring either occurrence would give the wrong gradient.

## 7. From pIC50 error back into the automaton

Suppose the regression prediction is

```math
\widehat y=\beta_0+\beta^\top z,
```

and the per-molecule squared-error loss is

```math
\mathcal L_{\rm reg}=\frac12(\widehat y-y)^2.
```

Then

```math
\frac{\partial\mathcal L_{\rm reg}}{\partial\widehat y}
=\widehat y-y,
```

```math
\frac{\partial\widehat y}{\partial z}=\beta,
```

so

```math
\frac{\partial\mathcal L_{\rm reg}}{\partial z}
=(\widehat y-y)\beta.
```

If fingerprint extraction is differentiable, the chain continues:

```math
\boxed{
\frac{\partial\mathcal L_{\rm reg}}{\partial\theta}
=
\frac{\partial\mathcal L_{\rm reg}}{\partial\widehat y}
\frac{\partial\widehat y}{\partial z}
\frac{\partial z}{\partial(X^{(0)},\ldots,X^{(T)})}
\frac{\partial(X^{(0)},\ldots,X^{(T)})}{\partial\theta}
}.
```

This is how an error measured in pIC50 can teach a local rule acting on atoms.

Some useful scientific summaries—such as a hard first-threshold crossing for convergence time—are not smoothly differentiable. We may therefore use differentiable approximations during end-to-end learning, retain non-differentiable quantities for analysis, or adopt staged training. This is a model-design decision to validate, not conceal.

## 8. Two tasks and a shared automaton

The planned model has a regression loss for pIC50 and a classification loss for time-dependent inhibition (TDI). A combined objective can be written

```math
\mathcal L_{\rm total}
=w_{\rm reg}\mathcal L_{\rm reg}
+w_{\rm TDI}\mathcal L_{\rm TDI}
+\lambda_\theta R(\theta),
```

where $w_{\rm reg}$ and $w_{\rm TDI}$ control task balance and $R(\theta)$ is an optional regulariser.

The gradient of the shared rule is

```math
\nabla_\theta\mathcal L_{\rm total}
=w_{\rm reg}\nabla_\theta\mathcal L_{\rm reg}
+w_{\rm TDI}\nabla_\theta\mathcal L_{\rm TDI}
+\lambda_\theta\nabla_\theta R(\theta).
```

Thus both experimental tasks can shape the common dynamics. Whether joint learning helps must be established on held-out molecules.

## 9. Vanishing and exploding gradients

Backpropagation through $T$ generations contains products of Jacobian matrices:

```math
\frac{\partial X^{(T)}}{\partial X^{(t)}}
=J^{(T-1)}J^{(T-2)}\cdots J^{(t)},
\qquad
J^{(k)}=\frac{\partial X^{(k+1)}}{\partial X^{(k)}}.
```

If their effective magnitudes are repeatedly below one, gradients shrink: **vanishing gradients**. Early generations then receive little learning signal. If they are repeatedly above one, gradients may grow: **exploding gradients**. Optimisation becomes unstable.

Relevant controls include:

- sensible parameter initialisation;
- bounded nonlinearities and awareness of tanh saturation;
- gradient-norm monitoring and clipping;
- an appropriate number of generations;
- smaller learning rates when necessary; and
- reporting the dynamics produced, not only predictive loss.

These numerical behaviours are connected to dynamical stability. A strongly contracting system can erase initial information; an expansive system can amplify perturbations and gradients.

## 10. Parameters, hyperparameters, and fixed scientific inputs

Backpropagation may learn:

- CA message and update weights $\theta$;
- parameters of differentiable fingerprint transformations; and
- in a joint differentiable design, readout parameters.

Validation selects hyperparameters such as:

- generation count $T$;
- state dimension $d$;
- learning rate and optimiser settings;
- regularisation strengths;
- task weights; and
- early-stopping rules.

Chemical identity, bond connectivity, experimental labels, and train/validation membership are data or experimental design—not quantities the optimiser is allowed to rewrite.

## 11. Gradient checking

An analytical derivative can be tested against a centred finite difference:

```math
g_{\rm numerical}
=\frac{\mathcal L(\theta_k+h)-\mathcal L(\theta_k-h)}{2h}.
```

Compare it with

```math
g_{\rm analytical}=\frac{\partial\mathcal L}{\partial\theta_k}
```

using relative error

```math
\frac{|g_{\rm analytical}-g_{\rm numerical}|}
{\max(1,|g_{\rm analytical}|,|g_{\rm numerical}|)}.
```

Agreement for several parameters and small test graphs provides evidence that the chain rule has been implemented correctly. It does not establish that the scientific model is useful; held-out validation addresses that question.

## 12. What success means

Successful optimisation means the chosen loss decreased. Scientific success requires more:

- improved performance on molecules excluded from fitting;
- comparison against chemically meaningful descriptor baselines;
- stable results across seeds and splits;
- interpretable and reproducible dynamical fingerprints;
- examination of learned attractors, oscillations, transients, and perturbation response; and
- strict isolation of blind test labels.

Backpropagation is the route by which experimental error can shape the automaton. Validation determines whether the learned dynamics contain transferable chemical information.

## Connection to the course

- [Emergence](Emergence.md) introduces local rules and collective behaviour.
- [Dynamics](Dynamics.md) defines the trajectories being differentiated.
- [Ridge Regression](Ridge_Regression.md) explains the pIC50 readout.
- [Validation and Statistics](Validation_and_Statistics.md) explains how learning claims are tested.

