# Gradient Clipping in a Recurrent Graph Cellular Automaton

## Learning objective

Understand why gradients can grow during backpropagation through graph-CA generations, how global norm clipping limits an optimisation step, and what clipping frequency tells us about dynamical stability.

## 1. Why recurrence can amplify gradients

Our graph CA applies one shared update rule repeatedly:

```math
H^{(t+1)}=F_\theta\left(H^{(t)},G,C,c_{\rm CYP}\right).
```

Backpropagation through 16 generations multiplies local Jacobian matrices through time. Schematically,

```math
\frac{\partial H^{(16)}}{\partial H^{(0)}}
=
\prod_{t=0}^{15}
\frac{\partial H^{(t+1)}}{\partial H^{(t)}}.
```

If repeated multiplication expands vectors, the gradient can become very large. This is an **exploding gradient**. If it repeatedly contracts vectors, the gradient can become very small, producing a **vanishing gradient**.

The gated residual update helps information survive through time, while gradient clipping protects individual optimisation steps from unusually large derivatives.

## 2. The global gradient norm

Let the complete trainable parameter gradient be $g$, formed by concatenating the gradients of all model parameters. Its Euclidean norm is

```math
\left\|g\right\|_2
=
\sqrt{
\sum_j g_j^2
}.
```

The norm describes the overall size of the proposed update direction before the optimiser rescales it.

## 3. Global norm clipping

For clipping threshold $c$, the clipped gradient is

```math
\widetilde g
=
g\,
\min\left(
1,
\frac{c}{\left\|g\right\|_2}
\right).
```

Therefore:

- when $\left\|g\right\|_2\leq c$, the gradient is unchanged;
- when $\left\|g\right\|_2>c$, every component is multiplied by the same factor; and
- the direction of the complete gradient is preserved while its magnitude is capped.

For this prototype, $c=1.0$.

### Numerical example

Suppose

```math
g=(3,4).
```

Its norm is

```math
\left\|g\right\|_2=5.
```

With $c=1$, the scaling factor is $1/5$, giving

```math
\widetilde g=(0.6,0.8),
\qquad
\left\|\widetilde g\right\|_2=1.
```

The direction is unchanged because both components receive the same multiplier.

## 4. Interaction with Adam

Clipping is applied after backpropagation has calculated the raw gradient and before Adam updates the parameters:

1. perform the forward graph-CA trajectory;
2. calculate prediction loss and regularisation;
3. backpropagate through all generations;
4. measure the raw global gradient norm;
5. clip it to 1.0 when necessary; and
6. let Adam transform the clipped gradient into a parameter update.

Gradient clipping and the learning rate have different roles:

- the clipping threshold limits unusually large gradient vectors;
- the learning rate controls the scale at which Adam changes parameters; and
- Adam's moment estimates adapt updates using recent gradient history.

## 5. What clipping frequency teaches us

The prototype records the fraction of optimiser steps for which the raw norm exceeded 1.0. Frequent clipping means the threshold is actively shaping training.

That observation can have several explanations:

- recurrent Jacobian products are amplifying derivatives;
- a subset of batches contains unusually difficult molecules or labels;
- the learning dynamics favour sharp regions of the loss surface; or
- the threshold is conservative relative to the model's natural gradient scale.

Clipping makes the run numerically manageable. It also creates a diagnostic: a model with similar validation error but calmer raw gradients may be preferable because its optimisation is more stable.

## 6. What clipping does scientifically

Gradient clipping changes training, not inference. Once parameters are frozen, the CA trajectory for a molecule contains no gradients and requires no clipping.

Clipping also does not force atom states themselves to be small. State magnitude, step-to-step change, Jacobian sensitivity, and gradient magnitude are related aspects of the system, but they are distinct measurements.

## 7. Accepted prototype policy

- use global gradient-norm clipping at 1.0;
- record the raw norm before clipping;
- record the proportion of steps clipped in every epoch;
- interpret persistent clipping as a stability diagnostic; and
- compare alternative thresholds only inside the grouped inner-validation procedure during production development.

