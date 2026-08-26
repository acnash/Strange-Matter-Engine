# Rule 10: delayed-memory graph cellular automaton

## Purpose

Delayed memory gives the update explicit access to an earlier graph state. The earlier state enters the transition directly instead of being represented only implicitly in the current state.

## Implemented update

The searched delay control $a$ selects an available integer lag:

```math
\tau_t=\max\!\left(1,
\min\!\left(t+1,\operatorname{round}(1+15a)\right)\right).
```

Let $h_i^{(t-\tau_t+1)}$ be the delayed state and $r_i^{(t)}$ the current reaction. The learned delayed reaction is

```math
\widetilde r_i^{(t)}=\tanh\!\left(
W_D\begin{bmatrix}r_i^{(t)}\\h_i^{(t-\tau_t+1)}\end{bmatrix}+b_D
\right).
```

Memory mixture and damping are

```math
\mu=0.1+0.8b,\qquad \delta=0.05+0.45d.
```

The combined drive is

```math
q_i^{(t)}=(1-\mu)r_i^{(t)}+\mu\widetilde r_i^{(t)}
+c\left(h_i^{(t-\tau_t+1)}-h_i^{(t)}\right),
```

and the update is

```math
h_i^{(t+1)}=\tanh\!\left(
(1-\delta\eta)h_i^{(t)}+\eta q_i^{(t)}
\right).
```

## Expected dynamics and cautions

Delayed feedback can produce slow oscillations, long transients, multistability, and bifurcation. Early generations cannot access the full requested lag, so the delay is clipped to available history. Full history also raises memory use with generation count. Oscillation requires longer perturbation analysis before any attractor claim.
