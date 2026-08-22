# Transition rule 3: activator–inhibitor graph CA

## Aim

This rule is designed to make excitation, inhibition, pulses and oscillations natural. Each atom carries a fast activator and a slower inhibitor.

## Two interacting fields

For atom i, let the activator be u and the inhibitor be v:

```math
u_i^{(t)},v_i^{(t)}\in\mathbb{R}.
```

In an eight-channel model, several channel pairs can act as activator–inhibitor systems.

## Activator equation

A FitzHugh–Nagumo-inspired update is:

```math
u_i^{(t+1)}=u_i^{(t)}+\Delta t\left[
u_i^{(t)}-\frac{(u_i^{(t)})^3}{3}-v_i^{(t)}+I_i
+D_u\Delta_Gu_i^{(t)}
\right].
```

The cubic term limits runaway activation. The learned input contains chemistry and CYP context:

```math
I_i=f_{\theta}(x_i,c,m_i^{(t)}).
```

## Inhibitor equation

```math
v_i^{(t+1)}=v_i^{(t)}+\Delta t\left[
\varepsilon\left(u_i^{(t)}+a-bv_i^{(t)}\right)
+D_v\Delta_Gv_i^{(t)}
\right].
```

A small epsilon makes inhibition slower than activation. This delay is what allows an excitation pulse to rise before being suppressed.

## Graph Laplacian

For mean-neighbour diffusion:

```math
\Delta_Gu_i^{(t)}=
\frac{1}{|N(i)|}\sum_{j\in N(i)}u_j^{(t)}-u_i^{(t)}.
```

Bond-dependent diffusion can replace the simple mean with learned edge weights.

## Worked local example

Suppose:

```math
u=0.60,
\quad
v=0.20,
\quad
I=0.10,
\quad
D_u\Delta_Gu=-0.05,
\quad
\Delta t=0.10.
```

The activator derivative is:

```math
0.60-\frac{0.60^3}{3}-0.20+0.10-0.05=0.378.
```

Therefore:

```math
u^{(t+1)}=0.60+(0.10)(0.378)=0.6378.
```

If inhibition is slow, activation initially rises. As v catches up, the negative inhibitor term can reverse the activator, producing a pulse.

## Nullclines and oscillation

Ignoring diffusion, the activator nullcline satisfies:

```math
v=u-\frac{u^3}{3}+I.
```

The inhibitor nullcline satisfies:

```math
v=\frac{u+a}{b}.
```

Their intersection is a fixed point. Its local stability depends on the Jacobian:

```math
J=
\begin{pmatrix}
1-u^2 & -1\\
\varepsilon & -\varepsilon b
\end{pmatrix}.
```

Complex eigenvalues with a positive real part can destabilise the fixed point and lead to a limit cycle; negative real parts produce damped oscillation.

## Expected molecular phenomena

- activation waves moving along bonded paths;
- refractory atoms that cannot immediately reactivate;
- synchronised or desynchronised atom groups;
- isolated pulses;
- damped oscillation; and
- sustained limit cycles.

## Hyperparameters to tune

- activator and inhibitor time-scale ratio;
- nullcline parameters a and b;
- activator and inhibitor diffusion;
- step size;
- chemical input strength;
- generation count;
- learning rates and regularisation; and
- state bounding or clipping.

This is the recommended next rule because it tests oscillation directly through a mathematically interpretable mechanism.
