# Transition rule 2: inertial reaction–diffusion graph CA

## Aim

This lesson explains the rule used by prototypes two and three. It combines nonlinear chemical response, neighbour diffusion, a restoring force and momentum-like memory.

## State and velocity

Each atom carries a state and a velocity:

```math
h_i^{(t)}\in\mathbb{R}^{d},
\qquad
v_i^{(t)}\in\mathbb{R}^{d}.
```

The state is learned molecular information. The velocity records how that information was changing.

## Learned reaction

The reaction term is chemically conditioned:

```math
r_i^{(t)}=\tanh\left(
W_sh_i^{(t)}+m_i^{(t)}+W_xx_i+W_cc+b
\right).
```

It is called a reaction term by analogy with reaction–diffusion systems: it changes information locally before diffusion redistributes it.

## Graph diffusion

The neighbour mean is:

```math
\bar h_{N(i)}^{(t)}=
\frac{1}{|N(i)|}\sum_{j\in N(i)}h_j^{(t)}.
```

The graph diffusion term is:

```math
D\left(\bar h_{N(i)}^{(t)}-h_i^{(t)}\right).
```

If an atom's channel is larger than its neighbourhood mean, diffusion pushes it down. If it is smaller, diffusion pushes it up.

## Restoring force and inertia

The complete velocity update is:

```math
v_i^{(t+1)}=
\gamma v_i^{(t)}+
\Delta t\left[
r_i^{(t)}+
D\left(\bar h_{N(i)}^{(t)}-h_i^{(t)}\right)-
\kappa h_i^{(t)}
\right].
```

The roles are:

- gamma preserves a fraction of the previous velocity;
- delta t is the update step;
- D controls neighbour diffusion;
- kappa pulls the state toward zero;
- the reaction term supplies learned chemical forcing.

The state then advances:

```math
h_i^{(t+1)}=\tanh\left(
h_i^{(t)}+\Delta t\,v_i^{(t+1)}
\right).
```

## Worked diffusion example

Suppose one channel of an atom has value 0.80 and its two neighbours have values 0.20 and 0.40:

```math
\bar h_{N(i)}=\frac{0.20+0.40}{2}=0.30.
```

With diffusion coefficient 0.10:

```math
D(\bar h_{N(i)}-h_i)=0.10(0.30-0.80)=-0.05.
```

Diffusion contributes a downward force of 0.05.

## Worked complete step

Let the previous velocity be 0.10, inertia 0.80, step size 0.20, reaction 0.30, diffusion contribution minus 0.05, and restoring contribution minus 0.08:

```math
v^{(t+1)}=(0.80)(0.10)+(0.20)(0.30-0.05-0.08).
```

```math
v^{(t+1)}=0.08+0.034=0.114.
```

For an old state of 0.40:

```math
h^{(t+1)}=\tanh(0.40+(0.20)(0.114)).
```

```math
h^{(t+1)}\approx\tanh(0.4228)\approx0.399.
```

## Expected dynamics

- Strong restoration and diffusion favour stable sinks.
- Moderate inertia can produce overshoot and damped oscillation.
- Weak damping can preserve long transients.
- Reaction–diffusion competition can create spatial patterns over the molecular graph.
- Excessive step size or inertia can destabilise training.

## Hyperparameters to tune

- generation count;
- maximum learned step size;
- initial inertia;
- diffusion and restoration initialisations;
- learning rates;
- regularisation;
- clipping threshold; and
- trajectory windows used by the fingerprint.

Prototype three showed long curved transients followed by contraction. Its negative perturbation slopes supported stable attractors rather than chaos.
