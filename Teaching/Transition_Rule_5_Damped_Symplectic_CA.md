# Transition rule 5: damped symplectic graph CA

## Aim

This rule gives every atom position-like and momentum-like latent variables. It is inspired by Hamiltonian mechanics, but includes controlled damping so the model can represent both persistent oscillation and relaxation.

## Latent coordinates and momenta

For each atom:

```math
q_i^{(t)},p_i^{(t)}\in\mathbb{R}^{d/2}.
```

Together they form an eight-channel state when each has four channels.

## Learned potential

The molecular graph defines a learned potential energy:

```math
U_{\theta}(q,x,c,E)=
\sum_i U_i(q_i,x_i,c)+
\sum_{(i,j)\in E}U_{ij}(q_i,q_j,e_{ij}).
```

The negative gradient supplies a force:

```math
F_i^{(t)}=-\nabla_{q_i}U_{\theta}(q^{(t)},x,c,E).
```

## Damped symplectic-Euler update

Momentum is updated first:

```math
p_i^{(t+1)}=\rho p_i^{(t)}+\Delta t\,F_i^{(t)}.
```

Position then uses the new momentum:

```math
q_i^{(t+1)}=q_i^{(t)}+\Delta t\,p_i^{(t+1)}.
```

The damping factor lies between zero and one. A value near one preserves motion; a smaller value dissipates it rapidly.

## Harmonic worked example

For a one-dimensional harmonic potential:

```math
U(q)=\frac{kq^2}{2},
```

the force is:

```math
F=-\frac{dU}{dq}=-kq.
```

Let the state be q equals 1.0, momentum equals 0, spring strength equals 0.5, damping equals 0.90, and step size equals 0.20.

First update the momentum:

```math
p^{(t+1)}=(0.90)(0)+(0.20)(-0.5)(1.0)=-0.10.
```

Then update the position:

```math
q^{(t+1)}=1.0+(0.20)(-0.10)=0.98.
```

The next force remains negative, so momentum grows in the negative direction. The system crosses the origin, overshoots, and oscillates. Damping gradually reduces the amplitude.

## Graph-coupled potential example

A simple bonded term is:

```math
U_{ij}=\frac{k_{ij}}{2}\lVert q_i-q_j\rVert^2.
```

Its force on atom i is:

```math
F_{ij\rightarrow i}=-k_{ij}(q_i-q_j).
```

This resembles a latent spring along each molecular bond. Learned bond-dependent stiffness can make single, double, aromatic and other bond classes transmit information differently.

## Energy and dissipation

Without damping, an ideal Hamiltonian system conserves:

```math
H(q,p)=U(q)+\frac{1}{2}\sum_i\lVert p_i\rVert^2.
```

With damping below one, energy decreases over time. Monitoring latent energy provides a direct diagnostic of persistent modes versus dissipative collapse.

## Expected molecular phenomena

- normal-mode-like collective oscillations;
- phase relationships between atom groups;
- long-lived ringing;
- damped oscillation;
- energy transfer between modes; and
- convergence when damping dominates.

## Hyperparameters to tune

- step size;
- damping range;
- potential-network size;
- bond stiffness parameterisation;
- generation count;
- energy regularisation;
- learning rates; and
- gradient clipping.

This rule is attractive when we want interpretable oscillatory motion without deliberately driving the system toward chaos.
