# Inertial reaction–diffusion graph cellular automata

## Purpose

This update rule lets an atom's latent state respond to chemistry, retain momentum, exchange information with bonded neighbours, and relax under a restoring force. It creates a discrete dynamical system on a molecular graph.

## State and velocity

Atom $i$ has an eight-channel state $h_i^{(t)}$ and an eight-channel velocity $v_i^{(t)}$. The state is learned information, not a directly measured physical field. Velocity records how the state was changing, giving the update memory.

## Reaction

The learned reaction term combines the atom's current state, neighbour and bond messages, fixed chemical descriptors, and CYP identity:

```math
r_i^{(t)}=\tanh\left(W_s h_i^{(t)}+m_i^{(t)}+W_x x_i+W_c c+b\right).
```

## Diffusion, restoration, and inertia

```math
v_i^{(t+1)}=\gamma v_i^{(t)}+\Delta t\left[r_i^{(t)}+D\left(\bar h_{N(i)}^{(t)}-h_i^{(t)}\right)-\kappa h_i^{(t)}\right].
```

- $\gamma$ is inertia: larger values preserve more of the previous motion.
- $\Delta t$ is the learned step size.
- $D$ controls diffusion between bonded neighbours.
- $\bar h_{N(i)}^{(t)}-h_i^{(t)}$ compares the atom with its neighbourhood.
- $\kappa$ is a restoring strength that discourages unbounded drift.

The state then advances through a bounded nonlinearity:

```math
h_i^{(t+1)}=\tanh\left(h_i^{(t)}+\Delta t\,v_i^{(t+1)}\right).
```

## Possible behaviours

The balance among reaction, inertia, diffusion, restoration, and nonlinear saturation can produce rapid convergence, slow transients, damped oscillation, sustained oscillation, or more complicated paths. A visually complicated trajectory is not automatically chaotic. Chaos requires controlled perturbations and reproducible positive finite-time Lyapunov evidence.

## Prototype-two observation

The 100-generation prototype predominantly produced attractor-seeking relaxation. A smaller family, especially in the CYP1A2 context, retained appreciable late motion. No sustained oscillator or chaos was established, making longer trajectories and perturbation analysis the appropriate next experiment.
