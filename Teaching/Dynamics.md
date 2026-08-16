# Dynamical Systems

## State and trajectory

At generation `t`, every atom has a state. Collecting all atom states gives the global molecular state `X⁽ᵗ⁾`. Repeated application of the CA rule produces the trajectory:

```text
X⁽⁰⁾, X⁽¹⁾, X⁽²⁾, …, X⁽ᵀ⁾
```

The complete route through state space is part of the molecular representation. Two molecules can reach similar final states by very different trajectories, so the project will not retain only `X⁽ᵀ⁾`.

## Convergence and sinks

A trajectory **converges** when its state approaches a stable value or pattern. A **fixed point** satisfies `X⁽ᵗ⁺¹⁾ = X⁽ᵗ⁾`. A stable fixed-point attractor is also called a **sink**: nearby trajectories move towards it.

## Attractors and basins

An **attractor** is a state or set of states towards which nearby trajectories evolve. Its **basin of attraction** is the collection of initial states that lead to it. An **attractor family** is an empirical grouping of trajectories with related long-term behaviour; defining such families will require explicit, reproducible criteria.

## Oscillators and limit cycles

An **oscillator** displays periodic or approximately periodic behaviour. A **limit cycle** is a repeating finite sequence of states. We may study cycle length, amplitude, autocorrelation, and temporal spectra to distinguish stable oscillation from noise or a long transient.

## Complex transients

A **transient** is the part of a trajectory before it reaches its long-term regime. A complex transient may be long and visually intricate even when the system eventually converges. Visual complexity alone is not evidence of chaos.

## Perturbation sensitivity

Perturbation analysis starts two trajectories from almost identical initial conditions and measures how their separation changes. This can reveal stability, basin boundaries, and sensitive regions near possible bifurcations or activity cliffs.

## Strange attractors

A **strange attractor** is a bounded attracting set with genuinely chaotic dynamics. A defensible claim would require evidence such as sustained sensitivity to initial conditions, appropriate Lyapunov analysis, recurrence structure, and meaningful dimensional or fractal properties. Until then, intricate behaviour will be called **complex dynamics**.

## Dynamical fingerprint

Candidate trajectory summaries include convergence time, step-to-step distance, variation across atoms and time, oscillation amplitude, periodicity, autocorrelation, entropy, spectral features, transient length, and perturbation response. Each quantity will be learned mathematically before adoption.

## Topics to develop

- phase and state space;
- fixed points and stability;
- attractors and basins;
- periodicity and Fourier analysis;
- bifurcations;
- recurrence;
- Lyapunov exponents; and
- dimensionality reduction for projected attractor portraits.
