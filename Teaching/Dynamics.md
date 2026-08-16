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

### Lyapunov analysis

Lyapunov analysis measures how rapidly two initially nearby states separate or converge as a dynamical rule repeatedly acts on them. Begin with a reference state `X⁽⁰⁾` and a slightly perturbed state whose initial separation is `δ₀`. After `t` generations, let the separation be `δₜ`. Over an interval in which the separation behaves approximately exponentially:

```text
δₜ ≈ δ₀ e^(λt)
```

The corresponding finite-time estimate is:

```text
λₜ ≈ (1/t) ln(δₜ / δ₀)
```

Here, `λ` is a **Lyapunov exponent**, `ln` is the natural logarithm, and distance must be defined in the full global CA state space—not merely in a two-dimensional visual projection.

The largest Lyapunov exponent describes the most rapidly separating direction:

- `λ < 0`: nearby trajectories tend to converge, as expected around a stable sink;
- `λ = 0`: perturbations are approximately preserved, as can occur along a neutral direction or ideal periodic motion; and
- `λ > 0`: nearby trajectories separate exponentially, providing evidence of sensitive dependence on initial conditions.

In a bounded deterministic system, a robust positive largest exponent is important evidence for chaos. It is not sufficient by itself to establish a strange attractor: we must also show that trajectories remain bounded, are attracted to a persistent set, and are not displaying only a finite transient or numerical artefact.

For long trajectories, the separation between two freely evolving states may become too large to measure the local divergence rate: bounded states eventually saturate at the size of the attractor. A standard calculation therefore evolves a very small perturbation for a short interval, measures its growth, rescales it to the original small magnitude, and repeats. The logarithmic growth factors are then averaged. For our differentiable CA, the same local stretching can also be studied through the Jacobian of the transition rule, but that method will be derived before use.

For the result to be reproducible and interpretable, our analysis must report:

- **Perturbation size:** the initial magnitude `δ₀` of the small displacement applied to the reference state. It must be small enough to measure local behaviour, but large enough to remain distinguishable from numerical rounding error.
- **Distance norm:** the mathematical rule used to convert the difference between two complete CA states into one separation value. Examples include the Euclidean (`L²`) norm and maximum (`L∞`) norm. Different norms can produce different finite-scale measurements, so the choice must be stated.
- **Rescaling interval:** the number of CA generations allowed between measuring perturbation growth and reducing the perturbed displacement back to magnitude `δ₀`. Intervals that are too long can allow separation to saturate; intervals that are too short can make the estimate sensitive to local fluctuations and numerical noise.
- **Discarded transient:** the initial generations excluded before Lyapunov growth factors are accumulated. This prevents the approach towards the long-term regime from being mistaken for behaviour on the attractor itself.
- **Trajectory length:** the number of post-transient generations over which growth factors are collected and averaged. Longer observations help reveal whether an estimated positive exponent persists rather than appearing briefly.
- **Numerical precision:** the floating-point representation used in the calculation, such as 32-bit or 64-bit values. Rounding error can create, conceal, or distort very small separations, so stability should be checked at sufficient precision.
- **Results across multiple initial conditions:** repeated estimates from different molecular states or controlled perturbations. Agreement across relevant starting points provides stronger evidence than a result obtained from one exceptional trajectory.

A positive value seen only briefly will be reported as **finite-time local divergence**, not immediately as chaos.

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
