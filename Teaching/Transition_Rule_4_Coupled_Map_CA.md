# Transition rule 4: coupled-map graph CA

## Aim

Coupled-map systems combine a nonlinear local map with information exchange between neighbours. They are classical laboratories for bifurcation, synchronisation, intermittency and chaos.

## Local nonlinear map

A familiar one-dimensional example is the logistic map:

```math
f(q)=rq(1-q).
```

For values between zero and one, the parameter r controls the behaviour. Low r produces a stable fixed point; increasing r can produce period doubling and chaos.

Our molecular version would use a bounded learned map conditioned on chemistry:

```math
f_{\theta}(q_i^{(t)},x_i,c).
```

## Neighbour coupling

One graph-coupled update is:

```math
q_i^{(t+1)}=(1-D)f_{\theta}(q_i^{(t)},x_i,c)
+\frac{D}{|N(i)|}\sum_{j\in N(i)}
w_{ij}f_{\theta}(q_j^{(t)},x_j,c).
```

The bond-dependent weight is learned from edge features:

```math
w_{ij}=\sigma(g_{\theta}(e_{ij})).
```

## Worked two-atom example

Consider two atoms with states 0.20 and 0.70, logistic parameter 3.5, equal bond weight, and coupling 0.10.

Their uncoupled map values are:

```math
f(0.20)=3.5(0.20)(0.80)=0.56,
```

```math
f(0.70)=3.5(0.70)(0.30)=0.735.
```

The first coupled update is:

```math
q_1^{(t+1)}=(0.90)(0.56)+(0.10)(0.735)=0.5775.
```

The second is:

```math
q_2^{(t+1)}=(0.90)(0.735)+(0.10)(0.56)=0.7175.
```

Coupling pulls the map outputs toward one another without making them identical immediately.

## Local stability

For the logistic map:

```math
f'(q)=r(1-2q).
```

A perturbation approximately evolves as:

```math
\delta_{t+1}\approx f'(q_t)\delta_t.
```

The Lyapunov exponent is:

```math
\lambda=\lim_{T\rightarrow\infty}
\frac{1}{T}\sum_{t=0}^{T-1}\log|f'(q_t)|.
```

A positive value indicates average local stretching. In a molecular graph, coupling can suppress chaos through synchronisation or create spatiotemporal complexity through competing local regimes.

## Expected molecular phenomena

- fixed points and periodic cycles;
- period-doubling sequences;
- synchronised atom groups;
- cluster synchronisation;
- intermittent bursts;
- travelling irregular patterns; and
- possible positive Lyapunov exponents.

## Scientific safeguards

A complicated plot is insufficient evidence of chaos. We must specify perturbation size, rescaling interval, discarded transient, trajectory duration, norm, numerical precision and results across initial conditions.

## Hyperparameters to tune

- local-map nonlinearity range;
- coupling strength;
- bond-weight temperature;
- generation count;
- state bounding;
- learning rates;
- regularisation; and
- perturbation-aware stability penalties.

This is the strongest deliberate chaos candidate in the five-rule programme.
