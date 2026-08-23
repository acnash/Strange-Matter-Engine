# Graph cellular-automata transition rules

## Why compare transition rules?

A transition rule determines how every atom updates its evolving state from its own state, its bonded neighbours, bond properties, fixed chemical information, and CYP context. Changing the rule changes the model's inductive bias: the kinds of propagation, memory, oscillation, convergence, and instability it can represent naturally.

Every proposed rule remains differentiable so that its parameters can be learned by backpropagation from experimental pIC50 values.

## Correct prototype lineage

| Prototype | Transition rule | Generations | Hyperparameter treatment |
|---|---|---:|---|
| Prototype 1 | Gated residual message-passing CA | 16 | Fixed configuration |
| Prototype 2 | Inertial reaction–diffusion graph CA | 100 | Fixed configuration |
| Prototype 3 | Inertial reaction–diffusion graph CA | 200 | Six-candidate multi-fidelity tuning |

Prototype 2 changed both the rule and generation count relative to prototype 1. Prototype 3 retained prototype 2's rule, doubled the generation count, and introduced tuning.

## Rule 1: gated residual message-passing CA

This was the prototype-one rule. Each atom constructs a candidate state from itself, its neighbours, bond features, chemical features, and CYP context:

```math
\widetilde h_i^{(t+1)}=\tanh\left(W_s h_i^{(t)}+m_i^{(t)}+W_xx_i+W_cc+b\right).
```

A learned gate decides how much of the old and candidate states to retain:

```math
\alpha_i^{(t)}=\sigma\left(W_g[h_i^{(t)},m_i^{(t)},x_i,c]+b_g\right),
```

```math
h_i^{(t+1)}=\left(1-\alpha_i^{(t)}\right)h_i^{(t)}+\alpha_i^{(t)}\widetilde h_i^{(t+1)}.
```

The residual path preserves information, while the gate supplies learned damping and update speed. This is a strong stable baseline.

## Rule 2: inertial reaction–diffusion graph CA

This was used by prototypes two and three. Each atom has both a state and a velocity. A nonlinear chemical reaction, graph diffusion, restoring force, and inertia jointly determine the next velocity:

```math
v_i^{(t+1)}=\gamma v_i^{(t)}+\Delta t\left[r_i^{(t)}+D\left(\bar h_{N(i)}^{(t)}-h_i^{(t)}\right)-\kappa h_i^{(t)}\right],
```

```math
h_i^{(t+1)}=\tanh\left(h_i^{(t)}+\Delta t\,v_i^{(t+1)}\right).
```

Memory and diffusion make waves, damped excursions, slow manifolds, and oscillatory transients more natural than in a purely first-order rule.

## Rule 3: activator–inhibitor graph CA

Each atom carries coupled activator and inhibitor fields. The activator can amplify local activity; the slower inhibitor suppresses it. A differentiable FitzHugh–Nagumo-inspired form is:

```math
u_i^{(t+1)}=u_i^{(t)}+\Delta t\left(u_i^{(t)}-\frac{(u_i^{(t)})^3}{3}-v_i^{(t)}+I_i+D_u\Delta_Gu_i^{(t)}\right),
```

```math
v_i^{(t+1)}=v_i^{(t)}+\Delta t\left(\varepsilon\left(u_i^{(t)}+a-bv_i^{(t)}\right)+D_v\Delta_Gv_i^{(t)}\right).
```

Here $I_i$ is learned chemical/CYP input and $\Delta_G$ is graph diffusion. This rule is especially attractive for travelling activation waves, pulses, refractory behaviour, and sustained oscillation.

## Rule 4: coupled-map graph CA

Each atom undergoes a nonlinear local map while bonded neighbours diffusively couple the maps:

```math
q_i^{(t+1)}=(1-D)f_{\theta}(q_i^{(t)},x_i,c)
+\frac{D}{|N(i)|}\sum_{j\in N(i)}f_{\theta}(q_j^{(t)},x_j,c,e_{ij}).
```

The local map may use a bounded learned polynomial, logistic-like map, or small neural transformation. Coupled-map lattices are classical models of synchronisation, spatiotemporal intermittency, bifurcation, and chaos. This is the clearest candidate when we deliberately want to test whether chemically conditioned chaotic regimes can emerge.

## Rule 5: damped symplectic graph CA

Each atom carries position-like and momentum-like latent variables. The update separates momentum and state advances:

```math
p_i^{(t+1)}=\rho p_i^{(t)}-\Delta t\,\nabla_{q_i}U_{\theta}(q,x,c,E),
```

```math
q_i^{(t+1)}=q_i^{(t)}+\Delta t\,p_i^{(t+1)}.
```

The learned potential $U_{\theta}$ couples bonded atoms and chemical context; $\rho$ introduces controlled damping. This structure naturally supports oscillatory modes and long-lived collective motion while making the source of dissipation explicit.

## Rule 6: FitzHugh–Nagumo excitable graph CA

Each channel pair contains a fast excitation variable and a slower recovery variable. Chemical and CYP information supplies a learned stimulus, while graph diffusion carries excitation to bonded atoms. The recovery variable creates a refractory period after a pulse. This is an explicitly excitable candidate distinct from the broader activator–inhibitor implementation used as rule 3.

The four generic controls represent excitation diffusion, excitation threshold, stimulus strength, and recovery timescale.

## Rule 7: Gray–Scott graph reaction–diffusion CA

Each channel pair contains two bounded fields. They diffuse at different rates and react through a cubic interaction. Feed and removal terms continually replenish one field and remove the other. On a molecular graph this can form localised activity, travelling patterns, multiple stable regimes, or convergence.

The four generic controls represent the two diffusion coefficients, feed rate, and removal rate.

## Rule 8: Kuramoto–Sakaguchi graph oscillator CA

Every dynamical channel is interpreted as a circular phase. Each atom has a chemically conditioned natural frequency and is coupled to the phases of bonded neighbours. A phase lag permits synchronisation, phase-locked clusters, travelling phase waves, and frustrated oscillation.

The principal generic controls represent natural-frequency scale, coupling strength, and phase lag. The fourth generic control remains available to the shared search interface but is not used by this rule.

## Rule 9: conservative graph-flux CA

This rule moves latent quantity across each bond using equal and opposite fluxes. Consequently, the sum of every dynamical channel over all atoms is conserved to floating-point precision during the transition. It tests whether pIC50-relevant information can be represented by redistribution through the molecule rather than by creating or destroying latent activity.

The first generic control scales the flux. The remaining three generic controls remain available to the shared search interface but are not used by this rule.

## Rule 10: delayed-memory graph CA

The next state depends on the present reaction and a state retained from an earlier generation. This gives the system an explicit time delay rather than relying only on information stored in the current state. Delayed feedback can produce slow oscillation, bifurcation, long transients, and multistability.

The four generic controls represent delay length, delayed-state mixture, feedback strength, and damping.

## Mandatory fair-tuning policy

From prototype four onward, every transition rule will receive its own hyperparameter tuning. We will not use hyperparameters optimised for one rule as evidence that another rule performs poorly.

For a fair comparison:

- every rule uses identical grouped folds and molecular standardisation;
- every rule receives a comparable tuning budget;
- rule-specific stability parameters are tuned within declared ranges;
- common parameters such as learning rates, regularisation, clipping, batch size, and generation count are tuned;
- the selection metric is grouped-validation RMSE inside the tuning loop;
- dynamical-interest scores never select a model using protected experimental labels;
- finalists are confirmed across multiple seeds; and
- final comparisons use protected outer folds or the accepted grouped nested-cross-validation design.

The tuning record will include every attempted configuration, failure, score, seed, computational budget, and promoted checkpoint.

## Original five-rule trial order

1. Retune the gated residual rule as the predictive baseline.
2. Retune the inertial reaction–diffusion rule as the current dynamical baseline.
3. Trial the activator–inhibitor rule for waves and oscillation.
4. Trial the coupled-map rule for bifurcation and possible chaos.
5. Trial the damped symplectic rule for persistent collective modes.

The second trial series comprises FitzHugh–Nagumo, Gray–Scott, Kuramoto–Sakaguchi, conservative graph flux, and delayed memory. Each receives the same independent tuning and seed-confirmation procedure before comparison with the original five.

Predictive performance and dynamical richness will be reported separately. A fascinating trajectory is not automatically a better pIC50 predictor, and a low RMSE does not by itself establish scientifically meaningful dynamics.

## Dedicated lessons

- [Rule 1: Gated Residual CA](Transition_Rule_1_Gated_Residual_CA.md)
- [Rule 2: Inertial Reaction–Diffusion CA](Transition_Rule_2_Inertial_Reaction_Diffusion_CA.md)
- [Rule 3: Activator–Inhibitor CA](Transition_Rule_3_Activator_Inhibitor_CA.md)
- [Rule 4: Coupled-Map CA](Transition_Rule_4_Coupled_Map_CA.md)
- [Rule 5: Damped Symplectic CA](Transition_Rule_5_Damped_Symplectic_CA.md)
