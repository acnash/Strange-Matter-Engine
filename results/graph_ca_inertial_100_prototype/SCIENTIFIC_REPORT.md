# Prototype two scientific report

## Executive result

The 100-generation inertial reaction–diffusion Graph-CA trained for 49 epochs. Its restored fit RMSE was **0.835 pIC50** and its grouped-validation RMSE was **0.893 pIC50**. It produced predictions for all 3,000 blinded molecule–CYP pairs and complete trajectories for dynamical screening.

The clearest dynamical finding is a family of **long, curved, damped transients approaching attractors**. Several trajectories remain visibly active at generation 100, particularly CYP1A2 examples, but the present evidence does not support a sustained oscillator, strange attractor, or chaos.

## Transition rule

Each atom carries an eight-channel state. The learned reaction term combines its present state, transformed neighbour messages, bond information, fixed chemical features, and the CYP context. The update also carries velocity and includes graph diffusion and a restoring force:

```math
v_i^{(t+1)}=\gamma v_i^{(t)}+\Delta t\left[r_i^{(t)}+D\left(\bar h_{N(i)}^{(t)}-h_i^{(t)}\right)-\kappa h_i^{(t)}\right]
```

```math
h_i^{(t+1)}=\tanh\left(h_i^{(t)}+\Delta t\,v_i^{(t+1)}\right)
```

This gives the system memory, local propagation, dissipation, and nonlinear saturation.

## Predictive result

| Quantity | Result |
|---|---:|
| Fit observations | 5,216 |
| Grouped-validation observations | 1,309 |
| Epochs | 49 |
| Fit RMSE | 0.835 pIC50 |
| Grouped-validation RMSE | 0.893 pIC50 |
| Blinded predictions | 3,000 |
| Visual trajectories | 20 |

The validation score is an honest grouped holdout result for this single prototype split. It is not the final nested-cross-validation estimate.

## Dynamical screening

Every blinded trajectory was screened using late step size, final step size, late amplitude, approximate recurrence, curvature, and spectral concentration. The 20 visual examples deliberately sample four behaviours: strongest apparent recurrence, concentrated spectra, curved paths through state space, and persistent late motion.

Across all 3,000 trajectories, mean late motion ranged from **0.00028** to **0.00811** state units per generation. Final step size ranged from **0.000009** to **0.00673**. Thus, many trajectories were close to stationary by generation 100, while a smaller slow-relaxing family had not fully settled.

### Attractors and long transients

The strongest supported interpretation is attraction toward fixed points. In the selected examples, successive late displacement vectors did not reverse direction. Their motion generally shrank while continuing along a curved or nearly monotonic path. This is consistent with damped relaxation into an attractor basin.

The most persistent examples were:

- `OCNT-2535122 / CYP1A2`: mean late motion 0.00811 and final step 0.00386;
- `OCNT-2535623 / CYP1A2`: mean late motion 0.00803 and the largest late amplitude, 0.03621;
- `OCNT-2534843 / CYP1A2`: mean late motion 0.00787 and final step 0.00420;
- `OCNT-2535578 / CYP1A2`: mean late motion 0.00777 and final step 0.00499;
- `OCNT-2535357 / CYP1A2`: mean late motion 0.00776 and final step 0.00585.

These are scientifically interesting **complex or extended transients**. A longer run is needed to determine their final limiting state.

### Oscillation

Some trajectories had concentrated low-frequency spectra or approximate recurrence scores. Detailed inspection showed no repeated reversal of the late state-space velocity among the selected 20. The spectral peaks therefore arise primarily from smooth curved relaxation over a finite observation window, rather than a completed repeating cycle.

The conclusion for this run is: **no sustained oscillator was demonstrated**.

### Strange attractors and chaos

No trajectory should presently be labelled a strange attractor or chaotic. The stored trajectories show structured transients, but chaos requires a perturbation experiment: start two copies infinitesimally close together, repeatedly measure their separation, and estimate a finite-time Lyapunov exponent.

```math
\lambda_T=\frac{1}{T}\sum_{t=1}^{T}\log\left(\frac{\delta_t}{\delta_{t-1}}\right)
```

A persistently positive value across perturbation sizes, initial conditions, trajectory lengths, and numerical precision would support local exponential divergence. That experiment remains the next dynamical test.

## PyMOL visualisation

Each PDB contains **102 display states**: graph-CA generations 0–100 followed by the labelled hydrogen coda. The B-factor field stores a 0–100 visual scaling of the eight-channel atom-state magnitude. The matching NPZ file retains the lossless eight-channel trajectory.

Load `load_20_trajectories.pml`, then use `gca_next`, `gca_previous`, `gca_state 51`, `gca_play`, and `gca_stop`. The controller embeds generation-specific values because some PyMOL builds collapse B-factors across multi-model states.

## Scientific conclusion

The new rule produced richer and much longer relaxation paths than the first prototype. The most credible phenomenon is a mixture of rapidly convergent attractor-seeking trajectories and a CYP1A2-enriched family of slow, persistent transients. This is a useful result: it identifies precisely which cases should receive longer simulations and formal perturbation/Lyapunov analysis next.
