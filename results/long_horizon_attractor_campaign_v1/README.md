# Long-horizon attractor campaign

This frozen-model campaign propagated 20 complete molecular Graph-CA states through 5,000 generations. It retained every atom and all 16 dynamical channels for all base trajectories, calculated detailed phase-space diagnostics for the ten established visual cases, and tested all 20 cases with eight independent full-state perturbation directions of magnitude 1e-5.

## Current result

Kuramoto-Sakaguchi is the only rule family in which all four screened molecules show replicated positive finite-time separation. Trajectories 7 and 8 remain the principal candidates. Inertial reaction-diffusion and delayed memory contract every tested perturbation. Gated residual has a contracting mean response in all four cases but some direction-dependent early growth. FitzHugh-Nagumo has a mixed, molecule-dependent response.

This is evidence of local finite-time sensitivity, not yet proof of a strange attractor. A defensible chaos claim still requires a renormalized largest Lyapunov exponent, stability across perturbation magnitudes and numerical precision, and exclusion of a very long complex transient.

## Rule summary

| transition_rule | cases | mean_slope | minimum_slope | maximum_slope | mean_positive_fraction |
| --- | --- | --- | --- | --- | --- |
| delayed_memory | 4 | -0.0187306 | -0.0288533 | -0.00328552 | 0 |
| fitzhugh_nagumo | 4 | -0.0106785 | -0.0293595 | 0.00584223 | 0.40625 |
| gated_residual | 4 | -0.0118963 | -0.0271531 | -0.000775959 | 0.5 |
| inertial_reaction_diffusion | 4 | -0.00820873 | -0.00948109 | -0.00756678 | 0 |
| kuramoto_sakaguchi | 4 | 0.0106298 | 0.00793559 | 0.0136388 | 1 |

## Leading sensitivity candidates

| visual_rank | transition_rule | molecule_id | cyp_target | direct_perturbation_slope | direct_perturbation_slope_std | direct_positive_fraction | correlation_dimension | spectral_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | kuramoto_sakaguchi | OCNT-0494110 | CYP2C9 | 0.0136388 | 0.000762712 | 1 | 4.4456 | 0.455339 |
| 8 | kuramoto_sakaguchi | OCNT-2328784 | CYP1A2 | 0.00793559 | 0.00229711 | 1 | 4.82352 | 0.450292 |
| 9 | fitzhugh_nagumo | OCNT-2328824 | CYP2D6 | 0.00584223 | 0.00449554 | 0.875 | 1.58355 | 0.440653 |
| 3 | delayed_memory | OCNT-0494638 | CYP2C9 | -0.00328552 | 0.00144716 | 0 | 2.54909 | 0.371764 |
| 5 | inertial_reaction_diffusion | OCNT-2312382 | CYP1A2 | -0.00783151 | 0.000800169 | 0 | 1.09341 | 0.288861 |
| 6 | inertial_reaction_diffusion | OCNT-2315030 | CYP2D6 | -0.00948109 | 0.000795698 | 0 | 0.719835 | 0.299688 |

## Figures

![Attractor screen](figures/01_attractor_screen.png)

![Phase portraits](figures/02_phase_portraits.png)

![Recurrence plots](figures/03_recurrence_gallery.png)

![Perturbation summary](figures/04_direct_perturbation_summary.png)

![Perturbation curves](figures/05_perturbation_curve_gallery.png)

![Kuramoto divergence](figures/06_kuramoto_trajectory_07_08_divergence.png)

## Retained data

- `base_trajectories`: lossless 5,001-frame atom-by-channel trajectories.
- `case_data`: PCA coordinates, recurrence matrices, spectra, correlation integrals, step energies, and nearest-neighbour divergence curves.
- `perturbations`: eight direct perturbation separation traces for every case.
- `combined_dynamics_evidence.csv`: one-row-per-case numerical summary.
