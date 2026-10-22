# Extended audit of the Kuramoto–Sakaguchi candidates

Completed 29 September 2026. This analysis used the original frozen checkpoint and molecular graphs, with no predictive training, model promotion, submission changes or blind-label use.

## Finding

Both candidates show sustained multidirectional expansion and average local volume contraction. The evidence supports bounded hyperchaotic dynamics and consistency with a sampled attracting distribution. A strict strange-attractor classification remains unresolved: dimension estimates depend on projection and lack a sufficiently wide scaling range, and the learned map is discontinuous across the wrapped phase boundary. The previous wording claiming confirmed fractal strange attractors has therefore been qualified in the revised manuscript.

| Case | Leading exponent, 20,000 generations | Eighth exponent | Positive full-spectrum components | Full-spectrum sum | Kaplan–Yorke heuristic |
|---|---:|---:|---:|---:|---:|
| OCNT-2328784 / CYP1A2 | 0.012115 | 0.004376 | 29 / 240 | -3.606412 | 68.19 |
| OCNT-0494110 / CYP2C9 | 0.012641 | 0.004421 | 30 / 320 | -5.144542 | 73.91 |

Exponents are per computational generation. The full spectra use a subsequent 10,000-generation interval after 1,000 generations of basis alignment. Positive counts near zero retain finite-time uncertainty. Kaplan–Yorke values are derivative-spectrum heuristics; they are not measured correlation dimensions or rigorous dimensions of the discontinuous map.

## Protocol and limitations

- Original candidates and frozen model; 5,000-generation burn-in and 20,000 measured generations in float64.
- Analytic tangent propagation with eight vectors and QR every ten generations, checked against automatic differentiation and the original step implementation at the initial states.
- Renormalized finite separations 1e-5, 1e-7 and 1e-9. The two smaller scales agree closely with each other; all estimates are positive.
- Complete 240- and 320-dimensional derivative spectra; the sums are negative. The second case has 28 and 30 positive components in the two halves of that interval.
- Four additional displaced initial states per molecule at radii 0.1, 0.5, 1 and 2. Their late distribution distances are 0.843–1.036 times a reference-window distance. This is a descriptive finite-sampling comparison, not a hypothesis test or proof of global attraction.
- Correlation integrals in 6, 12, 24 and full circular coordinates, with Theiler exclusions of 100, 500 and 1,000 generations. Projected slopes near 4.7 rise to about 27–29 in full coordinates. Radius ranges span less than 0.3 decades, so no robust fractal dimension is assigned.
- A circular input separation near 2e-10 across a phase boundary produces next-step separations 0.005422 and 0.005871. The raw-state frequency drive makes the learned rule piecewise smooth rather than a continuous map of the phase torus.
- Boundedness is enforced by wrapping. Longer observation cannot establish infinite-time behaviour; independent checkpoints, broader basins and stable dimensional scaling remain future tests.
- CUDA index accumulation can produce floating-point-order differences between reruns. Exact orbit agreement beyond the initial step is not an expected reproducibility criterion for unstable trajectories. Stored arrays preserve this realised run, and replication should compare diagnostic distributions and exponents.

## Plot-ready data

The supporting numerical archive is provided in this directory on the `Complex-Systems` branch. Download the NPZ files to inspect the arrays with NumPy. Manuscript drafts, videos and temporary progress files are excluded from this numerical archive.

- `OCNT-2328784_0_trajectories.npz`: `states` has shape (20001, 15, 16).
- `OCNT-0494110_1_trajectories.npz`: `states` has shape (20001, 20, 16).
- `states[t, atom, channel]` is the float64 wrapped state after burn-in plus t generations. All 16 channels are retained; values represent learned computational states, not atomic coordinates or physical motion.
- `basin_states[sample, displacement, atom, channel]` holds the four displaced trajectories every ten generations. Sample zero is measured generation 10. Displacements are ordered by radius 0.1, 0.5, 1 and 2.
- `src`, `dst`, `edge_features`, `atom_features` and `smiles` retain graph connectivity and chemical context. Generate display coordinates from the saved SMILES while verifying atom ordering against the saved graph. A common circular colour mapping avoids interpreting a wrap as a large physical state jump.
- `tangent_local_exponents` and `finite_difference_local_exponents` retain interval-by-interval rates; `epsilon` identifies finite-separation columns.
- `*_geometry.npz` retains radii and correlation integrals by embedding width and temporal exclusion.
- `*_full_spectrum.npz` retains all QR local rates. Columns track QR vector indices; the summary spectrum is sorted by its time average.
- `results.json` contains the long-horizon, boundary, basin and dimension diagnostics; `*_full_spectrum.json` contains complete spectrum summaries.
- `provenance.json` and `full_spectrum_provenance.json` identify the checkpoint and audit-script hashes. `source_snapshot/` preserves the implementation.

## Reproduction

The `source_snapshot/` directory preserves the analysis script and model implementation used for this run. Re-running the analysis requires the checkpoint identified by SHA-256 in `provenance.json`, the original dataset and the project's `strange-matter-gpu` Python environment; the checkpoint and dataset are not included in this directory. In a configured workspace, place the analysis script at `scripts/audit_strange_attractor_20260929.py` and run it for the long-horizon analysis, then repeat with `AUDIT_FULL_SPECTRUM=1` for the full derivative spectra. Preserve the current archive before a rerun if both numerical realisations are required. The complete spectrum uses the saved terminal state from the first stage.

## References

Grassberger and Procaccia (1983), https://doi.org/10.1016/0167-2789(83)90298-1.

Theiler (1986), https://doi.org/10.1103/PhysRevA.34.2427.

Eckmann and Ruelle (1992), https://doi.org/10.1016/0167-2789(92)90023-G.
