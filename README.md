# Strange Matter Engine

## Introduction

Strange Matter Engine is an experimental molecular machine-learning project for the OpenADMET CYP inhibition challenge. It will explore whether a compact, learned **graph cellular automaton (GCA)** can transform molecular structure into interpretable emergent dynamics that predict direct CYP inhibition (`pIC50`) and time-dependent inhibition (`is_TDI`).

Each molecule will be represented as a graph: atoms are cells, chemical bonds define their neighbourhoods, and a small shared local rule evolves every atom's state through time. The model will preserve and analyse the complete trajectory of this evolution—from the chemically initialised state through every subsequent generation—rather than reducing the molecule to its final state.

The project is both a predictive experiment and a study of molecular dynamical behaviour. It will investigate convergence, oscillations, attractor families, perturbation sensitivity, complex transients, and possible relationships between dynamical regimes and activity cliffs. Any claim of chaotic behaviour will require mathematical evidence.

The work is designed for a personal workstation, with an emphasis on compact models, scientific interpretability, honest benchmarking, and visualisations grounded in quantities genuinely produced or analysed by the model.

The evolving [Teaching curriculum](Teaching/README.md) records the scientific and mathematical background behind every component so that implementation and understanding advance together.

The [Visual Laboratory](notebooks/README.md) provides interactive, code-hidden Jupyter experiences with clean RDKit molecular depictions, a restrained cyberpunk visual system, graph views, atom and bond inspectors, encoding heatmaps, and a transparent first graph cellular automaton.

## Project visual standard

All future notebooks, plots, reports, molecular diagrams, trajectory visualisations, and presentation graphics must reproduce the established **clean cyberpunk** visual language. This is a persistent project requirement, not a one-off styling suggestion.

### Core palette

| Role | Colour | Hexadecimal value |
|---|---|---|
| Deep-ink background | Near-black | `#070914` |
| Raised panels and code backgrounds | Midnight blue | `#11152A` |
| Primary signal, graph connectivity, and starts | Electric cyan | `#27E1FF` |
| Secondary signal, selections, and endpoints | Hot magenta | `#FF3CAC` |
| Strong changes and checkpoints | Acid yellow–lime | `#F9F871` |
| Categorical accent | Cyber orange | `#FF9F43` |
| Intermediate states and aromatic edges | Muted violet | `#7A5CFA` |
| Main text and carbon skeletons | Cool off-white | `#DCE6F2` |
| Secondary lines and subdued annotations | Steel grey | `#65758B` |
| Fluorine and chlorine | Cyber green | `#43F6A7` |

Scientific plots may use the closely matched high-contrast plotting variants cyan `#00E5FF`, magenta `#FF1493`, lime `#A6FF00`, violet `#6C4CFF`, and orange `#FF9F1C` when stronger separation is required against the deep-ink background.

### Consistent meaning

- Use deep ink for figure and axes backgrounds, with cool off-white labels and restrained grid lines.
- Use cyan for primary information, graph connectivity, reference trajectories, and starting states.
- Use magenta for selected or contrasting information, predictions, and terminal states.
- Use lime or acid yellow for the strongest activity, important thresholds, and exceptional changes.
- Use violet and orange for additional categories rather than changing the principal cyan–magenta hierarchy.
- Preserve clean RDKit chemical rendering: carbon and ordinary bonds are off-white, nitrogen is cyan, oxygen is hot magenta, sulfur and phosphorus are acid yellow, fluorine and chlorine are green, and bromine is orange.
- Colours must encode a stated scientific quantity or category and must be accompanied by labels, legends, or colour bars. They must not imply molecular motion or chemical meaning that the model did not calculate.
- When an existing project graphic is extended or regenerated, retain this palette and semantic mapping unless a scientifically necessary visual encoding is documented explicitly.

## Cross-platform computation contract

Strange Matter Engine uses **one scientific codebase** on both computers. The molecular preparation, grouped data split, atom and bond features, Graph-CA equations, trajectory fingerprints, ridge readout, losses, metrics, and saved parameters must remain identical across platforms. Only the device performing the tensor arithmetic changes.

This contract is a persistent project requirement:

| Computer | Required device | Principal work |
|---|---|---|
| Windows PC with NVIDIA GPU | CUDA | Production training, hyperparameter searches, multi-seed confirmation, and large batched experiments |
| MacBook without a supported GPU | CPU | Inspecting saved results, generating reports and plots, testing code, and extended forward-only analysis of selected trajectories |

Do not create separate Windows and Mac model implementations. Changes to the scientific model must be made once in the shared repository and exercised through the common runner.

### Device selection

The shared runtime accepts three explicit device values:

- `auto`: select CUDA when PyTorch can access it; otherwise select CPU;
- `cuda`: require an available CUDA device and stop with a clear error if none is available; and
- `cpu`: require CPU execution even when CUDA exists.

The production controller defaults to `auto`. It uses the active Python interpreter, unless `--python` or the `SME_PYTHON` environment variable explicitly names another interpreter. No user-specific Windows or macOS path is stored in the controller.

Every completed training run records the operating system, machine architecture, Python version, Python executable, PyTorch version, CUDA runtime, resolved device, and GPU name in `metrics.json`. PyTorch checkpoints must always be loaded with an explicit `map_location`, allowing CUDA-trained parameters to be inspected on a CPU-only Mac.

### Windows GPU environment

From Anaconda Prompt or PowerShell in the repository root:

```powershell
conda env create -f environment-gpu.yml
conda activate strange-matter-gpu
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The first printed value must be `True` before beginning production training.

Run one complete production transition-rule study with:

```powershell
python scripts/run_production_transition_study.py --rule coupled_map --device auto
```

Use `--device cuda` when a run must fail rather than silently fall back to the CPU. The runner batches molecular graphs on the selected device; batch size remains a hyperparameter and may be reduced if GPU memory is insufficient.

### Mac CPU environment

From Terminal in the repository root:

```bash
conda env create -f environment-cpu.yml
conda activate strange-matter-cpu
python -c "import torch; print(torch.cuda.is_available())"
```

The expected value on this CPU-only Mac is `False`.

Use CPU explicitly for a production-controller invocation:

```bash
python scripts/run_production_transition_study.py --rule coupled_map --device cpu
```

A complete production search is computationally expensive on the Mac and is normally assigned to the Windows GPU. CPU execution is intended for verification and analysis, and for extended forward-only trajectories after a trained model and a small candidate set have been frozen.

### Reproducibility and transfer

1. Commit and push code, environments, configuration, metrics, selected checkpoints, prediction tables, and reports to `main` from the machine that produced them.
2. Pull `main` on the other computer before inspecting or extending the work.
3. Preserve seeds, grouped splits, hyperparameters, numerical precision, and model version when comparing CPU and CUDA results.
4. Expect small floating-point differences between CPU and CUDA. Compare them using declared numerical tolerances rather than requiring bit-for-bit equality.
5. Never use blinded challenge labels for training, hyperparameter selection, early stopping, or dynamical-candidate selection.

### Frozen long-horizon dynamics

Long-horizon dynamical investigation is performed only after production training and model selection. It loads a frozen checkpoint, selects validation molecule–CYP cases using several complementary short-trajectory criteria, and propagates those cases without backpropagation or parameter updates.

The coupled-map study defaults to 100 candidates, balanced as 25 cases per CYP target, 5,000 generations, and a 1,000-generation discarded transient:

```bash
conda activate strange-matter-cpu
sh scripts/run_extended_coupled_map_dynamics.sh
```

The analysis measures the complete atom-by-channel state rather than only the molecule-level mean. It writes recurrence, late-motion, frequency, autocorrelation, and finite-time perturbation summaries to `results/coupled_map_5000_generation_dynamics`. It stores molecule-level mean trajectories for plotting but deliberately creates no PyMOL or atom-coordinate files. PyMOL generation begins only after scientifically interesting structures have been reviewed and selected.

## Goal

The primary goal is to build, understand, validate, and submit a complete end-to-end model that maps:

> **SMILES + CYP identity → predicted direct-inhibition pIC50 + TDI probability**

The direct-inhibition and time-dependent-inhibition (TDI) challenge tracks are independent. A shared GCA trajectory encoder will support two task-specific readouts:

- a regression readout predicting `pIC50` for CYP1A2, CYP2C9, CYP2D6, and CYP3A4; and
- a binary-classification readout predicting `is_TDI` for CYP2D6 and CYP3A4.

The TDI readout will produce a probability during training and validation. A decision threshold selected using training data only will convert that probability into the Boolean value required by the challenge submission schema.

Success means:

- producing a valid competition submission;
- understanding and being able to explain every equation, operation, and trainable component;
- retaining a genuine graph-cellular-automata architecture built around a repeated local rule;
- using the full temporal trajectory as part of the molecular representation;
- comparing predictive performance fairly with conventional baselines;
- analysing emergent dynamics without overclaiming; and
- creating memorable scientific visualisations and a clear public account of the project.

Leaderboard victory is welcome, but learning, scientific defensibility, completion, and insight are the principal measures of success.

## Official challenge contract

The authoritative rules, submission form, and live leaderboard are hosted in the [OpenADMET CYP Inhibition Blind Challenge Space](https://huggingface.co/spaces/openadmet/cyp-challenge). The two tracks are independent: a team may enter either or both, each through a separate file and leaderboard.

### Evaluation

| Track | Targets | Primary metric | Secondary metrics |
|---|---|---|---|
| Direct inhibition (regression) | `pIC50` for CYP1A2, CYP2C9, CYP2D6, and CYP3A4 | Macro-Averaged Soft-Threshold Relative Absolute Error (MA-ST-RAE) across the four isoforms | MAE, R², Spearman ρ, and Kendall's τ |
| Time-dependent inhibition (classification) | Boolean `is_TDI` for CYP2D6 and CYP3A4 | Matthews Correlation Coefficient (MCC) | Accuracy, precision, recall, and F1 |

All reported metrics use 1,000 bootstrap resamples. For direct inhibition, values below `pIC50 = 4` lie outside the assay's reliable dynamic range. ST-RAE measures error from the nearest bound of the fitted dose-response credible interval, with zero error for predictions inside that interval.

Production hyperparameter promotion, early stopping, seed confirmation, and final model selection use grouped-validation **MA-ST-RAE**, with the four CYP isoforms weighted equally. The differentiable training objective remains mean squared error because the interval-based challenge metric contains flat and non-smooth regions; RMSE remains a secondary diagnostic and no longer determines which configuration advances. Credible-interval columns are stored in the prepared graph cache and written beside every labelled prediction for auditability.

The TDI label represents an IC50 shift after preincubation. For compounds with direct-inhibition `pIC50 > 4`, a shift greater than two-fold (`log10(2) = 0.301`) is positive and a smaller shift is negative. When direct-inhibition `pIC50 < 4`, a TDI-arm `pIC50 > 4.301` is an inferred positive, while a TDI-arm `pIC50 < 4` is assigned negative. Predictions are required for every test compound, although only confidently assigned labels contribute to the classification score.

### Submission files

Each submission must be a `.parquet` file (preferred by the organisers) or `.csv` file with exactly 750 rows, one for each blinded test compound. Column names are case-sensitive. Every prediction cell must be populated; regression values must be finite floats with no `NaN`, `inf`, or `-inf`, and classification values must be Boolean (`True`/`False` or `1`/`0`). Run the official tutorial repository's validation script before uploading.

The regression file must contain exactly these six columns:

```text
SMILES
Molecule_Name
CYP1A2_pIC50_direct_inhibition
CYP2C9_pIC50_direct_inhibition
CYP2D6_pIC50_direct_inhibition
CYP3A4_pIC50_direct_inhibition
```

The classification file must contain exactly these four columns:

```text
SMILES
Molecule_Name
CYP2D6_is_TDI
CYP3A4_is_TDI
```

### Participation rules and dates

- Submit under one Hugging Face account for the collaborating team or laboratory. Cooperating members may not enter separately.
- Submissions are limited to one every 12 hours. Only the latest valid submission determines the live leaderboard standing.
- External data and pretrained models are permitted. Use of proprietary training data must be disclosed during submission.
- Public source code is encouraged and is required for consideration for the Innovation in ML Award. Select the open-source disclosure and supply a publicly accessible repository link when submitting.
- The intermediate-submission deadline is September 24, 2026 at 23:59 UTC. The intermediate leaderboard is released September 25 using the full test set, without revealing ground-truth labels.
- Final submissions close November 3, 2026 at 23:59 UTC. Final results and challenge wrap-up begin November 4.

The blinded test labels must remain excluded from training, hyperparameter optimisation, early stopping, threshold selection, and model selection. The intermediate leaderboard is for reporting progress and must not become a tuning target.

## Schematic

```text
TRAINING DATA: SMILES + CYP identity + experimental pIC50
                              │
                              ▼
Molecular graph → initial atom states X⁽⁰⁾
                              │
                              ▼
Shared local CA rule, repeated for T generations
                              │
                              ▼
Complete trajectory X⁽⁰⁾, X⁽¹⁾, …, X⁽ᵀ⁾
                              │
                              ▼
Interpretable dynamical fingerprint z
                ┌─────────────┴─────────────┐
                ▼                           ▼
Regression readout                  Binary-classification readout
→ predicted pIC50                   → predicted TDI probability
                │                           │
                ▼                           ▼
Regression loss against             Classification loss against
experimental pIC50                  experimental is_TDI label
                └─────────────┬─────────────┘
                              ▼
LEARNING 1: shared CA rule    LEARNING 2: task-specific readouts
Backpropagation through       Differentiable ridge solves pIC50
all T CA generations learns   coefficients βreg; a regularised
the shared transition         logistic readout learns TDI
parameters θ                  coefficients βTDI
                              │
                              ▼
               Validate → select TDI threshold → freeze parameters
                              │
                              ▼
             Unseen SMILES + CYP → same frozen pipeline
                ┌─────────────┴─────────────┐
                ▼                           ▼
       submitted finite pIC50        submitted Boolean is_TDI
```

The direct-inhibition model backpropagates query error through a genuine ridge solve and through repeated CA evolution to optimise the shared transition parameters (`θ`). Ridge coefficients (`βreg`) are solved with `torch.linalg.solve` from support fingerprints and labels; Adam updates only `θ`. At validation and blind-prediction boundaries, scaling and ridge state are fitted using permitted fitting observations, then frozen. The later TDI programme will use its own regularised logistic readout and training protocol. Once training and validation are complete, all parameters and transformations will be frozen before generating blind-set predictions.

## Enhanced ten-rule production framework

The production encoder supports ten differentiable local transition laws: gated residual, inertial reaction-diffusion, activator-inhibitor, coupled map, damped symplectic, FitzHugh-Nagumo, Gray-Scott, Kuramoto-Sakaguchi, conservative graph flux, and delayed memory. Every rule updates atoms synchronously, shares its parameters across atoms and generations, and communicates only along declared chemical bonds. Bond descriptors gate neighbour messages, so bond type, aromaticity, conjugation, ring membership, and stereochemistry can modulate information transfer without introducing a global molecular shortcut.

The five new candidates use the same grouped-validation production controller:

```bash
python scripts/run_production_transition_study.py --rule fitzhugh_nagumo --device auto
python scripts/run_production_transition_study.py --rule gray_scott --device auto
python scripts/run_production_transition_study.py --rule kuramoto_sakaguchi --device auto
python scripts/run_production_transition_study.py --rule conservative_graph_flux --device auto
python scripts/run_production_transition_study.py --rule delayed_memory --device auto
```

All rules share tunable update scale, initial-state scale and training noise, support/query fraction, bond-gate temperature, channel count, generation count, optimisation controls, and a cosine learning-rate schedule. Rule-specific dynamical controls retain their scientific interpretation. The molecular fingerprint concatenates final-state means and variances with temporal means, temporal variances, and accumulated motion, allowing transient and terminal CA behaviour to enter the genuine closed-form ridge readout.

Production selection uses grouped validation only. Stable finalists are repeated across three seeds and ranked by mean RMSE plus one quarter of the seed standard deviation. Dynamical measurements remain descriptive secondary outputs and never influence predictive promotion. The blinded challenge set remains outside tuning, selection, and report generation.

For the chemistry-augmented Gray-Scott and coupled-map studies, atom feature profiles are also tuned. Every profile retains the original 25 atom channels and may add periodic properties, valence descriptors, electronic descriptors, ring geometry, or controlled combinations of those groups. Run the resumable CUDA/CPU searches with `python scripts/run_gray_scott_feature_search.py --device auto` or `python scripts/run_coupled_map_feature_search.py --device auto`; each dedicated graph cache is prepared with the blinded set excluded.
