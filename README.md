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
Backpropagation through       Ridge regression learns pIC50
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

The model therefore contains a shared learning process and two task-specific readouts. Direct-inhibition error and TDI-classification error can be backpropagated through the repeated CA evolution to optimise the shared transition parameters (`θ`). Ridge regression learns regularised coefficients (`βreg`) mapping the trajectory-derived fingerprint to `pIC50`; a regularised logistic readout learns coefficients (`βTDI`) mapping the same fingerprint to a TDI probability. The classification threshold will be selected without using blinded test outcomes, with Matthews correlation coefficient as the primary validation objective to match the challenge metric. The precise training schedule—joint, alternating, or staged—will be treated as an experimental design choice and taught before implementation. Once training, validation, and threshold selection are complete, all parameters will be frozen and the identical pipeline will generate the two independent blind-set submission files.
