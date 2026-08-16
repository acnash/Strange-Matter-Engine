# Strange Matter Engine

## Introduction

Strange Matter Engine is an experimental molecular machine-learning project for the OpenADMET CYP inhibition challenge. It will explore whether a compact, learned **graph cellular automaton (GCA)** can transform molecular structure into interpretable emergent dynamics that predict experimental CYP inhibition (`pIC50`).

Each molecule will be represented as a graph: atoms are cells, chemical bonds define their neighbourhoods, and a small shared local rule evolves every atom's state through time. The model will preserve and analyse the complete trajectory of this evolution—from the chemically initialised state through every subsequent generation—rather than reducing the molecule to its final state.

The project is both a predictive experiment and a study of molecular dynamical behaviour. It will investigate convergence, oscillations, attractor families, perturbation sensitivity, complex transients, and possible relationships between dynamical regimes and activity cliffs. Any claim of chaotic behaviour will require mathematical evidence.

The work is designed for a personal workstation, with an emphasis on compact models, scientific interpretability, honest benchmarking, and visualisations grounded in quantities genuinely produced or analysed by the model.

## Goal

The primary goal is to build, understand, validate, and submit a complete end-to-end model that maps:

> **SMILES + CYP identity → predicted pIC50**

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
                              │
                              ▼
Ridge-regression readout → predicted pIC50
                              │
                              ▼
Prediction error against experimental pIC50
                ┌─────────────┴─────────────┐
                ▼                           ▼
LEARNING 1: CA rule                 LEARNING 2: readout
Backpropagation through             Ridge regression learns
all T CA generations learns         coefficients linking each
the shared transition              dynamical feature to pIC50
parameters θ                        (coefficients β)
                │                           │
                └─────────────┬─────────────┘
                              ▼
               Validate → freeze parameters
                              │
                              ▼
Unseen SMILES + CYP → same frozen pipeline → submitted pIC50
```

The model therefore contains two learning processes. First, prediction error is backpropagated through the repeated CA evolution to optimise the shared transition parameters (`θ`). Second, ridge regression learns the regularised readout coefficients (`β`) that map the trajectory-derived fingerprint to `pIC50`. The precise training schedule—joint, alternating, or staged—will be treated as an experimental design choice and taught before implementation. Once training and validation are complete, all learned parameters will be frozen and the identical pipeline will generate blind-set predictions.
