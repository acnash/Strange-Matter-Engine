# Validation and Statistics

## Purpose

Validation asks whether the model has learned a relationship that generalises to unseen molecules rather than memorising the training data or exploiting leakage.

## Data partitions

Training data are used to fit parameters. Validation data are used to compare design choices and hyperparameters. Blind test data are used only to generate the final competition predictions; their unknown outcomes must not influence training choices.

## Molecular splitting

A random split is useful as an early diagnostic, but structurally related molecules can appear on both sides and produce an optimistic estimate. A scaffold-aware split groups compounds by core chemical structure and better tests generalisation to novel chemistry.

Performance will be examined both overall and separately for each CYP target. The official competition rules, released dataset structure, and scoring metric will govern the final evaluation protocol.

## Baselines

Conventional baselines are comparison instruments:

- simple molecular descriptors with linear or ridge regression;
- Morgan/ECFP fingerprints with a standard regressor; and
- a small conventional message-passing graph neural network if resources permit.

They show whether the graph CA and its trajectory add predictive information.

## Sanity checks

The first learned model should deliberately overfit a very small training subset. Failure to do so often reveals an error in the data flow, loss, gradients, or implementation before full training begins.

## Topics to develop

- bias and variance;
- training, validation, and test sets;
- data leakage;
- scaffold splitting;
- cross-validation;
- error metrics and uncertainty;
- ablation studies; and
- reproducibility, checkpoints, and random seeds.
