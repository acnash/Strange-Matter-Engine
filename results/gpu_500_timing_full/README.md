# RTX 5070 Ti 500-generation timing test

This directory contains a timing benchmark rather than a production model-selection study.

## Configuration

- Transition rule: inertial reaction-diffusion graph CA
- Generations: 500
- Dynamical channels: 8
- Device: NVIDIA GeForce RTX 5070 Ti
- PyTorch: 2.11.0+cu128
- Fit observations: 5,216
- Grouped-validation observations: 1,309
- Training epochs: 10
- CA learning rate: 0.002
- Readout learning rate: 0.003
- Readout L2 penalty labelled `ridge`: 0.001
- Gradient-clipping norm: 1.0

The four optimization hyperparameters were selected using a six-candidate, three-epoch screen on a fixed training-only subset of 600 fitting molecules and 200 grouped-validation molecules. The screen took 622.6 seconds. The promoted full-data training stage took 1,982.4 seconds. Their combined measured time was 2,605.1 seconds, or 43 minutes 25 seconds, excluding environment installation and graph-cache preparation.

## Timing result

- Full training stage: 33 minutes 2 seconds
- Moderate tuning screen: 10 minutes 23 seconds
- Combined GPU study: 43 minutes 25 seconds
- Peak PyTorch GPU allocation: 177.1 MiB

The low peak allocation shows that this is a conservative batching benchmark. Larger graph batches and prepacked tensors may improve production throughput further.

## Predictive checkpoint

- Restored fit RMSE: 0.879 pIC50
- Restored grouped-validation RMSE: 0.937 pIC50

These scores are engineering checks from one grouped split and one seed. They are not nested-cross-validation estimates and must not be presented as production performance.

## Blind-set isolation

The graph cache used for this test contained 4,905 labelled training molecules and zero blinded molecules. No blinded predictions were generated.

## Readout qualification

This historical benchmark used an Adam-optimized linear readout with an L2 coefficient penalty. It measured the former prototype and must not be compared directly with the current differentiable-ridge implementation. The production code now solves standardized ridge coefficients with `torch.linalg.solve`, differentiates query loss through that solve, and uses Adam only for graph-CA parameters.
