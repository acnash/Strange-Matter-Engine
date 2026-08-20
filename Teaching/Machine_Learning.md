# Machine Learning

## The two learned components

Strange Matter Engine currently proposes two conceptually distinct learned systems:

1. the shared graph-CA transition parameters `θ`; and
2. the readout coefficients `β` that map a dynamical fingerprint to predicted `pIC50`.

The precise joint, alternating, or staged training schedule remains an experimental question.

The full mathematical chapters are [Backpropagation](Backpropagation.md) and [Ridge Regression](Ridge_Regression.md).

## Prediction error and loss

For labelled training molecules, the predicted `pIC50` is compared with the experimental value. A provisional loss is mean squared error: the average squared prediction error. Squaring prevents positive and negative errors from cancelling and penalises larger errors more strongly.

The competition's actual metric must be verified before the final optimisation strategy is chosen.

## Backpropagation through the automaton

The same local transition rule is applied repeatedly for `T` generations. **Backpropagation** uses the chain rule to calculate how the final prediction error depends on each trainable parameter through every intermediate CA state. An optimiser then changes `θ` in a direction intended to reduce future error.

This is backpropagation through time: the computational graph is formed by unrolling the repeated CA update across generations. Gradients, the chain rule, learning rates, and parameter updates will be derived from first principles before implementation.

## Ridge regression

Ridge regression predicts `pIC50` as a weighted sum of dynamical fingerprint components plus an intercept. It fits the coefficients while adding a penalty proportional to the sum of their squared magnitudes.

The penalty discourages excessively large coefficients, helps when fingerprint components are correlated, and trades a small amount of bias for potentially improved stability and generalisation. The penalty strength is a hyperparameter selected using training data only.

## Why preserve the trajectory?

The machine-learning representation will include features derived from the entire trajectory, not just the last generation. Convergence rate, oscillation, transients, and sensitivity may contain predictive information that a final-state-only representation would discard.

## Evolutionary learning

An optional later experiment may evolve populations of CA rules through selection, mutation, and recombination. This is conceptually well matched to cellular automata but may be computationally expensive. It is a secondary experiment rather than the initial training method.

## Topics to develop

- supervised learning and regression;
- parameters, hyperparameters, and features;
- loss functions and gradient descent;
- derivatives, partial derivatives, and the chain rule;
- computational graphs and backpropagation through time;
- linear regression and ridge regularisation;
- overfitting and generalisation; and
- optimisation stability in recurrent systems.
