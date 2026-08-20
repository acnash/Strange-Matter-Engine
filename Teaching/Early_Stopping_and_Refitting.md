# Early Stopping and Leakage-Free Refitting

## Learning objective

Training for too few epochs can leave a model underfit. Training for too many can make it increasingly specialised to the training molecules. **Early stopping** uses validation performance to choose when optimisation should stop.

The validation data used for stopping become part of model selection. They cannot also serve as an untouched test of generalisation.

## The accepted production decision

For each inner-fold training run:

- maximum budget: 200 epochs;
- validation measurement: RMSE after every epoch;
- minimum meaningful improvement: 0.005 pIC50 RMSE;
- patience: 20 consecutive epochs without that improvement;
- retained model: checkpoint with the best inner-validation RMSE.

For outer-fold refitting, the outer-test fold is never used for stopping. We will train on the complete outer-training set for the median best-epoch count observed in the selected configuration's inner runs.

The final all-data model will use an epoch count derived from the completed nested-validation results before blinded-test prediction.

## 1. An epoch

An epoch is one pass through every molecule in the current training fold. If there are $N_{\rm mol}$ training molecules and 16 molecules per batch, the approximate update count per epoch is

```math
U_{\rm epoch}
=
\left\lceil
\frac{N_{\rm mol}}{16}
\right\rceil.
```

Epoch number is therefore a measure of how many times the optimiser has encountered the training set, not a universal measure of computational effort across datasets.

## 2. Training and validation curves

At epoch $e$, let

```math
R_{\rm train}(e)
```

be training RMSE and

```math
R_{\rm val}(e)
```

be inner-validation RMSE.

Training error often decreases as optimisation continues. Validation error may initially decrease and later flatten or increase:

```math
\text{continued training}
\quad\Longrightarrow\quad
\begin{cases}
\text{better fit to training molecules},\\
\text{possibly worse transfer to held-out scaffolds}.
\end{cases}
```

Early stopping retains the epoch with the strongest validation evidence rather than automatically retaining the last epoch.

## 3. Best checkpoint

Let the best validation RMSE observed by epoch $e$ be

```math
R_{\rm best}(e)
=
\min_{1\le j\le e}R_{\rm val}(j).
```

When a new best value is accepted, the checkpoint must retain:

- graph-CA parameters $\theta$;
- readout parameters $\beta$;
- Adam first and second moments if training may resume;
- epoch and update counts;
- preprocessing and fingerprint scaling;
- random states; and
- complete configuration metadata.

For final prediction from that run, the model weights from the best epoch are restored.

## 4. Minimum meaningful improvement

Small validation fluctuations should not continually reset patience. We define

```math
\delta_{\rm min}=0.005
```

pIC50 RMSE.

An epoch counts as a meaningful improvement only if

```math
R_{\rm val}(e)
<
R_{\rm best}-\delta_{\rm min}.
```

For example, if the best RMSE is 0.720:

- 0.718 improves by 0.002 and does not reset patience;
- 0.714 improves by 0.006 and does reset patience.

The threshold reduces sensitivity to numerical jitter. It does not claim that 0.005 is biologically important.

## 5. Patience

Patience $P=20$ means training stops after 20 consecutive validation checks without a meaningful improvement.

If the last accepted improvement occurred at epoch $e^*$, stopping occurs when

```math
e-e^*\ge20,
```

unless the 200-epoch maximum is reached first.

Patience allows the optimiser to move through short plateaus without running indefinitely.

## 6. Maximum budget

The hard limit is

```math
E_{\rm max}=200.
```

Every configuration receives the same maximum opportunity. A run can finish because:

- patience was exhausted;
- epoch 200 was completed;
- numerical failure occurred; or
- another prespecified safety rule was triggered.

The stopping reason is recorded.

## 7. A worked stopping example

Suppose the accepted improvements occur at epochs

```math
1,\ 4,\ 9,\ 17,\ 31.
```

If no later epoch improves RMSE by at least 0.005, patience expires at

```math
31+20=51.
```

Training stops at epoch 51, but the restored predictive model is the checkpoint from epoch 31.

The 20 patience epochs are evidence that the model did not produce a meaningfully better validation score; they are not the model selected for inference.

## 8. Why the outer test cannot stop training

The outer fold estimates generalisation of the entire design procedure. If we inspect its RMSE every epoch and stop at its minimum, the outer fold becomes training guidance.

This would select

```math
e_{\rm outer}^*
=
\underset{e}{\arg\min}\,
R_{\rm outer}(e),
```

making the resulting outer score optimistically biased.

The outer test fold is predicted once after all hyperparameters and the refit epoch count are fixed.

## 9. Deriving a refit duration

For selected configuration $h^*$, inner folds and confirmation seeds produce best epochs

```math
e_1^*,e_2^*,\ldots,e_M^*.
```

The outer-refit duration is

```math
E_{\rm refit}
=
\mathrm{median}
\left(
e_1^*,e_2^*,\ldots,e_M^*
\right),
```

rounded to an integer according to a declared rule.

The median is robust to an occasional run that stops exceptionally early or trains to the maximum.

## 10. Refitting the outer model

After inner selection:

1. combine all outer-training groups;
2. fit preprocessing on that complete outer-training set;
3. initialise the selected architecture with the declared refit seed;
4. train for exactly $E_{\rm refit}$ epochs;
5. freeze the model; and
6. predict the untouched outer-test fold once.

There is no outer-test monitoring during the refit.

## 11. Final all-data training

After nested validation is complete, all released labelled molecules can be used for the production fit.

The final epoch rule is derived from the nested experiment, for example the median of appropriate selected inner best epochs across outer folds. It is fixed before producing predictions for the 750 blinded challenge molecules.

The blinded data provide molecular inputs only. They do not select epoch count, checkpoint, hyperparameters, or preprocessing.

## 12. Interaction with random seeds

Best epoch can vary across seeds. We therefore record its distribution during three-seed confirmation:

```math
\{e_{h,\ell,r}^*\},
```

where $h$ is configuration, $\ell$ is inner fold, and $r$ is seed.

Large variability can indicate unstable optimisation even when mean RMSE is acceptable.

## 13. Monitoring beyond RMSE

At every epoch we also monitor:

- training MSE;
- per-CYP RMSE and MAE;
- regularisation losses;
- parameter and gradient norms;
- state ranges and tanh saturation;
- gate distributions;
- fingerprint variance; and
- numerical failures.

Best validation RMSE selects the checkpoint, while these diagnostics determine whether the run is scientifically and numerically credible.

## 14. Early stopping is regularisation

Stopping limits how far optimisation can adapt to the training data. It therefore acts as an additional form of regularisation.

Its effect interacts with:

- learning rate;
- L2 penalties;
- model width;
- generation count;
- batch ordering; and
- label noise.

The stopping policy must remain identical when comparing hyperparameter configurations.

## 15. Prototype specification

The accepted policy is:

- 200-epoch maximum;
- validation RMSE measured every epoch;
- 0.005 minimum improvement;
- patience of 20;
- best checkpoint restored;
- stopping epoch and reason recorded;
- median inner best epoch used for outer refitting;
- no outer-test early stopping; and
- final all-data duration derived before blinded inference.

## Connection to the course

- [Hyperparameter Search](Hyperparameter_Search.md) places stopping inside model selection.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) defines the protected outer fold.
- [Optimisation, Adam, and Learning Rates](Optimisation_and_Learning_Rates.md) explains epoch-by-epoch parameter updates.
- [Mini-Batching Molecular Graphs](Mini_Batching_Molecular_Graphs.md) defines an epoch.

