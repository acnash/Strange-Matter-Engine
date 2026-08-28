# Molecule Space-Time: Predicting Cytochrome P450 Inhibition with a Nonlinear Graph Cellular Automaton

**Anthony Nash**

## Abstract

Cytochrome P450 (CYP) inhibition is a major consideration in drug discovery because it can alter drug metabolism and contribute to clinically significant drug–drug interactions. The OpenADMET CYP Inhibition Challenge provides a blinded setting in which to evaluate computational prediction of direct-inhibition pIC50 across four CYP isoforms. Here, we introduce a molecular graph cellular automaton that represents atoms as cells, chemical bonds as local neighbourhoods, and molecular computation as the repeated evolution of a shared, learned transition rule. Unlike molecular models that reduce a structure directly to a fixed representation, our approach retains the complete sequence of atom states and uses its transient and terminal properties to predict CYP inhibition. We call this evolving representation **molecule space-time**: the joint description of molecular structure and its learned progression through computational time. Predictive performance across the CYP targets was [PREDICTION RESULTS]. As a secondary objective, we investigated the nonlinear dynamics contained within molecule space-time by extending selected trajectories over thousands of generations and examining convergence, recurrence, periodicity, perturbation sensitivity, strange-attractor candidates, and possible chaotic behaviour. This analysis identified [CAPTURED DYNAMICS]. The framework therefore treats prediction and dynamical exploration as complementary views of the same learned molecular process, offering a route toward CYP inhibition models whose internal evolution can be measured, visualised, and studied as a nonlinear system.

## Introduction

## Materials and Methods

### Dataset and Prediction Task

The study used the primary direct-inhibition dataset released for the 2026 OpenADMET CYP Inhibition Blind Challenge. It comprised 4,905 unique compounds represented by a molecule identifier and a SMILES string. Experimental direct-inhibition pIC50 values were provided for four major drug-metabolising cytochrome P450 isoforms: CYP1A2, CYP2C9, CYP2D6, and CYP3A4. Here, pIC50 denotes the negative base-10 logarithm of the half-maximal inhibitory concentration expressed in molar units. Each reported measurement was accompanied by lower and upper uncertainty bounds and an estimated standard deviation from the fitted concentration–response experiment.

The response matrix was incomplete because compounds were not necessarily measured against every CYP isoform. The dataset contained 6,525 observed compound–CYP pairs, distributed as follows:

| CYP isoform | Compounds with an observed direct-inhibition pIC50 |
|---|---:|
| CYP1A2 | 1,412 |
| CYP2C9 | 1,285 |
| CYP2D6 | 1,493 |
| CYP3A4 | 2,335 |
| **Total** | **6,525** |

Each observed compound–CYP pair constituted one supervised regression example. The prediction task was to learn a single CYP-conditioned mapping from molecular structure to direct-inhibition pIC50, rather than fitting an independent model for each isoform. Missing endpoint values were retained as missing and contributed neither targets nor loss terms. The single-concentration, time-dependent-inhibition, and Emax datasets distributed with the challenge were outside the scope of this direct-inhibition study.

To assess generalisation beyond closely related chemistry, compounds were grouped by their standardised Bemis–Murcko scaffold before data partitioning. A fixed 20% subset of scaffold groups was reserved as a sealed holdout, leaving 5,216 observed compound–CYP pairs for model fitting and internal selection and 1,309 pairs for final validation. No scaffold group occurred in both partitions. Hyperparameter selection was conducted within the fitting pool, while the reserved scaffold holdout remained unused until final evaluation.

The challenge test set contained 750 additional compounds for which all direct-inhibition labels were withheld. The trained system was required to generate four finite pIC50 predictions for each compound, giving 3,000 blinded compound–CYP predictions in total. Blinded compounds and their unreleased outcomes were excluded from parameter estimation, hyperparameter selection, early stopping, and dynamical candidate selection. Predictive evaluation followed the challenge formulation: the primary measure was the macro-averaged soft-threshold relative absolute error across the four CYP isoforms, with each isoform weighted equally and predictions falling within the reported experimental uncertainty interval assigned zero error. Conventional regression statistics were retained as complementary measures of predictive agreement.

### Molecular Graph Representation

### Nonlinear Graph Cellular Automaton

### Machine-Learning Training and Validation

### Long-Horizon Dynamical Analysis

## Results and Discussion

### CYP pIC50 Predictions

### Nonlinear Dynamics in Molecule Space-Time

## Conclusions
