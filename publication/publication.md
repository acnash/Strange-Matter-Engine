# Molecule Space-Time: Predicting Cytochrome P450 Inhibition with a Nonlinear Graph Cellular Automaton

**Anthony Nash**

## Abstract

Cytochrome P450 (CYP) inhibition is a major consideration in drug discovery because it can alter drug metabolism and contribute to clinically significant drug–drug interactions. The OpenADMET CYP Inhibition Challenge provides a blinded setting in which to evaluate computational prediction of direct-inhibition pIC50 across four CYP isoforms. Here, we introduce a molecular graph cellular automaton that represents atoms as cells, chemical bonds as local neighbourhoods, and molecular computation as the repeated evolution of a shared, learned transition rule. Unlike molecular models that reduce a structure directly to a fixed representation, our approach retains the complete sequence of atom states and uses its transient and terminal properties to predict CYP inhibition. We call this evolving representation **molecule space-time**: the joint description of molecular structure and its learned progression through computational time. Predictive performance across the CYP targets was [PREDICTION RESULTS]. As a secondary objective, we investigated the nonlinear dynamics contained within molecule space-time by extending selected trajectories over thousands of generations and examining convergence, recurrence, periodicity, perturbation sensitivity, strange-attractor candidates, and possible chaotic behaviour. This analysis identified [CAPTURED DYNAMICS]. The framework therefore treats prediction and dynamical exploration as complementary views of the same learned molecular process, offering a route toward CYP inhibition models whose internal evolution can be measured, visualised, and studied as a nonlinear system.

## Introduction

## Materials and Methods

### Dataset and Prediction Task

### Molecular Graph Representation

### Nonlinear Graph Cellular Automaton

### Machine-Learning Training and Validation

### Long-Horizon Dynamical Analysis

## Results and Discussion

### CYP pIC50 Predictions

### Nonlinear Dynamics in Molecule Space-Time

## Conclusions
