# Pharmacology

## Role in the project

The prediction target is experimental CYP inhibition expressed as `pIC50`. Pharmacology explains what that number means, how it is measured, and what limitations it carries.

## CYP enzymes

**CYP** refers to the cytochrome P450 enzyme family. These enzymes play major roles in drug metabolism. A molecule may inhibit different CYP family members by different amounts, so CYP identity is an input to the model rather than an incidental label.

## Inhibition

An inhibitor reduces an enzyme's activity. In a concentration–response experiment, several inhibitor concentrations are tested and the remaining enzyme activity is measured. The resulting curve is used to estimate the concentration producing half-maximal inhibition.

## IC50 and pIC50

`IC50` is the inhibitor concentration that produces 50% inhibition under the conditions of a particular assay. Lower `IC50` generally means that less compound is required to produce the measured effect.

`pIC50` is commonly defined as:

```text
pIC50 = −log10(IC50 expressed in molar units)
```

The logarithm reverses the direction: a larger `pIC50` corresponds to a smaller molar `IC50` and therefore stronger measured inhibition. For example, an `IC50` of `1 µM = 10⁻⁶ M` corresponds to a `pIC50` of `6`.

`IC50` is assay-dependent. It is not automatically an intrinsic binding constant, and comparisons require attention to experimental conditions and data quality.

## Mapping learned by the project

```text
molecular structure + CYP identity → predicted pIC50
```

The model must return a numerical prediction for every required molecule–target pair in the blind competition set.

## Topics to develop

- enzyme kinetics;
- concentration–response curves;
- competitive and non-competitive inhibition;
- assay variability and censoring;
- the distinction between potency, affinity, and efficacy; and
- why one ligand can inhibit different CYP enzymes differently.
