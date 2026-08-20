# Conservative Molecular Standardisation

## Learning objective

Molecular standardisation converts a supplied SMILES record into one reproducible molecular graph without silently rewriting chemically meaningful information.

The challenge data provide text representations, but our model consumes atoms, bonds, formal charges, aromaticity, stereochemistry, and other graph properties. Standardisation is therefore the boundary between a dataset record and the chemical object presented to the graph cellular automaton.

## The accepted production decision

Strange Matter Engine will use a **conservative, reproducible standardisation policy**:

1. retain the original supplied SMILES unchanged for provenance;
2. parse and chemically sanitise the representation;
3. separate disconnected fragments;
4. identify the principal organic component;
5. remove only recognised simple counterions and solvent fragments;
6. preserve formal charges rather than automatically neutralising the molecule;
7. preserve specified stereochemistry and isotopes;
8. apply one declared aromaticity model consistently;
9. generate a canonical isomeric SMILES for identity and duplicate checks;
10. record every transformation and warning; and
11. quarantine structures that cannot be resolved unambiguously.

The purpose is consistency, not cosmetic simplification.

## 1. A SMILES string and a molecule

A SMILES string is a sequence of characters describing a molecular graph. More than one valid string can describe the same graph.

For example, ethanol can be written with different traversal orders:

```text
CCO
OCC
```

The strings differ, but both describe two carbon atoms and one oxygen atom connected in the same order. Our scientific unit is the molecular graph, not the spelling of the SMILES.

We therefore distinguish:

- **original SMILES:** the exact dataset text;
- **parsed molecular graph:** the atoms, bonds, charges, and annotations interpreted from that text; and
- **canonical isomeric SMILES:** a deterministic serialization generated from the standardised graph.

## 2. Parsing

Parsing translates SMILES grammar into a graph

```math
G=(V,E),
```

where $V$ is the atom set and $E$ is the bond set.

Parsing identifies features such as:

- elements;
- explicit and implicit hydrogens;
- bond types;
- branches;
- ring closures;
- formal charges;
- aromatic atom notation;
- isotopes; and
- tetrahedral or double-bond stereochemistry.

A string can fail to parse because of invalid syntax, unsupported elements, unmatched ring markers, impossible annotations, or truncation. A failed record must receive an explicit status; it must not disappear silently.

## 3. Chemical sanitisation

Sanitisation checks whether the parsed graph is chemically and internally coherent under the declared cheminformatics model. Typical checks include:

- allowed valence;
- consistency of implicit hydrogen counts;
- ring perception;
- aromaticity assignment;
- conjugation;
- hybridisation; and
- consistency of stereochemical annotations.

Sanitisation is not an experimental proof that a molecule exists or has the supplied biological activity. It establishes that the representation can be interpreted consistently enough to generate graph features.

A sanitisation failure can indicate a malformed record, unusual chemistry outside the chosen model, or an ambiguity requiring review.

## 4. Disconnected fragments

In SMILES, a full stop separates disconnected components. A salt may be represented as

```text
[Na+].[O-]C(=O)c1ccccc1
```

This record contains a sodium cation and an organic carboxylate anion. If every disconnected component became one graph, the sodium would appear as an isolated node with no covalent path to the ligand.

Let the parsed record contain connected components

```math
G=G_1\cup G_2\cup\cdots\cup G_r.
```

The standardisation procedure must decide which component represents the modelled compound and what role the others play.

## 5. Principal organic component

The principal component is normally the covalently connected organic fragment representing the compound of interest. A transparent selection rule can consider:

- presence of carbon;
- number of heavy atoms;
- molecular mass;
- recognised counterion patterns; and
- whether one fragment clearly dominates the record.

For component $G_k$, let

```math
n_{\rm heavy}(G_k)
=
\text{number of non-hydrogen atoms in }G_k.
```

Heavy-atom count is useful, but it is not sufficient on its own. A record containing two similarly large organic fragments may represent a co-crystal, mixture, or multi-component compound. Such a case is ambiguous and should be quarantined for review.

The chosen fragment, discarded fragments, selection rule, and reason must all be recorded.

## 6. Counterions and solvents

Simple counterions balance charge without usually representing the pharmacologically modelled covalent structure. Examples include sodium, chloride, bromide, and simple protonated or deprotonated salt partners.

Solvent fragments may also appear in structure records, such as water or small crystallisation solvents.

Removal is conservative:

- a fragment is removed only when it matches a declared counterion or solvent rule;
- the removed fragment remains recorded in provenance;
- the charge of the retained component is not artificially changed to compensate; and
- unfamiliar multi-component records are reviewed rather than guessed.

Removing chloride from a protonated amine salt leaves the amine positively charged. Fragment removal and charge neutralisation are different operations.

## 7. Formal charge and protonation

Formal charge is part of the supplied molecular representation. It affects valence, hydrogen-bonding behaviour, electrostatic interactions, and our fixed atom channels.

For a molecular component, total formal charge is

```math
Q=\sum_{i=1}^{n}q_i,
```

where $q_i$ is the formal charge assigned to atom $i$.

Our initial policy preserves $q_i$ and therefore $Q$. We will not automatically add or remove protons merely to make the molecule neutral.

For example, a protonated amine and its neutral free base differ in:

- formal charge;
- hydrogen count;
- hydrogen-bond donor and acceptor status;
- valence representation; and
- likely interaction with the CYP environment.

Automatic neutralisation could erase this information.

### Protonation is condition-dependent

The predominant protonation state of a molecule depends on pH, microscopic pKa values, solvent, and environment. A SMILES record provides one represented state; it does not prove that this is the only state present in an assay.

For the first prototype, we preserve the supplied state. A future protonation-state ensemble would be a separate, explicitly validated model extension.

## 8. Tautomers

Tautomers have the same overall molecular formula but differ in proton position and associated bond placement. Keto–enol and some heteroaromatic systems are common examples.

Tautomer choice can alter:

- hydrogen-bond donors and acceptors;
- formal bond orders;
- aromaticity assignments;
- local atom environments; and
- the resulting graph channels.

Aggressive tautomer canonicalisation can merge representations that may behave differently in a biological environment. The initial conservative policy therefore does not silently transform every molecule into a preferred tautomer.

We will canonicalise the represented graph, flag suspected tautomeric duplicates when useful, and test any future tautomer-normalisation policy as a separate design decision.

## 9. Aromaticity

Aromaticity is not stored as an experimentally measured binary label. It is perceived from bonding and valence using a defined aromaticity model.

The same declared model must be used for:

- sanitising training molecules;
- constructing their atom and bond channels;
- standardising validation folds;
- processing blinded test molecules; and
- reproducing predictions later.

Changing aromaticity perception can change atom hybridisation, bond encoding, ring features, and canonical SMILES. The software version and aromaticity model are therefore part of the scientific provenance.

## 10. Stereochemistry

Stereoisomers can share atom connectivity while differing in spatial arrangement. A chiral CYP active site can distinguish them.

We preserve stereochemistry that is explicitly specified in the supplied SMILES, including:

- tetrahedral chirality;
- double-bond geometry; and
- other supported stereochemical annotations.

We do not invent unspecified stereochemistry. A structure with an undefined stereocentre remains marked as undefined rather than being assigned an arbitrary configuration.

A canonical **isomeric** SMILES is required because a non-isomeric canonical representation may collapse stereoisomers into the same text identity.

## 11. Isotopes

An isotope annotation changes atomic mass and may identify a labelled experimental compound. If an isotope is explicitly specified, it is retained.

For an atom with mass number $A$, the isotope label is part of its identity:

```math
{}^{A}\!X.
```

We will not replace an explicit isotope with the element's default natural-abundance mass during standardisation. Whether isotope information becomes a model channel is a later feature decision.

## 12. Canonical isomeric SMILES

Canonicalisation chooses a reproducible atom ordering and SMILES traversal for a given standardised graph. Conceptually,

```math
S_{\rm canonical}=f_{\rm canon}(G_{\rm standardised}).
```

Canonicalisation helps:

- identify exact duplicate graphs;
- create stable molecule identifiers;
- compare processing runs;
- group all CYP observations for the same molecule; and
- prevent exact-molecule leakage across folds.

Canonical does not mean experimentally correct, neutral, dominant at assay pH, or equivalent across different standardisation toolkits. It means deterministic under a declared method and software version.

## 13. Duplicate identity

Two source records may become identical after counterion removal and canonicalisation. Define a standardised identity key

```math
I(m)=
\left(
S_{\rm canonical},
\text{isotope state},
\text{stereochemical state}
\right).
```

If

```math
I(m_i)=I(m_j),
```

the records belong to the same standardised molecular identity for splitting. All associated CYP measurements must remain in the same cross-validation group.

Duplicate biological measurements are not automatically averaged. Replicates, conflicting labels, confidence intervals, and assay provenance must be examined before defining an aggregation rule.

## 14. Quarantine rather than silent repair

A record enters quarantine when automated processing cannot make a unique, defensible choice. Examples include:

- parsing or sanitisation failure;
- multiple similarly large organic components;
- unsupported organometallic bonding;
- impossible valence under the declared model;
- contradictory stereochemical annotations;
- no identifiable principal organic component; or
- a transformation that changes more chemistry than the accepted policy permits.

Quarantine means:

1. retain the original record;
2. record the failure stage and diagnostic;
3. exclude it from automatic model fitting temporarily;
4. review it scientifically; and
5. resolve it only through a documented rule.

This converts hidden data loss into an auditable scientific decision.

## 15. Provenance record

For every input molecule, we will retain fields equivalent to:

- source dataset and row identifier;
- original SMILES;
- parsing status;
- sanitisation status;
- number and identities of fragments;
- selected principal component;
- removed counterions or solvents;
- total formal charge before and after fragment selection;
- stereochemistry and isotope warnings;
- canonical isomeric SMILES;
- standardisation status;
- transformation log;
- standardisation-rule version; and
- software and chemistry-toolkit version.

The model-ready graph can therefore be traced back to the supplied challenge record.

## 16. Standardisation and cross-validation

Most chemical standardisation rules are fixed before model development and applied identically to all records. Any rule learned from dataset statistics belongs inside the cross-validation boundary.

For example:

- a fixed counterion dictionary is a declared preprocessing rule;
- feature means and standard deviations are fitted inside each training fold;
- data-driven feature selection occurs inside each training fold; and
- decisions made after studying outer-fold errors must not be retroactively presented as untouched validation.

All records mapping to the same standardised identity or scaffold group remain together.

## 17. Initial production workflow

```math
\begin{aligned}
\text{original SMILES}
&\longrightarrow \text{parse}\\
&\longrightarrow \text{sanitise}\\
&\longrightarrow \text{separate fragments}\\
&\longrightarrow \text{select principal organic component}\\
&\longrightarrow \text{remove declared counterions or solvents}\\
&\longrightarrow \text{preserve charge, stereochemistry, isotopes}\\
&\longrightarrow \text{canonical isomeric SMILES}\\
&\longrightarrow \text{identity and scaffold groups}\\
&\longrightarrow \text{fixed 2D molecular graph}.
\end{aligned}
```

Every arrow produces a recorded status and transformation entry.

## 18. Scientific validation checks

Before model training, we will report:

- total input records;
- successful and failed parses;
- sanitisation failures;
- fragment-count distribution;
- removed-fragment frequencies;
- formal-charge distribution;
- stereochemistry and isotope counts;
- exact duplicates before and after standardisation;
- quarantined structures and reasons;
- scaffold-group sizes; and
- examples of every transformation class.

We will also visually inspect representative molecules before and after standardisation. A numerically successful pipeline is acceptable only if the resulting chemistry remains defensible.

## Connection to the course

- [Chemistry](Chemistry.md) defines the molecular properties being preserved.
- [Graph Theory](Graph_Theory.md) explains the graph produced from the standardised structure.
- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) explains how graph chemistry becomes fixed channels.
- [Grouped Nested Cross-Validation](Grouped_Nested_Cross_Validation.md) uses canonical identities and scaffolds to prevent leakage.
- [Validation and Statistics](Validation_and_Statistics.md) explains why preprocessing provenance matters.

