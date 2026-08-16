# Chemistry

## Role in the project

Chemistry determines what information is present at generation zero. A ligand supplied as a SMILES string will be parsed into atoms and bonds, then converted into a chemically meaningful initial state for the graph cellular automaton.

## Ligands

A **ligand** is a molecule that can bind to a biological target. In this challenge, the molecules are tested for their ability to inhibit CYP enzymes. Molecular structure influences binding, but the supplied activity value is an experimental measurement rather than something that can be read directly from the molecular formula.

## SMILES

**SMILES** (Simplified Molecular Input Line Entry System) is a text notation for molecular structure. Symbols describe atoms, bonds, branches, rings, aromatic atoms, charge, and stereochemistry. The same molecular graph can sometimes be written using more than one valid SMILES string, so a SMILES string is a serialization of a structure, not the structure itself.

The first practical chemistry lesson will parse small examples by hand and identify their atoms, bonds, branches, and rings before software performs the conversion.

## Candidate atom properties

Properties raised so far include:

- element and atomic number;
- formal charge;
- aromaticity;
- hybridisation;
- number of bonded heavy atoms and attached hydrogens;
- hydrogen-bond donor and acceptor status;
- ring membership;
- atomic mass;
- electronegativity;
- van der Waals radius; and
- polarizability.

**Molecular weight** is a whole-molecule property obtained from its atoms. **Polarizability** describes how readily an atom's or molecule's electron distribution is distorted by an electric field. Each property must have a chemical justification and appropriate scale before it is used numerically.

## Candidate bond properties

The molecular graph may distinguish single, double, triple, and aromatic bonds, together with conjugation, ring membership, and relevant stereochemistry. This allows a local rule to respond differently to chemically different neighbourhoods.

## Design principle

The initial feature set will remain small and defensible. The purpose is to give the automaton useful chemical starting conditions while leaving it responsible for learning dynamics, rather than overwhelming it with a large conventional descriptor library.

## Topics to develop

- valence, bonding, and resonance;
- aromaticity and conjugation;
- functional groups;
- ionisation and formal charge;
- stereochemistry;
- intermolecular forces; and
- which atomic properties are measured, calculated, categorical, or context-dependent.
