# Chemistry

## Role in the project

Chemistry determines what information is present at generation zero. A ligand supplied as a SMILES string will be parsed into atoms and bonds, then converted into a chemically meaningful initial state for the graph cellular automaton.

## Ligands

A **ligand** is a molecule that can bind to a biological target. In this challenge, the molecules are tested for their ability to inhibit CYP enzymes. Molecular structure influences binding, but the supplied activity value is an experimental measurement rather than something that can be read directly from the molecular formula.

## SMILES

**SMILES** (Simplified Molecular Input Line Entry System) is a text notation for molecular structure. Symbols describe atoms, bonds, branches, rings, aromatic atoms, charge, and stereochemistry. The same molecular graph can sometimes be written using more than one valid SMILES string, so a SMILES string is a serialization of a structure, not the structure itself.

The first practical chemistry lesson will parse small examples by hand and identify their atoms, bonds, branches, and rings before software performs the conversion.

## Candidate atom properties

Each atom can be described by a small set of chemically meaningful properties:

- **Element:** the chemical identity of the atom, such as carbon, nitrogen, oxygen, or sulfur.
- **Atomic number:** the number of protons in the nucleus; it uniquely identifies the element.
- **Formal charge:** the integer charge assigned by electron-bookkeeping rules, such as `+1`, `0`, or `−1`. It is a representation of charge localisation, not a measured partial charge.
- **Aromaticity:** whether the atom participates in an aromatic electron system—a cyclic, approximately planar, conjugated system with exceptional electronic stabilisation.
- **Hybridisation:** a local bonding description such as `sp`, `sp²`, or `sp³`, related to orbital arrangement and molecular geometry.
- **Atom degree:** the number of atoms directly bonded to the atom. Heavy-atom degree excludes bonded hydrogen atoms.
- **Attached hydrogen count:** the number of hydrogen atoms bonded to the atom, whether written explicitly or implied by the molecular representation.
- **Hydrogen-bond donor status:** whether the atom can donate a hydrogen in a hydrogen bond, commonly an `O–H` or `N–H` group.
- **Hydrogen-bond acceptor status:** whether the atom has an available electron pair capable of accepting a hydrogen bond. Formal charge, resonance, and chemical environment affect this property.
- **Ring membership:** whether the atom belongs to at least one closed path of bonded atoms.
- **Aromatic-ring membership:** whether the atom belongs specifically to a ring classified as aromatic.
- **Atomic mass:** the mass associated with the element or isotope, usually expressed in unified atomic mass units.
- **Electronegativity:** an element's tendency to attract shared bonding electrons towards itself.
- **Van der Waals radius:** an approximate measure of the atom's non-bonded spatial extent.
- **Polarizability:** how readily the atom's electron cloud is distorted by a nearby charge or electric field.

**Molecular weight** is a whole-molecule property obtained by summing the atomic masses in its molecular formula. It is distinct from atomic mass, although it is derived from it.

Some properties are categorical, some are numerical, and some depend on chemical context. Each must have a chemical justification and an appropriate encoding or scale before it is supplied to the model.

## Aromaticity and conjugation

**Conjugation** occurs when a continuous sequence of neighbouring atoms has overlapping orbitals—usually `p` orbitals—through which electrons can be delocalised. A conjugated system may be an open chain or a ring. For example, the alternating double bonds in `1,3-butadiene` form a conjugated chain, but the molecule is not aromatic.

**Aromaticity** is a special case of cyclic conjugation. In the introductory Hückel model, an aromatic system must be cyclic, approximately planar, continuously conjugated, and contain `4n + 2` delocalised π electrons, where `n` is a non-negative integer. Benzene satisfies these conditions with six π electrons.

The central distinction is:

> Every aromatic system is conjugated, but not every conjugated system is aromatic.

A molecule can also contain a conjugated bond outside an aromatic ring. We therefore retain aromaticity and conjugation as separate candidate features: aromaticity describes membership in a special stabilised cyclic electron system, whereas conjugation describes the local possibility of electron delocalisation across connected atoms and bonds.

## Candidate bond properties

The molecular graph may distinguish single, double, triple, and aromatic bonds, together with conjugation, ring membership, and relevant stereochemistry. This allows a local rule to respond differently to chemically different neighbourhoods.

## Design principle

The initial feature set will remain small and defensible. The purpose is to give the automaton useful chemical starting conditions while leaving it responsible for learning dynamics, rather than overwhelming it with a large conventional descriptor library.

The Gray-Scott chemistry study treats additions as grouped hyperparameters. Periodic channels include scaled atomic number, atomic mass, covalent radius, van der Waals radius, and outer-electron count. Valence channels include total and implicit valence, heavy-atom degree, radical electrons, and absolute formal charge. Electronic channels include Pauling electronegativity, approximate atomic polarizability, heteroatom and halogen flags, and the fraction of incident conjugated bonds. Ring-geometry channels encode ring count and membership in three- through seven-membered or larger rings. The baseline and controlled combinations are compared under the same grouped validation split; the blinded set remains excluded.

## Topics to develop

- valence, bonding, and resonance;
- aromaticity and conjugation;
- functional groups;
- ionisation and formal charge;
- stereochemistry;
- intermolecular forces; and
- which atomic properties are measured, calculated, categorical, or context-dependent.
