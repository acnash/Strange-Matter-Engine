# Visualising Graph-CA Propagation in PyMOL

## Learning objective

The trained graph cellular automaton produces an atom-level trajectory over generations. We can preserve that trajectory and animate it on a generated three-dimensional molecular conformer so that information propagation becomes visible.

The animation is a scientific view of a 2D graph model. Three-dimensional coordinates provide a display scaffold; they do not become model inputs.

## The accepted visualisation concept

For each of the 750 blinded challenge molecules:

1. standardise the supplied SMILES;
2. build the accepted 2D graph;
3. run the frozen model for generations 0 through 16;
4. save the complete eight-channel atom trajectory losslessly;
5. generate one reproducible 3D conformer for display;
6. project the trajectory into a chosen atom-level display scalar;
7. write a multi-state structure and PyMOL session or script;
8. animate colour and glow over CA generations; and
9. append a clearly labelled hydrogen **visual coda** that is not a model generation.

## 1. The model remains two-dimensional

The predictive model uses:

- atom identities and fixed atom properties;
- bond connectivity and fixed bond properties;
- CYP context;
- eight evolving dynamical channels; and
- 16 graph-CA updates.

It does not use generated coordinates, distances, angles, or conformer energies.

The scientific prediction is

```math
\widehat y
=
f_{\rm model}
\left(
G_{\rm 2D},c_{\rm CYP}
\right).
```

The visualisation adds coordinates only after prediction:

```math
R_{\rm 3D}
=
f_{\rm display}
\left(
G_{\rm 2D},s_{\rm conformer}
\right),
```

where $s_{\rm conformer}$ is a recorded conformer-generation seed.

Changing the display conformer must not change predicted pIC50 or the stored CA trajectory.

## 2. What is stored from the model

For molecule $m$, CYP $c$, generation $t$, atom $i$, and channel $k$, the fundamental value is

```math
h_{mcik}^{(t)}.
```

With $n_m$ atoms, 17 stored states, and eight channels, the complete trajectory has shape

```math
n_m\times17\times8.
```

The lossless scientific record should include:

- standardised atom ordering;
- fixed atom and bond features;
- all $H^{(0)},\ldots,H^{(16)}$;
- CYP identity;
- 40-component fingerprint;
- predicted pIC50;
- model and preprocessing versions;
- configuration and checkpoint identifiers; and
- numerical precision.

PDB files are visualisation products, not the authoritative trajectory store.

## 3. Why PDB alone is insufficient

The PDB format offers one B-factor and one occupancy value per atom per model. Our trajectory contains eight values per atom.

PDB also has limited numeric field width and is not designed as a general tensor format. Reducing eight channels to one displayed value necessarily discards information.

We will therefore retain:

- a lossless trajectory file for analysis;
- a metadata file describing atoms, channels, CYP, prediction, and scaling; and
- one or more PDB/PyMOL visualisations derived from those records.

The animation can always be regenerated from the lossless data.

## 4. B-factor versus beta

Two similarly named quantities must remain distinct.

### Learned beta

The readout coefficients are

```math
\beta\in\mathbb R^{40}.
```

They weight molecule-level dynamical fingerprint components to predict pIC50.

### PDB B-factor

The PDB **B-factor field** is a scalar storage column attached to each atom in each model. We can repurpose it for visual colour or brightness.

The PDB field is not the learned $\beta$ vector. We will call it the **display B value** to prevent ambiguity.

## 5. Eight channels require a visual projection

The default atom activity magnitude is

```math
a_{mci}^{(t)}
=
\left\lVert
h_{mci}^{(t)}
\right\rVert_2
=
\sqrt{
\sum_{k=1}^{8}
\left(h_{mcik}^{(t)}\right)^2
}.
```

This gives one non-negative activity value per atom and generation.

It answers:

> How far is this atom's dynamical state from the neutral origin?

It does not show sign or identify which channel is active. We will therefore support several scientifically labelled views:

- total channel magnitude;
- one selected signed channel;
- step-change magnitude;
- gate openness;
- and later, a formal prediction-attribution measure.

No projection will be called “information” without defining what it measures.

## 6. Step-change animation

To emphasise propagation rather than accumulated activity, define

```math
\Delta a_{mci}^{(t)}
=
\left\lVert
h_{mci}^{(t+1)}
-
h_{mci}^{(t)}
\right\rVert_2.
```

Atoms glow when their state changes strongly between generations. A wave of bright atoms can therefore reveal where dynamical updates are occurring through the graph.

The activity-magnitude and step-change movies answer different questions and should be exported separately.

## 7. Readout-aware attribution

The 40 coefficients $\beta$ act on aggregated trajectory summaries, not directly on individual atoms. It would therefore be incorrect to assign one readout coefficient to each atom.

A later prediction-aware visual can use a defined attribution such as

```math
A_{mci}^{(t)}
=
\left\lVert
\frac{\partial\widehat y_{mc}}
{\partial h_{mci}^{(t)}}
\right\rVert_2.
```

This measures local prediction sensitivity to the atom state at that generation.

Gradient magnitude is sensitivity, not causal contribution. Integrated gradients or controlled atom-state interventions can later provide complementary views.

## 8. Mapping values to the B-factor field

Raw values can differ greatly between molecules, CYPs, channels, and visual modes. For display, map a declared reference range to

```math
B_i^{(t)}\in[0,100].
```

For reference limits $a_{\rm low}$ and $a_{\rm high}$,

```math
B_i^{(t)}
=
100\,
\mathrm{clip}
\left(
\frac{a_i^{(t)}-a_{\rm low}}
{a_{\rm high}-a_{\rm low}},
0,1
\right).
```

The limits must be stored. Per-frame scaling is unsuitable for comparing brightness over time because it can make a weak frame appear as bright as a strong one.

Useful scaling scopes include:

- one molecule–CYP trajectory;
- all four CYP trajectories for one molecule; or
- the complete blinded set.

The animation title and metadata will identify the scope.

## 9. Colour and brightness

The visual vocabulary can use a cyberpunk diverging or sequential palette:

- near-black or dark violet for low activity;
- cyan for intermediate activity;
- magenta for high activity; and
- lime or near-white highlights for the strongest changes.

For a signed channel, negative and positive values require a diverging scale centred on zero. For non-negative magnitude, a sequential scale is appropriate.

Brightness, colour, sphere radius, and glow can all encode the same scalar, but using too many effects simultaneously can obscure quantitative interpretation. Every movie will include a legend.

## 10. Multi-model PDB structure

A multi-model PDB uses repeated blocks:

```text
MODEL        1
ATOM ...
ENDMDL
MODEL        2
ATOM ...
ENDMDL
```

For CA generations 0 through 16, we need 17 scientific states:

```math
\text{PDB state number}=t+1.
```

All states use:

- the same atom ordering;
- the same coordinates;
- the same bond topology; and
- generation-specific display B values.

The molecule remains spatially stationary while colour or glow propagates. Camera movement can be added by PyMOL without pretending the atoms physically moved.

## 11. Why coordinates remain constant

Our CA state describes abstract learned dynamics, not molecular dynamics. Changing coordinates between generations would visually imply conformational motion unsupported by the model.

The 3D conformer is therefore generated once and repeated identically in every state:

```math
R_i^{(0)}
=
R_i^{(1)}
=\cdots=
R_i^{(16)}.
```

Only visual attributes change.

## 12. Generating the display conformer

The display process will:

1. add explicit hydrogens to a copy of the standardised molecule;
2. generate a 3D conformer with a recorded seed;
3. perform a declared geometry refinement;
4. preserve the heavy-atom mapping to the model graph;
5. record energy and convergence status; and
6. quarantine failed conformers for a fallback 2D or alternative display.

This conformer is illustrative. It is not claimed to be the CYP-bound pose or biologically dominant conformation.

## 13. Explicit hydrogens are not CA nodes

The accepted model uses attached-hydrogen count as a fixed atom property rather than adding every hydrogen as a graph node.

Display hydrogens are generated after the CA trajectory. For hydrogen $q$ attached to heavy atom $i$, the final coda value can be

```math
B_q^{\rm coda}
=
B_i^{(16)}.
```

The hydrogen copies the final display intensity of its parent heavy atom. It has no independently learned trajectory.

## 14. The hydrogen visual coda

The proposed final flash is a compelling presentation device. It must be labelled separately from scientific generations.

The sequence becomes:

- states 1–17: CA generations 0–16;
- coda-off state: same final heavy-atom display, hydrogens hidden or dark;
- coda-on state: hydrogens appear with their parent atoms' final brightness;
- optional repeated off/on coda states to create a blink.

The metadata will contain

```text
state_type = scientific_generation
```

or

```text
state_type = visual_coda
```

so the extra state can never be mistaken for model output.

## 15. Consistent topology across states

Multi-state molecular viewers behave most reliably when every state contains the same atoms in the same order.

Hydrogens should therefore exist in every PDB state, but their early-state visibility can be controlled through occupancy or the PyMOL script. In the coda-on state, their occupancy and visual styling change.

This avoids changing the atom count between models.

## 16. PyMOL control script

The generated PyMOL script or session will:

- load the multi-state structure;
- set a black background;
- display the molecule cleanly in sticks;
- map display B values to the chosen cyberpunk palette;
- keep the camera and molecular coordinates stable;
- show generation, molecule identifier, CYP, and predicted pIC50;
- iterate through states at a declared frame rate;
- apply hydrogen coda visibility;
- loop or play once; and
- render frames at a declared resolution.

The scientific data remain separate from cinematic camera and glow settings.

## 17. File organisation

For each molecule–CYP prediction, the output bundle should contain files equivalent to:

```text
trajectory.npz
metadata.json
trajectory_multistate.pdb
view_and_animate.pml
preview.mp4
```

A manifest across all 750 molecules will record:

- molecule and source identifiers;
- original and standardised SMILES;
- CYP;
- predicted pIC50;
- trajectory path;
- conformer status;
- display projection;
- scaling limits;
- model version; and
- rendering status.

## 18. Storage scale

The raw trajectory for molecule $m$ contains

```math
17\times n_m\times8
```

floating-point values per CYP.

For all four CYPs and 750 molecules, approximate value count is

```math
750\times4\times17\times8\times\bar n,
```

where $\bar n$ is mean heavy-atom count.

This is manageable with compressed binary arrays, but rendered frames and videos may consume much more space. Lossless data, PDBs, PyMOL scripts, and videos will therefore be versioned and manifested deliberately rather than committed indiscriminately.

## 19. Validation checks

Before accepting an animation, verify:

- predicted pIC50 matches the prediction manifest;
- atom ordering matches the model graph;
- all 17 scientific states are present;
- coordinates are identical across scientific states;
- all eight channels exist in the lossless file;
- display projection matches metadata;
- scaling is constant over the intended comparison scope;
- no cross-molecule or cross-CYP data were mixed;
- hydrogens map to the correct parent atoms;
- coda states are explicitly labelled non-scientific; and
- re-rendering from stored data gives the same result.

## 20. Scientific interpretation

The animation can reveal:

- where dynamical activity first appears;
- how quickly changes propagate through bonds;
- whether regions synchronise;
- whether activity persists, contracts, or oscillates;
- how CYP context changes the same molecule's trajectory; and
- which atoms are locally sensitive to the final prediction.

Visual appeal supports exploration and communication. Quantitative claims must still come from stored values and declared measurements.

## 21. Prototype visualisation plan

We will first generate:

1. one representative molecule;
2. all four CYP contexts;
3. activity-magnitude and step-change views;
4. one fixed 3D conformer;
5. 17 scientific states;
6. a hydrogen off/on visual coda;
7. a PyMOL session and short video; and
8. a validation report.

After this small pilot is verified, the workflow can scale to the 750 blinded molecules.

## Connection to the course

- [Scientific Visualisation](Scientific_Visualisation.md) defines the visual principles.
- [Hybrid Atom-State Channels](Hybrid_Atom_State_Channels.md) defines the eight evolving values.
- [Differentiable Dynamical Fingerprint](Differentiable_Dynamical_Fingerprint.md) distinguishes atom trajectories from molecular summaries.
- [Dynamics](Dynamics.md) defines propagation, convergence, oscillation, and perturbation response.
- [Molecular Standardisation](Molecular_Standardisation.md) preserves atom identity and provenance.

