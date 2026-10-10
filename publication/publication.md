# Molecular Space-Time: Emergent Prediction from Local Interactions in an Evolved Graph Cellular Automaton

**Anthony Nash, PhD**

**Independent Researcher, Warwickshire, United Kingdom**

**info@acnash.co.uk**

## Abstract

This study investigates how local interactions between atomic cells generate predictive information and complex dynamical behaviour. Molecular graphs form cellular automata whose numerical atom states evolve through interactions along chemical bonds. Their molecular space-time trajectories provide fingerprints for predicting enzyme inhibition. Genetic programming evolves the cellular automaton transition rule, while backpropagation trains numerical weights and ridge regression fits predictions. In the OpenADMET CYP Inhibition Challenge, genetic programming modestly reduced blind prediction error from 1.0092 to 0.9946 on the official metric. Separate dynamical analyses identified convergence, possible periodic behaviour and persistent complex variation. Two systems exhibited bounded hyperchaotic dynamics, with small initial differences growing in multiple independent directions. Discontinuities in the phase-based rule and unresolved fractal dimension leave their classification as strange attractors provisional. These findings support molecular space-time as a framework for studying how local interactions generate predictive information and dynamical complexity.

**Keywords:** cellular automata; complex systems; emergence; genetic programming; graph dynamics; molecular representation; nonlinear dynamics

## 1. Introduction

Cellular automata demonstrate how repeated local changes and interactions can produce organised behaviour across a system. Uniform rules on simple cell states can produce fixed configurations, oscillators, propagating structures, long transients, computational universality, and transitions between ordered and disordered behaviour [1–3]. A reaction–diffusion system provides a related physical example: local chemical reactions change species concentrations, while diffusion redistributes those species through space. Together, these processes can generate larger-scale concentration patterns [4]. This shared principle motivates our investigation of whether repeated interactions between atomic cells can generate molecule-wide numerical patterns useful for predicting chemical behaviour. To investigate this possibility, we construct a molecular graph cellular automaton in which each atom carries numerical state values and chemical bonds define which atoms exchange information. Physical reaction–diffusion systems provide the analogy for these local interactions, and training adjusts the cellular automaton so that its trajectory summaries predict enzyme inhibition.

Three related approaches motivate our use of repeated local interactions to learn a molecular representation. Learned graph cellular automata extend shared cellular automaton transition rules to irregular networks, providing a framework that accommodates atoms connected by chemical bonds [5]. Neural cellular automata demonstrate how training can adjust local interactions so that numerical state channels collectively support a desired behaviour; each channel acquires its role through training [6]. Molecular message-passing networks show how exchanging information between bonded atoms can support molecular-property prediction [7]. Together, these approaches motivate a model that learns local interactions on molecular graphs and retains the resulting atom-state trajectories as predictive features.

Our model uses atom and bond descriptors to encode chemical information and retains successive atom states as a trajectory, which we call **molecular space-time**. At generation *t*, a vector of learned numerical values describes each atom's cellular state. The possible combinations of these values across the molecule define its numerical state space. One point in this mathematical space represents all atom-state channel values at one generation, and successive generations trace a trajectory through it. A direction in state space describes a particular combination of changes to those values; the coordinates describe computational states. Typed bonds constrain the messages that can alter that state, and a shared nonlinear cellular automaton transition rule advances the entire molecular graph to generation *t* + 1. We summarise the early changes, states at several generations and final states into a molecular fingerprint, a vector of numerical features for predicting pIC50.

The cellular automaton transition rule determines which trajectories the system can express. We studied three methods that progressively expanded what training could change while preserving the atom-state trajectories as the basis for prediction. The first used fixed cellular automaton transition rule families. Backpropagation through time traced prediction errors backwards through successive generations to train the numerical weights. Ridge regression then combined trajectory features into pIC50 predictions; differentiating through this fit allowed prediction errors to guide the weights that generated those features. The second used a genetic algorithm to choose the Graph-CA configuration, including the rule family, trajectory depth, state width, feature profile, and temporal observables, while retaining backpropagation for continuous weights. Sections 2.2 and 2.3 explain these choices in detail, and Section 2.4.4 specifies the alternatives tested by the genetic search. The third used genetic programming to evolve the mathematical form of the cellular automaton transition rule itself. Genetic algorithms operate over encoded parameter or design vectors, whereas genetic programming applies selection, crossover, and mutation to executable symbolic structures [8, 9].

The 2026 OpenADMET CYP Inhibition Challenge provided an external validation task. Cytochrome P450 (CYP) enzymes influence the metabolism and clearance of many medicines, and inhibition can contribute to clinically important drug interactions [10]. The challenge examined four CYP isoforms, CYP1A2, CYP2C9, CYP2D6, and CYP3A4, which are distinct enzyme members of this family with different substrate preferences. Each isoform defined a separate prediction task: using the compound’s molecular structure to predict how potently it inhibits that enzyme. The organisers compared the resulting direct-inhibition pIC50 predictions with concealed experimental measurements. This independent evaluation tests the predictive usefulness of the molecular space-time fingerprints.

The retained molecular space-time trajectories also allow us to examine how numerical atom states behave over successive generations. Cellular-automaton and nonlinear-dynamics studies distinguish convergence towards a fixed state, repeating oscillations, persistent complex variation and chaos through quantitative measurements [1–3, 11–13]. We held the trained Graph-CA parameters fixed and followed the atom states for thousands of generations to investigate whether molecular structure and the cellular automaton transition rule produce repeatable patterns, such as convergence or oscillation. We analysed recurrence, frequency spectra, perturbation growth, Lyapunov exponents and attraction basins to characterise these patterns. To test sensitivity to initial conditions, we introduced small differences into the numerical atom states and measured whether those differences grew over subsequent generations. Sustained growth indicates that initially similar computational states follow increasingly different trajectories, providing evidence relevant to chaos. We also investigated whether trajectories approach a bounded set with fractal geometry. We estimated its fractal dimension, which describes how the detail of that set changes with the scale at which we examine it. These estimates require care: we excluded pairs of states close together in generation number and checked whether the estimated dimension remained consistent when we represented the trajectory using different numbers of coordinates [14–16].

Genetic programming produced a modest improvement in blind enzyme-inhibition prediction over the fixed-rule model. Separate analyses of the numerical atom states identified convergence, possible periodic behaviour and persistent complex variation. Two systems showed bounded hyperchaotic dynamics, with small initial differences growing in multiple independent directions. Discontinuities at the phase boundary and an unresolved fractal dimension leave their classification as strange attractors provisional.

## 2. Materials and Methods

### 2.1. Dataset and Prediction Task

The study used the primary direct-inhibition dataset released for the 2026 OpenADMET CYP Inhibition Blind Challenge (https://openadmet.org/blindchallenges/). It comprised 4,905 unique compounds represented by a molecule identifier and a Simplified Molecular Input Line Entry System (SMILES) string. The dataset provided experimental direct-inhibition pIC50 values for four drug-metabolising cytochrome P450 enzyme isoforms: CYP1A2, CYP2C9, CYP2D6, and CYP3A4. Each value described inhibition of one specific isoform by one compound. Here, pIC50 denotes the negative base-10 logarithm of the half-maximal inhibitory concentration expressed in molar units. Each reported measurement included lower and upper uncertainty bounds and an estimated standard deviation from the fitted concentration–response experiment.

The experiments did not necessarily measure each compound against every CYP isoform. The dataset contained 6,525 observed compound–CYP pairs, distributed as summarised in Table 1.

**Table 1. Number of observed direct-inhibition pIC50 measurements for each CYP isoform.**

| CYP isoform | Compounds with an observed direct-inhibition pIC50 |
|---|---:|
| CYP1A2 | 1,412 |
| CYP2C9 | 1,285 |
| CYP2D6 | 1,493 |
| CYP3A4 | 2,335 |
| **Total** | **6,525** |

Each available compound–CYP measurement formed one training case: the model received the compound's molecular structure and CYP isoform and was trained to predict the corresponding experimental pIC50 value. The Endpoint-Aligned Cross-Validated CYP-Specialist Graph Cellular Automaton (EA-CV-CYP-GCA; see Section 2.4 for the full training and validation procedure) comprised four separately trained nonlinear graph cellular automaton models, one for each isoform. Each graph cellular automaton model used only observations for its assigned isoform during backpropagation through successive generations, differentiable ridge fitting, validation and selection of the saved model. Here, an endpoint means the single prediction target defined for one CYP isoform: its direct-inhibition pIC50. A missing endpoint was therefore a compound–CYP combination for which the dataset reported no experimental direct-inhibition pIC50 value. We retained these values as missing, so they contributed neither targets nor loss terms.

To assess generalisation beyond closely related chemistry, we grouped compounds by their standardised Bemis–Murcko scaffold before data partitioning. A Bemis–Murcko scaffold is a simplified structural skeleton that retains a molecule's rings and the chains connecting those rings while removing side branches attached to this central framework. Compounds with the same underlying ring-and-connector framework therefore belonged to the same scaffold group. We reserved a fixed 20% subset of scaffold groups as a sealed holdout, leaving 5,216 observed compound–CYP pairs for model fitting and internal selection and 1,309 pairs for final validation. No scaffold group occurred in both partitions. We selected hyperparameters within the fitting pool, while the reserved scaffold holdout remained unused until final evaluation.

The challenge test set contained 750 additional compounds for which the organisers withheld all direct-inhibition labels. The challenge required the four trained CYP-specific graph cellular automaton models to generate one finite pIC50 prediction per isoform for each compound, giving 3,000 blinded compound–CYP predictions. We excluded blinded compounds and their unreleased outcomes from parameter estimation, hyperparameter selection, early stopping, and dynamical candidate selection. Predictive evaluation followed the challenge formulation: the primary measure was the macro-averaged soft-threshold relative absolute error (MA-ST-RAE) across the four CYP isoforms, with equal weight for each isoform and zero error for predictions falling within the reported experimental uncertainty interval. Lower MA-ST-RAE indicates better prediction under this uncertainty-aware measure. We retained conventional regression statistics as complementary measures of predictive agreement, including root mean squared error (RMSE), which summarises prediction-error magnitude, and rank correlations, which assess how well predictions preserve the ordering of experimental values.

### 2.2. Molecular Graph Representation

We used RDKit to standardise SMILES structures, including common groups of bonded atoms and their formal atomic charges (integer charges assigned to individual atoms, such as +1 or −1). We then retained the largest organic fragment and applied RDKit's sanitisation procedure to check the consistency of the molecular representation, including whether each atom's number and types of bonds were compatible with its assigned chemical state. We converted the resulting structure to canonical isomeric SMILES. We represented each standardised molecule as a graph G=(V,E), where every heavy atom (an atom other than hydrogen) i ∈  V was a cellular-automata cell and every covalent bond defined two directed message-passing edges, j →  i and i →  j. The predictive input, meaning the model inputs for predicting pIC50, comprised the standardized two-dimensional graph connectivity, atom and bond descriptors, and CYP context; it contained no molecular geometry.

The standard set of atomic descriptors contained one-hot element identity for H, C, N, O, F, P, S, Cl, Br, I, and other elements; for example, with this ordering the encoding represented carbon as (0,1,0,0,0,0,0,0,0,0,0) and oxygen as (0,0,0,1,0,0,0,0,0,0,0). Further descriptors comprised formal charge; aromaticity; sp, sp2, sp3 (types of atomic orbital hybridisation), and other hybridisation states; degree; total attached hydrogens; ring membership; hydrogen-bond donor and acceptor status; and tetrahedral chirality. We also tested additional atomic descriptors, organised into the five groups in Table 2. To choose which groups to include, we compared their predictive performance using training and validation subsets drawn entirely from the fitting data. The sealed holdout and blinded challenge set played no part in that selection.

**Table 2. Chemically organised atom-feature groups considered during training-only model selection.** A feature group is a named collection of related atom-level descriptors that we added to the baseline encoding as one candidate block.

| Feature group | Atom-level quantities |
|---|---|
| Periodic | Atomic number, atomic mass, covalent radius, van der Waals radius, outer electrons, period, group, and approximate atomic volume. |
| Valence | Total and implicit valence, heavy-atom degree, radical electrons, and absolute formal charge. |
| Electronic | Pauling electronegativity, approximate polarizability, heteroatom and halogen indicators, conjugated-bond fraction, first ionization energy, electron affinity, and missing-property indicator. |
| Ring geometry | Number of rings containing the atom and indicators for ring sizes 3, 4, 5, 6, 7, and 8 or greater. |
| Local environment | Mean neighbouring electronegativity, electronegativity difference from the neighbourhood, mean neighbouring atomic number, and mean neighbouring formal charge. |

The *feature profile* specifies which chemical descriptors form each atom's input vector: the baseline encoding together with the selected additional feature groups in Table 2. We stored the selected feature profile with every frozen checkpoint, meaning the saved model whose parameters we held fixed for evaluation. This preserved the exact feature names and order required to predict new molecules. We scaled continuous quantities by fixed chemically meaningful constants during graph construction. The bond vector 𝑒ⱼᵢ contained one-hot indicators for single, double, triple, or aromatic bonds, together with indicators for conjugation, ring membership, and three stereochemical properties. We attached the same bond vector to both directed representations of an undirected bond.

Each CYP endpoint defined a distinct prediction task: predicting direct-inhibition pIC50 for one isoform. A context vector, 𝒄, identified which CYP isoform the model was predicting for a given molecule. This vector contained four entries, one for each isoform: the entry for the target isoform was 1, and the other three were 0. This one-hot encoding supplied the target identity to the cellular automaton and its prediction readout, allowing the same molecular graph to generate a trajectory and prediction specific to CYP1A2, CYP2C9, CYP2D6, or CYP3A4.

Within this molecular graph, each atom carries H numerical state values, called channels, which change over successive generations. These channels are latent because training determines how they combine the atomic descriptors to support prediction. Figure 1A illustrates the atomic descriptors and typed bonds that define this connected molecular Graph-CA, with any additional feature groups selected from Table 2 forming part of the atomic input vector and retained in the frozen feature profile.

![Molecular graph encoding and initial dynamical channels](figures/01_graph_ca_encoding_initial_state.png)

**Figure 1. Molecular graph encoding and formation of the initial dynamical state.** **A**, N-methylacetamide illustrates the molecular graph: each heavy atom (an atom other than hydrogen) forms a cell, and each bond allows information to pass in both directions. The vector xᵢ records the atom's chemical properties, including its attached hydrogen count, while 𝑒ⱼᵢ records the bond's properties. **B**, the model combines each atom's F input descriptors into H numerical state values, called channels. Learned weights determine how strongly each descriptor contributes to each channel; scaling and the hyperbolic tangent then keep the initial values between −1 and 1. The same transformation applies to every atom. The example of 20 descriptors producing 8 channels is illustrative. These initial values subsequently change through interactions with bonded neighbours, forming the molecular space-time trajectory.

### 2.3. Nonlinear Graph Cellular Automaton

The Graph Cellular Automaton (Graph-CA) treats every heavy atom as a cell and every chemical bond as a route for local communication. Each atomic cell begins with fixed information about its chemical identity and surroundings. A shared initialisation process converts that information into a numerical state. The state has no assigned one-to-one chemical meaning. Its meaning is operational: it comes from how it affects neighbouring cells, how it changes over repeated generations and how the resulting molecular trajectory contributes to prediction.

All five Graph-CA approaches considered here share this local organisation. At each generation, an atom receives information from directly bonded neighbours and combines it with its own state, its fixed chemical features and the CYP isoform being predicted. The same local rule is applied to every atom and reused at every generation. Information can therefore spread beyond an atom's immediate neighbours through repeated updates, even though no individual update has access to the complete molecule. The molecular representation emerges from this repeated local exchange. Appendix A gives the complete mathematical definition of the initial states, bond-dependent messages, common reaction drive and trajectory summaries.

The five approaches differ in what they assume about change, memory and coordination. These differences are useful because a cellular automaton transition rule is more than a numerical update. It places a particular structure on how local information can persist and spread. Each structure provides a different route from symbolic atomic states to organised molecule-wide behaviour.

**Gated residual.** The gated-residual approach treats change as a controlled choice between retaining the present state and accepting a proposed new state [17]. Each atom makes that choice separately for every state channel and at every generation. This gives the rule a direct form of adaptive memory: a channel can remain almost unchanged when its current information should be preserved, or it can move rapidly when local chemical and neighbourhood information favours an update. The approach is closely related to gated recurrent neural networks and neural cellular automata. Its principal bias is towards stable, incremental change, because every update is formed as a bounded mixture of the old state and the proposed reaction.

**Inertial reaction–diffusion.** The inertial reaction–diffusion approach separates three influences on an atom: locally generated change, exchange with bonded neighbours and restoration towards a central state [4]. Here, diffusion describes the movement of numerical information across bonds rather than the physical diffusion of chemical material. The rule also retains a velocity-like quantity, so part of an earlier change continues into the next generation [18]. This inertia gives the system a short memory of its direction of travel. The rule can therefore overshoot, damp, propagate disturbances through the graph or settle gradually. It provides a natural framework for asking whether local disturbances spread, fade or organise into longer-lived graph-wide patterns.

**FitzHugh–Nagumo.** The FitzHugh–Nagumo approach divides the state of each atom into a faster excitation component and a slower recovery component [19, 20]. A strong enough local input can move the excitation component away from rest, while the recovery component acts later to regulate or reverse that change. This separation of timescales can support threshold responses, pulses and oscillatory return. In the Graph-CA, chemical information and bonded-neighbour exchange shape both components. The rule therefore tests whether a molecular representation benefits from excitable local cells whose immediate response and recovery unfold at different rates.

**Kuramoto–Sakaguchi.** The Kuramoto–Sakaguchi approach interprets each state channel as a phase within a repeating cycle [21, 22]. Bonded atoms influence one another according to their phase difference, while the atom's chemical information and CYP context determine its preferred rate of advance. This rule focuses on coordination rather than movement towards a single numerical value. Neighbouring atomic cells may synchronise, maintain a phase offset or resist collective alignment. Because phases repeat after a complete turn, the state space is circular. This geometry makes the rule well suited to collective rhythms and phase organisation, while also requiring care at the numerical boundary used to represent the circle. The long-horizon analysis in Sections 2.5 and 3.3 examines the complex regimes produced by this approach.

**Delayed memory.** The delayed-memory approach gives each atom explicit access to one of its earlier states [23]. The next state depends on the present reaction, a learned response to the retained earlier state and the difference between past and present. This creates temporal feedback that cannot be represented by a rule using the present state alone. A short delay can reinforce or oppose recent changes, while a longer delay can reintroduce information after several rounds of neighbour communication. The approach therefore tests whether molecular prediction benefits from local cells that compare their current condition with a stored point in their own history.

These approaches represent distinct computational hypotheses. Gated residual updates emphasise selective retention. Inertial reaction–diffusion emphasises propagation and momentum. FitzHugh–Nagumo dynamics emphasise excitation and recovery on different timescales. Kuramoto–Sakaguchi dynamics emphasise phase coordination. Delayed memory emphasises explicit temporal feedback. They are not claims that atoms literally follow these physical or biological equations. Instead, each rule supplies an inductive bias: a structured way for the model to organise local information over computational time.

The common Graph-CA design makes the approaches comparable. They use the same molecular graph, atomic descriptors, bond information, CYP context and overall prediction task. Their differences lie in how they transform local inputs into the next generation. This separation allows us to ask whether predictive information depends only on the starting chemical descriptors or also on the form of the repeated local interaction. It also allows the trained systems to be studied as dynamical objects, because each rule can generate characteristic forms of convergence, oscillation, persistence or sensitivity to initial conditions.

Every rule produces a trajectory containing the state of every atom at every generation. Molecules contain different numbers of atoms, so the complete trajectories cannot be passed directly to one fixed-size prediction model. We therefore summarised each trajectory across atoms and across selected generations. These summaries captured final state, variation between atoms, change through time and behaviour at several temporal scales. The resulting fixed-length molecular fingerprint was supplied to ridge regression. Optional summaries allowed the genetic search to test whether persistence, correlation, transition energy or the direction of change added useful information. Appendix A defines these quantities and explains the CYP-specific fingerprint construction.

Figure 2, presented in Appendix A alongside the full mathematical specification, shows recorded trajectories from the five approaches. The examples demonstrate that the same basic organisation—atomic cells, bond-limited communication and a shared local rule—can produce visibly different molecular space-time patterns. The important comparison is therefore between forms of local computation. Repeated generations turn those local choices into a global trajectory, and the prediction task tests whether that emergent trajectory contains chemically useful information.

### 2.4. Learning the Molecular Space-Time Representation

All four learning methods used the same route from a molecule to a prediction. First, the Graph-CA generated a molecular trajectory. The pooling calculations described in Section 2.3 converted that trajectory into a fixed-length molecular fingerprint. Ridge regression then assigned a coefficient to each fingerprint value and combined them to predict pIC50. Its penalty discourages very large coefficients, which helps control overfitting. A direct linear-equation solve calculated the coefficients from the training fingerprints and their measured pIC50 values. **All four methods used this fitted ridge readout.** They differed only in how they trained the Graph-CA or selected the system that generated its input fingerprints (Figure 3).

Three methods used backpropagation. Backpropagation starts with prediction error and works backwards through the calculations to determine how each trainable numerical parameter should change. In this study, it passed through the ridge fit and then through every retained Graph-CA generation. It trained continuous parameters, meaning numerical quantities that can vary smoothly, such as the weights controlling communication between neighbouring atoms. The evolutionary strategy trained these parameters differently: it deliberately perturbed their values, observed the resulting changes in prediction error, and used those comparisons to estimate an update without backpropagation.

- **Fixed-rule Graph-CA:** a chosen mathematical rule describes how atom states change between generations. Backpropagation learns the numerical weights within that rule and the other Graph-CA components; ridge regression fits the final prediction. The mathematical form of the chosen rule stays fixed during training.
- **Evolutionary strategy:** the optimiser makes small random changes to the current Graph-CA parameter values, producing a population of candidate models. Each candidate generates trajectories and receives a fitted ridge readout; the optimiser then assesses it on query molecules, which do not contribute to that candidate's ridge fit. Differences in prediction error indicate which directions of parameter change are useful, and these guide the next update. This method uses ridge regression and does not use backpropagation.
- **Hybrid genetic search:** a genetic algorithm chooses the model's *configuration*, meaning design choices such as the cellular automaton transition rule family, number of generations, number of dynamical channels, chemical descriptors, and trajectory summaries. Backpropagation and a fitted ridge readout train each candidate configuration. The genetic algorithm selects better-performing configurations and recombines or changes their design choices to produce new candidates.
- **Genetic programming:** genetic search changes the mathematical form of the cellular automaton transition rule used to update each atom. For example, it can change how operations such as addition, multiplication, or averaging combine the current atom state, the reaction signal, and differences from neighbouring states. Backpropagation learns the numerical weights for each candidate cellular automaton transition rule, and ridge regression fits its prediction readout. Validation error then guides the choice of cellular automaton transition rules for the next population. The candidate cellular automaton transition rule remains fixed throughout each molecular trajectory.

The following subsections describe these methods and their shared ridge readout in detail. Candidate configurations and transition rules were selected using scaffold-held-out development data, in which structurally related molecular frameworks were kept together. The sealed-holdout measurements and blind challenge labels were not used for training or selection.

#### 2.4.1. Fixed-rule Graph-CA with backpropagation

The fixed-rule reference was the Credible-Interval-Aligned Endpoint-Aligned Cross-Validated CYP-Specialist Graph Cellular Automaton (CIA-EA-CV-CYP-GCA). It contained several Graph-CA *experts*: independently trained models whose predictions could later be combined. Backpropagation through time traced prediction error backwards through every retained generation. It therefore calculated how changing an earlier local update would affect the final pIC50 prediction.

The Adam optimiser used these calculations to update the initialisation, bond-message, reaction-drive and transition-rule parameters. A cosine schedule gradually changed the learning rate during training. Gradient clipping limited exceptionally large updates, and an L2 penalty discouraged large cellular automaton weights. Using labelled development data alone, we selected the number of generations and dynamical channels, atom-feature profile, learning rate, ridge penalty, update scale, proportion assigned to the support subset, batch size, bond temperature, initialisation scale and noise, pooling design, and constants specific to each transition rule.

CIA-EA-CV-CYP-GCA trained four independent nonlinear prediction systems, one for each CYP isoform. A CYP-specific system used only measurements for its target isoform when fitting the ridge readout, calculating prediction error, deciding when to stop training and choosing which saved checkpoint to retain. Mean squared error measured the average squared prediction error. We supplemented it with a smooth credible-interval penalty, which added a cost when a prediction fell outside the experimental uncertainty interval reported for that measurement.

For each CYP prediction target, we first screened all ten transition-rule families using two scaffold-separated folds. The three strongest combinations of rule and model configuration then advanced to confirmation using five scaffold folds and two independent training seeds. Each out-of-fold prediction came from a model that had not used the corresponding validation fold for fitting. We used these predictions to choose a subset of experts and fit the ridge combination that joined their predictions. Only after completing this selection did we evaluate the sealed holdout.

The final CYP1A2 system combined FitzHugh-Nagumo, Gray-Scott and conservative graph flux experts. CYP2C9 combined Gray-Scott and damped symplectic experts; CYP2D6 combined delayed memory and FitzHugh-Nagumo experts; and CYP3A4 combined damped symplectic, FitzHugh-Nagumo and delayed memory experts. Figure 3A shows how gradients from the query prediction error pass through the differentiable ridge readout and the repeated Graph-CA updates to train the molecular space-time representation.



#### 2.4.2. Shared ridge readout and its role in training

All four learning strategies shared the same analytic ridge readout. For every training batch, we divided the molecules into a *support subset* and a *query subset*. The support molecules supplied trajectory fingerprints and measured pIC50 values for fitting the ridge coefficients. The fitted coefficients then predicted the query molecules. Because the query molecules did not contribute to that particular ridge fit, their errors measured whether the Graph-CA had generated fingerprints that transferred beyond the fitted support examples.

The fixed-rule method, hybrid genetic search and genetic programming all used a differentiable ridge calculation. This means that backpropagation could trace query error through the ridge solution and then through the recurrent Graph-CA trajectory. In the two genetic methods, backpropagation trained the continuous weights within each candidate model, while the outer genetic search compared candidates using their development performance. The evolutionary strategy also fitted a ridge readout for every candidate, but treated query error as evolutionary fitness. It estimated updates by comparing deliberately perturbed candidates and did not differentiate through the ridge calculation. Ridge regression therefore supplied the common prediction mechanism; the four strategies differed in how they trained or selected the Graph-CA that produced its fingerprints.

The support fingerprints formed the matrix Fₛ: each row represented a support molecule and each column represented one fingerprint feature. We standardised every column by subtracting its support-set mean and dividing by its support-set scale, producing Zₛ. We also centred the measured support targets by subtracting their mean, giving yₛ−ȳₛ. The following closed-form calculation then produced the ridge coefficients β [24]:

```math
\beta=\left(Z_s^{\mathsf T}Z_s+\lambda I\right)^{-1}
Z_s^{\mathsf T}(y_s-\bar y_s).
```

In this expression, ZₛᵀZₛ describes relationships among the standardised fingerprint features. The ridge term λI adds the non-negative penalty λ to the diagonal through the identity matrix I, reducing sensitivity to large or highly correlated coefficients. The inverse matrix then gives the coefficient vector that balances fit to the centred support targets against this penalty.

For a query molecule with fingerprint f[q], the predicted pIC50 was

```math
\widehat y_q=\bar y_s+
\left(\frac{f_q-\bar F_s}{s_F}\right)^{\mathsf T}\beta,
```

Here, F̄ₛ and s[F] are the support-set mean and scale for each fingerprint feature. Subtracting F̄ₛ and dividing by s[F] applies the same standardisation to the query fingerprint that was applied to the support fingerprints. The dot product with β produces the centred prediction, and adding ȳₛ restores the mean pIC50. The ridge penalty did not apply to this intercept.

Automatic differentiation retained the dependence of β on the support fingerprints. Query prediction errors could therefore update the Graph-CA through two connected routes: through the ridge fit and through the successive atom-state generations that produced the fingerprints. When highly correlated trajectory statistics made the matrix equations numerically singular, a Hermitian pseudoinverse supplied a numerically stable solution to the same ridge objective.

This support-query design trained the cellular automaton to produce fingerprints that predicted pIC50 for molecules outside the subset used for the current ridge fit. After Graph-CA training, we refitted the ridge readout using all observations permitted for fitting and stored it with the selected Graph-CA checkpoint. We used MA-ST-RAE on the relevant scaffold-held-out development fold to decide when to stop training and which checkpoint to advance. RMSE was recorded as a secondary diagnostic of prediction error.

#### 2.4.3. Evolutionary-strategy comparator

We constructed ES-EA-CV-CYP-GCA as a derivative-free comparison method (Figure 3B). *Derivative-free* means that it updated parameters without calculating gradients. To isolate this difference in training, it retained the same molecular graphs, chemical encodings, recurrent transition rules, stored trajectories, scaffold partitions and analytic ridge readout as the backpropagation system.

The evolutionary strategy created paired candidates by adding and subtracting the same random change from the current parameter values. These opposite, or *antithetic*, perturbations allowed a direct comparison of the two directions. For every candidate, the Graph-CA generated support and query fingerprints, the analytic solve fitted ridge coefficients from the support observations, and query error provided the evolutionary fitness. If one member of a pair produced a lower error, the direction of its parameter change was considered more useful. The optimiser combined evidence across all pairs and used Adam-style running averages to update the Graph-CA parameters without backpropagation.

The molecular representation, prediction readout and support-query separation remained fixed, so this comparison focused on how the nonlinear Graph-CA parameters were trained. Evolutionary fitness and checkpoint selection did not use the sealed holdout or blind labels. The backpropagation-through-time version remained the primary submitted method while we evaluated this comparator.

The campaign used the canonical damped-symplectic CYP3A4 expert throughout. It had 16 hidden channels, 32 recurrent generations, periodic-electronic atom features, typed-bond messages, multiscale trajectory pooling, an update scale of 0.25, a support fraction of 0.6 and a ridge penalty of 0.1. We tested 32 evolutionary configurations. Population size ranged from 32 to 128 candidates; the standard deviation of the Gaussian parameter perturbation ranged from 0.005 to 0.08; the evolutionary learning rate ranged from 0.0005 to 0.01; and batch size ranged from 256 to 1,600 molecules. We also compared random with activity-stratified batch composition and tested four stopping policies.

Each configuration was screened on two scaffold folds, producing 64 screening runs. The five strongest settings advanced to five-fold confirmation with two independent seeds, producing 50 more runs. Five CUDA workers executed the resulting 114 runs. The original two-fold result from the corresponding backpropagation-trained expert provided the matched development reference. The broader fold-and-seed confirmation measured the reproducibility of evolutionary performance. All evolutionary comparisons used development data only; the sealed holdout and blind challenge set remained unused.

#### 2.4.4. Hybrid genetic Graph-CA structure search

We next used a genetic algorithm to search the discrete architecture of the CYP3A4 specialist while retaining backpropagation for its continuous numerical parameters (Figure 3C). *Discrete architecture* refers to design choices selected from a defined list rather than numerical weights that vary continuously. Each candidate design was represented by a *genome*: a list containing its transition-rule family, trajectory depth, state width, atom-feature profile, trajectory-pooling design, degree normalisation, chemical-feature gating, initial-state anchoring, channel-specific timescales and optional trajectory summaries.

Trajectory depth and state width were defined in Section 2.3. The atom-feature profile selected descriptors from Section 2.2, while the pooling and temporal-observable choices selected which trajectory summaries entered ridge regression. The available transition rules were damped symplectic, FitzHugh-Nagumo and delayed memory. The search tested 16, 32, 64 or 128 recurrent generations and 8, 16 or 24 hidden channels. Atom encodings ranged from the periodic-electronic set to the comprehensive chemical profile. Numerical optimiser settings remained fixed at the previously established values for each rule.

We evolved four populations, called genetic generations, each containing 24 unique model structures. Tournament selection compared groups of three candidates when choosing parents. Uniform crossover formed a child by selecting individual design choices from either parent. Mutation independently changed each design choice with probability 0.25. Backpropagation trained every candidate on two scaffold-separated development folds for at most 35 epochs, producing 192 screening runs.

The four strongest evolved structures and the canonical reference advanced to confirmation using five scaffold folds, two training seeds and at most 80 epochs. These 50 additional runs brought the total to 242. Development MA-ST-RAE supplied the genetic fitness used for model selection, while RMSE was the secondary diagnostic. Five CUDA workers evaluated independent candidates at the same time.

The selected genome used a FitzHugh-Nagumo transition rule, 128 recurrent generations, 24 hidden channels, comprehensive atom features, multiscale trajectory pooling, unit degree normalisation, dynamic observables and multiscale transition-energy summaries. It did not use chemical-feature gating, initial-state anchoring, channel-specific timescales, recurrence at multiple generation lags, temporal extrema or directional-flux summaries. Backpropagation trained its continuous Graph-CA weights, and the analytic ridge readout remained differentiable. The sealed holdout and blind labels were excluded from genetic fitness, parent selection, crossover, mutation, early stopping and confirmation.

#### 2.4.5. Genetic programming of the cellular automaton transition rule

The leading method, GP-CIA-EA-CV-CYP-GCA, used genetic programming to evolve executable mathematical expressions for the CYP3A4 transition rule (Figure 3D). All other parts of the model retained the configuration selected by the hybrid genetic search: 128 recurrent generations, 24 hidden channels, comprehensive atom features, multiscale trajectory pooling, dynamic observables and multiscale transition-energy summaries. Holding these choices fixed allowed the experiment to isolate changes in the mathematical transition rule from changes in graph encoding, state dimension or training procedure.

Each candidate program could combine four inputs, called *terminals*: the learned reaction signal rᵢ⁽ᵗ⁾, current state hᵢ⁽ᵗ⁾, difference between the bonded-neighbour mean and current state h̄ᵢ⁽ᵗ⁾−hᵢ⁽ᵗ⁾, and difference between a delayed state and current state hᵢ⁽ᵗ⁻ᵈ⁾−hᵢ⁽ᵗ⁾. The permitted mathematical operations, called *primitives*, were addition, subtraction, multiplication, averaging, negation, hyperbolic tangent, sine and multiplication by a fixed scale. These operations were bounded and differentiable where required for training.

Candidate programs were expression trees, in which operations form branches and inputs form terminal leaves. We limited trees to depth four. Tournament selection chose parents. Subtree crossover exchanged complete parts of two parent expressions; subtree mutation replaced one part with a newly generated expression; and point mutation changed an individual operation or input. A small penalty for expression size discouraged additional complexity unless it improved prediction error. We screened four genetic generations of 24 programs on two scaffold folds. Five finalists then underwent five-fold confirmation with two seeds, producing 242 CUDA training runs.

Each symbolic candidate supplied one transition rule for a complete model evaluation. That rule remained unchanged while generating trajectories for every molecule in its training and validation folds; it did not mutate within an individual molecular trajectory. During each candidate evaluation, backpropagation through time trained the continuous bond-message, reaction and projection weights. The differentiable ridge solve fitted the prediction readout from support observations. Scaffold-validation MA-ST-RAE supplied the candidate fitness, and RMSE was a secondary diagnostic. Genetic programming therefore chose how the rule combined its local inputs, while backpropagation learned the numerical weights inside the resulting model.

The selected program was the depth-one expression

```math
u_i^{(t)}=\frac{1}{2}\left(r_i^{(t)}+\bar h_i^{(t)}-h_i^{(t)}\right),
```

This depth-one expression takes the average of two H-channel vectors: the learned reaction drive and the bonded-neighbour difference. The latter is the mean state of the atom's bonded neighbours minus the atom's own state. Repeatedly applying this evolved expression generated the molecular space-time trajectory. Multiscale pooling then converted that trajectory into the fingerprint used by the differentiable ridge readout. This procedure follows the established genetic-programming principle of evolving executable symbolic structures through selection and recombination [8].

![Four strategies for learning molecular space-time representations](figures/03_four_molecular_spacetime_learning_strategies.png)

**Figure 3. Four strategies for learning molecular space-time representations.** Every strategy begins with an encoded molecular graph. The Graph-CA produces a recurrent trajectory, pooling converts that trajectory into a molecular fingerprint, and an analytic ridge readout fitted on support observations predicts pIC50 for query observations. **A**, backpropagation through time carries query error through the ridge solution and every recurrent generation; Adam then updates the continuous Graph-CA parameters within the selected transition-rule family. **B**, the evolutionary strategy compares equal and opposite parameter perturbations. It fits a ridge readout for every candidate and estimates parameter updates from differences in query-error fitness, using Adam-style running averages. **C**, the hybrid genetic search evolves genomes containing discrete design choices, including rule family, trajectory depth, channel count, feature profile and pooling design. Backpropagation fits the continuous parameters of each candidate through the differentiable ridge solution, after which scaffold-held-out development performance determines genetic selection. **D**, genetic programming uses the same internal gradient-based fitting while evolving symbolic transition-rule trees through selection, crossover and mutation. Development MA-ST-RAE supplies the main fitness measure for the genetic searches. Solid blue arrows show forward calculation, dashed purple arrows show gradient-based learning, and green arrows show evolutionary updates or genetic search. Within a trajectory, the selected transition rule remains the same for every atom and generation; genetic changes occur only between candidate models. Learning and model selection exclude the sealed-holdout and blind labels.

#### 2.4.6. Final validation and blinded inference

Model development used five scaffold-grouped folds drawn from the fitting pool. Structurally related molecular frameworks were therefore kept within the same fold. We opened the previously defined sealed holdout only once, after selecting the experts and the sparse endpoint-specific ridge combinations. The hybrid genetic CYP3A4 structure and genetic-programming rule were fixed after five-fold development confirmation with two seeds. Each was then evaluated once on the sealed holdout. We used the primary and complementary metrics defined in Section 2.1 and estimated final uncertainty with 1,000 bootstrap resamples.

After all model parameters had been fixed, checkpoints trained across five scaffold folds and two seeds generated predictions for every blinded molecule-CYP pair under each selected rule. We first averaged the predictions from checkpoints using the same rule. The saved endpoint-specific ridge combination then produced the final pIC50 prediction.

HGS-CIA-EA-CV-CYP-GCA retained the frozen fixed-rule predictions for CYP1A2, CYP2C9 and CYP2D6, but replaced CYP3A4 with the ten-checkpoint specialist selected by the hybrid genetic search. GP-CIA-EA-CV-CYP-GCA used the same three frozen fixed-rule endpoint ensembles and replaced CYP3A4 with the ten-checkpoint genetic-programming ensemble. Concealed challenge labels were never loaded during development or local validation.

### 2.5. Long-Horizon Dynamical Analysis

#### 2.5.1. Selecting trajectories for tests of convergence, oscillation and chaos

We analysed the dynamics only after predictive training had finished. Model parameters and molecular graphs were frozen, so these tests observed the behaviour of already trained systems and could not change their predictions or validation scores. We repeatedly applied each trained transition rule and tracked the numerical state of every atom.

We used several broad descriptions of behaviour. *Convergence* means that a trajectory approaches a fixed state. *Periodic behaviour* means that it repeats a sequence of states. *Persistent complex variation* means that change continues without an immediately established classification. A *bounded* trajectory remains within a finite numerical range. A positive Lyapunov exponent indicates that an initially small difference between two states grows exponentially on average. More than one positive exponent indicates expansion along several independent directions, termed hyperchaotic dynamics over the measured interval.

A point in the model's state space contains every numerical channel value for every atom. The distance between two such points therefore compares their complete atom-by-channel states. Some transition rules also use an auxiliary variable, such as velocity, or retain earlier states for delayed memory. These additional quantities are part of the complete dynamical state because they are required to calculate the next generation.

For detailed analysis, we prioritised trajectories that continued to move, returned near previously visited regions, contained variation across a broad range of frequencies, or responded strongly to a small change in initial state. This selection identified informative dynamical examples but did not alter the trained regression models or their validation scores.

The initial screen used complete atom-by-channel validation trajectories. For each trajectory Hₜ, we first calculated the mean state across all atoms at every generation. *Late-time motion* was the mean Euclidean distance moved between successive states during the second half of the observed trajectory. We then assessed *recurrence*, meaning return near an earlier state, over separations of 2 to 64 generations. For each separation, we compared the mean distance between lagged states with the distance expected by accumulating the average step length. The smallest normalised lagged distance was the recurrence ratio; a smaller value indicated a closer return relative to ordinary movement.

We also calculated spectral entropy. A Fourier transform separates a changing signal into frequencies. Spectral entropy measures how broadly the signal's power is distributed across its non-zero frequencies: low values indicate concentration in a small number of frequencies, whereas higher values indicate a broader frequency distribution. We calculated it for every state channel and then averaged across channels. The screening score combined continued motion, frequency diversity and recurrence:

```math
S = \frac{v_{\mathrm{late}}\left(1 + H_{\mathrm{spectral}}\right)}{R_{\mathrm{recurrence}}},
```

Here, v[late] is late-time motion, H[spectral] is normalised spectral entropy, and R[recurrence] is the recurrence ratio. A trajectory received a high score when it continued to move, contained a broad range of frequencies and returned relatively close to earlier states. The score ranked trajectories for the more detailed tests of chaos described below. We selected candidates from the gated residual, delayed memory, inertial reaction-diffusion, Kuramoto-Sakaguchi and FitzHugh-Nagumo rule families. The resulting set included convergence, oscillation, repeated visits to similar states and separation of nearby trajectories.

We propagated 20 complete molecular states for 5,000 generations using frozen Graph-CA parameters. Every channel of every atom was retained at every generation. The first 1,000 generations were treated as *burn-in* and excluded from late-time measurements, allowing initial transient behaviour to pass before analysis. We examined ten representative cases in greater detail using principal-component projections, recurrence plots, frequency spectra, estimates of correlation dimension and curves describing the separation of neighbouring trajectories.

Principal-component analysis provides lower-dimensional coordinates that capture major patterns of variation. We used these coordinates only to display the trajectories. All perturbation and Lyapunov calculations used the complete atom-by-channel state, so dimensional reduction did not remove information from the quantitative dynamical tests.

#### 2.5.2. Direct perturbation screen

For each detailed case, we created eight companion states by making small changes in eight independently chosen directions through the complete state space. Every change had Euclidean magnitude 10⁻⁵. We then advanced the unchanged reference state and each perturbed companion under exactly the same frozen transition rule, molecular graph, bond features, atom features and CYP context. At every generation, we recorded the distance between the reference and companion states.

Candidates were prioritised when companions repeatedly moved away from the reference across several perturbation directions. We then tested whether this separation continued after repeatedly returning the companion to a small distance from the reference. Trajectories 7 and 8 showed the clearest reproducible expansion and advanced to the confirmatory tests. These trajectories represented `OCNT-0494110` with CYP2C9 context and `OCNT-2328784` with CYP1A2 context, respectively.

#### 2.5.3. Circular state-space distance

The Kuramoto-Sakaguchi rule represents a phase within the interval [−1,1]. The two ends of this interval correspond to the same point in the repeating cycle, just as −π and π represent the same direction on a circle. An ordinary numerical subtraction would incorrectly treat states close to opposite ends as far apart. We therefore measured their shortest separation around the circle. For reference state h and companion state h', the circular difference in each channel was

```math
\Delta(h',h) = \frac{1}{\pi}\mathrm{atan2}\!\left[
\sin\!\left(\pi(h'-h)\right),
\cos\!\left(\pi(h'-h)\right)
\right],
```

The sine and cosine convert the numerical difference into a circular direction, and atan2 returns its signed shortest angle. Dividing by π returns the result to the model's state scale. The full-state separation d=‖Δ(h',h)‖₂ is the Euclidean length of all these elementwise circular differences. It treats states near opposite ends of the interval as nearby phases and avoids an artificial distance caused only by the wrapped numerical representation. The extended analysis below separately tested whether the learned update itself remained continuous across this boundary.

#### 2.5.4. Confirmatory protocols

The table below summarises the numerical settings for the confirmatory and population experiments. A *repeat* used either an independently oriented initial perturbation or an independently constructed orthogonal perturbation basis, depending on the calculation.

| Analysis | Burn-in generations | Measured generations | Renormalization or sampling interval | Perturbation scale | Repeats |
|---|---:|---:|---|---|---:|
| Largest Lyapunov exponent | 1,000 | 4,000 | 10 | 10⁻⁴, 10⁻⁵, and 10⁻⁶ | 8 per scale and molecule |
| Float64 Lyapunov spectrum | 1,000 | 4,000 | 5, 10, and 20 | 10⁻⁷ | 2 per interval and molecule |
| Attraction basin | 1,000 | 6,000 | Sampled every 10 | Displacement radii 0.1, 0.5, 1.0, and 2.0 | 8 per radius and molecule |
| Population and structural interventions | 1,000 | 2,000 | 10 | 10⁻⁷ | 2 per system |

#### 2.5.5. Renormalized largest Lyapunov exponent

The direct perturbation screen could show that two nearby states separated once, but sustained chaos requires repeated local expansion. We therefore used a Benettin-style renormalisation calculation. After burn-in, we placed a companion state at a small distance ε from the reference state. Both states advanced for τ generations. We then measured their circular separation dₖ and recorded how much it had grown relative to ε. Before the next interval, we moved the companion back to distance ε while preserving the direction in which the separation had grown. Repeating this procedure tested whether local expansion was continually regenerated rather than caused by a single transient event.

The largest Lyapunov exponent was estimated as

```math
\lambda_1 = \frac{1}{K\tau}
\sum_{k=1}^{K}\log\!\left(\frac{d_k}{\varepsilon}\right).
```

Here, K is the number of measured intervals, τ is the number of generations in each interval, dₖ is the separation at the end of interval k, and ε is the separation imposed at its start. The logarithm converts multiplicative growth into an additive rate, and division by Kτ gives the mean rate per generation. The complete design produced 48 estimates. A positive value indicates that nearby states separated exponentially on average over the measured interval.

#### 2.5.6. Float64 Lyapunov spectrum

A single largest Lyapunov exponent measures the fastest average expansion. A *Lyapunov spectrum* extends this idea by measuring growth or contraction along several independent directions through state space. We calculated the eight leading exponents using double-precision float64 arithmetic and eight simultaneous perturbation vectors that began mutually perpendicular.

After each propagation block, the perturbations were no longer guaranteed to remain perpendicular. We assembled their full-state circular differences into a matrix and applied reduced QR decomposition. This standard matrix factorisation separates the matrix into an orthonormal basis Q and an upper-triangular matrix R. Q supplied a renewed set of perpendicular perturbation directions. The logarithms of the absolute diagonal elements of R measured local expansion or contraction along those directions. We compared results across the renormalisation intervals and repeats listed above. Stability across these settings helped distinguish persistent expansion in several directions from numerical precision effects or one isolated unstable direction.

#### 2.5.7. Boundedness and attraction-basin test

An attracting region should draw in trajectories that start from different nearby states. We tested this property by displacing the complete post-burn-in state by each radius listed above and propagating the resulting float64 trajectories. The full experiment contained 64 displaced trajectories.

We checked boundedness from the largest absolute state value reached by each trajectory. Because phases wrap around, we represented every phase by its sine and cosine. This circular embedding removes the artificial boundary between the two ends of the phase interval. We then compared how often the displaced and reference trajectories visited different parts of this embedded state space.

Direct comparison in the full high-dimensional space is difficult. We therefore projected the states onto 32 random one-dimensional directions and compared their sampled distributions along each direction. We calculated the distance from the displaced distribution to the reference distribution during both early and late portions of the trajectory. A late-to-early ratio below one indicated that the displaced trajectory had moved closer to the reference distribution. We also measured the distance from each displaced state to its nearest sampled reference state. This supplied a complementary test of whether the displaced trajectory approached the region visited by the reference.

#### 2.5.8. Population-level structure–dynamics analysis

We next tested whether molecular structure was associated with the strength of the learned instability across a broader population. We sampled 256 molecule-CYP cases that were held out by molecular scaffold: 64 cases for each of CYP1A2, CYP2C9, CYP2D6 and CYP3A4. To cover the range of observed dynamics, we sampled equally from the four quarters of the initial screening-score distribution. We then added the two leading candidates explicitly, producing 258 evaluated cases under the population protocol above.

We described each molecule using chemical, graph and three-dimensional structural measurements. The chemical measurements included molecular weight; calculated logP, an estimate of preference for oily rather than aqueous environments; topological polar surface area, a two-dimensional estimate of the surface associated with polar atoms; counts of hydrogen-bond donors and acceptors; rotatable bonds; ring composition; the proportions of aromatic atoms and atoms other than carbon or hydrogen; formal charge; the proportion of carbon atoms with sp3 hybridisation; and counts of each bond type.

The graph measurements described molecular connectivity without using three-dimensional coordinates. They included atom and bond counts; cyclomatic number, which counts independent cycles; graph density; summaries of the number of neighbours per atom; adjacency spectral radius; algebraic connectivity; the largest Laplacian eigenvalue; graph diameter, which is the longest shortest path between any two atoms; and mean shortest-path length. The spectral quantities summarise connectivity using eigenvalues of matrices constructed from the molecular graph.

We also generated reproducible three-dimensional conformations using the ETKDG procedure with random seed 260830, followed by relaxation with the Merck Molecular Force Field (MMFF). From these conformations we calculated radius of gyration, asphericity, eccentricity, inertial shape factor and spherocity, which describe molecular size and shape. Cartesian coordinates were not supplied to the Graph-CA. We therefore treated these three-dimensional measurements as structural correlates, not as direct inputs capable of determining its dynamics.

For each descriptor separately, we measured association with the largest Lyapunov exponent using Spearman rank correlation. The Benjamini-Hochberg procedure corrected for testing many descriptors within the same analysis. We obtained confidence intervals from 2,000 bootstrap resamples in which complete Bemis-Murcko scaffold groups, rather than individual molecules, were resampled together. This preserved dependence among molecules sharing the same central ring-and-linker framework.

We then tested whether combinations of descriptors gave reproducible predictions using Elastic Net and Extra Trees regression with five-fold scaffold-grouped cross-validation. Elastic Net is a penalised linear model, whereas Extra Trees combines many randomly varied decision trees. For each descriptor, we shuffled its held-out values and measured the resulting change in predictive performance. A larger decrease indicated a larger contribution. Scaffold separation was preserved between every training and evaluation fold.

#### 2.5.9. Frozen-model structural interventions

We performed controlled computational interventions on trajectories 7 and 8 to test whether particular structural inputs caused changes in the learned instability. All trained weights, the transition rule, CYP context and Lyapunov-calculation settings remained fixed. Because messages travel in both directions across a chemical bond, each undirected bond was represented by two directed message-passing edges. Deleting a bond removed both directions.

For a bond-identity intervention, we retained the connection but replaced its encoding as single, double, triple or aromatic with each alternative type. These categories describe how atoms share electrons and whether the bond belongs to an aromatic electron system. A ring-opening intervention removed one selected connection within a molecular ring and also removed the atom feature stating that the affected atoms belonged to that ring. Atom-feature interventions removed one interpretable group at a time: elemental identity; charge and aromaticity; hybridisation; local valence, which describes an atom's bonding capacity and current bonding pattern; hydrogen-bond donor or acceptor state; chirality; or the chemical properties of neighbouring atoms.

The campaign evaluated 187 baseline and modified systems using the population protocol. Effect size was the change in the mean largest Lyapunov exponent relative to the corresponding unmodified molecule-CYP system. A positive value meant that nearby trajectories separated faster after the intervention; a negative value meant that they separated more slowly. The workflow, numerical tables and publication figures are retained in the [structure-dynamics campaign archive](../results/structure_dynamics_publication_v1/README.md).

#### 2.5.10. Extended analysis of the two strange-attractor candidates

We extended the analysis of the two original candidates, `OCNT-2328784` with CYP1A2 context and `OCNT-0494110` with CYP2C9 context. Both used the frozen Kuramoto-Sakaguchi checkpoint. We made no further changes to its trained parameters or transition rule.

The first extended test used double-precision float64 trajectories with 5,000 burn-in generations followed by 20,000 measured generations. We propagated eight analytic tangent vectors alongside each trajectory. A tangent vector represents an infinitesimally small displacement and the local Jacobian describes how one update transforms that displacement. Every ten generations, QR decomposition restored eight mutually perpendicular directions. We verified this implementation in two ways: its one-step state update agreed with the original model, and its analytic Jacobian-vector product agreed with automatic differentiation to within 10⁻¹² at the checked initial states.

We also estimated the largest exponent using finite rather than infinitesimal separations. The three repeatedly renormalised starting distances were 10⁻⁵, 10⁻⁷ and 10⁻⁹. We examined the estimates in four consecutive windows of 5,000 generations to test whether expansion persisted through time. We then calculated the complete Lyapunov spectrum over a further 10,000 generations, after allowing 1,000 generations for the tangent basis to align with the local dynamics. Unlike the earlier leading spectrum, the complete spectrum included one exponent for every independent state-space direction.

The sum of all Lyapunov exponents measures whether a small volume of nearby states grows or shrinks on average along a trajectory. We also recorded Kaplan-Yorke estimates, which use the ordered exponent spectrum to infer an effective dimension. Interpreting this estimate as a fractal dimension requires additional assumptions about the underlying dynamics.

For each reference trajectory, we also propagated four additional displaced states. Each used an independent direction and one of the radii 0.1, 0.5, 1.0 or 2.0. We represented the full wrapped state using sine and cosine coordinates and compared sampled distributions over 32 random projections. Each late distribution was compared with the distance between two consecutive windows of the reference trajectory. Both reference windows contained 500 states sampled at intervals of ten generations.

The distance between consecutive reference windows estimates how much the sampled distribution varies even within the unperturbed reference trajectory. Comparing displaced trajectories with this internal variation provides a scale for interpreting the earlier attraction-basin test: a difference similar to the reference's own window-to-window variation is weaker evidence of distinct long-term behaviour than a much larger difference. The wrapped construction already bounds the numerical phases, so this extended boundedness test served as a consistency check.

Finally, we tested the sensitivity of the correlation-dimension analysis. We used 2,001 sampled states and drew 100,000 random pairs before excluding pairs that were too close in time. A *Theiler window* performs this exclusion: windows of 100, 500 and 1,000 generations prevented successive or near-successive states from dominating the geometric estimate. We repeated the calculation in projections containing 6, 12 and 24 components and in the complete circular embedding [14–16].

Fractal dimension describes how the number of regions required to cover a set changes as the measurement scale becomes finer. The correlation integral estimates this property by measuring how the fraction of state pairs lying within a given distance changes with that distance. A stable fractal-dimension estimate should appear as a consistent scaling region. We therefore retained the local slopes and the radius ranges used for fitting, allowing us to test whether such a stable region existed.

We also examined the phase boundary, where the numerical representation wraps from one end of the phase interval to the other. We placed one state coordinate on opposite sides of this boundary at circular separations ranging from 2 × 10⁻⁴ to 2 × 10⁻¹⁰ and measured the separation after one update. For reproducibility and later figure preparation, we archived every reference atom-by-channel state, displaced trajectories sampled every ten generations, molecular bond connectivity, chemical features and diagnostic curves.

## 3. Results

### 3.1. Internal Predictive Evaluation

We evaluated the fixed-rule CIA-EA-CV-CYP-GCA reference once on the sealed scaffold holdout. This holdout contained 1,309 observed molecule-CYP pairs. The point estimate of MA-ST-RAE was 0.748490, and the root mean squared error (RMSE) was 0.847738 pIC50. We also drew 1,000 bootstrap samples by repeatedly resampling the holdout observations. Across these samples, the mean MA-ST-RAE was 0.749064, with a 95% interval from 0.710791 to 0.789424. The individual CYP ST-RAE values were 0.816136 for CYP1A2, 0.713018 for CYP2C9, 0.936774 for CYP2D6, and 0.528032 for CYP3A4.

The evolutionary-strategy campaign completed 114 development runs. In the matched two-fold screen, its strongest configuration achieved an MA-ST-RAE of 0.638768 and an RMSE of 0.803659 pIC50. The fixed Graph-CA trained by backpropagation achieved a lower MA-ST-RAE of 0.626897 on the same screen. We then confirmed the evolutionary configurations using five folds and two random seeds. This stage selected a population of 128 candidate parameter sets, a perturbation scale of 0.02, an evolutionary learning rate of 0.01, batches of 256 molecules balanced by activity, and the patient stopping policy. Its mean confirmation MA-ST-RAE was 0.659505, and its mean RMSE was 0.817985 pIC50. This experiment showed that the recurrent Graph-CA could be trained without derivatives. Its development score, however, was not strong enough to justify evaluation on the sealed or blind data.

The hybrid genetic campaign completed 242 runs. The selected genome specified a FitzHugh-Nagumo rule, 128 generations, 24 hidden channels, the comprehensive set of atom features, summaries at several spatial and temporal scales, measurements of the trajectory dynamics, and transition-energy summaries at several scales. Across five folds and two random seeds, its development MA-ST-RAE was 0.596981 ± 0.042987, and its RMSE was 0.781116 ± 0.035890 pIC50. On the sealed holdout, HGS-CIA-EA-CV-CYP-GCA achieved a point MA-ST-RAE of 0.743723 and an RMSE of 0.841572 pIC50. Its CYP3A4 ST-RAE was 0.508964.

The genetic-programming campaign also completed 242 runs. It selected the cellular automaton transition rule `mean(reaction, neighbour_delta)`, which averages the atom's reaction signal and its state difference from bonded neighbours. Across five folds and two random seeds, this rule achieved a development MA-ST-RAE of 0.598075 and an RMSE of 0.772706 pIC50. On the sealed holdout, its point MA-ST-RAE was 0.745824 and its RMSE was 0.848265 pIC50. Its CYP3A4 ST-RAE was 0.517368. Across 1,000 bootstrap samples, the mean MA-ST-RAE was 0.746602, with a 95% interval from 0.708890 to 0.788464. Its sealed MA-ST-RAE therefore lay between those of the fixed-rule and genetic-structure systems. It also provided the lowest development RMSE among the genetic searches compared here.

**Table 3. Internal comparison of Graph-CA learning strategies.** Development values describe scaffold-held-out model selection and are not directly interchangeable across all rows because the fixed-rule and evolutionary-strategy figures use the matched CYP3A4 screen, whereas the genetic searches use five-fold, two-seed confirmation. Sealed values describe the four-endpoint ensemble.

| Method | Local-rule treatment | Development MA-ST-RAE | Sealed MA-ST-RAE | Sealed RMSE, pIC50 |
|---|---|---:|---:|---:|
| CIA-EA-CV-CYP-GCA | Fixed rule families, continuous weights by backpropagation | 0.626897 on matched CYP3A4 screen | 0.748490 | 0.847738 |
| ES-EA-CV-CYP-GCA | Fixed rule and structure, continuous weights by evolution strategy | 0.659505 ± 0.0430 | Not advanced | Not advanced |
| HGS-CIA-EA-CV-CYP-GCA | Genetic algorithm evolves discrete structure | **0.596981 ± 0.0430** | **0.743723** | **0.841572** |
| GP-CIA-EA-CV-CYP-GCA | Genetic programming evolves the mathematical form of the cellular automaton transition rule | 0.598075 | 0.745824 | 0.848265 |

### 3.2. Blind Challenge Evaluation

After each submission, the challenge organisers compared our fixed predictions with experimentally measured values. These measurements remained hidden from us throughout training, model selection, sealed evaluation and prediction. Table 4 gives the official results. We omit leaderboard positions because they can change when other teams submit new results. All three submissions predicted the same four CYP values for each of 750 blinded molecules.

**Table 4. Official blind OpenADMET CYP Inhibition Challenge results for the three submitted Graph-CA variants.** Lower values are favourable for MA-ST-RAE and macro MAE; higher values are favourable for macro R-squared, Spearman rho, and Kendall tau.

| Official blind metric | Fixed-rule CIA-EA-CV-CYP-GCA | Genetic algorithm HGS-CIA-EA-CV-CYP-GCA | Genetic programming GP-CIA-EA-CV-CYP-GCA |
|---|---:|---:|---:|
| MA-ST-RAE | 1.0092 | 1.0020 | **0.9946** |
| Macro MAE, pIC50 | 1.0789 | 1.0745 | **1.0688** |
| Macro R-squared | -0.0744 | -0.0724 | **-0.0656** |
| Macro Spearman rho | 0.5345 | 0.5444 | **0.5498** |
| Macro Kendall tau | 0.3762 | 0.3840 | **0.3878** |

Genetic programming produced the strongest blind result for every reported measure. Compared with the fixed-rule reference, its MA-ST-RAE decreased by 0.0146 and its macro mean absolute error (MAE) decreased by 0.0101 pIC50. Its Spearman rank correlation increased by 0.0153, and its Kendall rank correlation increased by 0.0116. The hybrid genetic structure search lay between the other two systems on all five measures. The improvement was modest but occurred consistently in both the error and rank-correlation measures.

### 3.3. Nonlinear Dynamics in Molecular Space-Time

![Four contrasting Graph-CA dynamical behaviours](../results/long_horizon_attractor_campaign_v1/figures/17_four_graph_ca_dynamical_behaviours.png)

**Figure 4. Four long-horizon Graph-CA dynamical behaviours.** **A**, the gated-residual trajectory of `OCNT-2328519`, using the CYP1A2 context, contracts towards a point attractor: a state that no longer changes. **B**, the Kuramoto–Sakaguchi hyperchaotic candidate for `OCNT-0494110`, using the CYP2C9 context, continues to explore a bounded region and repeatedly returns near earlier states. **C**, the Kuramoto–Sakaguchi trajectory of `OCNT-0495493`, using the CYP2D6 context, is classified by the initial screen as persistent or complex. It has late motion 0.148, spectral entropy 0.605, a dominant timescale of approximately 572 generations, and positive finite-time divergence. This candidate has not undergone the extended Lyapunov-spectrum and attraction-basin tests applied to panel B. Before projecting the Kuramoto phase channels into the plotted coordinates, we represented each circular phase by its sine and cosine. **D**, the coupled-map trajectory of `OCNT-0495275`, using the CYP3A4 context, is a candidate period-two oscillator with spectral concentration 0.802. It alternates between two states. The panel shows the mature behaviour from generation 1,000 to generation 5,000 using a three-coordinate delay representation of the leading full-state component; magenta and cyan markers identify the alternating states. Panels A–C show generations 0 to 5,000. Colour records computational time using the colourblind-safe Viridis scale. The plotted positions describe computational state space and show how the model's numerical atom states change across generations. Cyberpunk animations show the [point attractor](../results/long_horizon_attractor_campaign_v1/videos/trajectory_01_point_attractor_convergence.mp4), [strange attractor](../results/long_horizon_attractor_campaign_v1/videos/trajectory_07_hyperchaotic_strange_attractor.mp4), [persistent or complex candidate](../results/long_horizon_attractor_campaign_v1/videos/trajectory_kuramoto_persistent_complex_candidate.mp4), and [period-two oscillator candidate](../results/long_horizon_attractor_campaign_v1/videos/trajectory_coupled_map_period2_oscillator_candidate.mp4).

![Independent evidence supporting the Graph-CA strange-attractor classification](../results/long_horizon_attractor_campaign_v1/figures/18_strange_attractor_evidence_plate.png)

**Figure 5. Independent evidence for the Graph-CA strange-attractor classification.** **A**, we repeatedly introduced a small displacement and returned it to its original size after measuring its growth. This procedure produced positive largest Lyapunov exponents at displacement sizes from 10⁻⁶ to 10⁻⁴; all 48 repeated estimates were positive. **B**, the first eight Lyapunov exponents were positive for both selected trajectories. The states therefore expanded along several independent directions. **C**, the largest exponent remained positive and stable when the displacement was returned to its original size every 5, 10 or 20 generations. **D**, all 64 repeated perturbations of the initial state remained bounded. Ratios below one show that the late distributions were closer than the early distributions across four perturbation radii, consistent with movement towards the same attracting distribution. Error bars in panels A–C show standard deviations across repeated calculations. Horizontal black lines mark zero divergence, and the dashed line in panel D marks equal late and early distance. Together, these measurements support sustained separation along several independent directions while displaced trajectories approach similar sampled distributions. Section 3.3.2 describes the further evidence needed to classify these trajectories as fractal strange attractors. A vector version is available as [PDF](../results/long_horizon_attractor_campaign_v1/figures/18_strange_attractor_evidence_plate.pdf).

![Terminal atom cascades for the four patterns of Graph-CA dynamical behaviour](../results/long_horizon_attractor_campaign_v1/figures/19_four_graph_ca_terminal_atom_cascades.png)

**Figure 6. Molecular information cascades generated by the four Graph-CA trajectories in Figure 4.** Each panel shows the complete cascade retained at generation 5,000. The corresponding two-dimensional molecular structure, drawn using RDKit, appears directly below it. At each generation, the rendering rotated the molecule slightly, moved it downwards and retained every earlier atom position. This procedure produced the continuous atom ribbons used in the accompanying molecular space-time videos. Colour from cyan to magenta records the mean state of each atom across the 16 dynamical channels at that generation, after robust normalisation within the trajectory. The cascade therefore shows how the numerical atom states change across the bonded molecular graph. Vertical position marks computational generations. **A**, gated-residual point attractor. **B**, Kuramoto–Sakaguchi hyperchaotic candidate. **C**, Kuramoto–Sakaguchi persistent or complex candidate. **D**, coupled-map period-two oscillator candidate. Bonds are omitted from the cascades so that the accumulated atom-state structure remains visible. A high-resolution PDF version is available [here](../results/long_horizon_attractor_campaign_v1/figures/19_four_graph_ca_terminal_atom_cascades.pdf).

#### 3.3.1. Population-level dynamical screen

For each cellular automaton transition rule, the final production model retained summaries of 1,309 validation trajectories. The ten rules therefore produced 13,090 short-horizon screens. We classified a trajectory as a point-attractor candidate when its late motion was below 10⁻⁴ and its final update changed the state by less than 10⁻⁵. We classified it as an oscillator candidate when its recurrence ratio was below 0.25 and its spectral concentration was above 0.5. The recurrence ratio measures how closely the trajectory returns to an earlier state, while spectral concentration measures how strongly its variation is concentrated at a small number of frequencies. All other trajectories entered the persistent or complex screening class. Each trajectory belonged to exactly one of these three screening categories. Strange-attractor assessment was a separate, more demanding procedure. It used trajectories of 5,000 generations, controlled perturbations, repeated measurement and rescaling of divergence, Lyapunov spectra, and tests of whether displaced states approached the same attraction basin.

**Table 5. Population-level classification screen for frozen Graph-CA trajectories.** Analysed hyperchaotic candidates form a tested subset of the persistent or complex screen.

| Cellular automaton transition rule | Screened | Point-attractor candidates | Oscillator candidates | Persistent or complex | Analysed hyperchaotic candidates |
|---|---:|---:|---:|---:|---:|
| Gated residual | 1,309 | 0 | 0 | 1,309 | 0 |
| Delayed memory | 1,309 | 0 | 0 | 1,309 | 0 |
| Inertial reaction–diffusion | 1,309 | 0 | 0 | 1,309 | 0 |
| Gray–Scott | 1,309 | 0 | 0 | 1,309 | 0 |
| Coupled map | 1,309 | 0 | 1,309 | 0 | 0 |
| Activator–inhibitor | 1,309 | 0 | 0 | 1,309 | 0 |
| FitzHugh–Nagumo | 1,309 | 0 | 0 | 1,309 | 0 |
| Kuramoto–Sakaguchi | 1,309 | 0 | 0 | 1,309 | 2 |
| Damped symplectic | 1,309 | 0 | 0 | 1,309 | 0 |
| Conservative graph flux | 1,309 | 1,309 | 0 | 0 | 0 |
| **Total** | **13,090** | **1,309** | **1,309** | **10,472** | **2** |

The two analysed hyperchaotic candidates came from the Kuramoto–Sakaguchi trajectories assigned to the persistent or complex class. Both showed sustained expansion of small differences. The extended analysis below distinguishes this evidence for hyperchaos from the still unresolved question of whether the attracting sets are fractal. All 1,309 coupled-map trajectories met the initial oscillator criteria. Confirming them as true oscillators will require periodicity tests over longer trajectories. The full machine-readable table is available in the [dynamical population summary](../results/long_horizon_attractor_campaign_v1/validation_dynamics_population_summary.csv).

#### 3.3.2. Extended strange-attractor analysis

Both frozen candidates used the Kuramoto–Sakaguchi cellular automaton transition rule. After a burn-in of 5,000 generations, each retained eight positive leading tangent exponents throughout the following 20,000 measured generations. A positive exponent means that an initially small difference grows in the corresponding direction. The eight positive values therefore show growth along several independent combinations of numerical atom-state values. We also calculated the largest exponent in successive windows of 5,000 generations. The estimates ranged from 0.011870 to 0.012341 per generation for `OCNT-2328784` and from 0.012169 to 0.013314 per generation for `OCNT-0494110`. Tests using finite starting separations of 10⁻⁷ and 10⁻⁹ gave estimates of 0.012109 and 0.012704 for the respective systems. These results show that the instability persisted beyond the original short-horizon analysis.

**Table 6. Extended float64 analysis of the two Kuramoto–Sakaguchi candidates.** Leading exponents use the 20,000-generation analysis; complete Lyapunov-spectrum quantities use a subsequent 10,000-generation interval. Positive counts and Kaplan–Yorke estimates are finite-time numerical quantities.

| Quantity | OCNT-2328784, CYP1A2 | OCNT-0494110, CYP2C9 |
|---|---:|---:|
| Atom-by-channel state dimension | 240 | 320 |
| Largest leading exponent, per generation | 0.012115 | 0.012641 |
| Eighth leading exponent, per generation | 0.004376 | 0.004421 |
| Positive exponents in complete spectrum | 29 | 30 |
| Sum of complete spectrum, per generation | −3.606412 | −5.144542 |
| Kaplan–Yorke heuristic | 68.19 | 73.91 |

The sum of each complete spectrum was negative. A small volume formed by nearby states therefore shrank on average as it moved along the sampled trajectory. At the same time, the individual positive exponents show that the states separated along several directions within that shrinking volume. The largest exponents in the complete spectra were 0.012118 and 0.012480, respectively. For the second system, the number of positive exponents changed from 28 to 30 between the two halves of the 10,000-generation measurement. This variation shows that some finite-time estimates remained close to zero and were not fully stable. We also followed eight newly displaced trajectories. Their late distribution distances were 0.843–1.036 times the distance between two consecutive windows of the unperturbed reference. At this sampling resolution, the displaced trajectories approached distributions that differed from the reference by about as much as the reference differed from itself between successive time windows.

Two results limit the stronger claim that these systems contain strange attractors. First, we placed two initial states on opposite sides of a phase boundary with a circular separation of approximately 2 × 10⁻¹⁰. After one update, their separations were still 0.005422 and 0.005871. The learned frequency drive uses the raw values after they have been wrapped into the numerical phase interval. Two states can therefore represent almost identical phases on opposite sides of the boundary yet produce substantially different next states.

Second, the estimated correlation dimension did not settle at a stable value. The slopes of the correlation integral increased from approximately 4.6–4.8 when the trajectory was represented by six projected coordinates to approximately 27–29 when the complete circular representation was used. The fitted ranges of distance covered less than 0.3 logarithmic decades. A projection represents the trajectory using fewer coordinates than the full state. Because the estimate continued to change when we changed the representation, the tests did not identify a stable fractal dimension. The results support bounded trajectories in which small differences grow along several independent directions and displaced states later sample similar distributions. A strict strange-attractor classification requires further evidence of a robust fractal dimension [14–16]. We retained the complete states and diagnostic arrays in the [extended analysis archive](../results/strange_attractor_audit_20260929/README.md).

## 4. Discussion

### 4.1. Emergence from Local Molecular Interactions

The model builds a molecular fingerprint for predicting pIC50 from atom states recorded across the molecular graph and across successive generations. Fixed chemical descriptors define the starting state. Bonds define which atoms can exchange information. At every generation, the same cellular automaton transition rule updates each atom from its present state and messages received from directly bonded neighbours. The model then summarises the resulting trajectory and uses that summary to make a prediction. The fingerprint therefore contains information from many atoms and many computational time points. Our operational test of emergence asks whether these repeated local updates create a molecule-wide representation that contains useful predictive information [25, 26].

The blind challenge tested whether these trajectory fingerprints could predict inhibition for compounds whose measured pIC50 values remained hidden throughout model development. All three submitted systems retained a moderate correspondence between the predicted and experimental rankings. This occurred despite separating chemical scaffolds during internal model selection and never using the blind measurements. The genetically programmed transition rule gave the strongest external result and improved all five official measures. Its absolute errors and negative macro R-squared values show that the present predictive accuracy remains limited. The central result is therefore that repeated local atom-state updates can construct a fingerprint containing useful information about a molecular property. The trajectories that create this fingerprint can also be studied directly. Evolving the transition rule produced a modest further gain in blind prediction.

The connection between the local transition rule and the global prediction is central to this study. Backpropagation adjusted continuous numerical weights within transition-rule families chosen in advance. The hybrid genetic algorithm also searched choices such as model structure and which trajectory measurements to include. Genetic programming extended the search to the mathematical form of the transition rule itself. Blind performance improved in the same order: fixed rule, genetic structure search, then transition-rule search. This pattern suggests that changing how each atom processes local information changes both the trajectory and the information available to the final molecular readout. The compact rule selected by genetic programming averages the atom's reaction signal with the difference between its state and those of its bonded neighbours. For this prediction task and training procedure, that concise reaction-diffusion coupling was more useful than the larger formula selected by hand.

### 4.2. Molecular Space-Time as a Dynamical Object

Molecular space-time records how the numerical states on a molecular graph change during repeated local updates. Its coordinates are learned computational states. They are not positions in physical space or directly measured physical quantities. A coordinate does not need to correspond to one named chemical property for the complete state to carry measurable information [27]. Its operational meaning comes from how it changes messages between bonded atoms, later states, the trajectory fingerprint and the final prediction. That meaning belongs to this trained model and should not be interpreted as a direct physical measurement.

The long-horizon experiments revealed several forms of global organisation. Some frozen systems contracted towards a point attractor. Others produced candidate period-two oscillations or persistent complex motion. Two bounded trajectories generated by the Kuramoto–Sakaguchi rule repeatedly produced positive largest Lyapunov exponents. They also had several positive components in their Lyapunov spectra, and displaced states approached similar sampled distributions. The same Graph-CA framework can therefore produce different dynamical behaviours when the molecular graph, initial atom states, CYP context or transition-rule family changes.

We classified these dynamics using perturbations of the complete state, repeated measurement and rescaling of their growth, Lyapunov spectra, and repeated tests of the attraction basin [11–13]. These numerical tests separate evidence of chaos from patterns that merely appear persistent or visually complex. In the two analysed systems, nearby states separated along several independent directions while a small local volume of states shrank on average. Section 3.3.2 gives the important qualifications to this result. The larger persistent or complex class is only an initial screening category. Further work should test more candidates, measure effects caused by finite trajectory length and finite numerical resolution, resolve the discontinuity at the phase boundary, and determine whether independently trained models produce the same classifications.

### 4.3. Implications for Complex Systems and Drug Discovery

The CYP450 task is one example of a more general construction. In this construction, bonds connect atomic cells within a molecular graph. The cellular automaton transition rule generates a state trajectory for those cells. A final readout summarises that trajectory to predict a measured property of the whole molecule. Molecular graphs are useful because their local connections correspond to chemical bonds, while experimental measurements provide an external test of the resulting global information. The same framework could be tested on solubility, permeability, binding, reactivity and other properties of complete molecules. In each case, the local state dynamics would remain distinct from the global property being predicted.

For drug discovery, treating the representation as a dynamical system provides a way to study how molecular structure influences computational behaviour. We can compare known molecular properties with measurements of the trajectories. We can also change one bond at a time while keeping the trained model fixed. These tests can reveal which local graph features influence convergence, repetition or sensitivity to initial conditions. With further development, this analysis could test whether particular arrangements of atoms and bonds consistently produce convergence, oscillation, complex variation or chaos, and whether those behaviours relate to measured chemical properties. Before such patterns could be used as reliable chemical descriptors, they would need prospective validation on new datasets and independently trained models.

The evolutionary-strategy comparison shows where gradient-based optimisation was useful in this study. The derivative-free method successfully trained a working Graph-CA, but backpropagation produced the stronger score on the matched development test. Genetic search was more effective when it selected discrete model structures or transition rules while backpropagation continued to fit the continuous numerical parameters. The two methods therefore addressed different decisions. Gradients estimated how small changes in numerical weights altered the prediction error through successive generations. Population searches compared different model designs and different mathematical transition rules.

### 4.4. Limitations and Future Tests

The blind evaluation showed a moderate correspondence between the predicted and experimental rankings. This result indicates that the molecular trajectories contained predictive information. The absolute accuracy was still limited: MA-ST-RAE remained close to the baseline obtained from mean predictions, and the macro R-squared was negative. Genetic programming gave a modest improvement. Further testing is needed to determine whether that improvement is statistically reliable.

The challenge contains 4,905 labelled compounds and four related CYP prediction tasks. The blind results show that substantial prediction error remains. We evolved the transition rule only for the CYP3A4 specialist. For CYP1A2, CYP2C9 and CYP2D6, we retained the ensembles from the fixed-rule system. A larger study should evolve transition rules independently for all four CYPs. It should also test whether a rule evolved for one task can be reused successfully for another.

Our emergence claim is based on a measurable process: repeated interactions between bonded atomic cells generate trajectories, and summaries of those trajectories predict a property of the whole molecule. Stronger causal tests should measure when predictive information appears and how it accumulates over generations. Intact trajectories should be compared with controls in which time points are shuffled or communication across bonds is removed. Individual message pathways should also be removed in controlled tests, and the direction of information transfer across bonds should be measured. These experiments would show whether the gains come simply from applying many computational layers or specifically from organised local interactions.

The extended dynamical analysis followed longer trajectories and used several independent tests. It compared the growth of infinitesimal displacements with the separation of nearby finite trajectories. It used the complete Lyapunov spectrum to measure changes in a small local volume of states. It also compared displaced-trajectory distributions with the amount of variation seen between different windows of the unperturbed reference. Together, these tests support sustained hyperchaotic behaviour in two frozen systems.

A confirmed fractal-attractor claim requires more evidence. Dimension estimates must become stable across a sufficiently wide range of measurement scales. The convergence tests must cover longer trajectories, and the result must be reproduced in independently trained models. The transition rule changes smoothly within parts of the numerical phase interval, but it can jump at the boundary where the two ends of that interval represent neighbouring phases. This discontinuity complicates analyses that assume smooth motion on a circle. It also limits the interpretation of Kaplan–Yorke estimates as fractal dimensions. Longer tests using a wider range of initial states must determine whether the state distribution remains stable indefinitely, whether rare transitions occur, and which initial states lead to each long-term behaviour. Estimates of entropy rate, networks built from recurrent states, and computational-mechanics descriptions provide complementary future tests [13–16].

## 5. Conclusions

Repeated local interactions between chemically encoded atomic cells generated trajectory fingerprints that predicted CYP inhibition with a measurable correspondence between predicted and experimental rankings. The resulting molecular space-time extended across atoms and repeated generations. It supported both molecular prediction and direct analysis as a nonlinear dynamical system.

Three blind submissions tested three levels of learning: fitting numerical weights, searching model configurations, and evolving the mathematical transition rule. The fixed-rule system trained by backpropagation achieved an MA-ST-RAE of 1.0092. The hybrid genetic structure search achieved 1.0020, and genetic programming of the transition rule achieved 0.9946. The methods appeared in the same order for macro MAE, R-squared, Spearman correlation and Kendall correlation. Evolving the local transition rule therefore produced a small but consistent improvement on external data while preserving the cellular-automaton construction.

The retained trajectories included convergence towards fixed states, possible periodic behaviour and persistent complex variation. In two systems, the numerical states remained bounded while small initial differences grew along several independent directions. At the same time, small local volumes in state space contracted on average. These findings support hyperchaotic dynamics. Their classification as strange attractors remains provisional because the transition rule jumps at the phase boundary and the estimated dimension changes with the representation of the trajectory.

This range of global behaviour arose from local interactions constrained by molecular bonds. Molecular space-time therefore provides a framework for connecting chemical graph structure, emergent dynamics and molecular-property prediction. Future work should evolve rules for all four CYP tasks, test how communication between bonded cells creates predictive information, and determine which patterns of atom-state behaviour recur across molecular classes and independently trained models.

## Author Contributions

A.N. conceived the study, developed and implemented the computational methodology, conducted the experiments, analysed and interpreted the results, prepared the figures, and wrote and revised the manuscript. A.N. used AI to prepare the code.

## Funding

This research received no external funding.

## Data Availability

The 2026 OpenADMET CYP Inhibition Challenge provided the dataset that we analysed in this study; the challenge organiser's access conditions govern its availability.

## Code Availability

Source code, trained model configurations and scripts required to reproduce the reported analyses are available at https://github.com/acnash/Strange-Matter-Engine.git

## Appendix A. Mathematical Definition of the Graph Cellular Automaton

This appendix gives the complete mathematical specification and accompanying explanation of the Graph-CA described conceptually in Section 2.3.

For each atom i, the vector xᵢ listed fixed chemical features such as its element, charge, aromaticity, hybridisation and number of bonds. A trainable initialisation transformation combined these fixed chemical features to produce H starting values for each atom. On the first training pass, this transformation used its initial parameter values; subsequent training adjusted those values according to prediction error. The same transformation was applied to every atom. Each starting value was a continuous number that could be negative, positive or fractional. Together, the H values defined the atom's initial position in the model's computational state space. These values then changed over successive generations. Training determined how the chemical features were combined, so individual state values were not assigned fixed chemical meanings:

```math
h_i^{(0)}=\tanh\!\left(s_0 W_{\mathrm{init}}x_i\right),
```

The equation describes how the model created these H starting values. Wᵢₙᵢₜ is a trainable matrix: each of its rows forms a weighted sum of the F chemical features in xᵢ, producing one of the H values. The superscript (0) identifies generation zero and is not the number of channels. Thus, hᵢ⁽⁰⁾ ∈ ℝᴴ means that atom i has a vector of H continuous numerical values at generation zero. The scalar s₀, called the *initialisation scale*, controls the size of the weighted sums before the hyperbolic tangent is applied. The hyperbolic tangent converts each result to a value between −1 and 1. Large inputs approach these limits, where further changes to the input have progressively less effect; this is referred to as saturation. Figure 1B illustrates this sequence of weighted summation, scaling and bounding.

After forming the initial state, we applied the cellular automaton transition rule T times. Each application produced the next generation. We call T the *trajectory depth* and H the *state width*. A greater trajectory depth allows more rounds of communication between bonded atoms. A greater state width gives each atom more numerical values in which to represent the information it receives. The model used the same transition-rule parameters for every atom and every generation. At each update, an atom combined its own state with information from atoms directly connected to it by chemical bonds. Repeating this local calculation across the molecular graph produced the recurrent Graph-CA trajectory.

The first step in an update was to calculate the message sent from a neighbouring atom j to atom i at generation t:

```math
g_{ji}=\sigma\!\left(\frac{W_g e_{ji}}{\theta_b}\right),
\qquad
m_{ji}^{(t)}=g_{ji}\odot W_n h_j^{(t)}+W_e e_{ji},
```

The first expression calculates a *bond gate* gⱼᵢ, which controls how strongly neighbour j contributes to each of atom i's H channels. The vector eⱼᵢ describes the chemical bond connecting the two atoms. W[g] forms learned weighted sums of those bond features, and the positive parameter θ[b], called the *bond temperature*, controls how sharply the gate responds. Dividing by a smaller θ[b] moves gate values towards their extremes, so a contribution is more strongly passed or suppressed. A larger θ[b] moves the values towards one half, making the gate less decisive. The logistic sigmoid σ(z)=1/(1+exp(−z)) converts every gate value to the interval between zero and one.

The second expression calculates the message mⱼᵢ⁽ᵗ⁾. The vector hⱼ⁽ᵗ⁾ is neighbour j's H-channel state at generation t. Wₙ, where n denotes *neighbour*, forms learned weighted sums of this state. The symbol ⊙ means that corresponding elements are multiplied, so each bond-gate value scales the matching channel of the transformed neighbour state. Wₑ forms a further learned contribution directly from the bond features. In this context, a learned *linear map* such as W[g], Wₙ or Wₑ is a matrix of adjustable weights that forms new values by taking weighted sums of its inputs.

An atom may have more than one bonded neighbour. We therefore averaged its incoming messages and neighbouring states over its number of neighbours. This degree normalisation prevented atoms with more bonds from receiving larger totals merely because they had more neighbours:

```math
a_i^{(t)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}m_{ji}^{(t)},
\qquad
\bar h_i^{(t)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}h_j^{(t)}.
```
Here, N(i) is the set of atoms directly bonded to atom i, and |N(i)| is the number of atoms in that set. The vector aᵢ⁽ᵗ⁾ is the mean incoming message. The state hᵢ⁽ᵗ⁾ belongs to atom i itself, whereas h̄ᵢ⁽ᵗ⁾ is the mean state of its directly bonded neighbours.

The model next calculated a proposed change, called the *common reaction drive*. At every generation, this calculation combined the atom's present state, its mean incoming message, its fixed chemical features and the identity of the CYP isoform being predicted:

```math
r_i^{(t)}=\tanh\!\left(
W_s h_i^{(t)}+a_i^{(t)}+W_x x_i+W_c c+b
\right).
```

The resulting vector rᵢ⁽ᵗ⁾ ∈ ℝᴴ contains one proposed change for each of the H channels. Wₛ forms learned weighted sums of the atom's present state hᵢ⁽ᵗ⁾; Wₓ does the same for its fixed chemical features xᵢ; and W[c] transforms the CYP context vector c. The mean incoming message aᵢ⁽ᵗ⁾ adds information that has already been conditioned by the chemical bonds. The learned bias b ∈ ℝᴴ provides one adjustable offset for each channel. The hyperbolic tangent bounds the resulting reaction drive between −1 and 1.

The model still requires a rule for combining this proposed change with the current state. A *rule family* specifies the mathematical form of that cellular automaton transition. Training adjusts the numerical parameters within the chosen form. The five rule families below use the same bond-message and reaction-drive calculations, but they differ in how they combine the present state, neighbour information and any additional memory variables.

**Gated residual.** This rule was inspired by gates in recurrent neural networks [17]. For every atom and channel, it calculates how much of the current value to retain and how much of the proposed reaction to accept (Figure 2A). The model recalculates this choice from the local molecular state at every generation:

```math
\alpha_i^{(t)}=\sigma\!\left(W_\alpha
[h_i^{(t)},a_i^{(t)},x_i,c]\right),
\qquad
h_i^{(t+1)}=(1-s\alpha_i^{(t)})\odot h_i^{(t)}
+s\alpha_i^{(t)}\odot r_i^{(t)},
```

The acceptance gate αᵢ⁽ᵗ⁾ ∈ (0,1)ᴴ contains one value between zero and one for each channel. To calculate it, square brackets join, or concatenate, four vectors: the current atom state hᵢ⁽ᵗ⁾, mean incoming message aᵢ⁽ᵗ⁾, fixed chemical features xᵢ and CYP context c. W[α] transforms this combined vector into the H gate values. The positive parameter s is the update scale. We capped sαᵢ⁽ᵗ⁾ at one in each channel; the 1 in the equation is therefore an H-component vector of ones. The symbol ⊙ again denotes elementwise multiplication. The second expression then forms each new channel value as a weighted mixture of its previous value and the proposed reaction rᵢ⁽ᵗ⁾.

**Inertial reaction–diffusion.** Classical reaction–diffusion systems combine changes produced locally by a reaction with the movement of material through space [4]. Our graph version uses the same mathematical idea, but *diffusion* means exchange of numerical state between bonded atoms rather than the physical movement of chemical material. We added a velocity-like variable that carries part of a previous change into the next generation, providing a form of momentum [18] (Figure 2B). The reaction term drives local change, the graph-diffusion term exchanges state with bonded neighbours, and restoring and damping terms limit growth:

```math
f_i^{(t)}=r_i^{(t)}+D\odot(\bar h_i^{(t)}-h_i^{(t)})-R\odot h_i^{(t)},
```

```math
v_i^{(t+1)}=\eta\gamma\odot v_i^{(t)}+\delta\odot f_i^{(t)},
\qquad
h_i^{(t+1)}=\tanh\!\left(h_i^{(t)}+\delta\odot v_i^{(t+1)}\right).
```

The vector fᵢ⁽ᵗ⁾ ∈ ℝᴴ is the total force-like drive on atom i. It has three parts. The common reaction drive rᵢ⁽ᵗ⁾ proposes local change. The difference h̄ᵢ⁽ᵗ⁾−hᵢ⁽ᵗ⁾ compares the mean neighbour state with the atom's own state and therefore provides the graph-diffusion gradient. The non-negative H-channel vectors D and R set the strengths of diffusion and restoration, respectively; the restoring term draws the state back towards zero.

The auxiliary velocity vᵢ⁽ᵗ⁾ ∈ ℝᴴ carries change between generations and was set to zero at the start of each trajectory. The H-channel vector γ controls how strongly the previous velocity is damped, while δ gives the step size for each channel. The scalar η sets the overall strength of inertia. Sigmoid or softplus transformations kept these quantities within stable ranges; the softplus transformation was used where a parameter had to remain positive. Finally, tanh bounded the updated state.

**FitzHugh–Nagumo.** The classical FitzHugh–Nagumo model describes an excitable system using a fast-changing excitation variable and a slower recovery variable [19, 20]. Their interaction can produce threshold responses, pulses and oscillatory return towards rest. We adapted this idea by dividing each atom's H channels into an excitation half u and a recovery half v (Figure 2C). Learned chemical input and exchange with bonded neighbours make these behaviours specific to the molecule:

```math
\Delta u_i^{(t)}=u_i^{(t)}-\frac{(u_i^{(t)})^3}{3}-v_i^{(t)}
+\kappa_s\tanh(W_u r_i^{(t)})+D_u(\bar u_i^{(t)}-u_i^{(t)}),
```

```math
\Delta v_i^{(t)}=\epsilon\left[u_i^{(t)}+q-v_i^{(t)}+0.1\tanh(W_v r_i^{(t)})\right]
+D_v(\bar v_i^{(t)}-v_i^{(t)}),
```
We then updated the two channel groups together:

```math
h_i^{(t+1)}=\tanh\!\left(
[u_i^{(t)}+s\Delta u_i^{(t)},\;v_i^{(t)}+s\Delta v_i^{(t)}]
\right).
```

The vectors uᵢ⁽ᵗ⁾ and vᵢ⁽ᵗ⁾ each contain H/2 values and together form the complete state hᵢ⁽ᵗ⁾. The neighbour means ūᵢ⁽ᵗ⁾ and v̄ᵢ⁽ᵗ⁾ average the corresponding half-states over the atoms in N(i). The quantities Δuᵢ⁽ᵗ⁾ and Δvᵢ⁽ᵗ⁾ are the proposed changes to excitation and recovery. Wᵤ and Wᵥ transform the H-channel reaction drive into inputs of the appropriate H/2-channel size.

The scalar κₛ controls the strength of chemical stimulation. The non-negative parameters Dᵤ and Dᵥ control graph diffusion for excitation and recovery. The positive parameter ε sets the recovery timescale, q shifts the recovery response, and 0.1 is the fixed coefficient applied to the learned recovery drive. The positive parameter s is the numerical integration step that advances both halves. Square brackets concatenate the updated excitation and recovery vectors to reconstruct the H-channel state, and tanh bounds its values.

**Kuramoto–Sakaguchi.** A phase oscillator represents a repeating cycle by an angle. Angles separated by one complete turn describe the same point in the cycle, so the phase wraps around rather than increasing without limit. Following coupled phase-oscillator theory [21, 22], this rule converts every state channel to a phase using φᵢ⁽ᵗ⁾=πhᵢ⁽ᵗ⁾ (Figure 2D). Interaction through the molecular bonds encourages neighbouring phases to coordinate, while the chemical features and CYP context determine how quickly each atom's phases tend to advance:

```math
q_i^{(t)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}
g_{ji}\odot\sin\!\left(\phi_j^{(t)}-\phi_i^{(t)}-\psi\right),
```

```math
\phi_i^{(t+1)}=\phi_i^{(t)}+s\left[
\omega_i^{(t)}+Kq_i^{(t)}\right],
\qquad
\omega_i^{(t)}=A\tanh(W_\omega r_i^{(t)}).
```
The vector φᵢ⁽ᵗ⁾ ∈ [−π,π)ᴴ contains the H phases of atom i. For each bonded neighbour j, the sine term measures the phase difference between the two atoms after allowing for the preferred offset ψ, called the *phase lag*. The sine is calculated separately for each channel. The bond gate gⱼᵢ ∈ (0,1)ᴴ controls how strongly that bond contributes in each channel, and ⊙ applies this control element by element. Averaging over the neighbour set N(i) produces the mean coupling signal qᵢ⁽ᵗ⁾ ∈ ℝᴴ.

The parameter K sets the overall coupling strength and s is the integration step. The vector ωᵢ⁽ᵗ⁾ ∈ ℝᴴ is the atom's chemically conditioned *natural frequency*: the rate at which its phases would tend to advance before neighbour coupling is added. W[ω] is learned from the common reaction drive rᵢ⁽ᵗ⁾, and A sets the frequency scale. After each update, we wrapped every phase modulo 2π into the interval [−π,π), because angles differing by a complete turn are equivalent. Dividing by π converted the phase back to hᵢ⁽ᵗ⁺¹⁾ ∈ [−1,1]. A hyperparameter search conducted only on the training data selected the phase lag, coupling strength and frequency scale.

**Delayed memory.** This rule draws on delay-differential systems, in which the next state depends on both the present and an earlier state [23]. It can therefore represent delayed feedback and other history-dependent responses that cannot be expressed using the present state alone (Figure 2E). The delay d specifies how many generations back the rule looks. The calculation combines the present reaction, a learned response to the retained earlier state, and explicit feedback from the difference between the earlier and present states:

```math
\tilde r_i^{(t)}=\tanh\!\left(W_d[r_i^{(t)},h_i^{(t-d)}]\right),
```

```math
h_i^{(t+1)}=\tanh\!\left(
(1-\zeta s)h_i^{(t)}+s\left[(1-\mu)r_i^{(t)}
+\mu\tilde r_i^{(t)}+\kappa_d(h_i^{(t-d)}-h_i^{(t)})\right]
\right).
```
The vector hᵢ⁽ᵗ⁻ᵈ⁾ ∈ ℝᴴ is atom i's state d generations earlier. When fewer than d generations were available, the rule used the earliest retained state. To form the delay-conditioned reaction r̃ᵢ⁽ᵗ⁾ ∈ ℝᴴ, square brackets concatenate the present reaction rᵢ⁽ᵗ⁾ and the earlier state hᵢ⁽ᵗ⁻ᵈ⁾; W[d] then transforms this combined vector into H values. The hyperbolic tangent bounds both the delay-conditioned reaction and the final updated state.

The mixing coefficient μ ∈ [0,1] controls the balance between the present and delay-conditioned reactions. The 1 in (1−μ) is the scalar multiplicative identity. The parameter κ[d] controls feedback from the difference hᵢ⁽ᵗ⁻ᵈ⁾−hᵢ⁽ᵗ⁾, which measures how far the earlier state is from the present state. The parameter ζ damps the present state, and the positive parameter s sets the update scale. The rule-specific search selected d and these coefficients from their predefined search ranges.

Each rule produces a trajectory containing the state of every atom at every generation. The number of atoms differs between molecules, and the number of generations can also differ between model configurations. The prediction step therefore requires a fixed-length numerical description of each trajectory. We created this description by *pooling*: calculating summary statistics over atoms and generations. The resulting vector is the molecular fingerprint supplied to ridge regression.

The basic fingerprint contained five types of summary. The final atom-state mean described the average channel values at the last generation, while the final atom-state variance described how much atoms differed from one another at that point. The time-averaged atom state summarised channel values across the trajectory. The temporal variance of the molecular mean state measured how much the molecule-wide average changed over generations. The mean state-change energy summarised the magnitude of changes between successive states. A multiscale version also retained the molecular mean states at 12.5%, 25%, 50%, 75% and 100% of the trajectory, allowing the prediction step to distinguish early, intermediate and final behaviour.

The model predicted four different CYP isoforms. The one-hot CYP context vector therefore selected the copy of the fingerprint belonging to the target isoform: the relevant block was active and the other three blocks were zero. This produced CYP-specific readout features and allowed ridge regression to associate the same types of trajectory measurement with different isoforms.

The genetic search also tested optional summaries of trajectory change. The *dynamic observables* option added the difference between the initial and final molecular mean states and a normalised correlation between the molecular mean states at successive generations. The correlation measured how strongly one generation resembled the next. *Multiscale transition-energy summaries* added the squared changes between the molecular mean states at consecutive sampling points listed above. The search also tested correlations separated by several generation lags, smooth approximations to the largest and smallest molecular mean states, and separate summaries of increases and decreases in state values. Together, these options allowed the readout to use persistence, variation, direction and scale of change rather than relying only on the final state.

![One-dimensional displays of recorded molecular Graph-CA evolution for five cellular automaton transition rules](figures/02_five_rule_1d_cellular_evolution.png)

**Figure 2. One-dimensional displays of molecular cellular evolution under five cellular automaton transition rules.** **A**, gated residual (`OCNT-2328519`); **B**, inertial reaction–diffusion (`OCNT-2312382`); **C**, FitzHugh–Nagumo (`OCNT-2314750`); **D**, Kuramoto–Sakaguchi (`OCNT-2328784`); and **E**, delayed memory (`OCNT-2308483`). CYP1A2 supplies the context for all five recorded trajectories. Columns represent heavy atoms in their stored graph order, and rows show successive generations from 0 at the top to 256 at the bottom. Local interactions follow the molecular bonds; the horizontal arrangement is a display of the graph cells. Panels A, B, C, and E show the first dynamical channel, hᵢ₁⁽ᵗ⁾, on a common blue–white–red scale from −1 to 1. Panel D shows the first wrapped phase divided by π, φᵢ₁⁽ᵗ⁾/π, using a cyclic scale whose endpoints represent the same phase. The display retains every generation and draws colours from the recorded states without per-panel amplitude rescaling. These examples come from distinct molecules and trained systems and illustrate their observed evolution; they are not a matched comparison of rule effects on a single molecule. Each panel displays one channel of the full atom-by-channel trajectory.

## References

1. Wolfram S. Statistical mechanics of cellular automata. *Reviews of Modern Physics*. 1983;55(3):601–644. doi: [10.1103/RevModPhys.55.601](https://doi.org/10.1103/RevModPhys.55.601).
2. Wolfram S. Universality and complexity in cellular automata. *Physica D: Nonlinear Phenomena*. 1984;10(1–2):1–35. doi: [10.1016/0167-2789(84)90245-8](https://doi.org/10.1016/0167-2789(84)90245-8).
3. Langton CG. Computation at the edge of chaos: phase transitions and emergent computation. *Physica D: Nonlinear Phenomena*. 1990;42(1–3):12–37. doi: [10.1016/0167-2789(90)90064-V](https://doi.org/10.1016/0167-2789(90)90064-V).
4. Turing AM. The chemical basis of morphogenesis. *Philosophical Transactions of the Royal Society B*. 1952;237(641):37–72. doi: [10.1098/rstb.1952.0012](https://doi.org/10.1098/rstb.1952.0012).
5. Grattarola D, Livi L, Alippi C. Learning graph cellular automata. *Advances in Neural Information Processing Systems*. 2021;34:20983–20994. [https://proceedings.neurips.cc/paper/2021/hash/af87f7cdcda223c41c3f3ef05a3aaeea-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/af87f7cdcda223c41c3f3ef05a3aaeea-Abstract.html).
6. Mordvintsev A, Randazzo E, Niklasson E, Levin M. Growing neural cellular automata. *Distill*. 2020;5(2). doi: [10.23915/distill.00023](https://doi.org/10.23915/distill.00023).
7. Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. Neural message passing for quantum chemistry. *Proceedings of Machine Learning Research*. 2017;70:1263–1272. [https://proceedings.mlr.press/v70/gilmer17a.html](https://proceedings.mlr.press/v70/gilmer17a.html).
8. Goldberg DE. *Genetic Algorithms in Search, Optimization, and Machine Learning*. Reading, MA: Addison-Wesley; 1989.
9. Koza JR. *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. Cambridge, MA: MIT Press; 1992.
10. Zanger UM, Schwab M. Cytochrome P450 enzymes in drug metabolism: regulation of gene expression, enzyme activities, and impact of genetic variation. *Pharmacology & Therapeutics*. 2013;138(1):103–141. doi: [10.1016/j.pharmthera.2012.12.007](https://doi.org/10.1016/j.pharmthera.2012.12.007).
11. Benettin G, Galgani L, Giorgilli A, Strelcyn JM. Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems; a method for computing all of them. Part 1: theory. *Meccanica*. 1980;15:9–20. doi: [10.1007/BF02128236](https://doi.org/10.1007/BF02128236).
12. Oseledec VI. A multiplicative ergodic theorem: Lyapunov characteristic numbers for dynamical systems. *Transactions of the Moscow Mathematical Society*. 1968;19:197–231.
13. Crutchfield JP, Young K. Inferring statistical complexity. *Physical Review Letters*. 1989;63(2):105–108. doi: [10.1103/PhysRevLett.63.105](https://doi.org/10.1103/PhysRevLett.63.105).
14. Grassberger P, Procaccia I. Measuring the strangeness of strange attractors. *Physica D*. 1983;9:189–208. doi: [10.1016/0167-2789(83)90298-1](https://doi.org/10.1016/0167-2789(83)90298-1).
15. Theiler J. Spurious dimension from correlation algorithms applied to limited time-series data. *Physical Review A*. 1986;34:2427–2432. doi: [10.1103/PhysRevA.34.2427](https://doi.org/10.1103/PhysRevA.34.2427).
16. Eckmann J-P, Ruelle D. Fundamental limitations for estimating dimensions and Lyapunov exponents in dynamical systems. *Physica D*. 1992;56:185–187. doi: [10.1016/0167-2789(92)90023-G](https://doi.org/10.1016/0167-2789(92)90023-G).
17. Cho K, van Merriënboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*. 2014:1724–1734. doi: [10.3115/v1/D14-1179](https://doi.org/10.3115/v1/D14-1179).
18. Polyak BT. Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*. 1964;4(5):1–17. doi: [10.1016/0041-5553(64)90137-5](https://doi.org/10.1016/0041-5553(64)90137-5).
19. FitzHugh R. Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal*. 1961;1(6):445–466. doi: [10.1016/S0006-3495(61)86902-6](https://doi.org/10.1016/S0006-3495(61)86902-6).
20. Nagumo J, Arimoto S, Yoshizawa S. An active pulse transmission line simulating nerve axon. *Proceedings of the IRE*. 1962;50(10):2061–2070. doi: [10.1109/JRPROC.1962.288235](https://doi.org/10.1109/JRPROC.1962.288235).
21. Kuramoto Y. *Chemical Oscillations, Waves, and Turbulence*. Berlin: Springer; 1984. doi: [10.1007/978-3-642-69689-3](https://doi.org/10.1007/978-3-642-69689-3).
22. Sakaguchi H, Kuramoto Y. A soluble active rotator model showing phase transitions via mutual entrainment. *Progress of Theoretical Physics*. 1986;76(3):576–581. doi: [10.1143/PTP.76.576](https://doi.org/10.1143/PTP.76.576).
23. Hale JK, Verduyn Lunel SM. *Introduction to Functional Differential Equations*. New York: Springer; 1993. doi: [10.1007/978-1-4612-4342-7](https://doi.org/10.1007/978-1-4612-4342-7).
24. Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*. 1970;12(1):55–67. doi: [10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634).
25. Gershenson C. Emergence in Artificial Life. *Artificial Life*. 2023;29(2):153–167. doi: [10.1162/artl_a_00397](https://doi.org/10.1162/artl_a_00397).
26. Bedau MA. Artificial life: organization, adaptation and complexity from the bottom up. *Trends in Cognitive Sciences*. 2003;7(11):505–512. doi: [10.1016/j.tics.2003.09.012](https://doi.org/10.1016/j.tics.2003.09.012).
27. Shannon CE. A mathematical theory of communication. *Bell System Technical Journal*. 1948;27:379–423, 623–656. doi: [10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x); [10.1002/j.1538-7305.1948.tb00917.x](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x).
