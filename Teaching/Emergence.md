# Emergence and Cellular Automata

## Cellular automata

A cellular automaton consists of cells, cell states, neighbourhoods, a local transition rule, and repeated generations. Simple local interactions can create organised large-scale behaviour. This production of collective structure from local rules is **emergence**.

## Graph cellular automata

In Strange Matter Engine, atoms are cells and chemical bonds define their neighbourhoods. The same compact local rule is applied to every atom and reused at every generation. This repeated shared rule is the central cellular-automaton constraint.

The provisional update has two conceptual stages:

1. Each atom receives messages from its bonded neighbours, with bond properties affecting the messages.
2. The atom combines those messages with its own state and the CYP context to form its next state.

The exact equations will be derived before implementation.

## Chemical and dynamical channels

The initial atom state may contain:

- **chemical channels**, initialised from known atomic properties; and
- **free dynamical channels**, initially neutral and allowed to develop structure as the automaton evolves.

Chemical channels describe what an atom initially is. Dynamical channels record what the learned local rule causes it to do.

## Synchronous evolution

The first implementation will use synchronous updates: all next-generation states are calculated from the same current generation, then replaced together. This prevents the arbitrary order of atom processing from changing the result.

## Emergence is observed, not prescribed

We will preserve the possibility of fixed points, oscillations, multiple attractor basins, and complex transients. We will examine what the learned system produces instead of hard-coding a dynamical class for strong or weak inhibitors.

## Topics to develop

- classical one- and two-dimensional cellular automata;
- local rules and global patterns;
- synchronous and asynchronous updates;
- finite state versus continuous-state automata;
- neural or differentiable cellular automata; and
- the boundary between a compact graph CA and a conventional graph neural network.
