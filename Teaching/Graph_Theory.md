# Graph Theory

## Molecules as graphs

A molecular graph is written as `G = (V, E)`, where:

- `V` is the set of vertices or nodes, representing atoms; and
- `E` is the set of edges, representing chemical bonds.

For an atom `i`, its neighbourhood `N(i)` is the set of atoms directly connected to it by bonds. Unlike cells on a regular grid, molecular cells have neighbourhoods determined by chemical connectivity.

## Features on the graph

Each node carries an atom-state vector. Each edge can carry bond features. The graph therefore records both topology—what is connected to what—and attributes—what the atoms and bonds are like.

## Local information exchange

At each generation, atom `i` receives information from every bonded neighbour `j`. A message rule can depend on the current states of both atoms and on their bond. The neighbour messages are combined using an order-independent operation such as a sum.

Order independence matters because renumbering the atoms should not change the physical molecule or its prediction.

## Topics to develop

- adjacency lists and adjacency matrices;
- node degree, paths, cycles, and connected components;
- attributed and directed graphs;
- graph isomorphism and atom-order invariance;
- permutation-invariant aggregation; and
- batching molecular graphs of different sizes.
