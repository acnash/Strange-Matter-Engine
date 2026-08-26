# Rule 9: conservative graph-flux cellular automaton

## Purpose

This rule redistributes every latent channel across bonds without creating or destroying its graph-wide total. It tests whether prediction-relevant information can be encoded entirely through molecular redistribution.

## Implemented update

For directed bond $j\to i$, the learned flux is

```math
F_{j\to i}=g_{ji}\odot\tanh\!\left(W_F(h_j-h_i)\right),
```

where $g_{ji}$ is the bond-conditioned gate and $W_F$ has no bias. The node receives

```math
J_i=\sum_{j\in N(i)}F_{j\to i}.
```

Using maximum graph degree $d_{\max}$ for stable shared normalisation,

```math
h_i^{(t+1)}=h_i^{(t)}+
\frac{\eta a}{\max(1,d_{\max})}J_i.
```

Every undirected bond is stored in both directions. Since $\tanh$ is odd and the gate is identical in each direction,

```math
F_{i\to j}=-F_{j\to i},
```

so each channel obeys

```math
\sum_i h_i^{(t+1)}=\sum_i h_i^{(t)}
```

up to floating-point rounding. Only generic control $a$ is used.

## Expected dynamics and cautions

Activity can diffuse, concentrate, split around rings, or equilibrate while its total remains fixed. Conservation must be tested per molecule and channel. Asymmetric gates, a biased flux layer, or node-specific normalisation would invalidate the proof.
