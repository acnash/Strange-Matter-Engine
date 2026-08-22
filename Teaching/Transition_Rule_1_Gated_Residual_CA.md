# Transition rule 1: gated residual message-passing CA

## Aim

This lesson explains the transition rule used by prototype one. Its central idea is selective change: each atom constructs a proposed new state, then a learned gate decides how much of that proposal to accept.

## Molecular state

At generation t, atom i has a state vector with d channels:

```math
h_i^{(t)}\in\mathbb{R}^{d}.
```

Its fixed chemical feature vector is:

```math
x_i\in\mathbb{R}^{p}.
```

The CYP context vector is c, and the bond between atoms j and i has features denoted by the vector e with subscripts ji.

## Neighbour message

Each neighbour contributes a transformed state and a transformed bond description:

```math
m_{j\rightarrow i}^{(t)}=W_nh_j^{(t)}+W_ee_{ji}.
```

The atom receives the mean of its incoming messages:

```math
m_i^{(t)}=\frac{1}{|N(i)|}\sum_{j\in N(i)}m_{j\rightarrow i}^{(t)}.
```

Mean aggregation prevents an atom with many neighbours from automatically receiving a much larger signal solely because of its degree.

## Candidate state

The model combines self-information, neighbour information, fixed chemistry and CYP context:

```math
\widetilde h_i^{(t+1)}=\tanh\left(
W_sh_i^{(t)}+m_i^{(t)}+W_xx_i+W_cc+b
\right).
```

The hyperbolic tangent bounds every candidate channel between minus one and plus one.

## Learned gate

The gate examines the present state and the information available for the update:

```math
\alpha_i^{(t)}=\sigma\left(
W_g[h_i^{(t)},m_i^{(t)},x_i,c]+b_g
\right).
```

The sigmoid function places each gate channel between zero and one:

```math
0<\alpha_{ik}^{(t)}<1.
```

The final update is a convex mixture:

```math
h_i^{(t+1)}=
\left(1-\alpha_i^{(t)}\right)\odot h_i^{(t)}
+\alpha_i^{(t)}\odot\widetilde h_i^{(t+1)}.
```

The symbol with a circle and dot means channel-by-channel multiplication.

## Worked one-channel example

Suppose an atom currently has state 0.20, its candidate state is 0.80, and the gate is 0.25:

```math
h^{(t+1)}=(1-0.25)(0.20)+(0.25)(0.80).
```

```math
h^{(t+1)}=0.15+0.20=0.35.
```

The atom moves only one quarter of the way from its old state toward the candidate. If the gate were 0.90, the new state would be:

```math
h^{(t+1)}=(0.10)(0.20)+(0.90)(0.80)=0.74.
```

## Why it is residual

Subtract the old state from both sides:

```math
h_i^{(t+1)}-h_i^{(t)}=
\alpha_i^{(t)}\odot
\left(\widetilde h_i^{(t+1)}-h_i^{(t)}\right).
```

The update is therefore the old state plus a gated residual displacement. Small gates produce slow relaxation; large gates produce rapid change.

## Expected dynamics

- Gates near zero create memory and very slow transients.
- Gates near one create rapid nonlinear relaxation.
- Different channels can move at different speeds.
- Fixed points occur when the candidate equals the current state.
- Sustained oscillation is possible but is not structurally encouraged.

## Hyperparameters to tune

- number of generations;
- state width;
- Graph-CA learning rate and ridge penalty;
- ridge and CA regularisation;
- gradient-clipping threshold;
- batch size; and
- initial gate bias.

This rule is the stable predictive baseline against which the more dynamical rules should be compared.
