# Rule 7: Gray–Scott graph cellular automaton

## Purpose

Gray–Scott dynamics couple two diffusing fields with a cubic reaction, continuous feed, and removal. The molecular bond graph replaces the classical spatial grid.

## Implemented update

Stored values in $[-1,1]$ are mapped to concentrations:

```math
u_i=\frac{u_i^{\mathrm{raw}}+1}{2},\qquad
v_i=\frac{v_i^{\mathrm{raw}}+1}{2}.
```

The searched controls define

```math
D_u=0.01+0.19a,\quad D_v=0.005+0.095b,
\quad F=0.01+0.07c,\quad k=0.03+0.04d.
```

With learned bounded drives $s_{u,i}$ and $s_{v,i}$,

```math
\Delta u_i=D_u(\bar u_i-u_i)-u_iv_i^2+F(1-u_i)+0.02s_{u,i},
```

```math
\Delta v_i=D_v(\bar v_i-v_i)+u_iv_i^2-(F+k)v_i+0.02s_{v,i}.
```

Advance, clip, and return to the stored interval:

```math
u_i'=\operatorname{clip}_{[0,1]}(u_i+\eta\Delta u_i),\qquad
v_i'=\operatorname{clip}_{[0,1]}(v_i+\eta\Delta v_i),
```

```math
h_i^{(t+1)}=2\begin{bmatrix}u_i'\\v_i'\end{bmatrix}-1.
```

## Expected dynamics and cautions

Different diffusion rates and cubic conversion can support localised structures, fronts, multistability, or convergence. The channel count must be even. Clipping preserves valid concentrations but can create saturation; finite patterns do not establish chaos.
