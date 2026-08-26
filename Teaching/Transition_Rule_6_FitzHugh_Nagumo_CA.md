# Rule 6: FitzHugh–Nagumo graph cellular automaton

## Purpose

Every atom carries a fast excitation field $u_i$ and a slower recovery field $v_i$. Excitation can spread through bonded neighbours, while recovery creates a refractory interval after a pulse.

## Implemented update

Let $\bar u_i$ and $\bar v_i$ be bonded-neighbour means, and let $s_i$ and $r_i$ be learned bounded chemical/CYP drives. The implementation sets

```math
\varepsilon=0.01+0.19d,\qquad a_0=0.2+0.8b.
```

The increments are

```math
\Delta u_i=u_i-\frac{u_i^3}{3}-v_i+c\,s_i
             +a_D(\bar u_i-u_i),
```

```math
\Delta v_i=\varepsilon(u_i+a_0-v_i+0.1r_i)
             +0.25a_D(\bar v_i-v_i).
```

With shared update scale $\eta$,

```math
h_i^{(t+1)}=\tanh\!\left(
\begin{bmatrix}
u_i+\eta\Delta u_i\\
v_i+\eta\Delta v_i
\end{bmatrix}
\right).
```

Here $a_D$ controls graph diffusion, $b$ sets the excitation threshold, $c$ scales learned stimulus, and $d$ controls recovery timescale.

## Expected dynamics and cautions

Small stimuli decay, suprathreshold stimuli produce pulses, and recovery temporarily suppresses re-excitation. The channel count must be even. Large $\eta$, weak recovery, or strong stimulus can saturate the outer $\tanh$. Finite-time oscillation is not proof of a limit cycle.
