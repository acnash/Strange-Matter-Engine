# Rule 8: Kuramoto–Sakaguchi graph cellular automaton

## Purpose

Each atom-channel value is a circular phase. Chemical context determines natural angular velocity, while bonded atoms exchange phase-coupling signals.

## Implemented update

The stored state becomes phase:

```math
\theta_{ik}=\pi h_{ik}.
```

For phase lag

```math
\alpha=\pi\left(c-\frac{1}{2}\right),
```

the bond-gated mean coupling is

```math
C_{ik}=\frac{1}{\max(1,|N(i)|)}
\sum_{j\in N(i)}g_{ji,k}
\sin(\theta_{jk}-\theta_{ik}-\alpha).
```

A learned reaction supplies natural frequency,

```math
\omega_{ik}=a\tanh([W_\omega r_i]_k),
```

and

```math
\theta_{ik}'=\theta_{ik}+\eta(\omega_{ik}+bC_{ik}),
```

```math
h_{ik}^{(t+1)}=\frac{1}{\pi}
\operatorname{atan2}(\sin\theta_{ik}',\cos\theta_{ik}').
```

The fourth generic control is unused.

## Expected dynamics and cautions

The graph may synchronise, form phase-locked clusters, support travelling waves, or remain frustrated. Phase comparisons must be circular: a stored jump across $-1$ and $1$ can represent a continuous rotation.
