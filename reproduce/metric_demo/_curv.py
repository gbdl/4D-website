# SPDX-License-Identifier: GPL-3.0-or-later
"""Private minimal curvature for the WS5 cross-check (conventions.md, binding).

Gamma^i_{jk} = 1/2 g^{il}(d_j g_lk + d_k g_lj - d_l g_jk),  dg[i,j,k] = d_k g_ij
R^i_{jkl} = d_k Gamma^i_{lj} - d_l Gamma^i_{kj} + Gamma^i_{km}Gamma^m_{lj} - Gamma^i_{lm}Gamma^m_{kj}
Ric_{jl} = R^i_{jil},  S = g^{jl}Ric_{jl},  E = ||Ric+3g||_g/6.
Validated in tests/test_metrics_curv.py on H^4 (K=-1) and S^4 (K=+1).
The companion independently checks these contractions with NumPy index loops.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def jets(fn, p):
    g = fn(p)
    dg = jax.jacfwd(fn)(p)
    d2g = jax.jacfwd(jax.jacfwd(fn))(p)
    return g, dg, d2g


def christoffel(g, dg):
    gi = jnp.linalg.inv(g)
    # first[l,j,k] = d_j g_lk + d_k g_lj - d_l g_jk
    first = jnp.einsum("lkj->ljk", dg) + dg - jnp.einsum("jkl->ljk", dg)
    return 0.5 * jnp.einsum("il,ljk->ijk", gi, first), gi


def riemann_up(g, dg, d2g):
    """R^i_{jkl} with the convention above."""
    gi = jnp.linalg.inv(g)
    first = jnp.einsum("lkj->ljk", dg) + dg - jnp.einsum("jkl->ljk", dg)
    gamma = 0.5 * jnp.einsum("il,ljk->ijk", gi, first)
    # d_m first[l,j,k] = d_m d_j g_lk + d_m d_k g_lj - d_m d_l g_jk ; d2g[a,b,c,d] = d_c d_d g_ab
    dfirst = (jnp.einsum("lkjm->ljkm", d2g) + jnp.einsum("ljkm->ljkm", d2g) - jnp.einsum("jklm->ljkm", d2g))
    dgi = -jnp.einsum("ia,abm,bl->ilm", gi, dg, gi)
    dgamma = 0.5 * (jnp.einsum("ilm,ljk->ijkm", dgi, first) + jnp.einsum("il,ljkm->ijkm", gi, dfirst))
    # dgamma[i,j,k,m] = d_m Gamma^i_{jk}
    term1 = jnp.einsum("iljk->ijkl", dgamma)      # d_k Gamma^i_{lj}  -> index (i,j,k,l): dgamma[i,l,j,k]
    term2 = jnp.einsum("ikjl->ijkl", dgamma)      # d_l Gamma^i_{kj}: dgamma[i,k,j,l]
    quad1 = jnp.einsum("ikm,mlj->ijkl", gamma, gamma)
    quad2 = jnp.einsum("ilm,mkj->ijkl", gamma, gamma)
    return term1 - term2 + quad1 - quad2


def curvature(g, dg, d2g):
    Rup = riemann_up(g, dg, d2g)
    R = jnp.einsum("im,mjkl->ijkl", g, Rup)
    ric = jnp.einsum("ijil->jl", Rup)
    gi = jnp.linalg.inv(g)
    S = jnp.einsum("jl,jl->", gi, ric)
    A = ric + 3.0 * g
    E = jnp.sqrt(jnp.einsum("ia,jb,ij,ab->", gi, gi, A, A)) / 6.0
    return dict(R=R, ric=ric, S=S, E=E)


def einstein_E(fn, p):
    g, dg, d2g = jets(fn, p)
    return curvature(g, dg, d2g)["E"]


def sectional(R, g, u, v):
    num = jnp.einsum("ijkl,i,j,k,l->", R, u, v, u, v)
    den = (u @ g @ u) * (v @ g @ v) - (u @ g @ v) ** 2
    return num / den
