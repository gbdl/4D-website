# SPDX-License-Identifier: GPL-3.0-or-later
"""Network pieces written from the notes (constructions.tex, appendix.tex, math-appendix.tex).

Every place where the notes leave a choice open is a *reading* selected by a
string; the default reading is the one we consider most literal.  The black
box cross-check (ws5_reconstruction/) decides which reading the trained state
actually uses; see the ambiguity table in ws5_reconstruction/README.md.
"""
from __future__ import annotations

import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from . import _geom as G


def silu(a):
    return a / (1.0 + jnp.exp(-a))


ACTIVATIONS = {"silu": silu, "tanh": jnp.tanh}


def mlp(layers, v, activation):
    act = ACTIVATIONS[activation]
    for w, b in layers[:-1]:
        v = act(v @ w + b)
    w, b = layers[-1]
    return v @ w + b


# --- L(o): lower triangular with exponentiated diagonal (constructions eq. lower)
_ROW = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)]   # math-appendix: (00,10,11,20,...)
_COL = [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)]
LOWER_ORDERS = {"row": _ROW, "col": _COL}


def lower(o, order="row"):
    idx = LOWER_ORDERS[order]
    rows = np.array([i for i, _ in idx])
    cols = np.array([j for _, j in idx])
    diag = rows == cols
    vals = jnp.where(jnp.asarray(diag), jnp.exp(jnp.where(jnp.asarray(diag), o, 0.0)), o)
    return jnp.zeros((4, 4), dtype=o.dtype).at[rows, cols].set(vals)


# --- C_j: upper triangular 3x3 with exp diagonal from 6 parameters (constructions, "six shape parameters")
CUSP_LAYOUTS = {
    "triu-row": [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
    "triu-col": [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)],
    "diag-first": [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
    "tril-row": [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)],   # lower: deliberately "wrong" control
}


def cusp_matrix(c6, layout="triu-row"):
    idx = CUSP_LAYOUTS[layout]
    rows = np.array([i for i, _ in idx])
    cols = np.array([j for _, j in idx])
    diag = rows == cols
    c6 = jnp.asarray(c6)
    vals = jnp.where(jnp.asarray(diag), jnp.exp(c6), c6)
    return jnp.zeros((3, 3), dtype=c6.dtype).at[rows, cols].set(vals)


# --- black-hole compact tensor (geometry eq. bhregular) and its square-root factor
def h_bh(y):
    a = G.core_z(y) / G.RJ
    f = 1.0 + a + a * a
    w = y[:2]
    disk = f / 3.0 * jnp.eye(2) + (2.0 + a) * (3.0 + f) / (4.0 * G.RJ ** 2 * f) * jnp.outer(w, w)
    return jnp.eye(4, dtype=y.dtype).at[:2, :2].set(disk)


def chol2_lower(h):
    """Explicit lower Cholesky of a SPD 4x4 (generic formula, used for any anchor)."""
    return jnp.linalg.cholesky(h)


def sym_sqrt(h):
    w, v = jnp.linalg.eigh(h)
    return (v * jnp.sqrt(w)) @ v.T


ANCHOR_FACTORS = {"chol": chol2_lower, "sqrt": sym_sqrt}


# --- raw (g_loc) inputs (constructions eq. rawfeatures; appendix "raw scaling")
def raw_scaling(variant="note"):
    corners = np.array([[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 3.0) for c in (-1.0, 3.0)])
    tau = corners @ G.F[1:].T
    lo, hi = tau.min(0), tau.max(0)
    if variant == "note":
        m = np.r_[0.0, 0.0, (lo + hi) / 2]
        s = np.r_[G.DISK_SCALE, G.DISK_SCALE, (hi - lo) / 2]
    elif variant == "fullrange":
        m = np.r_[0.0, 0.0, (lo + hi) / 2]
        s = np.r_[G.DISK_SCALE, G.DISK_SCALE, (hi - lo)]
    elif variant == "unscaled":
        m, s = np.zeros(4), np.ones(4)
    else:
        raise ValueError(variant)
    return m, s


def raw_features(y, freqs, order="sin-block-coord-major", scaling="note", height=True, fourier=True):
    m, s = raw_scaling(scaling)
    q = (y - jnp.asarray(m)) / jnp.asarray(s)
    pieces = [q]
    if height:
        pieces.append(jnp.atleast_1d(2.0 * G.core_z(y) / G.RJ - 1.0))
    if fourier:
        k = jnp.asarray(freqs, dtype=y.dtype)
        ph = jnp.pi * q[:, None] * k[None, :]          # (coord i, freq k)
        if order == "sin-block-coord-major":            # all sines (i major, k minor), then all cosines
            pieces += [jnp.sin(ph).ravel(), jnp.cos(ph).ravel()]
        elif order == "sin-block-freq-major":           # all sines (k major, i minor), then cosines
            pieces += [jnp.sin(ph).T.ravel(), jnp.cos(ph).T.ravel()]
        elif order == "pairs-coord-major":              # (sin,cos) pairs, i major
            pieces.append(jnp.stack([jnp.sin(ph), jnp.cos(ph)], axis=-1).ravel())
        elif order == "pairs-freq-major":               # (sin,cos) pairs, k major
            pieces.append(jnp.stack([jnp.sin(ph).T, jnp.cos(ph).T], axis=-1).ravel())
        else:
            raise ValueError(order)
    return jnp.concatenate(pieces)


# --- periodic body (constructions eqs. untwist, periodicfeatures, periodictensor; appendix B_w)
def untwist(y):
    """(xi, eta) = (R(-beta.tau)(U,V), Lambda^{-1} tau) and its Jacobian d(xi,eta)/dy (by AD)."""
    def w(y):
        ang = -jnp.asarray(G.BETA) @ y[2:]
        c, s = jnp.cos(ang), jnp.sin(ang)
        xi = jnp.stack([c * y[0] - s * y[1], s * y[0] + c * y[1]])
        eta = jnp.asarray(G.LAMBDA_INV) @ y[2:]
        return jnp.concatenate([xi, eta])
    return w(y), jax.jacfwd(w)(y)


def periodic_features(xi, eta, z, freqs, order="sin-block", height=True):
    pieces = [xi / G.DISK_SCALE]
    if height:
        pieces.append(jnp.atleast_1d(2.0 * z / G.RJ - 1.0))
    k = jnp.asarray(freqs, dtype=xi.dtype)
    ph = 2.0 * jnp.pi * eta[:, None] * k[None, :]      # (eta_i, k)
    if order == "sin-block":                            # note eq. periodicfeatures for k=1: sin e1, sin e2, cos e1, cos e2
        pieces += [jnp.sin(ph).ravel(), jnp.cos(ph).ravel()]
    elif order == "sin-block-freq-major":
        pieces += [jnp.sin(ph).T.ravel(), jnp.cos(ph).T.ravel()]
    elif order == "pairs":
        pieces.append(jnp.stack([jnp.sin(ph), jnp.cos(ph)], axis=-1).ravel())
    else:
        raise ValueError(order)
    return jnp.concatenate(pieces)


def untwisted_anchor(xi):
    """H~ = B_w^T H_BH(xi1,xi2,0,0) B_w,  B_w = [[I2, (J2 xi) a^T],[0, Lambda]]."""
    j2xi = jnp.stack([-xi[1], xi[0]])
    B = jnp.eye(4, dtype=xi.dtype).at[:2, 2:].set(jnp.outer(j2xi, jnp.asarray(G.ANGLES)))
    B = B.at[2:, 2:].set(jnp.asarray(G.LAMBDA))
    H = h_bh(jnp.concatenate([xi, jnp.zeros(2, dtype=xi.dtype)]))
    return B.T @ H @ B


# --- native cusp networks (constructions eq. native)
def native_features(c, phase, periods, order="inv-sin-cos", offsets=True):
    t = c[0]
    dev = c[1:] + (jnp.asarray(phase) if offsets else 0.0)
    ph = 2.0 * jnp.pi * dev / jnp.asarray(periods)
    if order == "inv-sin-cos":
        return jnp.concatenate([jnp.atleast_1d(1.0 / t), jnp.sin(ph), jnp.cos(ph)])
    if order == "inv-pairs":
        return jnp.concatenate([jnp.atleast_1d(1.0 / t), jnp.stack([jnp.sin(ph), jnp.cos(ph)], axis=-1).ravel()])
    raise ValueError(order)
