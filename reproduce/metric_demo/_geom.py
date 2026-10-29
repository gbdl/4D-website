# SPDX-License-Identifier: GPL-3.0-or-later
"""Minimal M4 geometry reconstructed from the mathematical notes.

Sources (two-approach note TeX, sections/geometry.tex and sections/appendix.tex;
report TeX, sections/mathematics.tex and sections/math-appendix.tex):

* P^4: z>0, 0<=u_i<=1, z^2+|u-c|^2>=1 for c in C={(1,0,0),(0,1,0),(0,0,1),(1,1,1)}  (geometry eq. polytope)
* copies T_b(z,u)=(z,(-1)^b0 u1,(-1)^(b1+b3) u2+2b3,(-1)^(b2+b4) u3+2b4)          (geometry eq. copies)
* colour bits A1:0 A2:1 A3:2 B1:0 B2:3 B3:4 H1:4 H2:1 H3:3 H4:2                  (appendix table)
* slope k=(5,1,1), periods (2,4,4), v=(10,4,4), l=sqrt(132)                        (geometry eq. slope)
* (r^2-1/r)(4pi/3)^2=l^2                                                            (geometry eq. rj)
* frame: columns of diag(2,4,4)A_f, normalise the first, Gram-Schmidt the others    (appendix)
* rho^2=(4r/3)(r-z), theta=2 pi sigma/l, y=(rho cos, rho sin, tau1, tau2)           (geometry eq. core)
* cusp chart q=z^2+|x-p|^2, (t,X)=(z/q, O_p(x-p)/q)                                 (geometry eq. inversion)
* developed cubes, O_p=M_{v,b} diag(1-2v) A_b^T, phases 2pi(X+a)/P                  (appendix; math-appendix)

No geometric quantity is imported from WS3 (m4lc.geometry) or from project code; the two
implementations are cross-checked in tests/test_metrics_geom.py.

SITE NUMBERING: cusp class 0..3 first, then lexicographic integer centre.
All 37 centres, classes, orientations and periods are checked against the bundled
independent geometry fixture. No external geometry package is imported.
"""
from __future__ import annotations

import itertools
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# ----------------------------------------------------------------------------
# filled cusp, slope frame, r_j
# ----------------------------------------------------------------------------
SLOPE = np.array([5, 1, 1])
PERIODS = np.array([2.0, 4.0, 4.0])
SLOPE_VECTOR = PERIODS * SLOPE                     # v = (10,4,4)
ELL = float(np.linalg.norm(SLOPE_VECTOR))          # sqrt(132)
OMEGA = 2.0 * math.pi / ELL
A_FRAME = np.array([[5, -1, -5], [1, 0, -1], [1, 0, 0]], dtype=float)
A_PER = np.array([[5, 1, 0], [1, 0, 1], [1, 0, 0]], dtype=float)


def solve_rj(ell2=ELL ** 2):
    """Root r>1 of (r^2 - 1/r)(4 pi/3)^2 = ell^2 (monotone for r>=1): bisection + Newton."""
    beta2 = (4.0 * math.pi / 3.0) ** 2
    f = lambda r: (r * r - 1.0 / r) * beta2 - ell2
    lo, hi = 1.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    for _ in range(5):
        r -= f(r) / ((2 * r + 1.0 / (r * r)) * beta2)
    return r


RJ = solve_rj()


def gram_schmidt_frame(B):
    """Rows: normalised first column, then Gram-Schmidt of the projected other two."""
    e1 = B[:, 0] / np.linalg.norm(B[:, 0])
    P = np.eye(3) - np.outer(e1, e1)
    w2 = P @ B[:, 1]
    e2 = w2 / np.linalg.norm(w2)
    w3 = P @ B[:, 2]
    w3 = w3 - e2 * (e2 @ w3)
    e3 = w3 / np.linalg.norm(w3)
    return np.stack([e1, e2, e3])


F = gram_schmidt_frame(np.diag(PERIODS) @ A_FRAME)  # (sigma,tau1,tau2) = F x_hor
DISK_SCALE = math.sqrt(4.0 / 3.0) * RJ

# periodic (untwisting) data, appendix: Lambda=(F B_p)_{2:3,2:3}, a_i=(2pi/l)(F B_p)_{1,i+1}
_FBP = F @ (np.diag(PERIODS) @ A_PER)
LAMBDA = _FBP[1:, 1:].copy()
ANGLES = OMEGA * _FBP[0, 1:]
LAMBDA_INV = np.linalg.inv(LAMBDA)
BETA = np.linalg.solve(LAMBDA.T, ANGLES)            # beta = Lambda^{-T} a


def to_core(x):
    """Ordinary x=(z,x1,x2,x3) -> regular y=(U,V,tau1,tau2) (valid for z<=RJ)."""
    x = jnp.asarray(x)
    a = jnp.asarray(F) @ x[1:]
    rho = jnp.sqrt((4.0 / 3.0) * RJ * (RJ - x[0]))
    th = OMEGA * a[0]
    return jnp.stack([rho * jnp.cos(th), rho * jnp.sin(th), a[1], a[2]])


def core_z(y):
    return RJ - 3.0 * (y[0] ** 2 + y[1] ** 2) / (4.0 * RJ)


def from_core(y):
    """Principal-branch inverse (sigma in (-l/2,l/2]); never used at U=V=0."""
    y = jnp.asarray(y)
    sigma = jnp.arctan2(y[1], y[0]) / OMEGA
    hor = jnp.asarray(F).T @ jnp.stack([sigma, y[2], y[3]])
    return jnp.concatenate([core_z(y)[None], hor])


# ----------------------------------------------------------------------------
# polytope, facets, colours, copies
# ----------------------------------------------------------------------------
SPHERE_CENTERS = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1, 1]])   # H1..H4
SPHERE_CENTERS4 = np.concatenate([np.zeros((4, 1)), SPHERE_CENTERS], axis=1)
# facets: (label, kind, axis(0..2 horizontal) or sphere index, value, bit)
FACETS = (
    ("A1", "v", 0, 0.0, 0), ("A2", "v", 1, 0.0, 1), ("A3", "v", 2, 0.0, 2),
    ("B1", "v", 0, 1.0, 0), ("B2", "v", 1, 1.0, 3), ("B3", "v", 2, 1.0, 4),
    ("H1", "s", 0, None, 4), ("H2", "s", 1, None, 1), ("H3", "s", 2, None, 3), ("H4", "s", 3, None, 2),
)
SPHERE_BITS = np.array([4, 1, 3, 2])                 # H_i -> copy bit
SPHERE_COLOR_MASK = 1 << SPHERE_BITS                 # c(w)=16w1 + 2w2 + 8w3 + 4w4 (xor)


def bits_of(b):
    return [(int(b) >> k) & 1 for k in range(5)]


def copy_map(b):
    """(4x4 matrix, 4-offset) of T_b."""
    b0, b1, b2, b3, b4 = bits_of(b)
    M = np.diag([1.0, (-1.0) ** b0, (-1.0) ** (b1 + b3), (-1.0) ** (b2 + b4)])
    o = np.array([0.0, 0.0, 2.0 * b3, 2.0 * b4])
    return M, o


COPY_MATRICES = np.stack([copy_map(b)[0] for b in range(32)])
COPY_OFFSETS = np.stack([copy_map(b)[1] for b in range(32)])


def base_contains(u, tol=1e-12):
    """Closed base polytope (numpy, vectorised over leading axes)."""
    u = np.asarray(u, dtype=float)
    h = u[..., 1:]
    ok = (u[..., 0] > 0) & np.all(h >= -tol, axis=-1) & np.all(h <= 1 + tol, axis=-1)
    d2 = u[..., 0, None] ** 2 + np.sum((h[..., None, :] - SPHERE_CENTERS) ** 2, axis=-1)
    return ok & np.all(d2 >= 1 - tol, axis=-1)


def label_bits_np(h):
    """Copy bits of a horizontal point already in the box [-1,1)x[-1,3)^2 (half-open rule)."""
    h = np.asarray(h, dtype=float)
    b0 = h[..., 0] < 0
    b3 = h[..., 1] >= 1
    b1 = (h[..., 1] < 0) | (h[..., 1] >= 2)
    b4 = h[..., 2] >= 1
    b2 = (h[..., 2] < 0) | (h[..., 2] >= 2)
    return (b0.astype(int) | (b1.astype(int) << 1) | (b2.astype(int) << 2)
            | (b3.astype(int) << 3) | (b4.astype(int) << 4))


def fold_np(x):
    """Periodic translation into [-1,1)x[-1,3)^2 and copy label (numpy)."""
    x = np.array(x, dtype=float)
    h = x[..., 1:] - PERIODS * np.floor((x[..., 1:] + 1.0) / PERIODS)
    x[..., 1:] = h
    return x, label_bits_np(h)


def _label_jax(h):
    return (jnp.where(h[0] < 0, 1, 0) + 2 * jnp.where((h[1] < 0) | (h[1] >= 2), 1, 0)
            + 4 * jnp.where((h[2] < 0) | (h[2] >= 2), 1, 0) + 8 * jnp.where(h[1] >= 1, 1, 0)
            + 16 * jnp.where(h[2] >= 1, 1, 0))


def fold(x):
    """jax version of fold_np: (translated point, int label)."""
    x = jnp.asarray(x)
    P = jnp.asarray(PERIODS)
    h = x[1:] - P * jnp.floor((x[1:] + 1.0) / P)
    return jnp.concatenate([x[:1], h]), _label_jax(h)


BOX_LO = np.array([-1.0, -1.0, -1.0])
BOX_HI = BOX_LO + PERIODS


def fold_closed(x):
    """Translate x by a vector of the cusp-at-infinity lattice 2Z x 4Z x 4Z only where a horizontal
    coordinate lies OUTSIDE the closed box [-1,1]x[-1,3]x[-1,3]; the identity on the closed box.
    (The lattice translations are identifications of M4; on the closed box the raw field is evaluated
    as given, so the upper faces x_i = 1, 3 keep their own values, as in the project.)  jax-traceable;
    the map is piecewise a translation, so its Jacobian is the identity."""
    x = jnp.asarray(x)
    P = jnp.asarray(PERIODS)
    lo = jnp.asarray(BOX_LO)
    h = x[1:]
    k = jnp.floor((h - lo) / P)
    inside = (h >= lo) & (h <= lo + P)
    k = jnp.where(inside, 0.0, k)
    return jnp.concatenate([x[:1], h - P * k])


def fold_closed_np(x):
    x = np.array(x, dtype=float)
    h = x[..., 1:]
    k = np.floor((h - BOX_LO) / PERIODS)
    inside = (h >= BOX_LO) & (h <= BOX_HI)
    x[..., 1:] = h - PERIODS * np.where(inside, 0.0, k)
    return x


def copy_label(x):
    """Copy label (0..31) of a displayed point (half-open rule after periodic translation; numpy)."""
    return fold_np(x)[1]


def to_base(x, b):
    M, o = copy_map(b)
    return M.T @ (np.asarray(x, float) - o)


def contains(x, tol=1e-12):
    """Displayed slab membership: box, 0<z<=RJ, base point outside the hemispheres."""
    x = np.asarray(x, dtype=float)
    h = x[..., 1:]
    box = np.all(h >= np.array([-1, -1, -1]) - tol, axis=-1) & np.all(h <= np.array([1, 3, 3]) + tol, axis=-1)
    hc = np.clip(h, [-1, -1, -1], [1 - 1e-300, 3 - 1e-300, 3 - 1e-300])
    lab = label_bits_np(hc)
    M = COPY_MATRICES[lab]
    o = COPY_OFFSETS[lab]
    u = np.einsum("...ji,...j->...i", M, x - o)
    return box & base_contains(u, tol) & (x[..., 0] <= RJ + tol)


# ----------------------------------------------------------------------------
# finite cusps: developed inverted cubes, 37 sites, frames, phases
# ----------------------------------------------------------------------------
BASE_VERTICES = np.array([[0.0, 0, 0], [0.0, 1, 1], [1.0, 0, 1], [1.0, 1, 0]])
# Session-1 hard-coded table, kept ONLY as a regression reference; CUSP_PERIODS below is derived.
CUSP_PERIODS_NOTE = np.array([[2.0, 1, 2], [2.0, 1, 2], [2.0, 2, 1], [2.0, 2, 1]])


def _incident(facet, v):
    """(axis, value) of an incident facet in X=diag(1-2v)(u-v)/q, or None."""
    _, kind, idx, value, _ = facet
    if kind == "v":
        return (idx, 0.0) if v[idx] == value else None
    d = SPHERE_CENTERS[idx] - v
    if abs(d @ d - 1.0) > 1e-12:
        return None
    return (int(np.flatnonzero(np.abs(d) > 0.5)[0]), 0.5)


def cube_representatives(v):
    reps = []
    for bit in range(5):
        cands = [(f, _incident(f, v)) for f in FACETS if f[4] == bit]
        cands = [(f, a) for f, a in cands if a is not None]
        cands.sort(key=lambda fa: (fa[0][1] != "v", fa[0][0]))  # vertical first
        reps.append(cands[0][1])
    return reps


def developed_map(v, b):
    """X -> M X + a, composed in the bit order of T_b (bit 0 first)."""
    M = np.eye(3)
    a = np.zeros(3)
    for bit, (axis, value) in enumerate(cube_representatives(v)):
        if (b >> bit) & 1:
            R = np.eye(3)
            R[axis, axis] = -1.0
            s = np.zeros(3)
            s[axis] = 2.0 * value
            M, a = R @ M, R @ a + s
    return M, a


def periods_from_development(v):
    """Extent of the union of the 32 developed cubes M_{v,b}[0,1/2]^3 + a_{v,b} along each axis."""
    corners = np.array([[a, b, c] for a in (0.0, 0.5) for b in (0.0, 0.5) for c in (0.0, 0.5)])
    pts = []
    for b in range(32):
        M, a = developed_map(v, b)
        pts.append(corners @ M.T + a)
    pts = np.concatenate(pts)
    return pts.max(axis=0) - pts.min(axis=0)


def periods_from_colours(v):
    """Second route: on axis i the cube faces X_i=0 (vertical facet) and X_i=1/2 (sphere) generate the
    translation by 1; it is a deck translation iff their colours coincide (period 1), else period 2."""
    out = []
    for i in range(3):
        vert = [f for f in FACETS if f[1] == "v" and f[2] == i and f[3] == v[i]][0]
        sph = [f for f in FACETS if f[1] == "s" and _incident(f, v) == (i, 0.5)][0]
        out.append(1.0 if vert[4] == sph[4] else 2.0)
    return np.array(out)


CUSP_PERIODS = np.stack([periods_from_development(v) for v in BASE_VERTICES])
if not np.array_equal(CUSP_PERIODS, np.stack([periods_from_colours(v) for v in BASE_VERTICES])):
    raise AssertionError("cusp periods: development and colour routes disagree")


def _catalogue():
    recs = {}
    cells = []
    for j, v in enumerate(BASE_VERTICES):
        inward = np.diag(1.0 - 2.0 * v)
        for b in range(32):
            Mv, av = developed_map(v, b)
            Ab = COPY_MATRICES[b][1:, 1:]
            p = Ab @ v + COPY_OFFSETS[b][1:]
            O = Mv @ inward @ Ab.T
            key = tuple(np.round(p, 12))
            recs.setdefault(key, []).append(dict(cusp=j, copy=b, O=O, a=av, M=Mv))
            cells.append(dict(cusp=j, copy=b, key=key, M=Mv, a=av))
    for key, r in recs.items():
        if any(c["cusp"] != r[0]["cusp"] for c in r):
            raise AssertionError(f"site {key} shared by two cusp classes")
    # canonical rule: cusp class first, then lexicographic centre (= m4lc.geometry.cusps numbering)
    local = sorted(recs, key=lambda k: (recs[k][0]["cusp"], tuple(int(round(t)) for t in k)))
    keys, source = local, "local-rule(cusp,lexicographic)"
    return keys, recs, cells, source, local


_KEYS, _RECS, _CELLS, SITE_NUMBERING_SOURCE, _LOCAL_KEYS = _catalogue()
SITE_ORDER_MATCHES_LOCAL_RULE = list(_KEYS) == list(_LOCAL_KEYS)
NSITES = len(_KEYS)
SITES = np.array(_KEYS, dtype=float)
SITE_INDEX = {k: i for i, k in enumerate(_KEYS)}
SITE_CUSP = np.array([_RECS[k][0]["cusp"] for k in _KEYS], dtype=np.int32)
SITE_FRAMES = np.stack([_RECS[k][0]["O"] for k in _KEYS])
SITE_PHASE = np.stack([np.mod(_RECS[k][0]["a"], CUSP_PERIODS[_RECS[k][0]["cusp"]]) for k in _KEYS])
CELL_SITE = np.zeros((4, 32), dtype=np.int32)
CELL_M = np.zeros((4, 32, 3, 3))
CELL_A = np.zeros((4, 32, 3))
for _c in _CELLS:
    CELL_SITE[_c["cusp"], _c["copy"]] = SITE_INDEX[_c["key"]]
    CELL_M[_c["cusp"], _c["copy"]] = _c["M"]
    CELL_A[_c["cusp"], _c["copy"]] = _c["a"]

# session-1 numbering = global lexicographic sort of the centres (also the project's CUSP_CENTERS order)
LEXI_SITE_ORDER = sorted(_KEYS)
CANONICAL_OF_LEXI = np.array([SITE_INDEX[k] for k in LEXI_SITE_ORDER], dtype=np.int32)
LEXI_OF_CANONICAL = np.argsort(CANONICAL_OF_LEXI).astype(np.int32)


def site_index(center, cusp=None):
    """Canonical site index of an integer site centre (3,), optionally checking the cusp class."""
    k = tuple(np.round(np.asarray(center, dtype=float), 12))
    i = SITE_INDEX[k]
    if cusp is not None and int(SITE_CUSP[i]) != int(cusp):
        raise KeyError(f"site {k} has cusp class {SITE_CUSP[i]}, not {cusp}")
    return i


def site_map_from_centers(centers):
    """Map a foreign site table (list of centres in ITS order) to canonical indices: out[foreign] = ours."""
    return np.array([site_index(c) for c in np.asarray(centers, dtype=float)], dtype=np.int32)


def site_consistency():
    """Per site: max frame disagreement and max phase disagreement (mod P) among its cells."""
    out = []
    for k in _KEYS:
        r = _RECS[k]
        j = r[0]["cusp"]
        P = CUSP_PERIODS[j]
        dO = max(float(np.max(np.abs(c["O"] - r[0]["O"]))) for c in r)
        da = 0.0
        for c in r:
            d = np.mod(c["a"] - r[0]["a"] + P / 2, P) - P / 2
            da = max(da, float(np.max(np.abs(d))))
        same_class = all(c["cusp"] == j for c in r)
        out.append(dict(site=SITE_INDEX[k], center=list(k), cusp=j, cells=len(r),
                        frame_spread=dO, phase_spread=da, same_class=same_class,
                        orthogonal=float(np.max(np.abs(r[0]["O"] @ r[0]["O"].T - np.eye(3))))))
    return out


def to_cusp(x, site):
    x = jnp.asarray(x)
    d = x[1:] - jnp.asarray(SITES)[site]
    q = x[0] ** 2 + d @ d
    return jnp.concatenate([(x[0] / q)[None], (jnp.asarray(SITE_FRAMES)[site] @ d) / q])


def from_cusp(c, site):
    c = jnp.asarray(c)
    q = c @ c
    return jnp.concatenate([(c[0] / q)[None], jnp.asarray(SITES)[site] + jnp.asarray(SITE_FRAMES)[site].T @ c[1:] / q])


def heights(x):
    """t_p(x) for all 37 sites (jax)."""
    x = jnp.asarray(x)
    d = x[None, 1:] - jnp.asarray(SITES)
    return x[0] / (x[0] ** 2 + jnp.sum(d * d, axis=1))


def cusp_height(x):
    h = heights(x)
    s = jnp.argmax(h)
    return s, h[s]


def cusp_height_np(x):
    x = np.asarray(x, dtype=float)
    d = x[..., None, 1:] - SITES
    h = x[..., 0, None] / (x[..., 0, None] ** 2 + np.sum(d * d, axis=-1))
    s = np.argmax(h, axis=-1)
    return s, np.take_along_axis(h, s[..., None], axis=-1)[..., 0]


def region(x):
    x = np.asarray(x, dtype=float)
    s, t = cusp_height_np(x)
    if t > 1:
        return f"cusp{SITE_CUSP[s]}"
    if x[0] >= RJ:
        return "core"
    return "cap" if x[0] >= 1 else "body"


# ----------------------------------------------------------------------------
# reflections, collars, smooth step
# ----------------------------------------------------------------------------
def smooth_step(a):
    """s(a) of eq. (step); inactive branches kept finite for AD."""
    a = jnp.asarray(a)
    inside = (a > 0) & (a < 1)
    s = jnp.where(inside, a, 0.5)
    e1 = jnp.exp(-1.0 / s)
    e2 = jnp.exp(-1.0 / (1.0 - s))
    return jnp.where(a <= 0, 0.0, jnp.where(a >= 1, 1.0, e1 / (e1 + e2)))


def sphere_reflect(x, i):
    c = jnp.asarray(SPHERE_CENTERS4)[i]
    d = x - c
    return c + d / (d @ d)


def signed_distances(u):
    """d_i = arsinh((|u-(0,c_i)|^2-1)/(2z)), i=1..4, positive inside P."""
    u = jnp.asarray(u)
    d2 = jnp.sum((u[None, :] - jnp.asarray(SPHERE_CENTERS4)) ** 2, axis=1)
    return jnp.arcsinh((d2 - 1.0) / (2.0 * u[0]))


# ----------------------------------------------------------------------------
# small self-description for the README / cross-check with WS3
# ----------------------------------------------------------------------------
def manifest():
    return dict(rj=RJ, ell=ELL, omega=OMEGA, F=F.tolist(), Lambda=LAMBDA.tolist(), angles=ANGLES.tolist(),
                beta=BETA.tolist(), sites=SITES.tolist(), site_cusp=SITE_CUSP.tolist(),
                site_frames=SITE_FRAMES.tolist(), site_phase=SITE_PHASE.tolist(),
                cusp_site_counts=[int(np.sum(SITE_CUSP == j)) for j in range(4)])
