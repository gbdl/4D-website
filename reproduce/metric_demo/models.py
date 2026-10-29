# SPDX-License-Identifier: GPL-3.0-or-later
"""Clean-room g_loc (raw / round3 black-hole-relative), periodic and g_quot metrics.

Formulas: two-approach note constructions.tex eqs. (lower) (local) (rawfeatures)
(deformation) (untwist) (periodicfeatures) (periodictensor) (distance) (weights)
(projector) (native) (blend); appendix.tex (raw scaling, B_w, frames, phases);
report math-appendix.tex (logit order, gated cusps, reference tensor).

All public metric functions: one point (4,) -> 4x4 physical metric, jax-traceable.
Hyper-parameters come from the checkpoint metadata (never from code defaults).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from . import _geom as G
from . import _nets as N



DEFAULT_READINGS = dict(
    fourier_order="sin-block-coord-major",   # raw Fourier feature order (A1)
    scaling="note",                          # raw input scaling m,s (A2)
    lower_order="row",                       # L(o) logit order (A3)
    cusp_layout="triu-row",                  # C_j layout (A5)
    anchor_factor="chol",                    # A_BH = chol(H_BH) (lower Cholesky) (A4)
    deformation="native-DtgD",               # where/how D_j acts for raw/periodic (A6)
    periodic_order="sin-block",              # periodic body feature order (A7)
    native_inputs="inv-sin-cos",             # native-net input order (A8)
    native_offsets=True,                     # developed site offsets in the native phases (A8)
    native_gate="logits",                    # A = L(t^-3 f): t^-3 multiplies all ten logits (A8)
    native_D="DtAAtD",                       # g_nat = t^-2 D^T A A^T D (A8)
    outside="translate",                     # points outside the slab: periodic translation (A9)
    raw_fold="none",                       # optional alternative reading: translate points outside the CLOSED slab box by the
                                             # lattice 2Zx4Zx4Z before evaluating ('none' = project-literal) (A10)
)

# Red-team metric-level mutants (session 2, F6 item 5).  Each is a deliberate deviation from the note,
# selected with load(..., mutant=name); they are NOT readings (no reading choice is open there).
MUTANTS = {
    "eps0.15": dict(eps=0.15),                  # projector width 0.15 (code default) instead of 0.2 (quotient)
    "D_twice": dict(D_power=2),                 # cusp deformation D_j applied twice
    "L_transpose": dict(L_transpose=True),      # L^T (upper triangular) instead of L in every factor
    "site_renumber": dict(site_renumber=True),  # cusp-chart site index read in the session-1 (lexicographic) order
    "arctan2_slip": dict(arctan2_slip=True),    # inverse core chart on the branch theta in [0, 2 pi)
    "theta_flip": dict(theta_flip=True),        # theta -> -theta in the regular chart (V -> -V)
}


def _p(x):
    return jnp.asarray(x, dtype=jnp.float64)


# =============================================================================
# regular-chart maps (with the chart mutants)
# =============================================================================
class Chart:
    """x <-> y maps; the identity reading is G.to_core / G.from_core (principal branch)."""

    def __init__(self, theta_flip=False, arctan2_slip=False):
        self.theta_flip = bool(theta_flip)
        self.arctan2_slip = bool(arctan2_slip)
        self._flip = jnp.array([1.0, -1.0, 1.0, 1.0])
        self._v = jnp.concatenate([jnp.zeros(1), jnp.asarray(G.SLOPE_VECTOR, dtype=jnp.float64)])

    def to_core(self, x):
        y = G.to_core(x)
        return y * self._flip if self.theta_flip else y

    def from_core(self, y):
        yy = y * self._flip if self.theta_flip else y
        x = G.from_core(yy)
        if self.arctan2_slip:     # theta in [0, 2 pi): sigma -> sigma + l, i.e. x_hor -> x_hor + v
            x = jnp.where(yy[1] < 0.0, x + self._v, x)
        return x


IDENTITY_CHART = Chart()


# =============================================================================
# bodies: undeformed y-chart physical metric
# =============================================================================
class RawBody:
    """g_theta,y = z(y)^-2 A L(f(y)) L(f(y))^T A^T  (constructions eq. local)."""

    def __init__(self, layers, hyper, readings, anchor_body=None, L_transpose=False):
        self.layers = [(_p(w), _p(b)) for w, b in layers]
        self.h = hyper
        self.r = readings
        self.anchor_body = anchor_body
        self.L_transpose = bool(L_transpose)
        kind = hyper["model_kind"]
        if kind not in ("black_hole_relative", "smooth_relative", "raw", "learned_relative"):
            raise ValueError("unsupported raw model_kind " + kind)
        if (kind == "learned_relative") != (anchor_body is not None):
            raise ValueError("learned_relative needs exactly one embedded anchor body")
        mode = hyper.get("feature_mode", "cartesian")
        self.height = mode in ("height", "both")
        self.fourier = mode in ("fourier", "both")
        self.freqs = tuple(float(k) for k in hyper.get("fourier_frequencies", ()))
        nin = 4 + int(self.height) + (8 * len(self.freqs) if self.fourier else 0)
        if self.layers[0][0].shape[0] != nin:
            raise ValueError(f"input count {self.layers[0][0].shape[0]} != {nin} implied by metadata")

    def anchor(self, y):
        kind = self.h["model_kind"]
        if kind == "black_hole_relative":
            return N.h_bh(y)
        if kind == "smooth_relative":
            return reference_compact(y)
        if kind == "learned_relative":      # undeformed compact tensor of the frozen parent (report eq. relative)
            return self.anchor_body.compact(y)
        return None

    def logits(self, y):
        v = N.raw_features(y, self.freqs, self.r["fourier_order"], self.r["scaling"], self.height, self.fourier)
        return N.mlp(self.layers, v, self.h["activation"])

    def compact(self, y):
        Lm = N.lower(self.logits(y), self.r["lower_order"])
        if self.L_transpose:
            Lm = Lm.T
        H0 = self.anchor(y)
        A = Lm if H0 is None else N.ANCHOR_FACTORS[self.r["anchor_factor"]](H0) @ Lm
        return A @ A.T

    def bare_y(self, y):
        return self.compact(y) / G.core_z(y) ** 2


class PeriodicBody:
    """H_per = J^T A~ L L^T A~^T J in the untwisted coframe (constructions eq. periodictensor)."""

    def __init__(self, layers, hyper, readings, L_transpose=False):
        self.layers = [(_p(w), _p(b)) for w, b in layers]
        self.h = hyper
        self.r = readings
        self.L_transpose = bool(L_transpose)
        if hyper.get("model_kind") not in ("black_hole_relative", "smooth_relative"):
            raise ValueError("periodic body needs an analytic anchor")
        if list(hyper.get("slope", [5, 1, 1])) != [5, 1, 1] or [float(p) for p in hyper.get("periods", [2, 4, 4])] != [2.0, 4.0, 4.0]:
            raise ValueError("metadata slope/periods differ from the note")
        if "completion" in hyper and np.array(hyper["completion"]).tolist() != G.A_PER.astype(int).tolist():
            raise ValueError("metadata periodic completion differs from the note's A_p")
        self.height = bool(hyper.get("periodic_height", True))
        self.freqs = tuple(int(k) for k in hyper["fourier_frequencies"])
        nin = 2 + int(self.height) + 4 * len(self.freqs)
        if self.layers[0][0].shape[0] != nin:
            raise ValueError(f"periodic input count {self.layers[0][0].shape[0]} != {nin}")

    def anchor_untwisted(self, xi):
        if self.h["model_kind"] == "black_hole_relative":
            return N.untwisted_anchor(xi)
        j2xi = jnp.stack([-xi[1], xi[0]])
        B = jnp.eye(4).at[:2, 2:].set(jnp.outer(j2xi, jnp.asarray(G.ANGLES))).at[2:, 2:].set(jnp.asarray(G.LAMBDA))
        return B.T @ reference_compact(jnp.concatenate([xi, jnp.zeros(2)])) @ B

    def compact(self, y):
        w, J = N.untwist(y)
        xi, eta = w[:2], w[2:]
        v = N.periodic_features(xi, eta, G.core_z(y), self.freqs, self.r["periodic_order"], self.height)
        Lm = N.lower(N.mlp(self.layers, v, self.h["activation"]), self.r["lower_order"])
        if self.L_transpose:
            Lm = Lm.T
        A = N.ANCHOR_FACTORS[self.r["anchor_factor"]](self.anchor_untwisted(xi)) @ Lm
        A = J.T @ A
        return A @ A.T

    def bare_y(self, y):
        return self.compact(y) / G.core_z(y) ** 2


def reference_compact(y):
    """Smooth reference H_0 (geometry eq. reference; math-appendix 'formula finite at the tip')."""
    z = G.core_z(y)
    bh = N.h_bh(y)
    v = y[:2]
    rho2 = jnp.where(z < 1.5, v @ v, 1.0)
    Ar = 9.0 * rho2 / (4.0 * G.RJ ** 2)
    At = 1.0 / (G.OMEGA ** 2 * rho2)
    hyp = jnp.eye(4).at[:2, :2].set(At * jnp.eye(2) + (Ar - At) * jnp.outer(v, v) / rho2)
    chi = G.smooth_step(2.0 * (z - 1.0))
    return jnp.where(z >= 1.5, bh, (1.0 - chi) * hyp + chi * bh)


# =============================================================================
# common chart utilities
# =============================================================================
def _y_of_x(x):
    return G.to_core(x)


def _pull_y_to_x(gy_fn, x, to_core=G.to_core):
    """(dy/dx)^T g_y(y(x)) (dy/dx)."""
    J = jax.jacfwd(to_core)(x)
    return J.T @ gy_fn(to_core(x)) @ J


def _pull_y_to_c(gy_fn, c, site, to_core=G.to_core):
    f = lambda c: to_core(G.from_cusp(c, site))
    J = jax.jacfwd(f)(c)
    return J.T @ gy_fn(f(c)) @ J


# Session-2 precision fix: the physical tensor is (compact)/z^2 with z the ORDINARY height x[0] (exact in
# both charts), never z(y) = r_j - 3 rho^2/(4 r_j) recomputed from the regular chart, which cancels
# catastrophically for small z (relative error ~ eps r_j / z ~ eps t in the cusps; finding WS5-003).
def _pull_compact_to_x(compact_fn, x, to_core=G.to_core):
    """(dy/dx)^T H(y(x)) (dy/dx) / x_0^2."""
    J = jax.jacfwd(to_core)(x)
    return J.T @ compact_fn(to_core(x)) @ J / x[0] ** 2


def _pull_compact_to_c(compact_fn, c, site, to_core=G.to_core):
    """(dy/dc)^T H(y(x(c))) (dy/dc) / z(c)^2 with z(c) = t/|c|^2."""
    f = lambda cc: to_core(G.from_cusp(cc, site))
    J = jax.jacfwd(f)(c)
    z = c[0] / (c @ c)
    return J.T @ compact_fn(f(c)) @ J / z ** 2


SAFE_Y = jnp.array([1.0, 1.0, 0.3, 0.2])          # z(SAFE_Y) ~ 2.27: inactive-branch filler
SAFE_X = jnp.array([0.8, 0.5, 0.5, 0.5])          # inside P, no collar factor singularity
SAFE_C = jnp.array([1.5, 0.1, 0.1, 0.1])


class _MetricBase:
    family = "?"

    def __init__(self, name, params, meta, readings=None, mutant=None):
        self.name = name
        self.flat = params
        self.meta = meta
        self.r = dict(DEFAULT_READINGS)
        if readings:
            unknown = set(readings) - set(DEFAULT_READINGS)
            if unknown:
                raise ValueError(f"unknown readings {unknown}")
            self.r.update(readings)
        self.mutant = mutant
        self.mut = _mutant_dict(mutant)
        self.chart = Chart(theta_flip=self.mut.get("theta_flip", False),
                           arctan2_slip=self.mut.get("arctan2_slip", False))
        self.domain = G
        self._cusps = _p(params["cusps"])
        if self._cusps.shape != (4, 6):
            raise ValueError("cusps must have shape (4,6)")
        self._jit_cache = {}

    def _site_in(self, site):
        """Site index as received by a public cusp-chart call (mutant: session-1 numbering)."""
        if self.mut.get("site_renumber"):
            return jnp.asarray(G.CANONICAL_OF_LEXI)[site]
        return site

    # ---- cusp matrices
    def cusp_matrix(self, j):
        return N.cusp_matrix(self._cusps[j], self.r["cusp_layout"])

    # ---- dispatcher and helpers
    def physical(self, p, chart, site=None, copy=None):
        if chart == "core":
            return self.core(p)
        if chart == "cusp":
            return self.cusp(p, site)
        if chart == "ordinary":
            return self.ordinary(p, copy)
        if chart == "base":
            return self.base(p, copy)
        raise ValueError(chart)

    def _chart_fn(self, chart):
        """(point, site, copy) -> 4x4 for physical charts and a few diagnostic pieces."""
        if chart in ("core", "cusp", "ordinary", "base"):
            return lambda q, s, b: self.physical(q, chart, s, b)
        if chart == "bare_core":
            return lambda q, s, b: self.bare_core(q)
        if chart == "native":
            return lambda q, s, b: self.native_metric(q, self._site_in(s))
        raise ValueError(chart)

    def base(self, u, copy):
        M = jnp.asarray(G.COPY_MATRICES)[copy]
        o = jnp.asarray(G.COPY_OFFSETS)[copy]
        return M.T @ self.ordinary(M @ u + o, copy) @ M

    def jet(self, chart, p, site=None, copy=None):
        fn = lambda q: self.physical(q, chart, site, copy)
        p = _p(p)
        return fn(p), jax.jacfwd(fn)(p), jax.jacfwd(jax.jacfwd(fn))(p)

    # ---- batched numpy evaluation
    def evaluate(self, chart, points, sites=None, copies=None, order=0, batch=256, curvature=False):
        """Chunked jit(vmap) evaluation. Returns dict of numpy arrays: g (+dg,d2g) (+E,S)."""
        from . import _curv
        points = np.asarray(points, dtype=np.float64)
        n = len(points)
        sites = np.zeros(n, np.int32) if sites is None else np.asarray(sites, np.int32)
        use_copy = copies is not None
        copies = np.zeros(n, np.int32) if copies is None else np.asarray(copies, np.int32)
        key = (chart, order, curvature, use_copy)
        if key not in self._jit_cache:
            def one(p, s, b):
                fn = lambda q: self._chart_fn(chart)(q, s, b if use_copy else None)
                out = dict(g=fn(p))
                if order >= 1 or curvature:
                    out["dg"] = jax.jacfwd(fn)(p)
                if order >= 2 or curvature:
                    out["d2g"] = jax.jacfwd(jax.jacfwd(fn))(p)
                if curvature:
                    cv = _curv.curvature(out["g"], out["dg"], out["d2g"])
                    out["E"] = cv["E"]
                    out["S"] = cv["S"]
                return out
            self._jit_cache[key] = jax.jit(jax.vmap(one))
        f = self._jit_cache[key]
        res = []
        for i in range(0, n, batch):
            sl = slice(i, min(i + batch, n))
            k = sl.stop - sl.start
            pad = batch - k
            P, S, B = points[sl], sites[sl], copies[sl]
            if pad:
                P = np.concatenate([P, np.repeat(P[-1:], pad, 0)])
                S = np.concatenate([S, np.repeat(S[-1:], pad, 0)])
                B = np.concatenate([B, np.repeat(B[-1:], pad, 0)])
            out = f(jnp.asarray(P), jnp.asarray(S), jnp.asarray(B))
            res.append({kk: np.asarray(v)[:k] for kk, v in out.items()})
        return {kk: np.concatenate([r[kk] for r in res]) for kk in res[0]}


# =============================================================================
# raw / periodic families with the cusp deformation D_j(t) (constructions eq. deformation)
# =============================================================================
class DeformedMetric(_MetricBase):
    """g = deformation of the bare y-chart field by D_j(t) in the native chart of the
    horoball containing the point (t>1); D_j = diag(1,(1-chi)I + chi C_j), chi = s((t-t0)/(t1-t0)),
    (t0,t1) = representation 'cusp_transition' when recorded, else the note's (1,2) (report eq.
    cuspdeformation: chi_t = s(t-1)); hyper['transition_source'] says which.

    Optional internal folding (not used by this companion): points outside the CLOSED displayed box
    [-1,1]x[-1,3]^2 are first translated by the cusp-at-infinity lattice 2Zx4Zx4Z (identifications of
    M4).  Inside the closed box nothing changes, so the project's values are reproduced there; outside it
    the project evaluates the network at the unfolded point and looks up D_j among slab sites only
    (raw_fold='none' reproduces that).  core(y) evaluates the body at y as given (the regular chart has no
    fold at the core circle) and folds only to find the horoball."""

    family = "raw"

    def __init__(self, name, params, meta, body, readings=None, gate_power=None, transition=(1.0, 2.0),
                 transition_source="note-default s(t-1)", mutant=None):
        super().__init__(name, params, meta, readings, mutant)
        self.body = body
        self.gate_power = gate_power
        self.t_lo, self.t_hi = (float(v) for v in transition)
        if not self.t_hi > self.t_lo:
            raise ValueError("cusp transition must have t1 > t0")
        if self.mut.get("eps") is not None:
            raise ValueError("mutant eps applies to the quotient projector only")
        self.D_power = int(self.mut.get("D_power", 1))
        self.hyper = dict(body.h, gate_power=gate_power, readings=dict(self.r), cusp_transition=[self.t_lo, self.t_hi],
                          transition_source=transition_source, mutant=mutant,
                          site_numbering=G.SITE_NUMBERING_SOURCE)

    # ---- bare field
    def bare_core(self, y):
        return self.body.bare_y(_p(y))

    def _chi(self, t):
        return G.smooth_step((t - self.t_lo) / (self.t_hi - self.t_lo))

    def D(self, t, site):
        chi = self._chi(t)
        C = self.cusp_matrix(jnp.asarray(G.SITE_CUSP)[site])
        D = jnp.eye(4).at[1:, 1:].set((1.0 - chi) * jnp.eye(3) + chi * C)
        return D @ D if self.D_power == 2 else D

    def _Knat(self, t, site):
        mode = self.r["deformation"]
        if mode in ("native-DtgD", "ordinary-diag"):
            return self.D(t, site)
        if mode == "native-DgDt":
            return self.D(t, site).T
        if mode == "none":
            return jnp.eye(4)
        raise ValueError(mode)

    def _gate(self, t):
        if self.gate_power is None:
            return jnp.ones(())
        chi = self._chi(t)
        ts = jnp.where(t > 1.0, t, 1.0)
        return 1.0 - chi + chi * ts ** (-self.gate_power)

    def _deform_native(self, gnat, c, site):
        """Apply the (optional) whole-compact gate and D in the native chart of `site` (t=c0>1)."""
        t = c[0]
        a = self._gate(t)
        g = a * gnat + (1.0 - a) * jnp.eye(4) / t ** 2
        K = self._Knat(t, site)
        return K.T @ g @ K

    def _folded(self, x):
        return G.fold_closed(x) if self.r["raw_fold"] == "closed" else x

    # ---- charts
    def cusp(self, c, site):
        site = self._site_in(site)
        c = _p(c)
        x = G.from_cusp(c, site)
        outside = jnp.any(self._folded(x) != x)
        inside = (c[0] > 1.0) & jnp.logical_not(outside)
        gb = _pull_compact_to_c(self.body.compact, c, site, self.chart.to_core)
        own = self._deform_native(gb, c, site)
        # t<=1 at this site (the point may lie in ANOTHER site's horoball), or outside the closed slab box:
        # pull back the ordinary tensor (argmax site; folded).
        cs = jnp.where(inside, SAFE_C, c)
        Jxc = jax.jacfwd(lambda cc: G.from_cusp(cc, site))(cs)
        other = Jxc.T @ self.ordinary(G.from_cusp(cs, site)) @ Jxc
        return jnp.where(inside, own, other)

    def ordinary(self, x, copy=None):
        x = self._folded(_p(x))
        gb = _pull_compact_to_x(self.body.compact, x, self.chart.to_core)
        if self.r["deformation"] == "none":
            return gb
        s, t = G.cusp_height(x)
        if self.r["deformation"] == "ordinary-diag":
            K = self._Knat(t, s)
            a = self._gate(t)
            g = a * gb + (1.0 - a) * jnp.eye(4) / x[0] ** 2
            return jnp.where(t > 1.0, K.T @ g @ K, gb)
        c = G.to_cusp(x, s)
        Jcx = jax.jacfwd(lambda xx: G.to_cusp(xx, s))(x)
        gnat = jnp.linalg.solve(Jcx.T, jnp.linalg.solve(Jcx.T, gb).T).T   # Jcx^-T gb Jcx^-1
        gd = self._deform_native(gnat, c, s)
        out = Jcx.T @ gd @ Jcx
        return jnp.where(t > 1.0, out, gb)

    def core(self, y):
        y = _p(y)
        z = G.core_z(y)
        gb = self.body.bare_y(y)
        if self.r["deformation"] == "none":
            return gb
        ys = jnp.where(z < 1.0, y, SAFE_Y)
        xmap = lambda yy: self._folded(self.chart.from_core(yy))
        x = xmap(ys)
        s, t = G.cusp_height(x)
        if self.r["deformation"] == "ordinary-diag":
            Jxy = jax.jacfwd(xmap)(ys)
            gx = jnp.linalg.solve(Jxy.T, jnp.linalg.solve(Jxy.T, self.body.bare_y(ys)).T).T
            K = self._Knat(t, s)
            a = self._gate(t)
            gx = K.T @ (a * gx + (1.0 - a) * jnp.eye(4) / x[0] ** 2) @ K
            out = Jxy.T @ gx @ Jxy
            return jnp.where((z < 1.0) & (t > 1.0), out, gb)
        f = lambda yy: G.to_cusp(xmap(yy), s)
        c = f(ys)
        Jcy = jax.jacfwd(f)(ys)
        gnat = jnp.linalg.solve(Jcy.T, jnp.linalg.solve(Jcy.T, self.body.bare_y(ys)).T).T
        gd = self._deform_native(gnat, c, s)
        out = Jcy.T @ gd @ Jcy
        return jnp.where((z < 1.0) & (t > 1.0), out, gb)

    # ---- asymptotics
    def cusp_limit(self, site):
        """Q = lim t^2 g(t,X): the bare compact tensor at the ideal point, in the native
        frame at t=inf (reflection -> diag(-1,1,1,1)), deformed with C_j fully on."""
        site = int(site)
        p = G.SITES[site]
        # ordinary compact tensor A(x) = J^T H(y) J at z -> 0 (finite)
        x0 = jnp.array([0.0, *p])
        J = jax.jacfwd(self.chart.to_core)(x0)
        H = self.body.compact(self.chart.to_core(x0))
        A = np.asarray(J.T @ H @ J)
        R = np.eye(4)
        R[1:, 1:] = G.SITE_FRAMES[site].T
        R = R @ np.diag([-1.0, 1, 1, 1])
        Q0 = R.T @ A @ R
        if self.gate_power is not None:
            Q0 = np.eye(4)        # gate a(t)=t^-p -> 0: compact tensor -> I before D
        K = np.asarray(self._Knat(jnp.asarray(1e300), site))
        return K.T @ Q0 @ K


class PeriodicMetric(DeformedMetric):
    family = "periodic"


# =============================================================================
# quotient (g_quot): periodic body + reflection collars + native cusp networks
# =============================================================================
class QuotientMetric(_MetricBase):
    family = "quotient"

    def __init__(self, name, params, meta, body, native_layers, hyper, readings=None, mutant=None):
        super().__init__(name, params, meta, readings, mutant)
        self.body = body
        self.native = [(_p(w), _p(b)) for w, b in native_layers]     # w: (4,nin,nout)
        self.eps = float(hyper["projector_epsilon"]) if self.mut.get("eps") is None else float(self.mut["eps"])
        self.upper_z = math.exp(self.eps)
        self.D_power = int(self.mut.get("D_power", 1))
        self.L_transpose = bool(self.mut.get("L_transpose", False))
        self.t_lo, self.t_hi = (float(v) for v in hyper["cusp_transition"])
        self.gate_power = int(hyper["native"]["gate_power"])
        self.native_act = hyper["native"]["activation"]
        self.hyper = dict(hyper, readings=dict(self.r), mutant=mutant, projector_epsilon_used=self.eps,
                          site_numbering=G.SITE_NUMBERING_SOURCE)
        if hyper.get("deformation") != "native-C-once":
            raise ValueError("unexpected quotient deformation policy")
        if (self.t_lo, self.t_hi) != (1.0, 2.0):
            raise ValueError("the notes describe the transition s(t-1) on 1<t<2 only")
        words = np.arange(16)
        wb = (words[:, None] >> np.arange(4)) & 1
        self._words = jnp.asarray(wb)                               # (16,4) w_i
        self._colors = jnp.asarray(np.bitwise_xor.reduce(wb * G.SPHERE_COLOR_MASK, axis=1))

    # ---- body
    def bare_core(self, y):
        return self.body.bare_y(_p(y))

    def body_ordinary(self, x):
        return _pull_compact_to_x(self.body.compact, x, self.chart.to_core)

    # ---- reflection projector (constructions eqs. distance, weights, projector)
    def _reflect_word(self, u, wbits):
        """r_w(u) and D r_w(u) (spheres commute; composed in index order)."""
        pt, J = u, jnp.eye(4)
        for i in range(4):
            c = jnp.asarray(G.SPHERE_CENTERS4)[i]
            d = pt - c
            q = d @ d
            img = c + d / q
            Ji = jnp.eye(4) / q - 2.0 * jnp.outer(d, d) / q ** 2
            on = wbits[i] == 1
            pt, J = jnp.where(on, img, pt), jnp.where(on, Ji @ J, J)
        return pt, J

    def projected_base(self, u, b):
        """bar g_b(u) in base-cell coordinates."""
        d = G.signed_distances(u)
        sg = 1.0 - 2.0 * self._words                                 # (-1)^{w_i}
        om = jnp.prod(G.smooth_step((sg * d[None, :] + self.eps) / self.eps), axis=1)

        def term(wbits, col):
            pt, J = self._reflect_word(u, wbits)
            k = jnp.bitwise_xor(b, col)
            M = jnp.asarray(G.COPY_MATRICES)[k]
            o = jnp.asarray(G.COPY_OFFSETS)[k]
            gk = M.T @ self.body_ordinary(M @ pt + o) @ M            # T_k^* g_per
            return J.T @ gk @ J                                       # r_w^*

        T = jax.vmap(term)(self._words, self._colors)
        return jnp.einsum("w,wij->ij", om, T) / jnp.sum(om)

    def projected(self, x, b):
        """(T_b^{-1})^* bar g_b at displayed x of copy b."""
        M = jnp.asarray(G.COPY_MATRICES)[b]
        o = jnp.asarray(G.COPY_OFFSETS)[b]
        u = M.T @ (x - o)
        return M @ self.projected_base(u, b) @ M.T

    # ---- native networks (constructions eq. native)
    def native_metric(self, c, site):
        t = c[0]
        j = jnp.asarray(G.SITE_CUSP)[site]
        feats = N.native_features(c, jnp.asarray(G.SITE_PHASE)[site], jnp.asarray(G.CUSP_PERIODS)[j],
                                  self.r["native_inputs"], self.r["native_offsets"])
        layers = [(w[j], b_[j]) for w, b_ in self.native]
        f = N.mlp(layers, feats, self.native_act)
        mode = self.r["native_gate"]
        if mode == "logits":
            A = N.lower(f * t ** (-self.gate_power), self.r["lower_order"])
        elif mode == "offdiag":
            diag = np.array([i == k for i, k in N.LOWER_ORDERS[self.r["lower_order"]]])
            A = N.lower(jnp.where(jnp.asarray(diag), f, f * t ** (-self.gate_power)), self.r["lower_order"])
        elif mode == "none":
            A = N.lower(f, self.r["lower_order"])
        else:
            raise ValueError(mode)
        if self.L_transpose:
            A = A.T
        Dh = jnp.eye(4).at[1:, 1:].set(self.cusp_matrix(j))
        if self.D_power == 2:
            Dh = Dh @ Dh
        if self.r["native_D"] == "DtAAtD":
            return Dh.T @ A @ A.T @ Dh / t ** 2
        return Dh @ A @ A.T @ Dh.T / t ** 2

    # ---- folding of points outside the slab
    def _fold(self, x):
        if self.r["outside"] == "translate":
            return G.fold(x)
        # 'reflect': fold horizontal coordinates into the box by reflections (membership folding)
        h = x[1:]
        lo = jnp.array([-1.0, -1.0, -1.0])
        P = jnp.asarray(G.PERIODS)
        r = jnp.mod(h - lo, 2 * P)
        h2 = lo + jnp.where(r > P, 2 * P - r, r)
        xf = jnp.concatenate([x[:1], h2])
        return G.fold(xf)

    # ---- charts
    def _ordinary_parts(self, x, b):
        """Pieces of the ordinary-chart tensor at displayed x with copy label b."""
        upper = x[0] > self.upper_z
        s, t = G.cusp_height(x)
        deep = t >= self.t_hi
        M = jnp.asarray(G.COPY_MATRICES)[b]
        o = jnp.asarray(G.COPY_OFFSETS)[b]
        safe = M @ SAFE_X + o
        proj = self.projected(jnp.where(deep | upper, safe, x), b)
        gper = self.body_ordinary(jnp.where(upper, x, safe))
        c = G.to_cusp(x, s)
        Jcx = jax.jacfwd(lambda xx: G.to_cusp(xx, s))(x)
        tail = Jcx.T @ self.native_metric(jnp.where(t > 1.0, c, SAFE_C), s) @ Jcx
        w = G.smooth_step(t - self.t_lo)
        return upper, gper, proj, tail, w

    def _ordinary_labeled(self, x, b):
        upper, gper, proj, tail, w = self._ordinary_parts(x, b)
        return jnp.where(upper, gper, (1.0 - w) * proj + w * tail)

    def ordinary(self, x, copy=None):
        x = _p(x)
        if copy is None:
            x, b = self._fold(x)
        else:
            b = jnp.asarray(copy)
        return self._ordinary_labeled(x, b)

    def core(self, y):
        y = _p(y)
        upper = G.core_z(y) > self.upper_z
        ys = jnp.where(upper, jnp.array([1.5, 1.5, 0.3, 0.2]), y)     # finite filler for the unused branch
        Jxy = jax.jacfwd(self.chart.from_core)(ys)
        low = Jxy.T @ self.ordinary(self.chart.from_core(ys)) @ Jxy
        return jnp.where(upper, self.body.bare_y(y), low)

    def cusp(self, c, site):
        site = self._site_in(site)
        c = _p(c)
        t = c[0]
        deep = t >= self.t_hi
        nat = self.native_metric(jnp.where(t > 1.0, c, SAFE_C), site)
        cs = jnp.where(deep, SAFE_C, c)
        Jxc = jax.jacfwd(lambda cc: G.from_cusp(cc, site))(cs)
        xf, b = self._fold(G.from_cusp(cs, site))
        upper, gper, proj, tail, w2 = self._ordinary_parts(xf, b)
        w = G.smooth_step(t - self.t_lo)
        own = (1.0 - w) * (Jxc.T @ proj @ Jxc) + w * nat
        # t<=1 here: the point may lie above e^eps or in ANOTHER site's horoball: full ordinary tensor
        other = Jxc.T @ jnp.where(upper, gper, (1.0 - w2) * proj + w2 * tail) @ Jxc
        return jnp.where(deep, nat, jnp.where(t > 1.0, own, other))

    def cusp_limit(self, site):
        """lim t^2 g = diag(1, C^T C) since A=L(t^-3 f) -> I (constructions after eq. blend)."""
        j = int(G.SITE_CUSP[int(site)])
        C = np.asarray(self.cusp_matrix(j))
        if self.D_power == 2:
            C = C @ C
        out = np.eye(4)
        out[1:, 1:] = C.T @ C if self.r["native_D"] == "DtAAtD" else C @ C.T
        if self.r["native_gate"] == "none":
            raise ValueError("no closed-form limit for an ungated native head")
        return out


# =============================================================================
# loader
# =============================================================================
def _flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten(v, f"{prefix}/{k}" if prefix else k))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = np.asarray(tree)
    return out


def _layers(flat, prefix):
    out = []
    i = 0
    while f"{prefix}[{i}]/w" in flat:
        out.append((flat[f"{prefix}[{i}]/w"], flat[f"{prefix}[{i}]/b"]))
        i += 1
    if not out:
        raise KeyError(prefix)
    return out



def _mutant_dict(mutant):
    if mutant is None:
        return {}
    raise ValueError("This frozen companion does not support alternative readings")
