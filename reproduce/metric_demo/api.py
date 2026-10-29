# SPDX-License-Identifier: GPL-3.0-or-later
"""Small public API; loading never accepts pickle or imports training code."""
from pathlib import Path
import hashlib
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from . import _geom as geometry
from . import _curv
from .models import RawBody, PeriodicBody, DeformedMetric, QuotientMetric, DEFAULT_READINGS, _layers

ROOT = Path(__file__).resolve().parents[1]


def _core_contains(y, tol=1e-10):
    z = float(geometry.core_z(y))
    if z <= 0 or z > geometry.RJ + tol:
        return False
    if np.hypot(y[0], y[1]) > 1e-14:
        return bool(geometry.contains(np.asarray(geometry.from_core(y)), tol=tol))
    # The exact core is the projection of the horizontal box along the filled slope.
    # A sigma exists iff the three intervals implied by the box have nonempty intersection.
    offset = geometry.F[1:].T @ y[2:]
    v = geometry.F[0]
    a, b = (geometry.BOX_LO-offset)/v, (geometry.BOX_HI-offset)/v
    return bool(np.max(np.minimum(a,b)) <= np.min(np.maximum(a,b)) + tol)


class Metric:
    """Use metric(point, chart='core'|'ordinary'|'cusp', site=None).

    Invalid points raise ValueError. Native cusp inputs have t>=1. The ordinary
    chart cannot be used at the filled tip. Native site numbering is explicitly
    recorded in fixtures/geometry.json. No raw-coordinate folding is performed.
    """
    def __init__(self, name, model, metadata):
        self.name, self._model, self.metadata = name, model, metadata
        self._compiled = {}

    def validate(self, point, chart, site=None):
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (4,) or not np.all(np.isfinite(p)):
            raise ValueError('point must contain four finite numbers')
        if chart == 'ordinary':
            if not 0 < p[0] < geometry.RJ:
                raise ValueError('ordinary coordinates require 0 < z < r_j; use core at the tip')
            ok = bool(geometry.contains(p))
        elif chart == 'core':
            ok = _core_contains(p)
        elif chart == 'cusp':
            if site is None or not isinstance(site, (int,np.integer)) or not 0 <= int(site) < 37:
                raise ValueError('cusp chart requires an integer site in 0..36')
            if p[0] < 1:
                raise ValueError('native cusp inputs require t >= 1')
            ok = bool(geometry.contains(np.asarray(geometry.from_cusp(p,int(site)))))
        else:
            raise ValueError('chart must be ordinary, core or cusp')
        if not ok:
            raise ValueError('point lies outside the displayed coordinate domain')
        return p

    def jet(self, point, chart='core', site=None, order=2):
        p = self.validate(point, chart, site)
        if order not in (0,1,2):
            raise ValueError('order must be 0, 1 or 2')
        key = (chart,order)
        if key not in self._compiled:
            def one(q, s):
                fn=lambda y: self._model.physical(y,chart,s)
                out={'g':fn(q)}
                if order>=1:out['dg']=jax.jacfwd(fn)(q)
                if order>=2:out['d2g']=jax.jacfwd(jax.jacfwd(fn))(q)
                return out
            self._compiled[key]=jax.jit(one)
        out={k:np.asarray(v) for k,v in self._compiled[key](jnp.asarray(p),jnp.asarray(0 if site is None else site,dtype=jnp.int32)).items()}
        if not all(np.all(np.isfinite(v)) for v in out.values()):
            raise FloatingPointError('nonfinite metric or derivatives; point is not discarded')
        if np.linalg.eigvalsh(out['g'])[0] <= 0:
            raise FloatingPointError('metric is not numerically positive definite')
        return out

    def metric(self, point, chart='core', site=None):
        return self.jet(point,chart,site,order=0)['g']

    def diagnostics(self, point, chart='core', site=None):
        out=self.jet(point,chart,site,order=2)
        cv=_curv.curvature(*[jnp.asarray(out[k]) for k in ('g','dg','d2g')])
        result={'E':float(cv['E']), 'S':float(cv['S']), 'scalar_deviation':float((cv['S']+12)/12),
                'condition_number':float(np.linalg.cond(out['g'])), 'valid':True}
        if not all(np.isfinite(v) for k,v in result.items() if k!='valid'):
            raise FloatingPointError('nonfinite curvature; point is not discarded')
        return result


def load_metric(name='local'):
    """Load local (R4-022) or compatible (R4-087), checking the export hash."""
    if name not in ('local','compatible'):
        raise ValueError('name must be local or compatible')
    path=ROOT/'weights'/f'{name}.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['export_sha256']:
        raise ValueError('parameter export hash mismatch')
    with np.load(path,allow_pickle=False) as archive:
        params={key:np.array(archive[key]) for key in archive.files}
    if any(v.dtype!=np.float64 or not np.all(np.isfinite(v)) for v in params.values()):
        raise ValueError('all parameter arrays must be finite float64')
    if {key:list(value.shape) for key,value in params.items()}!=meta['parameter_shapes']:
        raise ValueError('parameter shapes differ from the frozen specification')
    rep=meta['representation']; rd=dict(DEFAULT_READINGS,raw_fold='none')
    if name=='local':
        body=RawBody(_layers(params,'layers'),rep,rd)
        model=DeformedMetric(name,params,meta,body,readings=rd)
    else:
        body=PeriodicBody(_layers(params,'body/layers'),rep['body'],rd)
        model=QuotientMetric(name,params,meta,body,_layers(params,'native'),rep,readings=rd)
    return Metric(name,model,meta)
