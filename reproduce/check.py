#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify the frozen companion. Run with --metrics to re-evaluate learned tensors.

Default checks use the saved jets and analytic metrics and take a few seconds.
--metrics compiles and evaluates each learned metric sequentially on 12 points
and two interface pairs. Compilation is measured separately from warm calls.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false --xla_cpu_parallel_codegen_split_count=1')
if hasattr(os,'sched_getaffinity'):
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from metric_demo import load_metric,geometry as G
from metric_demo import _curv
from metric_demo.numpy_curvature import diagnostics as numpy_diagnostics
ROOT=Path(__file__).resolve().parent


def assert_close(actual,expected,tolerance,label):
    a,b=np.asarray(actual),np.asarray(expected)
    error=float(np.max(np.abs(a-b))/max(1.,float(np.max(np.abs(b)))))
    if not np.isfinite(error) or error>tolerance:
        raise AssertionError(f'{label}: normalized difference {error:g} > {tolerance:g}')
    return error


def controls():
    p=jnp.array([1.1,.2,-.3,.1]);results=[]
    for label,fn,K in [('flat',lambda x:jnp.eye(4),0.),('hyperbolic',lambda x:jnp.eye(4)/x[0]**2,-1.),('sphere',lambda x:4*jnp.eye(4)/(1+x@x)**2,1.),('scaled-hyperbolic',lambda x:2*jnp.eye(4)/x[0]**2,-.5)]:
        jet=_curv.jets(fn,p); cv=_curv.curvature(*jet);ind=numpy_diagnostics(*map(np.asarray,jet))
        value=float(_curv.sectional(cv['R'],jet[0],jnp.eye(4)[0],jnp.eye(4)[1]))
        assert_close(value,K,2e-12,label+' K');assert_close(cv['S'],12*K,2e-12,label+' S');assert_close(ind['S'],cv['S'],2e-12,label+' independent S');assert_close(ind['E'],cv['E'],2e-12,label+' independent E')
        results.append({'control':label,'K':value,'S':float(cv['S']),'E':float(cv['E'])})
    caught=[]
    for label,actual,expected in [('wrong curvature sign',-results[1]['K'],-1.),('wrong normalization',results[3]['K'],-1.)]:
        try:assert_close(actual,expected,1e-10,label)
        except AssertionError:caught.append(label)
        else:raise AssertionError('negative control failed to reject '+label)
    # Nonzero extrinsic-curvature control: Euclidean unit sphere at its north pole.
    p=np.array([1.,0,0,0]);g=np.eye(4);dg=np.zeros((4,4,4));T=np.eye(4)[:,1:];B=np.zeros((4,3,3));B[0]=-np.eye(3)
    h,K=induced(g,dg,T,B,p);assert_close(K,h,1e-14,'unit sphere K=h')
    try:assert_close(-K,h,1e-14,'incorrect normal sign')
    except AssertionError:caught.append('incorrect nonzero-K normal sign')
    else:raise AssertionError('orientation negative control not detected')
    return {'analytic':results,'deliberate_failures_detected':caught}


def induced(g,dg,T,B,nu):
    gi=np.linalg.inv(g);first=np.empty((4,4,4))
    for l in range(4):
        for j in range(4):
            for k in range(4):first[l,j,k]=.5*(dg[l,k,j]+dg[l,j,k]-dg[j,k,l])
    gamma=np.tensordot(gi,first,axes=([1],[0]))
    ncov=nu/np.sqrt(nu@gi@nu)
    return T.T@g@T,-np.einsum('i,iab->ab',ncov,B+np.einsum('ijk,ja,kb->iab',gamma,T,T))


def main(args):
    start=time.perf_counter();manifest=json.loads((ROOT/'fixtures/manifest.json').read_text());geometry=json.loads((ROOT/'fixtures/geometry.json').read_text())
    report={'date':'2026-10-25','scope':'Bounded two-state reproduction; no training or new manifold-wide claim.','python':sys.version.split()[0],'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend(),'precision':'float64','controls':controls(),'states':{},'sample_count_per_state':len(manifest['samples'])}
    for key,actual in [('sites',G.SITES),('site_cusp',G.SITE_CUSP),('site_frames',G.SITE_FRAMES),('cusp_periods',G.CUSP_PERIODS)]:
        if not np.array_equal(actual,geometry[key]):raise AssertionError('geometry fixture mismatch '+key)
    report['canonical_sites_verified']=37
    for name in ('local','compatible'):
        model=load_metric(name);path=ROOT/f'fixtures/{name}-reference.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[name+'_fixture_sha256']:raise AssertionError('reference fixture hash')
        z=np.load(path,allow_pickle=False);rows=[]
        for index,sample in enumerate(manifest['samples']):
            ref={k:z[f'{index}/{k}'] for k in ('g','dg','d2g')};cv=numpy_diagnostics(**ref)
            assert_close(cv['E'],z[f'{index}/E'],1e-10,'saved NumPy E');assert_close(cv['S'],z[f'{index}/scalar'],1e-10,'saved NumPy S')
            row={'id':sample['id'],'E':cv['E'],'S':cv['S'],'saved_derivative_reconstruction_passed':True}
            model.validate(sample['point'],sample['chart'],sample['site'])
            if args.metrics:
                t=time.perf_counter();got=model.jet(sample['point'],sample['chart'],sample['site']);row['cold_or_cached_seconds']=time.perf_counter()-t
                row['errors']={k:assert_close(got[k],ref[k],tol,sample['id']+' '+k) for k,tol in [('g',2e-10),('dg',2e-9),('d2g',2e-8)]}
                t=time.perf_counter();warm=model.jet(sample['point'],sample['chart'],sample['site']);row['warm_seconds']=time.perf_counter()-t
                independent=numpy_diagnostics(**got);row['E_difference']=assert_close(independent['E'],cv['E'],2e-9,'recomputed E');row['S_difference']=assert_close(independent['S'],cv['S'],2e-9,'recomputed S')
                print(name,sample['id'],row['cold_or_cached_seconds'],row['warm_seconds'],flush=True)
            rows.append(row)
        faces=np.load(ROOT/f'fixtures/{name}-interfaces.npz',allow_pickle=False);face_rows=[]
        for f in range(2):
            r={'id':f,'length_distortion':float(faces[f'{f}/length_distortion']),'bending_max':float(faces[f'{f}/K_directional_max'])}
            for side in ('a','b'):
                g,dg=faces[f'{f}/g_{side}'],faces[f'{f}/dg_{side}']
                if args.metrics:
                    jet=model.jet(faces[f'{f}/native_{side}'],'core',order=1)
                    assert_close(jet['g'],g,2e-10,'interface g');assert_close(jet['dg'],dg,2e-9,'interface dg');g,dg=jet['g'],jet['dg']
                h,K=induced(g,dg,faces[f'{f}/tangent_{side}'],faces[f'{f}/embedding_hessian_{side}'],faces[f'{f}/canonical_conormal_{side}'])
                assert_close(h,faces[f'{f}/h_{side}'],2e-10,'interface h');assert_close(K,faces[f'{f}/K_{side}'],2e-9,'interface K')
            face_rows.append(r)
        rejected=0
        for p,c,s in [([G.RJ,0,0,0],'ordinary',None),([.5,.2,.2,.2],'cusp',0),([1,9,9,9],'ordinary',None),([0,0,100,100],'core',None)]:
            try:model.validate(p,c,s)
            except ValueError:rejected+=1
            else:raise AssertionError('invalid point accepted')
        report['states'][name]={'checkpoint':model.metadata['checkpoint'],'step':model.metadata['step'],'checkpoint_sha256':model.metadata['checkpoint_sha256'],'export_sha256':model.metadata['export_sha256'],'samples':rows,'interfaces':face_rows,'invalid_inputs_rejected':rejected,'fresh_metric_evaluation':args.metrics}
        del model
        jax.clear_caches()
    report['total_seconds']=time.perf_counter()-start;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;report['status']='passed'
    if args.output:
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'seconds':report['total_seconds'],'peak_rss_mib':report['peak_rss_mib'],'fresh_metrics':args.metrics},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--metrics',action='store_true');parser.add_argument('--output',type=Path);main(parser.parse_args())
