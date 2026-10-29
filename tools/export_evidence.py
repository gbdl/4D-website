#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Export public research evidence, or verify/regenerate tables from this bundle.

The optional --source-root argument reads the earlier research tree without
modifying it. No local source paths or execution records enter the export.
Default operation uses only bundled JSON and the Python standard library.
"""
from __future__ import annotations
import argparse, csv, hashlib, html, json, math, statistics
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'metrics': 'fillings-m4-lean/two-approach-note/data/evidence.json',
    'experiments': 'fillings-m4-lean/report/data/r4_evidence.json',
    'audit': 'fillings-m4-lean/two-approach-note/data/audit-20261023.json',
}
ROLES = ('raw022', 'quotient087')

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')

def public(x):
    """Remove operational locations/commands, retaining scientific identities."""
    if isinstance(x, list): return [public(v) for v in x]
    if isinstance(x, dict):
        return {k: public(v) for k,v in x.items()
                if k not in {'path', 'command', 'job_id', 'job_ids', 'hostname', 'username',
                             'audit_export_path', 'checkpoint_path', 'metadata_path'}
                and not k.endswith('_path')}
    if isinstance(x, str) and x.startswith(('/', 'file:', 'ssh:')):
        return '[operational location omitted]'
    return x

def export(source):
    data = {key: read(source / rel) for key,rel in FILES.items()}
    m,e,a = (public(data[k]) for k in ('metrics','experiments','audit'))
    # Relative provenance paths are also operational. Numbered records retain
    # source-file hashes, not a misleading promise that private files are bundled.
    for i,item in enumerate(m['provenance']): item['artifact'] = f'metric-source-{i:03d}'
    for i,item in enumerate(e['source_files']): item['artifact'] = f'experiment-source-{i:03d}'
    for name,item in a['sources'].items():
        a['sources'][name] = {k:v for k,v in item.items() if k in {'sha256','bytes','date','git_commit'}}
    e['accounting']['basis'] = 'Finite allocation records; child-step rows excluded. These are allocated CPU-hours, not productive compute estimates.'
    e['cohort'] = {'planned':96, 'submitted':80, 'completed':79, 'node_failure':1, 'unreleased':16,
                  'note':'This dated cohort is one part of the earlier search; continuations share founders.'}
    m['units'] = {'headline_rms_q99_maximum':'percent', 'regions_panels_maps_profiles_E':'dimensionless E; multiply by 100 for percent',
                  'scalar':'curvature-radius units; Einstein target -12', 'core_signed_angle_defect':'radians',
                  'length':'fractional tangent-length disagreement', 'bending':'inverse curvature-radius units',
                  'status':{'0':'outside saved domain mask', '1':'valid', '2':'numerical failure; never discard'}}
    m['map_labels'] = {
        'body':{'x':'x1','y':'z','fixed':'x2 = x3 = 0.5','chart':'ordinary'},
        'core-copy30':{'x':'U','y':'V','fixed':'tau1 = 2.8311042925, tau2 approximately 0','chart':'regular core'},
        **{f'cusp{j}':{'x':'base X1','y':'base X2','fixed':f'cusp {j}: t = 2, base X3 = 0.25','chart':'native inverted cusp'} for j in range(4)}}
    for name in m['states']['raw022']['profiles']:
        for role in ROLES:
            summary = m['states'][role]['profile_summary']['profiles'][name]
            m['states'][role]['profiles'][name]['description'] = summary['selector']
            m['states'][role]['profiles'][name]['chart'] = summary['chart']
    a['headlines'] = {}
    for role,prefix in [('raw022','raw'),('quotient087','quotient')]:
        a['headlines'][role] = {
            'observed_search_maximum_percent':100*a['measurements'][prefix+'_search_max']['value'],
            'ordinary_ridge_maximum_radians':a['measurements']['raw_ridge_search']['value'] if prefix=='raw' else a['measurements']['quotient_ridge_sample']['value']['max_abs'],
            'scope':'Finite search and separate ordinary-ridge audit; not a uniform bound.'}
        a['headlines'][role]['ordinary_ridge_maximum_degrees'] = math.degrees(a['headlines'][role]['ordinary_ridge_maximum_radians'])
    for name,obj in [('metrics',m),('experiments',e),('audit',a)]:
        obj['export'] = {'source_sha256':digest(source/FILES[name]),'license':'CC-BY-4.0','scientific_snapshot':'2026-10-23',
                         'note':'Saved evidence, with operational locations and execution commands omitted. No new metric evaluation.'}
        write(ROOT/'data'/f'{name}.json',obj)
    # Images are historical static fallbacks; retain their actual scientific dates.
    names = {'geometry':'geometry-charts','profiles':'matched-profiles','profiles-supplement':'cusp-profiles',
             'spatial':'matched-sections','gluing':'face-and-core-compatibility'}
    figures=ROOT/'assets/figures';figures.mkdir(parents=True,exist_ok=True)
    import shutil
    for old,new in names.items():
        shutil.copyfile(source/'fillings-m4-lean/two-approach-note/figures'/f'{old}.png', figures/f'{new}.png')

def verify_original_arrays(source):
    """Additional export-time reduction of hash-bound saved arrays, no model calls."""
    import numpy as np
    original=read(source/FILES['metrics']);audit=read(source/FILES['audit'])
    for item in original['provenance']:
        assert digest(source/item['path'])==item['sha256']
    for item in audit['sources'].values():
        assert digest(source/item['path'])==item['sha256']
    rows=[]
    def same(a,b): np.testing.assert_allclose(a,b,rtol=5e-10,atol=2e-12)
    for role in ROLES:
        state=original['states'][role]
        candidates=[source/p['path'] for p in original['provenance'] if '/'+role+'/' in p['path']]
        find=lambda ending:next(p for p in candidates if str(p).endswith(ending))
        counts={}
        with np.load(find('/after-bulk.npz'),allow_pickle=False) as z:
            for region,stats in state['regions'].items():
                e=z['heldout/'+region+'/E'];assert z['heldout/'+region+'/valid'].all()
                for key,value in [('rms',np.sqrt(np.mean(e**2))),('q99',np.quantile(e,.99)),('maximum',max(e))]:same(value,stats[key])
            counts['bulk_samples']=sum(v['count'] for v in state['regions'].values())
        pp=next(p for p in candidates if p.name.endswith('panels.npz'))
        with np.load(pp,allow_pickle=False) as z:
            for region,stats in state['panels'].items():
                e=z[region+'/E']
                for key,value in [('rms',np.sqrt(np.mean(e**2))),('q99',np.quantile(e,.99)),('maximum',max(e))]:same(value,stats[key])
            counts['dedicated_panel_samples']=sum(v['count'] for v in state['panels'].values())
        with np.load(find('/profiles/profiles.npz'),allow_pickle=False) as z:
            pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
            worst_e=worst_s=0
            for name in state['profiles']:
                packed=z[name+'/riemann_packed'];g=z[name+'/g'];gi=np.linalg.inv(g)
                tensor=np.zeros((len(g),4,4,4,4));offset=0
                for a,(i,j) in enumerate(pairs):
                    for k,l in pairs[a:]:
                        value=packed[:,offset];offset+=1
                        for u,v,sgn in [(i,j,1),(j,i,-1)]:
                            for w,x,sgn2 in [(k,l,1),(l,k,-1)]:
                                tensor[:,u,v,w,x]=sgn*sgn2*value;tensor[:,w,x,u,v]=sgn*sgn2*value
                ric=np.einsum('nik,nijkl->njl',gi,tensor);scalar=np.einsum('nij,nij->n',gi,ric)
                f=ric+3*g;err=np.sqrt(np.maximum(0,np.einsum('nai,nbj,nab,nij->n',gi,gi,f,f)))/6
                same(err,z[name+'/E']);same(scalar,z[name+'/scalar'])
                worst_e=max(worst_e,float(np.max(abs(err-z[name+'/E']))));worst_s=max(worst_s,float(np.max(abs(scalar-z[name+'/scalar']))))
            counts.update(profile_E_reconstruction_max_abs=worst_e,profile_scalar_reconstruction_max_abs=worst_s)
        with np.load(find('/faces/faces.npz'),allow_pickle=False) as z:
            ha,hb=z['h_a'],z['h_b'];ev=np.linalg.eigvals(np.linalg.solve(ha,hb));assert np.max(abs(ev.imag))<1e-11
            length=np.maximum(np.sqrt(ev.real),1/np.sqrt(ev.real)).max(axis=1)-1
            k=z['K_a']-z['normal_sign'][:,None,None]*z['K_b'];ev=np.linalg.eigvals(np.linalg.solve((ha+hb)/2,k));assert np.max(abs(ev.imag))<1e-11
            bending=abs(ev.real).max(axis=1);same(length,z['length_distortion']);same(bending,z['K_directional_max'])
            for p in state['physical_gluing']['by_pair']:
                mask=z['pair_id']==p['pair'];same(max(length[mask]),p['length_max']);same(max(bending[mask]),p['bend_max'])
            counts['face_samples']=len(length)
        path=find('/core-links.npz');incidence=read(path.parent/'incidence.json')
        with np.load(path,allow_pickle=False) as z:
            angles=[]
            for i,fibre in enumerate(incidence):
                g=z['g'][z['node_offsets'][i]:z['node_offsets'][i+1]]
                n=g[:,:2,:2]-g[:,:2,2:]@np.linalg.solve(g[:,2:,2:],g[:,2:,:2]);total=0
                for sector in fibre['sectors']:
                    l,r=np.asarray(sector['ray_left']),np.asarray(sector['ray_right']);h=n[sector['node']]
                    total+=np.arctan2(np.sqrt(np.linalg.det(h))*np.linalg.det(np.column_stack((l,r))),l@h@r)
                angles.append(2*np.pi-total)
            same(angles,state['core_links']['signed_angle_defect']);counts['core_strata']=len(angles)
        rows.append({'role':role,**counts})
    write(ROOT/'verification/source-arrays.json',{'metric_source_hashes_verified':len(original['provenance']),
          'audit_source_hashes_verified':len(audit['sources']),'reductions':rows,
          'scope':'Hash-bound saved arrays reduced without calling the learned metric. Source arrays are not all included in the public package.'})

def near(a,b): assert math.isclose(a,b,rel_tol=3e-10,abs_tol=2e-12),(a,b)
def csvfile(name,rows):
    p=ROOT/'data/tables'/f'{name}.csv';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return p

def verify_and_tables():
    m,e,a=(read(ROOT/'data'/f'{n}.json') for n in ('metrics','experiments','audit'))
    checks=[]
    assert len(e['records'])==79 and len(e['curves'])==79 and len(e['comparisons'])==63
    for role in ROLES:
        s=m['states'][role];rs=list(s['regions'].values())+list(s['panels'].values())
        for field in ['rms','q99','maximum']: near(s[field],100*max(r[field] for r in rs))
        near(s['Q'],max(s['rms']/.3,s['q99'],s['maximum']))
        assert len(s['physical_gluing']['by_pair'])==96 and len(s['core_links']['stratum'])==352
        assert s['checkpoint_sha256']==a['identities'][role]['checkpoint_sha256']
        for name,p in s['profiles'].items():
            assert len(p['parameter'])==len(p['E'])==len(p['scalar'])==len(p['status'])
            assert all(v==1 for v in p['status'])
            near(max(p['E']),s['profile_summary']['profiles'][name]['maximum_E'])
            near(max(p['E'][i] for i in p['coarse_indices']),s['profile_summary']['profiles'][name]['coarse_maximum_E'])
        for name,mp in s['maps'].items():
            vals=[v for row,stat in zip(mp['E'],mp['status']) for v,st in zip(row,stat) if st==1]
            near(max(vals),s['map_statistics'][name]['maximum'])
            near(math.sqrt(sum(v*v for v in vals)/len(vals)),s['map_statistics'][name]['rms'])
            assert len(vals)==s['map_statistics'][name]['count']
        checks.append({'role':role,'headline_reduction':'pass','checkpoint_identity':'pass','maps_profiles':'pass',
                       'face_pairs':96,'core_strata':352,'step':s['step'],'checkpoint_sha256':s['checkpoint_sha256']})
    for name in m['states']['raw022']['maps']:
        x,y=(m['states'][r]['maps'][name] for r in ROLES)
        for key in ('axis0','axis1','mask','coords','ordinary','sites'):
            if key in x: assert x[key]==y[key],(name,key)
    for name in m['states']['raw022']['profiles']:
        for key in ('parameter','coords','ordinary','sites','coarse_indices'):
            assert m['states']['raw022']['profiles'][name][key]==m['states']['quotient087']['profiles'][name][key]
    groups={}
    for p in e['comparisons']:
        x,y=(e['records'][str(p[k])] for k in ('control','treatment'))
        assert x['parent']==y['parent'] and x['lr']==y['lr']
        lab=p['label']
        if lab!='periodic_newpool': assert x['pool_hashes']['training_pool']==y['pool_hashes']['training_pool']
        else: assert x['pool_hashes']['training_pool']!=y['pool_hashes']['training_pool'] and x['ridge']==y['ridge']
        if lab.startswith('raw_eta='):assert x['eta']==0 and y['eta']==float(lab.split('=')[1])
        if lab=='periodic_ridge':assert (x['ridge'],y['ridge'])==(0,1)
        if lab=='quotient_stratification':assert x['collar']==y['collar']==0 and x['sampling']!=y['sampling']
        if lab.startswith('quotient_collar='):assert x['sampling']==y['sampling'] and x['collar']==0 and y['collar']==float(lab.split('=')[1])
        near(p['Q_ratio'],y['states']['last']['Q']/x['states']['last']['Q'])
        near(p['bulk_RMS_ratio'],y['states']['last']['bulk_rms']/x['states']['last']['bulk_rms'])
        groups.setdefault(lab,[]).append(p)
    summaries=[{'comparison':lab,'n':len(rows),'improved_Q':sum(p['Q_ratio']<1 for p in rows),
                'median_Q_ratio':statistics.median(p['Q_ratio'] for p in rows),
                'median_bulk_RMS_ratio':statistics.median(p['bulk_RMS_ratio'] for p in rows)} for lab,rows in groups.items()]
    # Tables preserve fraction/percent units in column names and every saved row.
    experiments=[]
    for key,r in sorted(e['records'].items(),key=lambda kv:int(kv[0])):
        for state in ('last','joint-best'):
            s=r['states'][state]
            experiments.append({'id':r['id'],'family':r['family'],'state':state,'step':s['step'],'worst_regional_RMS_percent':s['rms'],
                                'largest_sampled_E_percent':s['maximum'],'largest_regional_q99_percent':s['q99'],'Q':s['Q'],
                                'checkpoint_sha256':s['checkpoint_sha256'],'parent':r['parent'],'lr':r['lr'],'eta':r['eta'],
                                'ridge':r['ridge'],'collar':r['collar'],'sampling':r['sampling']})
    csvfile('experiments',experiments);csvfile('paired-comparisons',e['comparisons']);csvfile('paired-summary',summaries)
    faces=[];core=[];maps=[];profiles=[]
    for role in ROLES:
        s=m['states'][role]
        for p in s['physical_gluing']['by_pair']:faces.append({'construction':role,**p})
        c=s['core_links']
        for i in range(352):core.append({'construction':role,'index':i,**{k:v[i] for k,v in c.items() if isinstance(v,list)},'angle_degrees':math.degrees(c['signed_angle_defect'][i])})
        for name,mp in s['maps'].items():
            for j,y in enumerate(mp['axis1']):
                for i,x in enumerate(mp['axis0']):maps.append({'construction':role,'section':name,'axis0':x,'axis1':y,'inside_domain':mp['mask'][j][i],
                    'status':mp['status'][j][i],'E':mp['E'][j][i],'E_percent':100*mp['E'][j][i] if mp['E'][j][i] is not None else ''})
        for name,p in s['profiles'].items():
            coarse=set(p['coarse_indices'])
            for i,t in enumerate(p['parameter']):profiles.append({'construction':role,'profile':name,'index':i,'parameter':t,'E':p['E'][i],
                'E_percent':100*p['E'][i],'scalar':p['scalar'][i],'scalar_plus_12':p['scalar'][i]+12,'status':p['status'][i],'coarse_sample':i in coarse})
    for n,rows in [('faces',faces),('core-strata',core),('maps',maps),('profiles',profiles)]:csvfile(n,rows)
    tables={'experiments':experiments,'paired-summary':summaries,'faces':faces,'core-strata':core}
    for name,rows in tables.items():
        head=''.join(f'<th scope="col">{html.escape(k)}</th>' for k in rows[0])
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in r.values())+'</tr>' for r in rows)
        page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name} — saved numerical evidence</title><style>body{{font:16px/1.5 system-ui;background:#faf8f2;color:#172e38;margin:2rem}}table{{border-collapse:collapse;font-size:.85rem}}td,th{{padding:.45rem;border-bottom:1px solid #ccc;text-align:left}}th{{background:#ebe8df}}.scroll{{overflow:auto}}a{{color:#245e7a}}</style><main><a href="../../index.html">Return to the notebook</a><h1>{name.replace('-',' ').title()}</h1><p>Audited snapshot: 23 October 2026. Saved samples are nonuniform and repeated, not volume fractions. Numeric-only downloads preserve scientific precision.</p><p><a href="{name}.csv" download>Download CSV</a></p><div class="scroll" tabindex="0"><table><caption>{len(rows)} saved rows; units are included in column names or the data dictionary.</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></main></html>'''
        (ROOT/'data/tables'/f'{name}.html').write_text(page)
    verification={'schema':1,'scientific_snapshot':'2026-10-23','checks':checks,'matched_map_profile_grids':True,
                  'paired_comparisons_recomputed':63,'paired_configuration_checks':'pass','paired_summary':summaries,
                  'counts':{'completed_experiments':79,'face_pair_rows':len(faces),'core_stratum_rows':len(core),'map_samples':len(maps),'profile_samples':len(profiles)},
                  'limits':['Checks reduce bundled evidence; no new learned metric evaluation.','Raw arrays behind historical common-panel summaries are not all distributed.',
                            'Observed maxima are lower bounds on a supremum; no global smooth Einstein metric is certified.'],
                  'data_sha256':{n:digest(ROOT/'data'/f'{n}.json') for n in ('metrics','experiments','audit')}}
    write(ROOT/'verification/evidence.json',verification)
    print('Verified 2 frozen states, 79 experiments, 63 paired comparisons, 192 face rows and 704 core rows.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path)
    args=p.parse_args()
    if args.source_root:
        export(args.source_root)
        verify_original_arrays(args.source_root)
    verify_and_tables()
    subprocess.run([sys.executable, str(ROOT/"tools/build_audit_supplement.py")], check=True)
    subprocess.run([sys.executable, str(ROOT/"tools/update_audit_display.py")], check=True)
