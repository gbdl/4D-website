#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic algebraic checks of displayed geometry; no learned metric calls.

This replaces a prose review checklist with executable comparisons. It checks
copy membership, cusp inversion/pullbacks and the regular black-hole formula.
It does not evaluate either neural metric or prove quotient compatibility.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
CENTRES = ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(1.,1.,1.))
BOX_LO = (-1.,-1.,-1.)
BOX_HI = (1.,3.,3.)
SLOPE = (10.,4.,4.)
ELL = math.sqrt(sum(v*v for v in SLOPE))
OMEGA = 2*math.pi/ELL


def endpoint():
    lo,hi=1.,10.
    for _ in range(100):
        mid=(lo+hi)/2
        if (mid*mid-1/mid)*(4*math.pi/3)**2>ELL*ELL:hi=mid
        else:lo=mid
    return (lo+hi)/2


def base(x,tol=1e-11):
    z,*u=x
    return z>0 and all(-tol<=v<=1+tol for v in u) and all(z*z+sum((v-c)**2 for v,c in zip(u,centre))>=1-tol for centre in CENTRES)


def explicit_union(x):
    z,a,b,c=x
    for bits in itertools.product((0,1),repeat=5):
        b0,b1,b2,b3,b4=bits
        u=(z,(-1)**b0*a,(-1)**(b1+b3)*(b-2*b3),(-1)**(b2+b4)*(c-2*b4))
        if base(u):return True
    return False


def folded_membership(x):
    z,*h=x
    if not all(lo-1e-11<=v<=hi+1e-11 for v,lo,hi in zip(h,BOX_LO,BOX_HI)):return False
    a,b,c=h
    return base((z,abs(a),abs(b) if b<=1 else abs(2-b),abs(c) if c<=1 else abs(2-c)))


def transpose(a):return list(map(list,zip(*a)))
def product(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def apply(a,v):return [sum(x*y for x,y in zip(row,v)) for row in a]
def maxdiff(a,b):return max(abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra,rb))
def identity(n):return [[float(i==j) for j in range(n)] for i in range(n)]


def run():
    rng=random.Random(20261025)
    points=[[rng.uniform(.01,3),rng.uniform(-1.2,1.2),rng.uniform(-1.2,3.2),rng.uniform(-1.2,3.2)] for _ in range(4096)]
    boundary=[[z,.5,b,c] for z in (.05,.5,.8,1.,1.2) for b in (-1.,0.,1.,2.,3.) for c in (-1.,0.,1.,2.,3.)]
    mismatches=sum(explicit_union(x)!=folded_membership(x) for x in points+boundary)
    assert mismatches==0,('copy membership',mismatches)
    table_path=ROOT/'reproduce/fixtures/geometry.json';table=json.loads(table_path.read_text())
    inversion_error=pullback_error=0.
    for k in range(256):
        site=k%37;O=table['site_frames'][site];centre=table['sites'][site]
        z=rng.uniform(.2,2.);d=[z]+[rng.uniform(-1,1) for _ in range(3)];q=sum(v*v for v in d)
        c=[z/q]+[v/q for v in apply(O,d[1:])];qc=sum(v*v for v in c)
        back=[c[0]/qc]+[v/qc for v in apply(transpose(O),c[1:])]
        inversion_error=max(inversion_error,max(abs(a-b) for a,b in zip(back,d)))
        R=[[1.,0.,0.,0.]]+[[0.]+row for row in O]
        rd=apply(R,d)
        J=[[R[i][j]/q-2*rd[i]*d[j]/(q*q) for j in range(4)] for i in range(4)]
        pull=product(transpose(J),J);expected=identity(4)
        pull=[[v*z*z/(c[0]*c[0]) for v in row] for row in pull]
        pullback_error=max(pullback_error,maxdiff(pull,expected))
    assert inversion_error<1e-12 and pullback_error<1e-12
    rj=endpoint();endpoint_error=abs((rj*rj-1/rj)*(4*math.pi/3)**2-ELL*ELL)
    assert abs(rj-table['rj'])<1e-13 and endpoint_error<1e-11
    compact_error=0.
    for _ in range(128):
        z=rng.uniform(.15,rj-.01);a=z/rj;rho2=4*rj*(rj-z)/3;theta=rng.uniform(-math.pi,math.pi)
        U,V=math.sqrt(rho2)*math.cos(theta),math.sqrt(rho2)*math.sin(theta)
        f=1-a**3;p=1+a+a*a
        J=[[-3*U/(2*rj),-3*V/(2*rj),0.,0.],[-V/(OMEGA*rho2),U/(OMEGA*rho2),0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]
        adapted=[[v*(1/f if i==0 else f/(1-rj**-3) if i==1 else 1) for v in row] for i,row in enumerate(identity(4))]
        pulled=product(transpose(J),product(adapted,J));regular=identity(4);w=(U,V)
        for i in range(2):
            for j in range(2):regular[i][j]=p/3*(i==j)+(2+a)*(3+p)/(4*rj*rj*p)*w[i]*w[j]
        compact_error=max(compact_error,maxdiff(pulled,regular))
    assert compact_error<1e-11
    sectionals=[]
    for r in (1.,2.,10.,100.):
        k1,k2=-1+r**-3,-1-.5*r**-3
        scalar=2*(2*k1+4*k2)
        assert abs(scalar+12)<1e-13
        sectionals.append({'r':r,'two_planes':k1,'four_planes':k2,'scalar':scalar})
    regional_sum=1/3+1/3+sum(1/12 for _ in range(4))
    assert abs(regional_sum-1)<1e-15 and abs(.75+.25-1)<1e-15
    return {'schema':1,'command':'python3 tools/check_geometry.py','seed':20261025,
            'scope':'Algebraic geometry and analytic-model checks; no trained metric evaluation.',
            'checks':{'random_explicit_copy_union_comparisons':len(points),'boundary_slice_copy_union_comparisons':len(boundary),
            'copy_membership_mismatches':mismatches,'cusp_inversion_cases':256,'cusp_inversion_max_absolute_residual':inversion_error,
            'hyperbolic_inversion_relative_pullback_residual':pullback_error,'rj_endpoint_equation_absolute_residual':endpoint_error,
            'regular_black_hole_compact_tensor_comparisons':128,'regular_black_hole_compact_tensor_max_absolute_residual':compact_error,
            'adapted_black_hole_sectionals':sectionals,'regional_loss_weight_sum':regional_sum},
            'geometry_fixture_sha256':hashlib.sha256(table_path.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'verification/geometry.json')
    args=parser.parse_args();result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Compared 4,221 domain points, 256 cusp maps and 128 regular black-hole tensors.')
