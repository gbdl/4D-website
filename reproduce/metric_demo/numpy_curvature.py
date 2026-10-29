# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent NumPy contraction of metric derivatives, using explicit index loops.

Arrays use dg[i,j,k]=partial_k g_ij and d2g[i,j,k,l]=partial_k partial_l g_ij.
R^i_jkl = partial_k Gamma^i_lj - partial_l Gamma^i_kj + Gamma^i_km Gamma^m_lj - Gamma^i_lm Gamma^m_kj.
"""
import numpy as np

def diagnostics(g,dg,d2g):
    gi=np.linalg.inv(g)
    T=np.empty((4,4,4));dT=np.empty((4,4,4,4))
    for l in range(4):
        for j in range(4):
            for k in range(4):
                T[l,j,k]=.5*(dg[l,k,j]+dg[l,j,k]-dg[j,k,l])
                for m in range(4):
                    dT[l,j,k,m]=.5*(d2g[l,k,j,m]+d2g[l,j,k,m]-d2g[j,k,l,m])
    gamma=np.tensordot(gi,T,axes=([1],[0]))
    dgamma=-np.einsum('ia,abm,bjk->ijkm',gi,dg,gamma)+np.einsum('il,ljkm->ijkm',gi,dT)
    R=np.empty((4,4,4,4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for l in range(4):
                    R[i,j,k,l]=dgamma[i,l,j,k]-dgamma[i,k,j,l]+gamma[i,k]@gamma[:,l,j]-gamma[i,l]@gamma[:,k,j]
    ric=np.einsum('ijil->jl',R); A=ric+3*g
    S=float(np.einsum('ij,ij',gi,ric))
    norm=float(np.einsum('ia,jb,ij,ab',gi,gi,A,A))
    if norm < -1e-20:raise FloatingPointError('negative squared Einstein norm')
    E=float(np.sqrt(max(norm,0))/6)
    return {'E':E,'S':S,'Ric':ric,'R':np.tensordot(g,R,axes=([1],[0]))}
