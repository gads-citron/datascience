# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:47:58 2024

@author: Myriam Lopez
"""

import math
import numpy as np
from scipy.linalg import toeplitz
from cvxopt import matrix, spdiag, mul, div, solvers, lapack
solvers.options['maxiters'] = 5
solvers.options['show_progress'] = False
def l1(P, q):
    m, n = P.size
    c = matrix(n*[0.0] + m*[1.0])
    h = matrix([q, -q])

    def Fi(x, y, alpha = 1.0, beta = 0.0, trans = 'N'):    
        if trans == 'N':
            u = P*x[:n]
            y[:m] = alpha * ( u - x[n:]) + beta*y[:m]
            y[m:] = alpha * (-u - x[n:]) + beta*y[m:]
        else:
            y[:n] =  alpha * P.T * (x[:m] - x[m:]) + beta*y[:n]
            y[n:] = -alpha * (x[:m] + x[m:]) + beta*y[n:]

    def Fkkt(W): 
        d1, d2 = W['d'][:m], W['d'][m:]
        D = 4*(d1**2 + d2**2)**-1
        A = P.T * spdiag(D) * P
        lapack.potrf(A)

        def f(x, y, z):
            x[:n] += P.T * ( mul( div(d2**2 - d1**2, d1**2 + d2**2), x[n:]) 
                + mul( .5*D, z[:m]-z[m:] ) )
            lapack.potrs(A, x)

            u = P*x[:n]
            x[n:] =  div( x[n:] - div(z[:m], d1**2) - div(z[m:], d2**2) + 
                mul(d1**-2 - d2**-2, u), d1**-2 + d2**-2 )

            z[:m] = div(u-x[n:]-z[:m], d1)
            z[m:] = div(-u-x[n:]-z[m:], d2)
        return f

    uls =  +q
    lapack.gels(+P, uls)
    rls = P*uls[:n] - q 

    x0 = matrix( [uls[:n],  1.1*abs(rls)] ) 
    s0 = +h
    Fi(x0, s0, alpha=-1, beta=1) 

    if max(abs(rls)) > 1e-10:  
        w = .9/max(abs(rls)) * rls
    else: 
        w = matrix(0.0, (m,1))
    z0 = matrix([.5*(1+w), .5*(1-w)])

    dims = {'l': 2*m, 'q': [], 's': []}
    sol = solvers.conelp(c, Fi, h, dims, kktsolver = Fkkt,  
        primalstart={'x': x0, 's': s0}, dualstart={'z': z0})
    return sol['x'][:n]


def bilateral_filter(j, t, y_j, y_t, delta1=1.0, delta2=1.0):
    idx1 = -1.0 * (math.fabs(j-t) ** 2.0)/(2.0*delta1**2)
    idx2 = -1.0 * (math.fabs(y_j-y_t) ** 2.0)/(2.0*delta2**2)
    weight = (math.exp(idx1)*math.exp(idx2))
    return weight


def get_neighbor_idx(total_len, target_idx, H=3):
    """
    Let i = target_idx.
    Then, return i-H, ..., i, ..., i+H, (i+H+1)
    """
    return [np.max([0, target_idx-H]), np.min([total_len, target_idx+H+1])]


def get_neighbor_range(total_len, target_idx, H=3):
    start_idx, end_idx = get_neighbor_idx(total_len, target_idx, H)
    return np.arange(start_idx, end_idx)


def get_relative_trends(delta_trends):
    init_value = np.array([0])
    idxs = np.arange(len(delta_trends))
    relative_trends = np.array(
        list(map(lambda idx: np.sum(delta_trends[:idx]), idxs)))
    relative_trends = np.concatenate([init_value, relative_trends])
    return relative_trends


def get_toeplitz(shape, entry):
    h, w = shape
    num_entry = len(entry)
    assert np.ndim(entry) < 2
    if num_entry < 1:
        return np.zeros(shape)
    row = np.concatenate([entry[:1], np.zeros(h-1)])
    col = np.concatenate([np.array(entry), np.zeros(w-num_entry)])
    return toeplitz(row, col)


def get_season_idx(total_len, target_idx, T=10, K=2, H=5):
    num_season = np.min([K, int(target_idx/T)])
    if target_idx < T:
        key_idxs = target_idx + np.arange(0, num_season+1)*(-1*T)
    else:
        key_idxs = target_idx + np.arange(1, num_season+1)*(-1*T)

    idxs = list(map(lambda idx: get_neighbor_range(
        total_len, idx, H), key_idxs))
    season_idxs = []
    for item in idxs:
        season_idxs += list(item)
    season_idxs = np.array(season_idxs)
    return season_idxs

