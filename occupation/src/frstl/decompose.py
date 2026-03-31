# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:46:56 2024

@author: Myriam Lopez
"""

import numpy as np
from cvxopt import matrix
#custom packages
from frstl import tools

def denoise_step(sample, H=3, dn1=1., dn2=1.):
    def get_denoise_value(idx):
        start_idx, end_idx = tools.get_neighbor_idx(len(sample), idx, H)
        idxs = np.arange(start_idx, end_idx)
        weight_sample = sample[idxs]

        weights = np.array(list(map(lambda j: tools.bilateral_filter(
            j, idx, sample[j], sample[idx], dn1, dn2), idxs)))
        return np.sum(weight_sample * weights)/np.sum(weights)

    idx_list = np.arange(len(sample))
    denoise_sample = np.array(list(map(get_denoise_value, idx_list)))
    return denoise_sample

def seasonality_extraction(s, season_len, alpha, z, K=2, H=5, ds1=50., ds2=1.):
    sample_len = len(s)
    idx_list = np.arange(sample_len)

    def get_season_value(idx, season):
        idxs = tools.get_season_idx(sample_len, idx, season, K, H)
        if idxs.size == 0:
            return s[idx]

        weight_sample = s[idxs]
        weights = np.array(list(map(lambda j: tools.bilateral_filter(
            j, idx, s[j], s[idx], ds1, ds2), idxs)))
        season_value = np.sum(weight_sample * weights)/np.sum(weights)
        return season_value

    seasons_tilda = 0.    
    id_season = [get_season_value(idx, season_len) for idx in idx_list]
    seasons_tilda += alpha * np.array(id_season)
    seasons_tilda /= z

    return seasons_tilda

def trend_extraction(s, season_len, reg1=10., reg2=0.5):
    sample_len = len(s)
    season_diff = s[season_len:] - s[:-season_len]

    assert len(season_diff) == (sample_len - season_len)
    q = np.concatenate([season_diff, np.zeros([sample_len*2-3])])
    q = np.reshape(q, [len(q), 1])

    M = tools.get_toeplitz([sample_len-season_len, sample_len-1],
                     np.ones([season_len]))
    D = tools.get_toeplitz([sample_len-2, sample_len-1], np.array([1, -1]))
    P = np.concatenate([M, reg1*np.eye(sample_len-1), reg2*D], axis=0)
    
    # l1 approx solver
    q = matrix(q)
    P = matrix(P)
    delta_trends = tools.l1(P, q)

    relative_trends = tools.get_relative_trends(delta_trends)

    return s-relative_trends, relative_trends

def adjustment(s, relative_trends, seasons_tilda, season_len):
    num_season = int(len(s)/season_len)

    trend_init = np.mean(seasons_tilda[:season_len*num_season])

    trends_hat = relative_trends + trend_init
    seasons_hat = seasons_tilda - trend_init
    remainders_hat = s - trends_hat - seasons_hat
    return [trends_hat, seasons_hat, remainders_hat]

def check_converge_criteria(prev_remainders, remainders):
    diff = np.sqrt(np.mean(np.square(remainders-prev_remainders)))
    if diff < 1e-6:
        return True
    else:
        return False

def fast_robustSTL(s, season_len, trend_regs, season_regs, alpha, z, denoise_ds, season_ds, K=2, H=5, max_iter=5000):
    trial = 1
    prev = None
    while True:
        #Step 1: Denoising using bilateral filter
        dn1, dn2 = denoise_ds
        denoise = denoise_step(s, H, dn1, dn2)

        # Step 2: Trend extraction by Robust Sparse Model
        reg1, reg2 = trend_regs
        detrend_sample, relative_trends =\
            trend_extraction(denoise, season_len, reg1, reg2)

        # Step 3: Seasonal component extraction by non-local seasonal filter
        ds1, ds2 = season_ds
        seasons_tilda = seasonality_extraction(
            detrend_sample, season_len, alpha, z, K, H, ds1, ds2)

        trends_hat, seasons_hat, remainders_hat =\
            adjustment(s, relative_trends, seasons_tilda, season_len)

        if trial != 1:
            converge = check_converge_criteria(
                prev, remainders_hat)
            if converge:
                break

        trial += 1
        prev = remainders_hat[:]
        s = trends_hat + seasons_hat + remainders_hat

    return (s, trends_hat, seasons_hat, remainders_hat)