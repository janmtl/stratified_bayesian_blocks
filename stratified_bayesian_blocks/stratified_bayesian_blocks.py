# -*- coding: utf-8 -*-
"""
Stratified Bayesian Blocks
--------------------------


"""

import numpy as np
from astroML.density_estimation import bayesian_blocks


def _regularize(x):
    r, s = np.unique(x, return_counts=True)
    s = (np.log(s)+1).astype(int)
    y = np.repeat(r, s)
    return y


def _normalize(x):
    r, s = np.unique(x, return_counts=True)
    s = (s - np.min(s)+1).astype(int)
    y = np.repeat(r, s)
    return y


def stratified_bayesian_blocks(x, p0=0.01, min_bin_width=0.01):
    """
    Creates smart histogram bins for mixed continuous/categorical data
    """

    # Create strata
    r, s = np.unique(x, return_counts=True)
    strata_edges = bayesian_blocks(s, p0=0.01)
    strata_bins = zip(strata_edges[:-1], strata_edges[1:])

    # Iterate over each strata
    data_bins = []
    for strata_bin in strata_bins:
        # Select the data pertaining only to a given strata
        sel = (strata_bin[0] <= s) & (s < strata_bin[1])
        strata_data = np.repeat(r[sel], s[sel])

        # Normalize and regularize the data within the strata
        strata_data = _regularize(_normalize(strata_data))

        # Perform Bayesian Blocks and append the bins
        edges = bayesian_blocks(strata_data, p0=0.01)
        data_bins.append(edges)

    # Collect the bins from all strata together
    data_bins = np.sort(np.concatenate(data_bins))

    # Clean up with the min_bin_width heuristic
    sel = (data_bins[1:]-min_bin_width) < data_bins[:-1]
    sel = np.r_[False, sel]
    data_bins = data_bins[~sel]

    return data_bins


def hist(x):
    pass
