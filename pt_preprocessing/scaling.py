import numpy as np

def standardize(a, with_mean=None, with_std=None):
    m = np.nanmean(a, axis=0) if with_mean is None else with_mean
    s = np.nanstd(a, axis=0) if with_std is None else with_std
    a = (a - m) / s
    return a, m, s

def normalize(a, t_min=0, t_max=1):
    maxes = np.nanmax(a, axis=0)
    mins = np.nanmin(a, axis=0)
    a = (a - mins)/(maxes - mins)
    return a, maxes, mins