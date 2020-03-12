# Python built in modules
import copy
from collections import OrderedDict
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import time # Time the code
import sys # sys utils
from sys import platform # plotting backend
import pdb # debugging
stop = pdb.set_trace

# Multiprocessing
from joblib import Parallel, delayed

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting

# Science/math
import scipy
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# User defined/pip modules
import pychell_rvs.pychell_model_components as pcmodelcomponents # the data objects
import pychell_rvs.pychell_math as pcmath

def rms_model(gp, v, fwm, iter_num, templates_dict, gpars):

    gp_ = pcmodelcomponents.Parameters.from_numpy(list(fwm.initial_parameters.keys()), values=gp, varies=v)

    # Generate the forward model
    wave_lr, model_lr = fwm.build_full(gp_, templates_dict, gpars)

    weights = np.copy(fwm.data.badpix)

    # RMS
    diffs2 = (fwm.data.flux - model_lr)**2
    good = np.where(np.isfinite(diffs2) & (weights > 0))[0]
    residuals2 = diffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    residuals2[:4] *= left_taper
    residuals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(residuals2)
    weights[ss[-1*gpars['flag_n_worst_pixels']:]] = 0
    residuals2[ss[-1*gpars['flag_n_worst_pixels']:]] = np.nan
    
    # Compute rms ignoring bad pixels
    rms = (np.nansum(residuals2 * weights) / np.nansum(weights))**0.5
    cons = np.nanmin(fwm.models_dict['lsf'].build(gp_)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return rms, cons


def weighted_rms_model(gp, v, fwm, iter_num, templates_dict, gpars):

    gp_ = pcmodelcomponents.Parameters.from_numpy(list(fwm.initial_parameters.keys()), values=gp, varies=v)

    # Generate the forward model
    wave_lr, model_lr = fwm.build_full(gp_, templates_dict, gpars)
    
    # Build weights
    if gp_[fwm.models_dict['star'].par_names[0]].vary and np.sum(v) < 3:
        star_flux = fwm.models_dict['star'].build(gp_, templates_dict['star'][:, 0], templates_dict['star'][:, 1], wave_lr)
        weights = pcmath.rv_content_per_pixel(wave_lr, star_flux, snr=100, use_blaze=False)
        bad = np.where(~np.isfinite(weights))[0]
        if bad.size > 0:
            weights[bad] = 0
        weights *= fwm.data.badpix
    else:
        weights = np.copy(fwm.data.badpix)
        
        

    # Force weights to contain bad pixels
    weights = np.copy(fwm.data.badpix)

    # weighted RMS
    wdiffs2 = (fwm.data.flux - model_lr)**2
    good = np.where(np.isfinite(diffs2) & (weights > 0))[0]
    wresiduals2 = diffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    wresiduals2[:4] *= left_taper
    wresiduals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(wresiduals2)
    weights[ss[-1*gpars['flag_n_worst_pixels']:]] = 0
    wresiduals2[ss[-1*gpars['flag_n_worst_pixels']:]] = np.nan
    
    # Compute rms ignoring bad pixels
    wrms = (np.nansum(residuals2 * weights) / np.nansum(weights))**0.5
    cons = np.nanmin(fwm.models_dict['lsf'].build(gp_)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return wrms, cons