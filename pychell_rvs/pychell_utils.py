from functools import reduce
import operator
import numpy as np
import pychell_rvs.pychell_math as pcmath
from pdb import set_trace as stop

def sort_data(data_list, gpars):
    bjds_temp = np.array([getattr(data_list[ispec], 'BJD') for ispec in range(gpars['n_spec'])]).astype(np.float64)
    sorting_inds = np.argsort(bjds_temp)
    return sorting_inds
    

def find_all_items(obj, key, keys=None):
    ret = []
    if not keys:
        keys = []
    if key in obj:
        out_keys = keys + [key]
        ret.append((out_keys, obj[key]))
    for k, v in obj.items():
        if isinstance(v, dict):
            found_items = find_all_items(v, key, keys=(keys+[k]))
            ret += found_items
    return ret

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
    
    
# spec num starts at 0
def get_night_index(spec_num, gpars):
    n = gpars['n_obs_nights'][0]
    for i in range(gpars['n_nights']):
        if spec_num < n:
            return i
        n += gpars['n_obs_nights'][i+1]
    return n

def get_spec_indices_from_night(night_index, gpars):

    if night_index == 0:
        f = 0
        l = f + gpars['n_obs_nights'][0]
    else:
        f = np.sum(gpars['n_obs_nights'][0:night_index])
        l = f + gpars['n_obs_nights'][night_index]

    return list(np.arange(f, l).astype(int))


# Computes the nightly jds from each epoch and the number of obs per night
def get_nightly_jds(jds, gpars):

    prev_i = 0
    # Calculate mean JD date and number of observations per night for nightly
    # coadded RV points; assume that observations on separate nights are
    # separated by at least 0.5 days.
    jds_nightly = []
    n_obs_nights = []
    if gpars['n_spec'] == 1:
        jds_nightly.append(jds[0])
        n_obs_nights.append(1)
    else:
        for i in range(gpars['n_spec']-1):
            if jds[i+1] - jds[i] > 0.5:
                jd_avg = np.average(jds[prev_i:i+1])
                n_obs_night = i - prev_i + 1
                jds_nightly.append(jd_avg)
                n_obs_nights.append(n_obs_night)
                prev_i = i + 1
        jds_nightly.append(np.average(jds[prev_i:]))
        n_obs_nights.append(gpars['n_spec'] - prev_i)

    jds_nightly = np.array(jds_nightly) # convert to np arrays
    n_obs_nights = np.array(n_obs_nights).astype(int)

    return jds_nightly, n_obs_nights


# Estimates the endpoints of the wavelength grid for each order
# using an estimation for the wavelength grid from the init parameters
# This is useful for knowing where to set the spline knots for the blaze and wavelength solution
def estimate_wavegrid_endpoints(data, order_num, blueprint, gpars):
    
    if hasattr(data, 'wave_grid'):
        wave_pad = 1 # Angstroms
        wave_estimate = data.wave_grid # use the first osbervation
        wave_left, wave_right = wave_estimate[gpars['pix_left']] - wave_pad, wave_estimate[gpars['pix_right']] + wave_pad
    else:
        # Make an array for the base wavelengths
        wavesol_base_wave_set_points = np.array([blueprint['base_set_point_1'][order_num], blueprint['base_set_point_2'][order_num], blueprint['base_set_point_3'][order_num]])
    
        # Get the polynomial coeffs through matrix inversion.
        wave_estimate_coeffs = pcmath.poly_coeffs(np.array(blueprint['base_pixel_set_points']), wavesol_base_wave_set_points)
    
        # The estimated wavelength grid
        wave_estimate = np.polyval(wave_estimate_coeffs, np.arange(gpars['n_data_pix']))
    
        # Wavelength end points are larger to account for changes in the wavelength solution
        # The stellar template is further padded to account for barycenter sampling
        wave_pad = 1 # Angstroms
        wave_left, wave_right = wave_estimate[gpars['pix_left']] - wave_pad, wave_estimate[gpars['pix_right']] + wave_pad
    
    return wave_left, wave_right