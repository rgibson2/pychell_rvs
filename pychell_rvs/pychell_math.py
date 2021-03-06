#### A helper file containing math routines
import scipy.interpolate # spline interpolation
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays


# LLVM
from numba import jit, njit
import numba
from llc import jit_filter_function

import pdb
stop = pdb.set_trace

@jit_filter_function
def fmedian(x):
    if np.sum(np.isfinite(x)) == 0:
        return np.nan
    else:
        return np.nanmedian(x)

def rv_content_per_pixel(wave, flux, snr=100, use_blaze=False):

    gain = 1.8
    counts = snr**2
    pe = gain * counts
    A_center = pe
    good = np.where(np.isfinite(wave) & np.isfinite(flux))[0]
    if use_blaze:
        A = A_center * np.abs(np.sinc(0.01 * (wave - np.nanmean(wave))))**1.6 * flux # modulate by a true blaze
    else:
        A = A_center * flux
    rvc_per_pix = np.empty(wave.size, dtype=np.float64)
    A_spline = scipy.interpolate.CubicSpline(wave, A, extrapolate=True, bc_type='not-a-knot')
    for j in range(wave.size-1):
        if j in good:
            slope = A_spline(wave[j], 1)
            rvc_per_pix[j] = cs.c * np.sqrt(A[j]) / (wave[j] * np.abs(slope))
        else:
            rvc_per_pix[j] = np.nan
    return rvc_per_pix

# Fast median filter 1d over a fixed box width in "pixels"
def median_filter1d(x, width, preserve_nans=True, med_val=0.5):
    
    bad = np.where(~np.isfinite(x))[0]
    good = np.where(np.isfinite(x))[0]
    
    if good.size == 0:
        return np.full(x.size, fill_value=np.nan)
    else:
        if med_val == 0.5:
            out = scipy.ndimage.filters.generic_filter(x, fmedian, size=width, cval=np.nan, mode='constant')
        else:
            out = scipy.ndimage.filters.generic_filter(x, weighted_median_rolling, size=width, cval=np.nan, mode='constant')
        
    if preserve_nans:
        out[bad] = np.nan # If a nan is overwritten with a new value, rewrite with nan
        
    return out

# Returns the hermite polynomial of degree deg over the variable x

def hermfun(x, deg):
    herm0 = np.pi**-0.25 * np.exp(-1.0 * x**2 / 2.0)
    herm1 = np.sqrt(2) * herm0 * x
    if deg == 0:
        herm = herm0
    elif deg == 1:
        herm = np.array([herm0, herm1]).T
    else:
        herm = np.zeros(shape=(x.size, deg+1))
        herm[:, 0] = herm0
        herm[:, 1] = herm1
        for k in range(2, deg+1):
            herm[:, k] = np.sqrt(2 / k) * (x * herm[:, k-1] - np.sqrt((k - 1) / 2) * herm[:, k-2])
    return herm

# This calculates the median absolute deviation of array x
def mad(x):
    return np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826

# This calculates the weighted median of a data set.
def weighted_median(data, weights=None, med_val=0.5):

    if weights is None:
        weights = np.ones(shape=data.shape, dtype=float)
    bad = np.where(~np.isfinite(data))[0]
    if bad.size > 0:
        data = np.delete(data, bad)
        weights = np.delete(weights, bad)
    data = data.flatten()
    weights = weights.flatten()
    inds = np.argsort(data)
    data_s = data[inds]
    weights_s = weights[inds]
    med_val = med_val * np.nansum(weights)
    if np.any(weights > med_val):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.nancumsum(weights_s)
        idx = np.where(cs_weights <= med_val)[0][-1]
        if weights_s[idx] == med_val:
            w_median = np.nanmean(data_s[idx:idx+2])
        else:
            w_median = data_s[idx+1]
    return w_median

# This calculates the weighted median of a data set for rolling calculations
# width is in Angstroms
def estimate_continuum(x, y, width=7, n_knots=14, cont_val=0.9):
    nx = x.size
    continuum_coarse = np.ones(nx, dtype=np.float64)
    for ix in range(nx):
        use = np.where((x > x[ix]-width/2) & (x < x[ix]+width/2))[0]
        if np.all(~np.isfinite(y[use])):
            continuum_coarse[ix] = np.nan
        else:
            continuum_coarse[ix] = weighted_median(y[use], weights=None, med_val=cont_val)
    
    good = np.where(np.isfinite(y))[0]
    knot_points = x[np.linspace(good[0], good[-1], num=n_knots).astype(int)]
    interp_fun = scipy.interpolate.CubicSpline(knot_points, continuum_coarse[np.linspace(good[0], good[-1], num=14).astype(int)], extrapolate=False, bc_type='not-a-knot')
    continuum = interp_fun(x)
    return continuum

# This calculates the unbiased weighted standard deviation of array x with weights w
def weighted_stddev(x, w):
    weights = w / np.nansum(w)
    wm = weighted_mean(x, w)
    dev = x - wm
    bias_estimator = 1.0 - np.nansum(weights ** 2) / np.nansum(weights) ** 2
    var = np.nansum(dev ** 2 * weights) / bias_estimator
    return np.sqrt(var)

# This calculates the weighted mean of array x with weights w
def weighted_mean(x, w):
    return np.nansum(x * w) / np.nansum(w)

# Rolling function f over a window given w of y given the independent variable x
def rolling_fun_true_window(f, x, y, w):

    output = np.empty(x.size, dtype=float)

    for i in range(output.size):
        locs = np.where((x > x[i] - w/2) & (x <= x[i] + w/2))
        if len(locs) == 0:
            output[i] = np.nan
        else:
            output[i] = f(y[locs])

    return output

# Locates the closest value to a given value in an array
# Returns the value and index.
def find_closest(x, val):
    diffs = np.abs(x - val)
    loc = np.argmin(diffs)
    return loc, x[loc]

# Calculates the reduced chi squared
def reduced_chi_square(x, err):

    # Define the weights as 1 over the square of the error bars. 
    weights = 1.0 / err**2

    # Calculate the reduced chi square defined around the weighted mean, 
    # assuming we are not fitting to any parameters. 
    redchisq = (1.0 / (x.size-1)) * np.nansum((x - weighted_mean(x, weights))**2 / err**2)

    return redchisq

#This calculates the median absolute deviation of array x
def mad(x):
    return np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826

# Given 3 data points this returns the polynomial coefficients via matrix inversion, effectively
# In theory equivalent to np.polyval(x, y, deg=2)
@jit
def poly_coeffs(x, y):
    a0 = (-x[2] * y[1] * x[0]**2 + x[1] * y[2] * x[0]**2 + x[2]**2 * y[1] * x[0] - x[1]**2 * y[2] * x[0] - x[1] * x[2]**2 * y[0] + x[1]**2 * x[2] * y[0])/((x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]))
    a1 = (y[1] * x[0]**2 - y[2] * x[0]**2 - x[1]**2 * y[0] + x[2]**2 * y[0] - x[2]**2 * y[1] + x[1]**2 * y[2]) / ((x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2]))
    a2 = (x[1] * y[0] - x[2] * y[0] - x[0] * y[1] + x[2] * y[1] + x[0] * y[2] - x[1] * y[2]) / ((x[1] - x[0]) * (x[1] - x[2]) * (x[2] - x[0]))
    p = np.array([a2, a1, a0])
    return p

# Converts a mask array to a binary array. Note: Also needs size of full array. Only 1d arrays
def mask_to_binary(arr, l):
    binary = np.zeros(l, dtype=bool)
    if len(arr) == 0:
        return binary
    for i in range(l):
        if i in arr:
            binary[i] = True
    return binary.astype(bool)

# Returns the closest value and index
def find_closest_val(x, y, val):
    diffs = np.abs(y - val)
    index = np.argmin(diffs)
    val = x[index]
    return index, val


def compute_bisector_span(cc_vels, ccf, gpars):
        
    # B(d) = (v_l(d) + v_r(d)) / 2
    # v_l = velocities located on the left side from the minimum of the CCF peak and v_r are the ones on the right side
    # Mean bisector is computed at two depth ranges:
    # d = (0.1, 0.4), (0.6, 0.85)
    # B_(0.1, 0.4) = E(B(d)) for 0.1 to 0.4
    # B_(0.6, 0.85) = E(B(d)) for 0.6 to 0.85
    # BS = B_(0.1, 0.4) - B_(0.6, 0.85) = E(B(d)) for 0.1 to 0.4 - E(B(d)) for 0.6 to 0.85
    # .. = Average(B(d)) for 0.1 to 0.4 = Average((v_l(d) + v_r(d)) / 2) for 0.1 to 0.4
    
    # The bottom "half"
    dr1 = (0.1, 0.4)
    
    # The top "half"
    dr2 = (0.6, 0.85)
    
    # The depths are from 0 to 1 for the normalized CCF
    depths = np.linspace(0, 1, num=100)

    # Initialize the line bisector array (a function of CCF depth)
    line_bisectors = np.empty(depths.size, dtype=np.float64)

    # First normalize the CCF function
    ccf = ccf - np.nanmin(ccf)
    ccf = ccf / np.nanmax(ccf)
    
    # Get the velocities and offset such that the best vel is at zero
    best_vel = cc_vels[np.nanargmin(ccf)]
    cc_vels = cc_vels - best_vel
    
    # The vels on the left and right of the best vel.
    use_left = np.where(cc_vels < 0)[0]
    use_right = np.where(cc_vels > 0)[0]
    vel_max_ind_left, vel_max_ind_right = use_left[np.nanargmax(ccf[use_left])], use_right[np.nanargmax(ccf[use_right])]
    use_left = np.where((cc_vels > cc_vels[vel_max_ind_left]) & (cc_vels < 0))[0]
    use_right = np.where((cc_vels > 0) & (cc_vels < cc_vels[vel_max_ind_right]))[0]
    
    # Compute the line bisector
    for idepth in range(depths.size):
        d = depths[idepth]
        vl = find_closest_val(cc_vels[use_left], ccf[use_left], d)[1]
        vr = find_closest_val(cc_vels[use_right], ccf[use_right], d)[1]
        line_bisectors[idepth] = (vl + vr) / 2

    # Compute the bisector span
    top = np.where((depths > dr1[0]) & (depths < dr1[1]))[0]
    bottom = np.where((depths > dr2[0]) & (depths < dr2[1]))[0]
    avg_top = np.average(line_bisectors[top])
    avg_bottom = np.average(line_bisectors[bottom])
    
    # Store the bisector span
    bisector_span = (avg_top - avg_bottom) + best_vel
    
    return bisector_span