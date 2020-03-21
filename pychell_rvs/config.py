import numpy as np
import os

pipeline_path = os.path.dirname(os.path.realpath(__file__)) + os.sep
default_templates_path = pipeline_path + 'defualt_templates' + os.sep

# A few global configurations to get things going
# The use can override these settings.
# When the code is run, it first makes the global parameters dictionary below.
# Then it updates entries according to any matching keys in the parameters file.
# Then it updates entries according to any matching keys in the run file.

default_config = {
    
    # Supported Instruments
    'supported_instruments': ['iSHELL', 'CHIRON', 'PARVI'],
    
    # Star Settings
    'n_template_fits': 10, # a zeroth iteration (flat template) does not count towards this number.
    
    # Cross correlation / bisector span stuff for each iteration. Will take longer, but can be useful
    # A cross correlation will still be run on a "zeroth" iteration.
    'do_xcorr': False,
    
    # Model Resolution (n_model_pixels = model_resolution * n_data_pixels)
    # This is only important because of the instrument line profile (LSF)
    # In theory 8 is sufficient for any instrument. Some are probably fine at 4.
    'model_resolution': 8,
    
    # Which nights to use for the stellar template (empty list for all nights)
    'nights_for_template': [],
    
    # The target function. Must live in pychell_target_functions.py
    'target_function': 'rms_model',
    
    # Number of cores to use (for Nelder-Mead fitting and cross corr analysis)
    'n_cores': 8,
    
    # Flags the N worst pixels in fitting
    'flag_n_worst_pixels': 20,
    
    # If True, the best fit parameters are printed after each iteration
    'verbose_plot': True,
    'verbose_print': False,
    
    # No bary corr file
    'bary_corr_file': None,
    
    # If the user only wishes to compute the BJDS and barycorrs for later.
    'compute_bc_only': False,
    
    # Super simps: Normally the nelder mead optimizes the whole space followed by all consecutive pairs.
    # If true, instead of pairs, individual models are optimized alone.
    "super_simps": False,
    
    # The number of pixels to crop on each side of the spectrum
    'crop_pix': [50, 50],
    
    # Plotting parameters
    'dpi': 200, # the dpi used in plots
    'plot_wave_unit': 'nm', # The units for plots. Options are nm, ang, microns
    'lw': 0.8, # linewidth on fits
    'spec_img_width_pix': 2000, # in pixels
    'spec_img_height_pix': 720, # in pixels
    'rv_img_width_pix': 1800, # in pixels
    'rv_img_height_pix': 600, # in pixels
    'colors': {'blue': (0, 114/255, 189/255), 'gold': (244/255, 200/255, 66/255), 'red': (217/255, 83/255, 25/255), 'purple': (89/255, 23/255, 130/255), 'green': (0.1, 0.8, 0.1), 'light_blue': (66/255, 167/255, 244/255), 'orange': (255/255, 169/255, 22/255), 'darkpink': (153/255, 18/255, 72/255)}
    
}