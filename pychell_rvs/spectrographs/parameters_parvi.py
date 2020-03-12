import numpy as np
import os
from pathlib import Path

this_path = Path(os.path.dirname(os.path.realpath(__file__)) + os.sep)
default_templates_path = str(this_path.parent) + os.sep + 'default_templates' + os.sep

# PARVI CONFIGURATION

# Instrument parameters contains general information for this instrument.
default_instrument_parameters = {
    
    # The Spectrograph and observatory
    "spectrograph": "PARVI",
    "observatory": "Palomar",
    
    # The JPL ephemeris for barycorrpy
    "bary_corr_file": None,
    
    # Default PARVI instrument settings
    "n_orders": 7,
    "n_data_pix": 2038,
    "crop_pix": [10, 10],
    
    # Star Settings
    "n_template_fits": 40,
    
    "model_resolution": 8,
    
    # Whether or not to print best fit parameters and add template to plots
    "verbose": False
}


# Construct the default PARVI forward model
# Each entry must have a name and class.
# A given model can be effectively not used if n_delay is greater than n_template_fits
# Mandatory: Wavelength solution and star. Technically the rest are optional.
# Keywords are special, but their classes they point to can be anything.
# Keywords are rarely used explicitly in the code, but they are.
# Keywords:
# 'star' = star
# 'wavelength_solution' = wavelength solution
# 'lsf' = the line spread function
# 'tellurics' = the telluric model
# Remaining model components can have any keys, since the code won't be doing anything special with them.
default_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-np.inf, 0, np.inf]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'yjhband_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-4000, -1300, 1000],
        'components': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas_palomar.npz',
                'depth': [0.01, 1.5, 5.0],
            },
            'methane': {
                'input_file': default_templates_path + 'telluric_methane_tapas_palomar.npz',
                'depth': [0.1, 1.0, 3.0],
            },
            'nitrous_oxide': {
                'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0],
            },
            'carbon_dioxide': {
                'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0],
            },
            'oxygen': {
                'input_file': default_templates_path + 'telluric_oxygen_tapas_palomar.npz',
                'depth': [0.1, 1.1, 3.0],
            },
            'ozone': {
                'input_file': default_templates_path + 'telluric_ozone_tapas_palomar.npz',
                'depth': [0.05, 0.65, 3.0],
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'ResidualBlazeModel',
        'n_splines': 0,
        'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
        'base_lin': [-0.001, 1E-5, 0.001],
        'base_zero': [0.96, 1.0, 1.15],
        'spline': [-0.025, 0.001, 0.025],
        
        # Blaze is centered on the blaze wavelength.
        'blaze_wavelengths': [14807.35118155, 14959.9938945 , 15115.82353631, 15274.96519038,
       15946.37631354, 16123.53607164, 16304.66829244]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 4,
        'compress': 64,
        'width': [0.15, 0.21, 0.28], # LSF width, in angstroms (slightly larger than this for PARVI)
        'ak': [-0.075, 0.001, 0.075] # See cale et al 2019 or arfken et al some year for definition of ak > 0
    },
    
    # Frequency comb (no splines since no gas cell)
    'wavelength_solution': {
        'name': 'laser_comb_wls',
        'class_name': 'WaveModelKnown',
        'n_splines': 0,
        'spline': [-0.15, 0.01, 0.15]
    }
    
}
    
    