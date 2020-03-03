import numpy as np
import os
from pathlib import Path

this_path = Path(os.path.dirname(os.path.realpath(__file__)) + os.sep)
default_templates_path = str(this_path.parent) + os.sep + 'default_templates' + os.sep

# Instrument parameters contains general information for this instrument.
default_instrument_parameters = {
    
    # The Spectrograph and observatory
    "spectrograph": "CHIRON",
    "observatory": "CTIO",
    
    # Default CHIRON instrument settings
    "n_orders": 63,
    "n_data_pix": 3200,
    "crop_pix": [400, 400],
    
    # Star Settings
    "n_template_fits": 10,
    
    "model_resolution": 8,
    
    # MCMC Settings, sort of implemented but not working yet.
    "mcmc_burn_in_steps": 500,
    "mcmc_main_steps": 50000,
    "mcmc_walkers_factor": 8,
    "min_GR": None,
    
    # By default the Nelder-Mead fitter fits the full par space, then consecutive pairs of N pars.
    # So, [par1, par2], [par2, par3], [par3, par4], ..., [parN, par1]
    # It does this whole process N times.
    # Super does a custom Nelder-Mead where entire model components are varied individually instead of pairs.
    # This can be more effective if a given parameter doesn't affect the entire spectrum.
    "do_super": False,
    
    # Whether or not to print best fit parameters and add template to plots
    "verbose": False
}


# Construct the default iSHELL forward model
# Each entry must have a name and class.
# A given model can be effectively not used if n_delay is greater than n_template_fits
# Mandatory: Wavelength solution and star. Technically the rest are optional.
# Keywords are special, but their classes they point to can be anything.
# Keywords are rarely used explicitly in the code, but they are.
# Keywords:
# 'star' = star
# 'wavelength_solution' = wavelength solution
# 'lsf' = the line spread function
# 'gas_cell' = any gas cell
# 'tellurics' = the telluric model
# Remaining model components can be anything, since the code won't be doing anything special with them.
default_model_blueprints = {
    
    # The star
    'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-np.inf, 0, np.inf]
    },
    
    # The methane gas cell
    'gas_cell': {
        'name': 'iodine_gas_cell', # NOTE: full parameter names are name + base_name.
        'class_name': 'GasCellModel',
        'input_file': default_templates_path + 'iodine_gas_cell_chiron_master_40K.npz',
        'depth': [1, 1, 1],
        'shift': [-1.4956800000032473 - 0.1, -1.4956800000032473, -1.4956800000032473 + 0.1] # min, guess, max
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'vis_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-250, -100, 250],
        'components': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas.npz',
                'depth': [0.01, 1.5, 4.0]
            },
            'methane': {
                'input_file': default_templates_path + 'telluric_methane_tapas.npz',
                'depth': [0.1, 1.0, 3.0]
            },
            'nitrous_oxide': {
                'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'carbon_dioxide': {
                'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas.npz',
                'depth': [0.05, 0.65, 3.0]
            },
            'oxygen': {
                'input_file': default_templates_path + 'telluric_oxygen_tapas.npz',
                'depth': [0.1, 1.1, 3.0]
            },
            'ozone': {
                'input_file': default_templates_path + 'telluric_ozone_tapas.npz',
                'depth': [0.05, 0.65, 3.0]
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'full_blaze', # The blaze model after a division from a flat field
        'class_name': 'FullBlazeModel',
        'n_splines': 0,
        'base_amp': [1.02, 1.05, 1.1],
        'base_b': [0.008, 0.01, 0.04],
        'base_c': [-15, 0.1, 15],
        'base_d': [0.51, 0.7, 0.9],
        'spline': [-0.135, 0.01, 0.135],
        
        # Blaze is centered on the blaze wavelength.
        'blaze_wavelengths': [4542.565197891619, 4574.359030513886, 4607.24727150439, 4641.229920863132, 4676.306978590112, 4712.478444685328, 4749.744319148783, 4788.104601980475, 4827.559293180405, 4868.108392748572, 4909.751900684978, 4952.4898169896205, 4996.3221416625, 5041.248874703617, 5087.270016112973, 5134.385565890565, 5182.5955240363965, 5231.899890550464, 5282.29866543277, 5341.791848683312, 5386.379440302093, 5440.061440289111, 5494.837848644367, 5550.708665367861, 5607.673890459591, 5665.73352391956, 5724.887565747766, 5785.136015944209, 5846.47887450889, 5908.9161414418095, 5972.447816742965, 6037.073900412359, 6102.7943924499905, 6169.60929285586, 6237.518601629966, 6306.522318772311, 6376.620444282892, 6447.812978161712, 6520.099920408768, 6593.481271024063, 6667.957030007596, 6743.527197359364, 6820.191773079372, 6897.950757167617, 6976.8041496240985, 7056.751950448818, 7137.794159641776, 7219.9307772029715, 7303.161803132403, 7387.487237430073, 7472.907080095982, 7559.421331130126, 7647.02999053251, 7735.733058303131, 7825.530534441988, 7916.422418949083, 8008.408711824418, 8101.489413067987, 8195.664522679795, 8290.93404065984, 8387.297967008126, 8484.756301724647]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 6,
        'compress': 64,
        'width': [0.005, 0.0075, 0.015], # LSF width, in angstroms
        'ak': [-0.1, 0.001, 0.1] # See cale et al 2019 for definition of ak > 0
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        
        'name': 'lagrange_wavesol_splines',
        'class_name': 'WaveModelKnown',
        'n_splines': 20,
        'spline': [-0.05, 0.01, 0.05]
    }
}