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
        'class_name': 'GasCellModelOrderDependent',
        'input_file': default_templates_path + 'iodine_gas_cell_chiron_master_40K.npz',
        'depth': [1, 1, 1],
        'shifts': [-1.28151621, -1.28975381, -1.29827329, -1.30707465, -1.31615788, -1.32552298, -1.33516996, -1.34509881, -1.35530954, -1.36580215, -1.37657662, -1.38763298, -1.3989712 , -1.4105913 , -1.42249328, -1.43467713, -1.44714286, -1.45989046, -1.47291993, -1.48623128, -1.49982451, -1.5136996 , -1.52785658, -1.54229543, -1.55701615, -1.57201875, -1.58730322, -1.60286957, -1.61871779, -1.63484788, -1.65125985, -1.6679537 , -1.68492942, -1.70218701, -1.71972648, -1.73754783, -1.75565104, -1.77403614, -1.79270311, -1.81165195, -1.83088267, -1.85039526, -1.87018972, -1.89026606, -1.91062428, -1.93126437, -1.95218634, -1.97339018, -1.99487589, -2.01664348, -2.03869294, -2.06102428, -2.08363749, -2.10653258, -2.12970954, -2.15316838, -2.17690909, -2.20093168, -2.22523614, -2.24982247, -2.27469068, -2.29984077, -2.32527273],
        'shift_range': [0, 0]
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
        'base_amp': [1.02, 1.05, 1.15],
        'base_b': [0.008, 0.01, 0.04],
        'base_c': [-1, 0.01, 1],
        'base_d': [0.51, 0.7, 0.9],
        'spline': [-0.135, 0.01, 0.135],
        
        # Blaze is centered on the blaze wavelength.
        'blaze_wavelengths': np.array([4576.37529117, 4606.99031402, 4638.67632316, 4671.43331859, 4705.26130031, 4740.16026832, 4776.13022262, 4813.1711632, 4851.28309008, 4890.46600324, 4930.7199027, 4972.04478844, 5014.44066047, 5057.9075188 , 5102.44536341, 5148.05419431, 5194.7340115, 5242.48481498, 5291.30660475, 5341.1993808, 5392.16314315, 5444.19789179, 5497.30362671, 5551.48034793, 5606.72805543, 5663.04674923, 5720.43642931, 5778.89709568, 5838.42874834, 5899.03138729, 5960.70501253, 6023.44962406, 6087.26522188, 6152.15180599, 6218.10937638, 6285.13793307, 6353.23747604, 6422.40800531, 6492.64952086, 6563.9620227, 6636.34551084, 6709.79998526, 6784.32544597, 6859.92189297, 6936.58932626, 7014.32774584, 7093.1371517, 7173.01754386, 7253.96892231, 7335.99128704, 7419.08463807, 7503.24897538, 7588.48429898, 7674.79060888, 7762.16790506, 7850.61618753, 7940.13545629, 8030.72571134, 8122.38695268, 8215.1191803, 8308.92239422, 8403.79659443])
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 6,
        'compress': 64,
        'width': [0.009, 0.014, 0.018], # LSF width, in angstroms
        'ak': [-0.01, 0.001, 0.01] # See arken et al for definition of ak
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'name': 'wavesol_ThAr_I2',
        'class_name': 'WaveModelKnown',
        'n_splines': 20,
        'spline': [-0.005, 0.0005, 0.005]
    }
}