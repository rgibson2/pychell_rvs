import numpy as np
import os
from pathlib import Path
# Other notes for iSHELL:
# blaze_model parameters for a sinc model (full unmodified blaze)
# a: [1.02, 1.05, 1.08], b: [0.008, 0.01, 0.0115], c: [-5, 0.1, 5], d: [0.51, 0.7, 0.9]

# Pipeline path acquired at runtime in order to make use of the default templates
this_path = Path(os.path.dirname(os.path.realpath(__file__)) + os.sep)
default_templates_path = str(this_path.parent) + os.sep + 'default_templates' + os.sep


# Instrument parameters contains general information for this instrument.
default_instrument_parameters = {
    
    # The Spectrograph and observatory
    "spectrograph": "iSHELL",
    "observatory": "IRTF",
    
    # Default iSHELL instrument KGAS settings
    "n_orders": 29,
    "n_data_pix": 2048,
    "crop_pix": [200, 200],
    
    # Star Settings
    "n_template_fits": 40,
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
# Remaining model components can be anything, since the code doesn't rely on their existence, but will use them
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
        'name': 'methane_gas_cell', # NOTE: full parameter names are name + base_name.
        'class_name': 'GasCellModel',
        'input_file': default_templates_path + 'methane_gas_cell_ishell_kgas.npz',
        'depth': [0.97, 0.97, 0.97],
        'shift': [0, 0, 0]
    },
    
    # Tellurics (from TAPAS)
    'tellurics': {
        'name': 'kband_tellurics', # NOTE: full parameter names are name + component + base_name.
        'class_name': 'TelluricModelTAPAS',
        'vel': [-250, -100, 100],
        'components': {
            'water': {
                'input_file': default_templates_path + 'telluric_water_tapas.npz',
                'depth': [0.01, 1.5, 4.0],
            },
            'methane': {
                'input_file': default_templates_path + 'telluric_methane_tapas.npz',
                'depth': [0.1, 1.0, 3.0],
            },
            'nitrous_oxide': {
                'input_file': default_templates_path + 'telluric_nitrous_oxide_tapas.npz',
                'depth': [0.05, 0.65, 3.0],
                'airmass_correlation': [ 0.225356741957, 1.40889772648] # linear with airmass. called with np.polyval(pars, am)
            },
            'carbon_dioxide': {
                'input_file': default_templates_path + 'telluric_carbon_dioxide_tapas.npz',
                'depth': [0.05, 0.65, 3.0],
            }
        }
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'name': 'residual_blaze', # The blaze model after a division from a flat field
        'class_name': 'ResidualBlazeModel',
        'n_splines': 14,
        'base_quad': [-5.5E-5, -2E-6, 5.5E-5],
        'base_lin': [-0.001, 1E-5, 0.001],
        'base_zero': [0.96, 1.0, 1.08],
        'spline': [-0.135, 0.01, 0.135],
        
        # Blaze is centered on the blaze wavelength. Crude estimates
        'blaze_wavelengths': [24623.42005657, 24509.67655586, 24396.84451226, 24284.92392579, 24173.91479643, 24063.81712419, 23954.63090907, 23846.35615107, 23738.99285018, 23632.54100641, 23527.00061976, 23422.37169023, 23318.65421781, 23215.84820252, 23113.95364434, 23012.97054327, 22912.89889933, 22813.7387125,  22715.48998279, 22618.1527102, 22521.72689473, 22426.21253637, 22331.60963514, 22237.91819101, 22145.13820401, 22053.26967413, 21962.31260136, 21872.26698571, 21783.13282718]
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'name': 'lsf_hermite',
        'class_name': 'LSFHermiteModel',
        'hermdeg': 6,
        'compress': 64,
        'width': [0.055, 0.12, 0.2], # LSF width, in angstroms
        'ak': [-0.03, 0.001, 0.2] # Hermite polynomial coefficients
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        
        'name': 'lagrange_wavesol_splines',
        'class_name': 'WaveSolModelFull',
        
        # The three pixels to span the detector corresponding to the above wavelengths
        # They are chosen as such because we typically use pixels 200-1848 only.
        # These pixels must correspond to the wavelengths in the array wavesol_base_wave_set_points_i[order]
        'base_pixel_set_points': [199, 1023.5, 1847],
        
        # Left most set point for the quadratic wavelength solution
        "base_set_point_1": [24545.57561435, 24431.48444449, 24318.40830764, 24206.35776048, 24095.33986576, 23985.37381209, 23876.43046386, 23768.48974584, 23661.54443537, 23555.56359209, 23450.55136357, 23346.4923953, 23243.38904298, 23141.19183839, 23039.90272625, 22939.50127095, 22840.00907242, 22741.40344225, 22643.6481698, 22546.74892171, 22450.70934177, 22355.49187891, 22261.08953053, 22167.42305394, 22074.72848136, 21982.75611957, 21891.49178289, 21801.07332421, 21711.43496504],

        # Middle set point for the quadratic wavelength solution
        "base_set_point_2": [24628.37672608, 24513.79686837, 24400.32734124, 24287.85495107, 24176.4424356, 24066.07880622, 23956.7243081, 23848.39610577, 23741.05658955, 23634.68688897, 23529.29771645, 23424.86836784, 23321.379387, 23218.80573474, 23117.1876433, 23016.4487031, 22916.61245655, 22817.65768889, 22719.56466802, 22622.34315996, 22525.96723597, 22430.41612825, 22335.71472399, 22241.83394135, 22148.73680381, 22056.42903627, 21964.91093944, 21874.20764171, 21784.20091295],

        # Right most set point for the quadratic wavelength solution
        "base_set_point_3": [24705.72472863, 24590.91231465, 24476.99298677, 24364.12010878, 24252.31443701, 24141.55527091, 24031.82506843, 23923.12291214, 23815.40789995, 23708.70106907, 23602.95596074, 23498.18607941, 23394.35163611, 23291.44815827, 23189.49231662, 23088.42080084, 22988.26540094, 22888.97654584, 22790.57559244, 22693.02942496, 22596.33915038, 22500.49456757, 22405.49547495, 22311.25574559, 22217.91297633, 22125.33774808, 22033.50356525, 21942.41058186, 21852.24253555],
        
        'n_splines': 6,
        'base': [-0.35, -0.05, 0.2],
        'spline': [-0.15, 0.01, 0.15]
    },
    
    # Fabry Perot cavity with two parameters
    'fringing_first_pass': {
        'name': 'fringing_first_pass',
        'class_name': 'BasicFringingModel',
        'd': [183900000.0, 183911000.0, 183930000.0],
        'fin': [0.01, 0.04, 0.08],
        'n_delay': 0
    },
    
    # Super fun fringing with 5 parameters
    'fringing_second_pass': {
        'name': 'fringing_second_pass',
        'class_name': 'ComplexFringingModel',
        'amp': [1E-10, 0.02, 0.08],
        'lam0': [-0.75, 0.1, 0.75],
        'lam2': [-0.5, 0.1, 0.5],
        'phase': [0, 3.14159265358979, 6.28318530717959],
        'tilt': [-0.8, -0.1, 0.8],
        'n_delay': 0,
        # Lambda 0 (reflection point) for the outgoing AR fringing model
        "fringing_2_reflection": [24554.94658131, 24441.67257011, 24329.02834104, 24217.58727579, 24107.09653424, 23997.63334112, 23889.26703348, 23781.78021455, 23675.33283695, 23569.78328332, 23465.22515237, 23361.62178871, 23258.90950661, 23157.06884695, 23056.31359839, 22956.28955486, 22857.17461502, 22759.01364935, 22661.57962376, 22565.19441016, 22469.39690361, 22374.50795128, 22280.46526063, 22187.23350991, 22094.6910732, 22003.04874085, 21912.1427246, 21822.06907833, 21732.54779138],

        # Lambda 2 (set point) for the outgoing AR fringing model
        "fringing_2_set_point": [24671.16805981, 24556.82399424, 24443.52432104, 24331.37943116, 24220.1755842, 24110.0661197, 24001.01700546,  23892.91367454, 23785.75185787, 23679.64221501, 23574.44372992, 23470.24056899, 23366.92230799, 23264.55020808, 23163.16930502, 23062.55316678, 22962.90881225, 22864.11979489, 22766.18872077, 22669.17453629, 22572.91920278, 22477.51781644, 22382.97401317, 22289.20108882, 22196.23740506, 22104.0878579, 22012.67884279, 21922.00392053, 21832.18039698]
    }
}
    
    