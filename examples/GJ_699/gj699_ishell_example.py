
import pychell_rvs.pychell_rvs as pychell_rvs
import os

# This provides access to default templates in code_path + 'default_templates' + os.sep
code_path = os.path.dirname(pychell_rvs.__file__) + os.sep

user_input_options = {
    "instrument": "iSHELL",
    "data_input_path": "data/",
    "filelist": "filelist_example.txt",
    "output_path": os.path.dirname(os.path.realpath(__file__)) + os.sep, # For current directory
    "bary_corr_file": None, # BJD and bc vels are computed from barycorrpy
    "star_name": "GJ_699",
    "tag": "defaul_test_run",
    "do_orders": [11, 13, 15],
    "overwrite_output": True,
    "n_template_fits": 3,
    "n_threads": 8,
    "verbose_plot": True,
    "verbose_print": True,
    "nights_for_template": [],
    "model_resolution": 8
}

user_model_blueprints = {
    
    # The star
    'star': {
        'input_file': None
    },
    
    # The default blaze is a quadratic + splines for iSHELL.
    'blaze': {
        'n_splines': 6,
        'n_delay': 0
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'hermdeg': 2
    },
    
    # Quadratic (3 Lagrange points) + splines
    'wavelength_solution': {
        'n_splines': 6,
        'n_delay': 0
    },
    
    # Fabry Perot cavity with two parameters
    # Disable completely, since this example data has fringign removed through flat division
    'fringing_first_pass': {
        'n_delay': 100
    },
    
    # Super fun fringing with 5 parameters
    # Disable completely, since this is an example and doesn't seem to improve the RVs
    'fringing_second_pass': {
        'n_delay': 100
    }
}

pychell_rvs.pychell_rvs_main(user_input_options, user_model_blueprints)