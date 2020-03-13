import pychell_rvs.pychell_rvs as pychell_rvs

user_input_options = {
    "instrument": "iSHELL",
    "data_input_path": "/Users/gj_876/Research/Data/iSHELL/Reduced/ByTarget/Fringing_in_flats/TOI_836/",
    "filelist": "filelist_all.txt",
    "output_path": "/Users/gj_876/Research/RV_Runs/",
    "bary_corr_file": None,
    "star_name": "CD-23_12010",
    "tag": "new_format",
    "do_orders": [11],
    "overwrite_output": True,
    "n_template_fits": 3,
    "n_threads": 8,
    "verbose_plot": True,
    "verbose_print": True,
    "nights_for_template": 'all',
    "templates_to_optimize": [],
    "model_resolution": 4
}

user_model_blueprints = {
    
    # The star
    'star': {
        'input_file': '/Users/gj_876/Research/Synthetic_Templates/BT_Settl/toi836_btsettl_ishell1.npz'
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'n_splines': 6,
        'n_delay': 0
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'hermdeg': 2
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'n_splines': 50,
        'n_delay': 0
    },
    
    # Fabry Perot cavity with two parameters
    'fringing_first_pass': {
        'n_delay': 100
    },
    
    # Super fun fringing with 5 parameters
    'fringing_second_pass': {
        'n_delay': 100
    }
}

pychell_rvs.pychell_rvs_main(user_input_options, user_model_blueprints)