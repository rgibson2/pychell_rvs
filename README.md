# pychell_rvs

Extracts radial-velocities from reduced 1-dimensional echelle spectra by forward modeling the single orders.

from the directory this file is in, install pychell_rvs with
pip install .

The code is generalized to support any instrument, but before doing so at least two classes need to be implemented:
Until better documentation exists, also see the iSHELL, CHIRON,or PARVI implementations for examples.
All abstract methods existing for these classes must be implemnented for a given instrument.
1. SpecData
2. ForwardModel

A file in the "spectrographs" folder must be created with two dictionaries:
1. default_instrument_parameters
2. default_model_blueprints

The default_instrument_parameters dictionary contains any instrument dependent parameters. It must define:
1. spectrograph: the name of the spectrograph. Can be anything. (str)
2. observatory: The name of the the observatory. This must be recognized by astropy if not supplying own barycenter vels (str)
3. n_orders: the total number of possible orders (int)
4. n_data_pix: the number of data pixels present in the data (int)

It can define anything else helpful for this instrument used in the instrument specific forward model, model component, or data objects.

Each instrument must also define a dictionary called default_model_blueprints. This dictionary contains the blueprints to construct the forward model. Some keys in this dictionary are special. It must contain a 'star', 'lsf', and 'wavelength_solution'. Each item is then a dictionary which contains helpful info to construct that model component. Each model component must be tied to a class which implements/extends the SpectralComponent abstract class. An example entry is below:

```
'star': {
        'name': 'star',
        'class_name': 'StarModel',
        'input_file': None,
        'vel': [-np.inf, 0, np.inf]
    }
```

The name can be anything. The class_name must point to the class and live in the file pychell_rvs_spectral_components.py.
The input_file is the full path+filename to the stellar template file used. If None, things will start from a flat template.
The 'vel' key is the initial conditions for the stellar doppler shift parameter. These classes can have any remaining keywords that inform the model. When each class is initialized, it is given the above dictionary and the order number.

Example Run For Barnard's Star (GJ 699) for iSHELL:

Within a file called gj699.py:

```
import pychell_rvs.pychell_rvs as pychell_rvs

user_input_options = {
    "instrument": "iSHELL",
    "data_input_path": "/path/to/data/",
    "filelist": "some_filelist.txt", # Contains the names of the files to be read in.
    "output_path": "/path/to/output/",
    "bary_corr_file": None, # calcualting bc vels can be incredibly slow depending on versions
    "star_name": "Star_Name", # Use underscores for spaces
    "tag": "example",
    "do_orders": [15, 16, 17], # np.arange(number_of_orders).astype(int) for all orders
    "overwrite_output": 1,
    "n_template_fits": 0,
    "n_threads": 1,
    "nights_for_template": 'all',
    "model_resolution": 4
}

user_model_blueprints = {
    
    # The star
    'star': {
        'input_file': None
    },
    
    # The default blaze is a quadratic + splines.
    'blaze': {
        'n_splines': 5,
    },
    
    # Hermite Gaussian LSF
    'lsf': {
        'hermdeg': 0
    },
    
    # Quadratic (Lagrange points) + splines
    'wavelength_solution': {
        'n_splines': 6,
        'spline': [-0.15, -0.01, 0.15],
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

pyshell_rvs.pyshell_rvs_main(user_input_options, user_model_blueprints)
```