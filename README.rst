===========
pychell_rvs - The tldr;
===========

Extracts radial-velocities from reduced 1-dimensional echelle spectra by forward modeling the full single orders. This code can be adapted to a wide variety of spectrographs. For more information on adapting this code, see link here.

============
installation
============

from the directory this file is in, install pychell_rvs with

```pip install .```

This should install the relevant dependencies.


A quick start guide which uses data from the iSHELL spectrograph can be found here

The code is generalized to support any instrument, but before doing so at least two abstarct classes need to be implemented:
1. SpecData

This class contains information on how to handle the data for a given instrument. Each single order observation will correspond to a unique instance of the implemented class. These implementations of the abstract class all must belong in pychell_data.py. The __init__ function will by default take in the arguments
input_file : The name of the file to be read in for this observation. Discussed later on (str).
order_num : The image order number, starting at 0 (int).
spec_num : The spectrum number, starting at 0 (int).
gpars : a helpful dictionary containing pipeline wide general parameters. See below. (dict)

The implemented __init__ method should call the parant class init method through:

```
super().__init__(input_file, order_num, spec_num, gpars)
```

Checking the abstract class __init__ method, we see this stores the input_file, order_num, and spec_num. Then it calls the parse method. This abstract method needs to implemented by the user. This method only takes the gpars dictionary as input. The parse method must define the following class members for this order:

flux : The normalized flux for this observation. Unused ("bad") pixels must be converted to nans. (np.ndarray)
badpix : A bad pix array of ones and zeros. If not known, just set to all ones. Cropped pixels are still handled below. (np.ndarray).
BJD : The bary-centric julian date. If known a priori, set from the gpars key BJDS. Otherwise, one can call get_BC_vel if the JD UTC is known. (float)
bary_corr : The bary-center RV correction. If known a priori, set from the gpars key bary_corrs. Otherwise, one can call JDUTC_to_BJDTDB if the JD UTC is known. (float)
See https://github.com/shbhuk/barycorrpy for more information on these last two members.
obs_details : a dictionary containing any other remaining parameters for this observation one wants to "remember". This dict is stored in the outputs. (dict)

OPTIONAL:
If the wavelength solution is known a priori, even if it's just a "starting wavelength", it should be defined using the 'wave_grid' member. This will correspond to the flux array.

If the LSF for this observation and this order is known, it must be defined using the 'lsf' member. Since LSFs can be tricky and aren't used outside of the fitting, this can be anything that captures the LSF for this order. For example, if the LSF is known a priori from say a laser frequency comb, this member will contain that info. The implementation of the custom LSF for this instrument must know how to work with this variable, which the user can define (see custom model components later).

Finally, the parse method forces cropped pixels to be zero. That's it for this class!

2. ForwardModel

This class contains more general information necessary to build the forward model for this instrument.


A file in the "spectrographs" folder called parameters_insname.py must be created with the following two dictionaries. insname must further be all lowercase, otherwise identical to the given instrument name.
1. default_instrument_parameters
2. default_model_blueprints

The default_instrument_parameters dictionary contains any instrument dependent parameters. It must define:
1. spectrograph: the name of the spectrograph. Can be anything. (str)
2. observatory: The name of the the observatory. This must be recognized by astropy (EarthLocations) if not supplying own barycenter vels (str)
3. n_orders: the total number of possible orders (int)
4. n_data_pix: the number of data pixels present in the data (int)

It can also define anything else helpful for this instrument used in the instrument specific forward model, model component, or data objects. Otherwise, the following optional keywords are available to overwrite:

n_template_fits : The number of iteration a stellar template is fit to the data. A zeroth iteration does not count towards this number. If you only want a single run with a flat stellar template, set to zero and don't pass a stellar template input file. (int). Default: 10

do_xcorr : Whether or not a cross correlation analysis is performed after the fit. This takes time, but provides the bisector span of the ccf function which can be useful (bool). Default: False

model_resolution : The resolution of the model. It's important this is greater than 1 to ensure the convolution with the LSF is accurate. n_model_pix = n_data_pix * model_resolution. (int) Default: 8

flag_n_worst_pixels : The number of worst pixels to flag in the forward model (after weights are applied) (int). Default: 20

verbose_plot : Whether or not to add templates to the plots. (bool) Default: False

verbsoe_print : Whether or not to print the optimization results after each fit. (bool) Default: False

crop_pix : The number of data pix that are cropped on each side of the spectrum. The badpix array is updated to reflect these values. list; [left_most_pix, n_data_pix - right_most_pix] Default: [50, 50]

dpi : The dpi used for making plots (int). Default: 200

plot_wave_unit : The wavelength units in plots (str). Option are 'nm', 'ang', 'microns'. Default: 'nm'

lw: The linewidth in fits (float) Default: 0.8

spec_img_width_pix : The width in pixels of the fits (int). Default: 2000

spec_img_height_pix: The height in pixels of the fits (int). Default: 720

rv_img_width_pix : The width in pixels of the rv plots (int). Default: 1800

rv_img_height_pix: The height in pixels of the rv plots (int). Default: 600

target_function : The optimization function that minimizes some helpful quantity to fit the spectra. See custom target functions below (str)

That's it for the default_instrument_parameters dictionary.

Each instrument must also define a dictionary called default_model_blueprints. This dictionary contains the blueprints to construct the forward model. Some keys in this dictionary are special. It must contain a 'star' and 'wavelength_solution'. Each item is then a dictionary which contains helpful info to construct that model component. Each model component must be tied to a class which implements/extends the SpectralComponent abstract class in pychell_model_components.py. An example entry for a star:

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
The 'vel' item is [lower_bound, guess, upper_bound] for the stellar doppler shift parameter. These can have any remaining keywords that inform the model. When each class is initialized, it is given the above "blueprint" sub dictionary, the gpars dictionary, and the order number. The corresponding class for this model is StarModel.

Below is an example of a model component unique to iSHELL, and provides an idea of how to implement other custom model components.

The entry in default_model_blueprints:

```
'fringing_first_pass': {
    'name': 'fringing_first_pass',
    'class_name': 'BasicFringingModel',
    'd': [183900000.0, 183911000.0, 183930000.0],
    'fin': [0.01, 0.04, 0.08],
    'n_delay': 0
}
```

This will model one of the fringing components present in iSHELL spectra. It has parameters 'd' and 'fin'. The corresponding class is:

```
class BasicFringingModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = True
        self.base_par_names = ['_d', '_fin']
        self.name = blueprint['name']
        self.n_delay = blueprint['n_delay']
        self.par_names = [self.name + s for s in self.base_par_names]
    
    def build(self, pars, wave_final):
        if self.enabled:
            d = pars[self.par_names[0]].value
            fin = pars[self.par_names[1]].value
            theta = (2 * np.pi / wave_final) * d
            fringing = 1 / (1 + fin * np.sin(theta / 2)**2)
            return fringing
        else:
            return self.build_fake(wave_final.size)
    
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['d'][1], minv=blueprint['d'][0], maxv=blueprint['d'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=blueprint['fin'][1], minv=blueprint['fin'][0], maxv=blueprint['fin'][2], mcmcscale=0.1))
        return pars
    
    def modify(self, v):
        self.enabled = v
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
```

To run the code, a python config script must be created. This file must contain two dictionaries:
1. 

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




Custom optimization functions must be placed in the file pychell_target_functions.py. A custom target functions should take as input (gp, v, fwm, iter_num, templates_dict, gpars).

gp : the current parameters as a numpy array. (values only)
v : a boolean numpy array of which pars in gp are varied.
fwm : The forward model object for this observation / order
iter_num : The iteration number (int)
templates_dict : The templates dictionary.
gpars : The global parameters dictionary.

This function should first convert the parameters back to Parameter objects through:

```
gp_objects = pcmodelcomponents.Parameters.from_numpy(list(fwm.initial_parameters.keys()), values=gp, varies=v)
```

From here, the fwm.build() method can be called and a model returned. The data is accessible through fwm.data. From here, residuals and an effective RMS can be computed. The function must return (rms, cons) where rms is the minimization quantity, and cons is a constraint that must further be greater than zero or the target function is further penalized. For example, the LSF must be greater than zero, so we may wish to set cons=np.min(lsf). Multiple constraints can be included through a cons = np.min([cons1, cons2, ...])