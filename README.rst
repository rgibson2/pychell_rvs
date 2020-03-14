=======================
pychell_rvs - The tldr;
=======================

Extracts radial-velocities from reduced 1-dimensional echelle spectra by forward modeling the full single orders. This code can be adapted to a wide variety of spectrographs. Currently this code supports the following instruments

- iSHELL (0.375" slit, *KGAS* mode, methane gas cell)
- CHIRON (narrow slit mode, R~136K, iodine cell)
- PARVI (under development)
- NIRSPEC (under development)

For more information on adapting this code, see link here.

============
Installation
============

from the directory this file is in, install pychell_rvs with

``pip install .``

This should install the relevant dependencies.

===============
Getting Started
===============

Below is a quick-start guide which fits 4 nights (8 spectra) of Barnard's Star spectra using spectra from the iSHELL spectrograph. Only orders 11, 13, and 15 are fit. The wavelength solution and LSF are obtained from a methane gas cell, which are provided in default_templates.

First, copy the example folder GJ_699 to a new location of your choice. Open a terminal window in this new location. Run:

``python gj699_ishell_example.py``

If all goes well, the code should start printing helpful messages as it runs, including best fit parameters as fits finish. An output directory will also be created in the current folder called ``GJ_699_default_test_run``. This folder contains the global parameters dictionary (stored in a .npz file) used throughout the code and sub folders for each order. Each sub folder contains the following:

1. Fits - Contains the forward model plots for each iteration. Also contains ``.npz`` files with the following keys:
    - wave : The wavelegnth solutions for this spectrum, np.ndarray, shape=(n_data_pix, n_template_fits)
    - models : The best fit constructed forward models for this spectrum, np.ndarray, shape=(n_data_pix, n_template_fits)
    - residuals : The residuals for this spectrum (data - model), np.ndarray, shape=(n_data_pix, n_template_fits)
    - data : The data for this observation, np.ndarray, shape=(n_data_pix, 3), col1 is flux, col2 is flux_unc, col3 is badpix

2. Opt - Contains the optimization results from the Nelder-Mead fitting stored in .npz files. This must be loaded with allow_pickle=True. Keys are:
    - best_fit_pars : The best fit parameters, np.ndarray, shape=(n_template_fits,). Each entry is a Parameters object.
    - opt : np.ndarray, shape=(n_template_fits, 2). col1=final RMS returned by the solver. col2=total target function calls.

3. RVs - Contains the RVs for this order. Fits for each iteration are in .png files. The RVs are stored in the per iteration .npz files with the following keys:
    - rvs : The best fit RVs, np.ndarray, shape=(n_spec, n_template_fits)
    - rvs_nightly : The co-added ("nightly" RVs), np.ndarray, shape=(n_nights, n_template_fits)
    - rvs_unc_nightly : The corresponding 1 sigma error bars for the nightly RVs, same shape
    - BJDS : The bary-centric Julian dates which correspond to the single RVs.
    - BJDS_nightly : The nightly BJDS which correspond to the nightly RVs.
    - n_obs_nights : The number of observation observed on each night, np.ndarray, shape=(n_nights,)

4. Stellar_Templates - Contains the stellar template over iterations. For now contains a single .npz file with key:
    - stellar_templates : The stellar template used in each iteration, np.ndarray, shape=(n_model_pix, n_template_fits+1). col1 is wavelength, remaining cols are flux.

And that's it!


=====================
Supported Instruments
=====================



To use the code on a supported instrument but using ones own data, we look closer at the the example file ``gj699_example.py``, which defines two dictionaries and passes these to the pipeline:

1. user_input_options
2. user_model_blueprints

Below is information on how to use these dictionaries.

******************
user_model_options
******************

This dictionary contains both necessary entries to get things going, and optional entries which have default settings the user may wish to overwrite.

REQUIRED
########

- instrument : The spectrogrpah the data was taken with. Must be in the supported instruments - iSHELL, PARVI, CHIRON, NIRSPEC. (str).
- data_input_path : The data input path. All spectra must be stored in this single directory. (str)
- filelist : The text file containing the files (one per line) to be used in this run. This file must be stored in data_input_path. Order is not important (str).
- output_path : The output path to store the run in. A single directory is created per run. (str).
- star_name : The name of the star. If fetching bary-center info from barycorrpy, it must be searchable on simbad with this entry. FOr spaces, use an underscore. (str)
- tag : A tag to uniquely identfiy this run. The main level path for this run will be called star_name + tag. All files will include star_name + tag.
- do_orders : Which echelle orders to do. e.g., np.arange(1, 30).astype(int) would do all 29 iSHELL orders. Or a list of orders [4, 5, 6] will only fit orders 4-6. Orders are fit in numerical order, not the order they are provided. (iterable)


OPTIONAL
########

- bary_corr_file : A csv file in data_input_path containing the bary-center info. col1=BJDS, col2=bc_vels. The order must be consistent with the order provided in filelist. (str), DEFAULT: None, and bc info is calculated with barycorrpy.
- overwrite_output : If True, the output path is overwritten as the code runs. (bool). DEFAULT: True.
- n_cores : The number of cores used in the Nelder-Mead fitting and possible cross corr analysis. (int). DEFAULT: 1
- verbose_plot : Whether or not to add templates to the plots. (bool) DEFAULT: False
- verbose_print : Whether or not to print the optimization results after each fit. (bool) DEFAULT: False
- nights_for_template : Which nights to include when updating the stellar template. e.g., [1,2] will only use the first and second nights. Use an empty list to use all nights. (list). DEFAULT: [] for all nights.
- n_template_fits : The number of times a real stellar template is fit to the data. DEFAULT: 10
- model_resolution : The resolution of the model. It's important this is greater than 1 to ensure the convolution with the LSF is accurate. n_model_pix = n_data_pix * model_resolution. (int) DEFAULT: 8
- do_xcorr : Whether or not a cross correlation analysis is performed after the fit. This takes time, but provides the bisector span of the ccf function which can be useful (bool). DEFAULT: False
- flag_n_worst_pixels : The number of worst pixels to flag in the forward model (after weights are applied) (int). DEFAULT: 20
- plot_wave_unit : The wavelength units in plots (str). Option are 'nm', 'ang', 'microns'. DEFAULT: 'nm'
- lw : The linewidth in fits (float) DEFAULT: 0.8
- spec_img_width_pix : The width in pixels of the fits (int). DEFAULT: 2000
- spec_img_height_pix : The height in pixels of the fits (int). DEFAULT: 720
- rv_img_width_pix : The width in pixels of the rv plots (int). DEFAULT: 1800
- rv_img_height_pix: The height in pixels of the rv plots (int). DEFAULT: 600
- crop_pix : The number of pixels cropped on the ends each order; [crop_from_left, crop_from_right]. If the bad pix array provided with the data allows for a wider window, the window is still cropped according to this entry. If the bad pix array is smaller, the entry is irrelevant. (list). DEFAULT: [10, 10]
- target_function : The optimization function that minimizes some helpful quantity to fit the spectra. As of now, only two functions are implemented (basic and weighted RMS). See ``pychell_target_functions.py`` for more info. (str)

*********************
user_model_blueprints
*********************

Each instrument defines its own default_model_blueprints dictionary, stored in pychell_rvs/spectrographs/parameters_insname.py. This dictionary contains the blueprints to construct the forward model. Some keys in this dictionary are special. It must contain a 'star' and 'wavelength_solution'. Each item is then a dictionary which contains helpful info to construct that model component. Each model component must be tied to a class which implements/extends the SpectralComponent abstract class in pychell_model_components.py. For a given run, the user may wish to overwrite some of these defaults. This is done through defining the user_model_blueprints dictionary in their run file. From here, the user can add new model components by adding new keys, or updating existing ones by redefining an existing key. Three cases exist:

1. Key is common to both dictionaries - The item will only be updated according to the sub keys.
2. Key exists only in the user blueprints but not the default - The new model is added and must contain all information necessary (see below on defnining new models).
3. Key exists only in the default blueprints - Default settings are used.

Example of overriding blueprints model to start from a synthetic stellar template. The default setting was ``None`` - to start from a flat stellar template. This will now start things from a real template.

 ``
'star' : {
    'input_file' : '/path/to/input_file/'
}
 `` 
 

There are a few special keys required for each entry in this dictionary (see defining new models below). The format of each sub dictionary can be anything that the model supports. So, to know how to override settings for other mode components, one must look at the default model (in default_model_blueprints) to see what is available.

=========
Templates
=========

Custom (synthetic or empirical) templates may be used. Templates must be stored in .npz files and have the following keywords: wave (in angstroms), flux. Templates are always cropped to the order (with small padding).

===========================
Support for New Instruments
===========================

Coming soon!