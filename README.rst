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
installation
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

