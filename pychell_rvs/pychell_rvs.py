# Python built in modules
import copy
from collections import OrderedDict
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import time # Time the code
import sys # sys utils
from sys import platform # plotting backend
import pdb # debugging
stop = pdb.set_trace

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting

# Science/math
import scipy
from scipy import constants as cs # cs.c = speed of light in m/s
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# User defined/pip modules
import pychell_rvs.pychell_math as pcmath # mathy equations
import pychell_rvs.pychell_forward_models as pcforwardmodels # the various forward model implementations
import pychell_rvs.pychell_data as pcdata # the data objects
import pychell_rvs.pychell_model_components as pcmodelcomponents # the data objects
import pychell_rvs.pychell_solver as pcsolver # nelder mead solver
import pychell_rvs.pychell_utils as pcutils # random helpful functions

# Main function
def pychell_rvs_main(user_input_options, user_model_blueprints):

    # Start the main clock!
    stopwatch = pcutils.StopWatch()
    stopwatch.lap(name='ti_main')

    # Set things up and create a dictionary gpars used throughout the code
    gpars, model_blueprints, data_all_first_order = init_pipeline(user_input_options, user_model_blueprints)
    
    # Main loop over orders
    for order_num in range(gpars['n_orders']):

        # Move on if not fitting this order
        if order_num not in gpars['do_orders']:
            continue

        print('Starting order ' + str(order_num + 1) + ' (of ' + str(gpars['n_do_orders']) + ' order(s) ...)', flush=True)
        
        # Load in the data for this order
        if order_num == gpars['do_orders'][0]:
            data_all = data_all_first_order
        else:
            data_all = []
            data_class_init = getattr(pcdata, 'SpecData' + gpars['instrument'])
            for ispec in range(gpars['n_spec']):
                print('Loading In Observation ' + str(ispec+1) + ' ...', flush=True)
                data_all.append(data_class_init(gpars['data_filenames'][ispec], order_num, ispec, gpars))

        # First estimate the end points of the wavelength grid
        gpars['wave_left'], gpars['wave_right'] = pcutils.estimate_wavegrid_endpoints(data_all[0], order_num, model_blueprints['wavelength_solution'], gpars)
        
        # Stores the stellar templates over iterations. The plus 1 is for the wave grid
        stellar_templates = np.empty(shape=(gpars['n_model_pix'], gpars['n_template_fits'] + 1), dtype=np.float64)
        
        # templates_dict is a dictionary to store the large array synthetic templates (star, gas, tellurics)
        # models_dict is a dictionary to store the large array synthetic templates (star, gas, tellurics)
        # We only need a single instance of templates_dict for realistic memory usage
        # models_dict is per spectrum dependent and will belong to each forward model object as a class member
        # This allows for per spectrum customization if need be (not implemented) but templates_dict will still
        # store the majority of the synethetic data.
        
        templates_dict = OrderedDict()
        models_dict = OrderedDict()
        init_pars = pcmodelcomponents.Parameters()

        # First generate the wavelength solution model
        model_class_init = getattr(pcmodelcomponents, model_blueprints['wavelength_solution']['class_name'])
        models_dict['wavelength_solution'] = model_class_init(order_num, model_blueprints['wavelength_solution'], gpars)
        # Construct the initial parameters for the wavelength solution
        pars = models_dict['wavelength_solution'].initialize_parameters(model_blueprints['wavelength_solution'], gpars)
        for par in pars:
            init_pars.add_from_parameter(par)
        
        # Second generate the Stellar Model
        model_class_init = getattr(pcmodelcomponents, model_blueprints['star']['class_name'])
        models_dict['star'] = model_class_init(order_num, model_blueprints['star'], gpars)
        templates_dict['star'] = models_dict['star'].load_template(gpars)
        stellar_templates[:, 0] = templates_dict['star'][:, 0]
        if gpars['do_init_guess']:
            stellar_templates[:, 1] = templates_dict['star'][:, 1]
        # Construct the initial parameters for the star
        pars = models_dict['star'].initialize_parameters(model_blueprints['star'], gpars)
        for par in pars:
            init_pars.add_from_parameter(par)
        
        # Define the following value for the LSF to properly be in units of Angstroms.
        # The stellar template must be evenly spaced, since it's the master grid.
        gpars['dl'] = (templates_dict['star'][-1, 0] - templates_dict['star'][0, 0]) / gpars['n_model_pix']
        
        # Generate the remaining model components from their blueprints and load any input templates
        for blueprint in model_blueprints:
            if blueprint in ['wavelength_solution', 'star']:
                continue
            
            # Construct the model
            model_class_init = getattr(pcmodelcomponents, model_blueprints[blueprint]['class_name'])
            models_dict[blueprint] = model_class_init(order_num, model_blueprints[blueprint], gpars)
            
            # Load templates need be
            if 'input_file' in model_blueprints[blueprint].keys() or 'components' in model_blueprints[blueprint].keys():
                templates_dict[blueprint] = models_dict[blueprint].load_template(gpars)
            
            # Construct the initial parameters for this model component
            pars = models_dict[blueprint].initialize_parameters(model_blueprints[blueprint], gpars)
            for par in pars:
                init_pars.add_from_parameter(par)
                
        # Construct the forward model objects for seamless forward modeling
        forward_models = pcforwardmodels.ForwardModels(gpars) # basically a fancy list
        forward_model_class_init = getattr(pcforwardmodels, gpars['instrument'] + 'ForwardModel')
        for ispec in range(gpars['n_spec']):
            forward_models.append(forward_model_class_init(ispec, order_num, models_dict, data_all[ispec], init_pars, gpars))

        # Get better estimations for some parameters (eg xcorr for star)
        forward_models.optimize_guess_parameters(model_blueprints, templates_dict, gpars)

        # Zeroth Iteration - No doppler shift if using a flat template.
        # We could also flag the stellar lines, but this has minimal impact on the RVs
        if not gpars['do_init_guess']:

            print('  Iteration: 0 (flat stellar template, no RVs)', flush=True)
            forward_models.fit_spectra(0, templates_dict, gpars)

            if gpars['n_template_fits'] == 0:
                # Output and continue to next order
                for i in range(gpars['n_spec']):
                    forward_models[i].save_final_outputs(gpars)
                continue
            else:
                update_stellar_template(templates_dict, forward_models, 0, gpars)
                stellar_templates[:, 1] = templates_dict['star'][:, 1]
                forward_models.update_model_params(0, gpars)

        # Iterate over remaining stellar template generations
        for iter_num in range(gpars['n_template_fits']):

            print('Starting Iteration: ' + str(iter_num+1) + ' of ' + str(gpars['n_template_fits']), flush=True)
            stopwatch.lap(name='ti_iter')

            # Run the fit for all spectra and do a cross correlation analysis as well.
            forward_models.fit_spectra(iter_num+gpars['ndi'], templates_dict, gpars)
            
            print('Finished Iteration ' + str(iter_num+1) + ' in ' + str(round(stopwatch.time_since(name='ti_iter'), 2)/3600) + ' hours', flush=True)
            
            # Compute the RVs and output after each iteration
            forward_models.compute_nightly_rvs(iter_num, gpars)
            forward_models.plot_rvs(iter_num, gpars)
            forward_models.save_rvs(gpars)

            # Print RV Diagnostics
            if gpars['n_nights'] > 1:
                rvscd_std = np.std(forward_models.rvs[:, iter_num])
                print('  Stddev of all nightly RVs: ' + str(round(rvscd_std, 4)) + ' m/s', flush=True)

            # Compute the new stellar template, update parameters.
            if iter_num + 1 < gpars['n_template_fits']:

                # Update the forward model initial_parameters.
                forward_models.update_model_params(iter_num+gpars['ndi'], gpars)
                
                # Update the template through least squares cubic spline interpolation if there are more than 5 nights.
                # Otherwise interpolate each grid of residuals onto a common grid with cubic spline interpolation, then crunch using weighted median
                if gpars['n_nights'] >= 5:
                    cubic_spline_lsq_template(templates_dict, forward_models, iter_num+gpars['ndi'], gpars)
                else:
                    update_stellar_template(templates_dict, forward_models, iter_num+gpars['ndi'], gpars)
                stellar_templates[:, iter_num+1] = templates_dict['star'][:, 1]

        # Save forward model outputs
        print('Saving Final Outputs ... ', flush=True)
        forward_models.save_final_outputs(gpars)

        # Save Stellar Template Outputs
        np.savez(gpars['run_output_path'] + 'Order' + str(order_num+1) + os.sep + 'Stellar_Templates' + os.sep + gpars['full_tag'] + '_stellar_templates_ord' + str(order_num+1) + '.npz', stellar_templates=stellar_templates)

    # End the clock!
    print('ALL DONE! Runtime: ' + str(round(stopwatch.time_since(name='ti_main') / 3600, 2)) + ' hours', flush=True)

def cubic_spline_lsq_template(templates_dict, forward_models, iter_num, gpars):
    
    current_stellar_template = np.copy(templates_dict['star'])
    
    # Storage Arrays for the low res grid
    # This is for the low res reiduals where the star is constructed via a least squares cubic spline.
    # Before the residuals are added, they are normalized.
    waves_shifted_lr = np.empty(shape=(gpars['n_data_pix'], gpars['n_spec']), dtype=np.float64)
    residuals_lr = np.empty(shape=(gpars['n_data_pix'], gpars['n_spec']), dtype=np.float64)
    tot_weights_lr = np.empty(shape=(gpars['n_data_pix'], gpars['n_spec']), dtype=np.float64)
    
    # Weight by 1 / rms^2
    rms = np.array([forward_models[ispec].opt[iter_num][0] for ispec in range(gpars['n_spec'])])
    rms_weights = 1 / rms**2
    if iter_num > 1:
        bad = np.where(rms_weights < 10)[0]
        if bad.size > 0:
            rms_weights[bad] = 0
            
    if len(gpars['nights_for_template']) == 0: # use all nights
        template_spec_indices = np.arange(gpars['n_spec']).astype(int)
    else: # use specified nights
        template_spec_indices = []
        for inight in gpars['nights_for_template']:
            template_spec_indices += pcutils.get_spec_indices_from_night(inight - 1, gpars)
            
    # Loop over spectra
    for ispec in range(gpars['n_spec']):

        # De-shift residual wavelength scale according to the barycenter correction
        # Or best doppler shift if using a non flat initial template
        if gpars['do_init_guess']:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[:, iter_num] * np.exp(-1 * forward_models[ispec].best_fit_pars[iter_num][forward_models[ispec].models_dict['star'].par_names[0]].value / cs.c)
            waves_shifted_lr[:, ispec] = wave_stellar_frame
        else:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[:, iter_num] * np.exp(forward_models[ispec].data.bary_corr / cs.c)
            waves_shifted_lr[:, ispec] = wave_stellar_frame
            
        residuals_lr[:, ispec] = forward_models[ispec].residuals[:, iter_num]

        # Telluric weights
        tell_flux_hr = forward_models[ispec].models_dict['tellurics'].build(forward_models[ispec].best_fit_pars[iter_num], templates_dict['tellurics'], current_stellar_template[:, 0])
        tell_flux_hr_convolved = forward_models[ispec].models_dict['lsf'].convolve_flux(tell_flux_hr, pars=forward_models[ispec].best_fit_pars[iter_num])
        tell_flux_lr_convolved = np.interp(forward_models[ispec].wavelength_solutions[:, iter_num], current_stellar_template[:, 0], tell_flux_hr_convolved, left=np.nan, right=np.nan)
        tell_weights = tell_flux_lr_convolved**2
        
        # Final weights
        tot_weights_lr[:, ispec] = forward_models[ispec].data.badpix * rms_weights[ispec] * tell_weights
        
    # Generate the histogram
    hist_counts, histx = np.histogram(gpars['bary_corrs'], bins=int(np.min([gpars['n_spec'], 10])), range=(np.min(gpars['bary_corrs'])-1, np.max(gpars['bary_corrs'])+1))
    
    # Check where we have no spectra (no observations in this bin)
    hist_counts = hist_counts.astype(np.float64)
    bad = np.where(hist_counts == 0)[0]
    if bad.size > 0:
        hist_counts[bad] = np.nan
    number_weights = 1 / hist_counts
    number_weights = number_weights / np.nansum(number_weights)

    # Loop over spectra and check which bin an observation belongs to
    # Then update the weights accordingly.
    if len(gpars['nights_for_template']) == 0:
        for ispec in range(gpars['n_spec']):
            vbc = forward_models[ispec].data.bary_corr
            y = np.where(histx >= vbc)[0][0] - 1
            tot_weights_lr[:, ispec] = tot_weights_lr[:, ispec] * number_weights[y]
            
    # Now to co-add residuals according to a least squares cubic spline
    # Flatten the arrays
    waves_shifted_lr_flat = waves_shifted_lr.flatten()
    residuals_lr_flat = residuals_lr.flatten()
    tot_weights_lr_flat = tot_weights_lr.flatten()
    
    # Remove all bad pixels.
    good = np.where(np.isfinite(waves_shifted_lr_flat) & np.isfinite(residuals_lr_flat) & (tot_weights_lr_flat > 0))[0]
    waves_shifted_lr_flat, residuals_lr_flat, tot_weights_lr_flat = waves_shifted_lr_flat[good], residuals_lr_flat[good], tot_weights_lr_flat[good]

    # Sort the wavelengths
    sorted_inds = np.argsort(waves_shifted_lr_flat)
    waves_shifted_lr_flat, residuals_lr_flat, tot_weights_lr_flat = waves_shifted_lr_flat[sorted_inds], residuals_lr_flat[sorted_inds], tot_weights_lr_flat[sorted_inds]
    
    # Knot points are roughly the detector grid. 
    knots_init = np.linspace(waves_shifted_lr_flat[0]+0.01, waves_shifted_lr_flat[-1]-0.01, num=gpars['n_use_data_pix'])
    bad_knots = []
    for iknot in range(len(knots_init) - 1):
        n = np.where((waves_shifted_lr_flat > knots_init[iknot]) & (waves_shifted_lr_flat < knots_init[iknot+1]))[0].size
        if n == 0:
            bad_knots.append(iknot + 1)
    bad_knots = np.array(bad_knots)
    knots = np.delete(knots_init, bad_knots)

    # Do the fit
    tot_weights_lr_flat /= np.nansum(tot_weights_lr_flat)
    spline_fitter = scipy.interpolate.LSQUnivariateSpline(waves_shifted_lr_flat, residuals_lr_flat, t=knots, w=tot_weights_lr_flat, k=3, ext=1, bbox=[waves_shifted_lr_flat[0], waves_shifted_lr_flat[-1]], check_finite=True)
    
    # Use the fit to determine the hr residuals to add
    residuals_hr_fit = spline_fitter(current_stellar_template[:, 0])

    # Remove bad regions
    bad = np.where((current_stellar_template[:, 0] <= knots[0]) | (current_stellar_template[:, 0] >= knots[-1]))[0]
    if bad.size > 0:
        residuals_hr_fit[bad] = 0

    # Augment the template
    new_flux = current_stellar_template[:, 1] + residuals_hr_fit
    
    bad = np.where(new_flux > 1)[0]
    if bad.size > 0:
        new_flux[bad] = 1
        
    templates_dict['star'] = np.array([current_stellar_template[:, 0], new_flux]).T
            
    
def update_stellar_template(templates_dict, forward_models, iter_num, gpars):
    
    current_stellar_template = np.copy(templates_dict['star'])

    # Stores the shifted high resolution residuals (all on the star grid)
    residuals_hr = np.empty(shape=(gpars['n_model_pix'], gpars['n_spec']), dtype=np.float64)
    bad_pix_hr = np.empty(shape=(gpars['n_model_pix'], gpars['n_spec']), dtype=bool)
    tot_weights_hr = np.zeros(shape=(gpars['n_model_pix'], gpars['n_spec']), dtype=np.float64)
    
    # Stores the weighted median grid. Is set via loop, so pre-allocate.
    residuals_median = np.empty(gpars['n_model_pix'], dtype=np.float64)
    
    # These show the min and max of of the residuals for all observations, useful for plotting if desired.
    residuals_max = np.empty(gpars['n_model_pix'], dtype=np.float64)
    residuals_min = np.empty(gpars['n_model_pix'], dtype=np.float64)
    
    # Weight by 1 / rms^2
    rms = np.array([forward_models[ispec].opt[iter_num][0] for ispec in range(gpars['n_spec'])]) 
    rms_weights = 1 / rms**2
    
    if len(gpars['nights_for_template']) == 0: # use all nights
        template_spec_indices = np.arange(gpars['n_spec']).astype(int)
    else: # use specified nights
        template_spec_indices = []
        for inight in gpars['nights_for_template']:
            template_spec_indices += pcutils.get_spec_indices_from_night(inight - 1, gpars)

    # Loop over spectra
    for ispec in range(gpars['n_spec']):

        # De-shift residual wavelength scale according to the barycenter correction
        # Or best doppler shift if using a non flat initial template
        if gpars['do_init_guess']:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[:, iter_num] * np.exp(-1 * forward_models[ispec].best_fit_pars[iter_num][forward_models[ispec].models_dict['star'].par_names[0]].value / cs.c)
        else:
            wave_stellar_frame = forward_models[ispec].wavelength_solutions[:, iter_num] * np.exp(forward_models[ispec].data.bary_corr / cs.c)

        # Telluric Weights
        tell_flux_hr = forward_models[ispec].models_dict['tellurics'].build(forward_models[ispec].best_fit_pars[iter_num], templates_dict['tellurics'], current_stellar_template[:, 0])
        tell_flux_hr_convolved = forward_models[ispec].models_dict['lsf'].convolve_flux(tell_flux_hr, pars=forward_models[ispec].best_fit_pars[iter_num])
        tell_weights_hr = tell_flux_hr_convolved**2

        # For the high res grid, we need to interpolate the bad pixel mask onto high res grid.
        # Any pixels not equal to 1 after interpolation are considered bad.
        bad_pix_hr[:, ispec] = np.interp(current_stellar_template[:, 0], wave_stellar_frame, forward_models[ispec].data.badpix, left=0, right=0)
        bad = np.where(bad_pix_hr[:, ispec] < 1)[0]
        if bad.size > 0:
            bad_pix_hr[bad, ispec] = 0

        # Weights for the high res residuals
        tot_weights_hr[:, ispec] = rms_weights[ispec] * bad_pix_hr[:, ispec] * tell_weights_hr

        # Only use finite values and known good pixels for interpolating up to the high res grid.
        # Even though bad pixels are ignored later when median combining residuals,
        # they will still affect interpolation in unwanted ways.
        good = np.where(np.isfinite(forward_models[ispec].residuals[:, iter_num]) & (forward_models[ispec].data.badpix == 1))
        residuals_interp_hr = scipy.interpolate.CubicSpline(wave_stellar_frame[good], forward_models[ispec].residuals[good, iter_num].flatten(), bc_type='not-a-knot', extrapolate=False)(current_stellar_template[:, 0])

        # Determine values with np.nans and set weights equal to zero
        bad = np.where(~np.isfinite(residuals_interp_hr))[0]
        if bad.size > 0:
            tot_weights_hr[bad, ispec] = 0
            bad_pix_hr[bad, ispec] = 0

        # Also ensure all bad pix in hr residuals are nans, even though they have zero weight
        bad = np.where(tot_weights_hr[:, ispec] == 0)[0]
        if bad.size > 0:
            residuals_interp_hr[bad] = np.nan

        # Pass to final storage array
        residuals_hr[:, ispec] = residuals_interp_hr

    # Additional Weights:
    # Up-weight spectra with poor BC sampling.
    # In other words, we weight by the inverse of the histogram values of the BC distribution
    # Generate the histogram
    hist_counts, histx = np.histogram(gpars['bary_corrs'], bins=int(np.min([gpars['n_spec'], 10])), range=(np.min(gpars['bary_corrs'])-1, np.max(gpars['bary_corrs'])+1))
    
    # Check where we have no spectra (no observations in this bin)
    hist_counts = hist_counts.astype(np.float64)
    bad = np.where(hist_counts == 0)[0]
    if bad.size > 0:
        hist_counts[bad] = np.nan
    number_weights = 1 / hist_counts
    number_weights = number_weights / np.nansum(number_weights)

    # Loop over spectra and check which bin an observation belongs to
    # Then update the weights accordingly.
    if len(gpars['nights_for_template']):
        for ispec in range(gpars['n_spec']):
            vbc = forward_models[ispec].data.bary_corr
            y = np.where(histx >= vbc)[0][0] - 1
            tot_weights_hr[:, ispec] = tot_weights_hr[:, ispec] * number_weights[y]

    # Only use specified nights
    tot_weights_hr = tot_weights_hr[:, template_spec_indices]
    bad_pix_hr = bad_pix_hr[:, template_spec_indices]
    residuals_hr = residuals_hr[:, template_spec_indices]

    # Co-add residuals according to a weighted median crunch
    # 1. If all weights at a given pixel are zero, set median value to zero.
    # 2. If there's more than one spectrum, compute the weighted median
    # 3. If there's only one spectrum, use those residuals, unless it's nan.
    for ix in range(gpars['n_model_pix']):
        if np.nansum(tot_weights_hr[ix, :]) == 0:
            residuals_median[ix] = 0
        else:
            if gpars['n_spec'] > 1:
                # Further flag any pixels larger than 3*wstddev from a weighted average, but use the weighted median.
                #wavg = pcmath.weighted_mean(residuals_hr[ix, :], tot_weights_hr[ix, :]/np.nansum(tot_weights_hr[ix, :]))
                #wstddev = pcmath.weighted_stddev(residuals_hr[ix, :], tot_weights_hr[ix, :]/np.nansum(tot_weights_hr[ix, :]))
                #diffs = np.abs(wavg - residuals_hr[ix, :])
                #bad = np.where(diffs > 3*wstddev)[0]
                #if bad.size > 0:
                    #tot_weights_hr[ix, bad] = 0
                    #bad_pix_hr[ix, bad] = 0
                residuals_median[ix] = pcmath.weighted_median(residuals_hr[ix, :], weights=tot_weights_hr[ix, :]/np.nansum(tot_weights_hr[ix, :]))
            elif ~np.isfinite(residuals_hr[ix, 0]):
                residuals_median[ix] = residuals_hr[ix, 0]
            else:
                residuals_median[ix] = 0

        # Store the min and max
        residuals_max[ix] = np.nanmax(residuals_hr[ix, :] * bad_pix_hr[ix, :])
        residuals_min[ix] = np.nanmin(residuals_hr[ix, :] * bad_pix_hr[ix, :])
        
    # Change any nans to zero
    bad = np.where(~np.isfinite(residuals_median))[0]
    if bad.size > 0:
        residuals_median[bad] = 0

    # Augment the template
    new_flux = current_stellar_template[:, 1] + residuals_median

    # Force the max to be less than 1.
    bad = np.where(new_flux > 1)[0]
    if bad.size > 0:
        new_flux[bad] = 1.0
        
    templates_dict['star'] = np.array([current_stellar_template[:, 0], new_flux]).T
                    

# Initialize the pipeline based on input_options file
def init_pipeline(user_input_options, user_model_blueprints):

    # Numpy warnings
    np.seterr(divide='ignore')
    warnings.filterwarnings('ignore')

    # Use a dictionary to store global pars.
    global_pars = {}

    # First the pipeline path to the dict
    global_pars['pipeline_path'] = os.path.dirname(os.path.realpath(__file__)) + os.sep
    
    # Load the config file and add to dict.
    spec = importlib.util.spec_from_file_location('config', global_pars['pipeline_path'] + 'config.py')
    module_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_)
    config_parameters = module_.default_config
    global_pars.update(config_parameters)
    #for config_key in config_parameters:
        #global_pars[config_key] = config_parameters[config_key]
    
    # Load in the default instrument settings and add to dict.
    spec = importlib.util.spec_from_file_location('parameters_' + user_input_options['instrument'].lower(), global_pars['pipeline_path'] + 'spectrographs' + os.sep + 'parameters_' + user_input_options['instrument'].lower() + '.py')
    module_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_)
    # instrument parameters (combine with global_parameters)
    instrument_parameters = module_.default_instrument_parameters
    # Default blueprints (new dictionary since config.py does not define any default models)
    model_blueprints = module_.default_model_blueprints
    # Merge default global pars with instrument specific settings
    global_pars.update(instrument_parameters)
    # Now do the same for the user input dict
    global_pars.update(user_input_options)

    # Now do something similar for the model blueprints dictionary
    # The model blueprints always contain all the model compnents (keys) defined in the defauly blueprints.
    # If a user wishes to disable a model component, they can set the n_delay keyword to anything larger than n_template_fits
    # If a user wishes to use a different model component in place of default one, they can do so by changing the class keyword
    # If a user wishes to add a new model comopnent, they can add it. But it must have a new name.
    # If a user wishes to modify the default settings, they can do so by creating a new branch.
    # So, if a key is common to the user blueprints and instrument blueprints, the sub keys are updated.
    for user_key in user_model_blueprints:
        if user_key in model_blueprints: # Key is common, update sub keys only
            model_blueprints[user_key].update(user_model_blueprints[user_key])
        else: # Key is new, just add
            model_blueprints[user_key] = user_model_blueprints[user_key]
            
    # Helpful things if we do a zeroth iteration with no stellar template (and thus no rvs)
    if model_blueprints['star']['input_file'] is not None:
        # Starting from a synthetic stellar template
        global_pars['do_init_guess'] = True
        global_pars['di'] = 1
        global_pars['ndi'] = 0
    else:
        # Starting from a flat star
        global_pars['do_init_guess'] = False
        global_pars['di'] = 0
        global_pars['ndi'] = 1

    # The number of echelle orders
    global_pars['do_orders'] = np.atleast_1d(global_pars['do_orders']) - 1 # zero indexing
    global_pars['n_do_orders'] = len(global_pars['do_orders'])
    
    # The template directory within the instrument directory
    global_pars['default_templates_path'] = global_pars['pipeline_path'] + 'default_templates' + os.sep

    # The number of model pixels = resolution x number of data pixels
    global_pars['n_model_pix'] = int(global_pars['model_resolution'] * global_pars['n_data_pix'])

    # The full tag is the name of the star + tag
    global_pars['full_tag'] = global_pars['star_name']+ '_' + global_pars['tag']

    # The number of data pixels used (ignoring the cropped pixels at the ends). Possibly not used
    global_pars['n_use_data_pix'] = int(global_pars['n_data_pix'] - global_pars['crop_pix'][0] - global_pars['crop_pix'][1])
    
    # The left and right pixels. This should roughly match the bad pix arrays
    global_pars['pix_left'], global_pars['pix_right'] = global_pars['crop_pix'][0], global_pars['n_data_pix'] - global_pars['crop_pix'][1]

    # Make output directories
    global_pars['run_output_path'] = global_pars['output_path'] + global_pars['full_tag'] + os.sep
    isdir = os.path.isdir(global_pars['run_output_path'])
    overwrite = global_pars['overwrite_output']
    if isdir and not overwrite:
        sys.exit('ERROR: This output folder already exists! Come up with a new tag name or choose to overwrite it.')
    elif not isdir and overwrite:
        os.makedirs(global_pars['run_output_path'])
        for i in range(global_pars['n_orders']):
            if i in global_pars['do_orders']:
                order = str(i+1)
                os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'RVs')
                os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'Fits')
                os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'Opt')
                os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'Stellar_Templates')
    elif isdir and overwrite:
        for i in range(global_pars['n_orders']):
            if i in global_pars['do_orders']:
                order = str(i+1)
                issubdir = os.path.isdir(global_pars['run_output_path'] + 'Order' + order)
                if not issubdir:
                    os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'RVs')
                    os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'Fits')
                    os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'Opt')
                    os.makedirs(global_pars['run_output_path'] + 'Order' + order + os.sep + 'Stellar_Templates')

    # Update plotting parameters
    global_pars['spec_img_w'] = int(global_pars['spec_img_width_pix'] / global_pars['dpi']) # plot img width for spectral models
    global_pars['spec_img_h'] = int(global_pars['spec_img_height_pix'] / global_pars['dpi']) # plot img height for spectral models
    global_pars['rv_img_w'] = int(global_pars['rv_img_width_pix'] / global_pars['dpi']) # plot img width for rvs
    global_pars['rv_img_h'] = int(global_pars['rv_img_height_pix'] / global_pars['dpi']) # plot img width for rvs
    
    # Units for plotting wavelength
    plot_wave_units = {
        'microns': 1E-4,
        'nm' : 1E-1,
        'ang' : 1 }    
    global_pars['plot_wave_factor'] = plot_wave_units[global_pars['plot_wave_unit']]

    # For now load in bary corr file before this code is run.
    # This is a workaround because astropy does not play well with the ARGO cluster
    global_pars['bary_corrs'] = None
    if global_pars['bary_corr_file'] is not None:
        global_pars['BJDS'], global_pars['bary_corrs'] = np.loadtxt(global_pars['data_input_path'] + global_pars['bary_corr_file'], delimiter=',', unpack=True, comments='#')

    # Grab some relevant details of the observations that will be the same for all orders
    global_pars['data_filenames'] = np.atleast_1d(np.genfromtxt(global_pars['data_input_path'] + global_pars['filelist'], dtype='<U100', comments='#'))
    global_pars['n_spec'] = len(global_pars['data_filenames'])
    data_all_first_order = []
    data_class_init = getattr(pcdata, 'SpecData' + global_pars['instrument'])
    for ispec in range(global_pars['n_spec']):
        print('Loading In Observation ' + str(ispec+1) + ' to determine order independent obs details ...', flush=True)
        data_all_first_order.append(data_class_init(global_pars['data_filenames'][ispec], global_pars['do_orders'][0], ispec, global_pars))
        
    # Now set the global values if they were not set
    if global_pars['bary_corr_file'] is None:
        global_pars['BJDS'] = np.array([getattr(data_all_first_order[ispec], 'BJD') for ispec in range(global_pars['n_spec'])]).astype(np.float64)
        global_pars['bary_corrs'] = np.array([getattr(data_all_first_order[ispec], 'bary_corr') for ispec in range(global_pars['n_spec'])]).astype(np.float64)
        
    np.savetxt('bary_corrs_gj740.txt', np.array([global_pars['BJDS'], global_pars['bary_corrs']]).T, delimiter=',')
    
    # Sort by BJD
    # Resort the filenames as well so we only have to do this once.
    global_pars['sorting_inds'] = np.argsort(global_pars['BJDS'])
    global_pars['BJDS'] = global_pars['BJDS'][global_pars['sorting_inds']]
    global_pars['bary_corrs'] = global_pars['bary_corrs'][global_pars['sorting_inds']]
    global_pars['data_filenames'] = global_pars['data_filenames'][global_pars['sorting_inds']]
    data_all_first_order = [data_all_first_order[global_pars['sorting_inds'][ispec]] for ispec in range(global_pars['n_spec'])]

    # Calculate some more parameters from the observations
    global_pars['BJDS_nightly'], global_pars['n_obs_nights'] = pcutils.get_nightly_jds(global_pars['BJDS'], global_pars)
    global_pars['n_nights'] = len(global_pars['n_obs_nights'])

    # Save the global parameters dictionary to the output directory
    np.savez(global_pars['run_output_path'] + 'global_parameters_dictionary.npz', global_pars)
    
    # Matplotlib backend
    if global_pars['n_cores'] > 1 or platform != 'darwin':
        matplotlib.use("AGG")
    else:
        matplotlib.use("MacOSX")
        
    # Print summary
    print('TARGET: ' + global_pars['star_name'], flush=True)
    print('INSTRUMENT: ' + global_pars['observatory'] + ' / ' + global_pars['instrument'], flush=True)
    print('TOTAL SPECTRA: ' + str(global_pars['n_spec']), flush=True)
    print('TOTAL NIGHTS: ' + str(global_pars['n_nights']), flush=True)
    print('MODEL RESOLUTION: ' + str(global_pars['model_resolution']) + ' x THE DATA', flush=True)
    print('TAG: ' + global_pars['tag'], flush=True)
    print('N TEMPLATE ITERATIONS: ' + str(global_pars['n_template_fits']), flush=True)
    print('N ECHELLE ORDERS TO FIT: ' + str(global_pars['n_do_orders']), flush=True)
    print('N CORES USED: ' + str(global_pars['n_cores']), flush=True)

    return global_pars, model_blueprints, data_all_first_order

################################################################################################################
