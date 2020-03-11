# Python built in modules
import copy
from collections import OrderedDict
import glob # File searching
import os # Making directories
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import importlib.util # importing other modules from files
import warnings # ignore warnings
import time # Time the code
import sys # sys utils
from sys import platform # plotting backend
import pdb # debugging
stop = pdb.set_trace

# Multiprocessing
from joblib import Parallel, delayed

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
        
        # templates_dict is a dictionary to store the large array synthetic templates (star, gas, tellurics)
        # models_dict is a dictionary to store the large array synthetic templates (star, gas, tellurics)
        # We only need a single instance of templates_dict for realistic memory usage
        # models_dict is per spectrum dependent and will belong to each forward model object as a class member
        # This allows for per spectrum customization if need be (not implemented) but templates_dict will still
        # store the majority of the synethetic data.
        stellar_templates = np.empty(shape=(gpars['n_model_pix'], gpars['n_template_fits'] + 1), dtype=np.float64)
        rvs = np.empty(shape=(gpars['n_spec'], gpars['n_template_fits']), dtype=np.float64)
        rvs_xcorr = np.empty(shape=(gpars['n_spec'], gpars['n_template_fits']), dtype=np.float64)
        rvs_cd = np.empty(shape=(gpars['n_nights'], gpars['n_template_fits']), dtype=np.float64)
        rvs_unc_cd = np.empty(shape=(gpars['n_nights'], gpars['n_template_fits']), dtype=np.float64)
        bispans_all = np.full(shape=(gpars['n_spec'], gpars['n_template_fits']), dtype=np.float64, fill_value=np.nan)
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
        forward_models = []
        forward_model_class_init = getattr(pcforwardmodels, gpars['instrument'] + 'ForwardModel')
        for ispec in range(gpars['n_spec']):
            forward_models.append(forward_model_class_init(ispec, order_num, models_dict, data_all[ispec], init_pars, gpars))

        # Get better estimations for some parameters (eg xcorr for star)
        optimize_guess_parameters(forward_models, model_blueprints, templates_dict, gpars)


        # Zeroth Iteration - No doppler shift if using a flat template.
        # We could also flag the stellar lines, but this has minimal impact on the RVs
        if not gpars['do_init_guess']:

            print('  Iteration: 0 (flat stellar template, no RVs)', flush=True)
            fit_spectra(forward_models, 0, templates_dict, gpars)

            if gpars['n_template_fits'] == 0:
                # Output and continue to next order
                for i in range(gpars['n_spec']):
                    forward_models[i].save_final_outputs(gpars)
                continue
            else:
                stellar_templates[:, 1] = update_stellar_template(templates_dict, forward_models, 0, order_num, gpars)
                update_model_params(forward_models, 0, gpars)
                templates_dict['star'] = np.array([stellar_templates[:, 0], stellar_templates[:, 1]]).T

        # Iterate over remaining stellar template generations
        for iter_num in range(gpars['n_template_fits']):

            print('Starting Iteration: ' + str(iter_num+1) + ' of ' + str(gpars['n_template_fits']), flush=True)
            stopwatch.lap(name='ti_iter')

            fit_spectra(forward_models, iter_num+gpars['ndi'], templates_dict, gpars)
            print('Finished Iteration ' + str(iter_num+1) + ' in ' + str(round(stopwatch.time_since(name='ti_iter'), 2)/3600) + ' hours', flush=True)

            # Generate RVs
            # Extract doppler shifts from the Parameters Objects and compute nightly RVs
            if gpars['do_xcorr']:
                
            rvs[:, iter_num], rvs_cd[:, iter_num], rvs_unc_cd[:, iter_num], rvs_xcorr[:, iter_num] = compute_rvs(forward_models, iter_num, gpars)
            # Save RVs, will continuously overwrite, but useful to look at on the fly
            np.savez(gpars['run_output_path'] + 'Order' + str(order_num+1) + os.sep + 'RVs' + os.sep + gpars['full_tag'] + '_rvs_ord' + str(order_num+1) + '.npz', rvs=rvs, rvs_cd=rvs_cd, rvs_xcorr=rvs_xcorr, BJDS_cd=gpars['BJDS_nightly'], BJDS=gpars['BJDS'], n_obs_nights=gpars['n_obs_nights'], bisector_spans=bispans_all, unc=rvs_unc_cd)

            # Print RV Diagnostics
            if gpars['n_nights'] > 1:
                rvscd_std = np.std(rvs_cd[:, iter_num])
                print('  RV Stddev (Nightly): ' + str(round(rvscd_std, 4)) + ' m/s', flush=True)

            # Compute the new stellar template, update parameters.
            if iter_num + 1 < gpars['n_template_fits']:

                # Update the forward model initial_parameters.
                update_model_params(forward_models, iter_num+gpars['ndi'], gpars)
                
                # Update the template through least squares cubic spline interpolation if there are more than 5 nights.
                # Otherwise interpolate each grid of residuals onto a common grid with cubic spline interpolation, then crunch using weighted median
                if gpars['n_nights'] >= 5:
                    cubic_spline_lsq_template(templates_dict, forward_models, iter_num+gpars['ndi'], order_num, gpars)
                else:
                    update_stellar_template(templates_dict, forward_models, iter_num+gpars['ndi'], order_num, gpars)

        # Save forward model outputs
        for i in range(gpars['n_spec']):
            print('Creating Final Outputs For Spec ' + str(i+1) + ' Of ' + str(gpars['n_spec']) + ' ...', flush=True)
            forward_models[i].save_final_outputs(gpars)

        # Save Stellar Template Outputs
        np.savez(gpars['run_output_path'] + 'Order' + str(order_num+1) + os.sep + 'Stellar_Templates' + os.sep + gpars['full_tag'] + '_stellar_templates_ord' + str(order_num+1) + '.npz', stellar_templates=stellar_templates)

    # End the clock!
    print('ALL DONE! Runtime: ' + str(round(stopwatch.lap(name='ti_main') / 3600, 2)) + ' hours', flush=True)

def cross_correlate_all(forward_models, templates_dict, iter_num, gpars, update_from=False):

    # Fit in Parallel
    ti = time.time()
    print('Cross Correlating Spectra ... ', flush=True)

    if gpars['n_threads'] > 1:
        #pool = mp.Pool(gpars['n_threads'])

        # Construct the arguments
        iter_pass = []
        for ispec in range(gpars['n_spec']):
            iter_pass.append((forward_models[ispec], templates_dict, iter_num, gpars, update_from))

        presults = Parallel(n_jobs=gpars['n_threads'], verbose=0, batch_size=1)(delayed(cc_wrapper)(*iter_pass[ispec]) for ispec in range(gpars['n_spec']))
            

        # Run
        #presults = pool.starmap(cc_wrapper, iter_pass)

        #pool.close()

        tf = time.time()

        # Fit in Parallel
        print('Cross Correlation Finished in ' + str(round((tf-ti)/60, 3)) + ' min ', flush=True)

        # Sort of redundant
        for ispec in range(gpars['n_spec']):
            forward_models[ispec] = presults[ispec]
    else:
        for ispec in range(gpars['n_spec']):
            forward_models[ispec] = cc_wrapper(forward_models[ispec], templates_dict, iter_num, gpars, update_from)

def cc_wrapper(forward_model, templates_dict, iter_num, gpars, update_from):
    forward_model.cross_correlate(templates_dict, iter_num, gpars, update_from)
    return forward_model


# Wrapper to fit all spectra
# Given full orders
def fit_spectra(forward_models, iter_num, templates_dict, gpars):

    if gpars['n_threads'] > 1:

        # Fit in Parallel
        ti = time.time()
        #pool = mp.Pool(gpars['n_threads'])
        #print('    Opened Parallel Pool of ' + str(gpars['n_threads']) + ' Workers For ' + str(gpars['n_spec']) + ' Spectra ', flush=True)

        # Construct the arguments
        iter_pass = []
        for spec_num in range(gpars['n_spec']):
            iter_pass.append((forward_models[spec_num], iter_num, templates_dict, gpars))

        # Run
        #presults = pool.starmap(solver_wrapper, iter_pass)

        #pool.close()
        
        presults = Parallel(n_jobs=gpars['n_threads'], verbose=0, batch_size=1)(delayed(solver_wrapper)(*iter_pass[ispec]) for ispec in range(gpars['n_spec']))

        tf = time.time()

        # Fit in Parallel
        print('    Parallel Fitting Finished in ' + str(round((tf-ti)/60, 3)) + ' min ', flush=True)

        # Sort of redundant, list to array
        for i in range(gpars['n_spec']):
            forward_models[i] = presults[i]

    else:
        # Fit one at a time
        for spec_num in range(gpars['n_spec']):
            print('    Performing Nelder-Mead Fit For Spectrum '  + str(spec_num+1) + ' of ' + str(gpars['n_spec']), flush=True)
            ti = time.time()
            forward_models[spec_num] = solver_wrapper(forward_models[spec_num], iter_num, templates_dict, gpars)
            tf = time.time()

def rms_model(gp, v, fwm, iter_num, templates_dict, gpars, weights=None):

    gp_ = pcmodelcomponents.Parameters.from_numpy(list(fwm.initial_parameters.keys()), values=gp, varies=v)

    # Generate the forward model
    wave_lr, model_lr = fwm.build_full(gp_, templates_dict, gpars)

    if weights is None:
        weights = np.copy(fwm.data.badpix)
    else:
        weights *= fwm.data.badpix

    # RMS
    diffs2 = (fwm.data.flux - model_lr)**2
    good = np.where(np.isfinite(diffs2) & (weights > 0))[0]
    residuals2 = diffs2[good]
    weights = weights[good]

    # Taper the ends
    left_taper = np.array([0.2, 0.4, 0.6, 0.8])
    right_taper = np.array([0.8, 0.6, 0.4, 0.2])

    residuals2[:4] *= left_taper
    residuals2[-4:] *= right_taper

    # Ignore worst 20 pixels
    ss = np.argsort(residuals2)
    weights[ss[-20:]] = 0
    
    # Compute rms ignoring bad pixels
    #n_use_pix = np.nansum(fwm.data.badpix) - 20
    #rms = (np.nansum(residuals2) / n_use_pix)**0.5
    wrms = (np.nansum(residuals2 * weights) / np.nansum(weights))**0.5
    cons = np.nanmin(fwm.models_dict['lsf'].build(gp_)) # Ensure LSF is greater than zero

    # Return rms and constraint
    return wrms, cons

# Returns the closest value and index in a numpy array
def find_closest_depth(x, y, val):
    diffs = np.abs(y - val)
    index = np.argmin(diffs)
    vel = x[index]
    return index, vel

# Computes the bisector span
# If the rvs are correlated with bisector span, we model this with linear regression (rvs=rvs(bs))
# Then we subtract off the best fit model from the RVs.
def compute_bisector_spans(forward_models, templates_dict, rvs, iter_num, gpars):
    
    stellar_template = templates_dict['star']
    
    # B(d) = (v_l(d) + v_r(d)) / 2
    # v_l = velocities located on the left side from the minimum of the CCF peak and v_r are the ones on the right side
    # Mean bisector is computed at two depth ranges:
    # d = (0.1, 0.4), (0.6, 0.85)
    # B_(0.1, 0.4) = E(B(d)) for 0.1 to 0.4
    # B_(0.6, 0.85) = E(B(d)) for 0.6 to 0.85
    # BS = B_(0.1, 0.4) - B_(0.6, 0.85) = E(B(d)) for 0.1 to 0.4 - E(B(d)) for 0.6 to 0.85
    # .. = Average(B(d)) for 0.1 to 0.4 = Average((v_l(d) + v_r(d)) / 2) for 0.1 to 0.4
    dr1 = (0.1, 0.4)
    dr2 = (0.6, 0.85)

    bs = np.empty(gpars['n_spec'], dtype=np.float64)

    for ispec in range(gpars['n_spec']):
        vels = forward_models[ispec].cross_correlation_vels[iter_num]
        ccf = forward_models[ispec].cross_correlations[iter_num]
        ccf = ccf - np.nanmin(ccf)
        ccf_norm = ccf / np.nanmax(ccf)
        best_index = np.nanargmin(ccf)
        best_vel = vels[best_index]
        vels = vels - best_vel
        depths = np.linspace(0, 1, num=100)
        bisectors = np.empty(depths.size, dtype=np.float64)
        use_left = np.where(vels < 0)[0]
        use_right = np.where(vels > 0)[0]
        vel_max_ind_left, vel_max_ind_right = use_left[np.nanargmax(ccf_norm[use_left])], use_right[np.nanargmax(ccf_norm[use_right])]
        use_left = np.where((vels > vels[vel_max_ind_left]) & (vels < 0))[0]
        use_right = np.where((vels > 0) & (vels < vels[vel_max_ind_right]))[0]
        for i in range(depths.size):
            d = depths[i]
            vl = find_closest_depth(vels[use_left], ccf_norm[use_left], d)[1]
            vr = find_closest_depth(vels[use_right], ccf_norm[use_right], d)[1]
            bisectors[i] = (vl + vr) / 2

        top = np.where((depths > dr1[0]) & (depths < dr1[1]))[0]
        bottom = np.where((depths > dr2[0]) & (depths < dr2[1]))[0]
        avg_top = np.average(bisectors[top])
        avg_bottom = np.average(bisectors[bottom])
        bs[ispec] = avg_top - avg_bottom
        forward_models[ispec].bisectors[iter_num] = bisectors
        forward_models[ispec].bisector_spans[iter_num] = bs[ispec] + best_vel

    return bs

def cubic_spline_lsq_template(templates_dict, forward_models, iter_num, order_num, gpars):
    
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
            
    if gpars['nights_for_template'] == 'all': # use all nights
        template_spec_indices = np.arange(gpars['n_spec']).astype(int)
    elif type(gpars['nights_for_template']) is list: # use specified nights
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
    if gpars['nights_for_template'] == 'all':
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
            
    

def update_stellar_template(templates_dict, forward_models, iter_num, order_num, gpars):
    
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
    
    if gpars['nights_for_template'] == 'all': # use all nights
        template_spec_indices = np.arange(gpars['n_spec']).astype(int)
    elif type(gpars['nights_for_template']) is list: # use specified nights
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
    if gpars['nights_for_template'] == 'all':
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
    
    

# Later on the individual RVs are typically combined in a user specific way
# The co-added (nightly) RVs are mainly for output plots of individual orders
def compute_rvs(forward_models, iter_num, gpars):

    # These are estimates
    rvs_cd = np.full(gpars['n_nights'], fill_value=np.nan)
    unc_cd = np.full(gpars['n_nights'], fill_value=np.nan)

    star_vels = pcforwardmodels.ForwardModel.extract_parameter_values(forward_models, forward_models[0].models_dict['star'].par_names[0], iter_num+gpars['ndi'], gpars)
    star_vels = np.array([forward_models[ispec].best_fit_pars[iter_num][forward_models.models_dict['star'].par_names[0]] for ispec in range(gpars['n_spec'])])
    rms = np.array([forward_models[ispec].opt[iter_num][0] for ispec in range(gpars['n_spec'])])

    # Per epoch RVs
    rvs = star_vels + gpars['bary_corrs']

    # Co-add to get nightly RVs
    # If only one spectrum and no initial guess, no rvs!
    if gpars['n_spec'] == 1 and not gpars['do_init_guess']:
        rvs[0] = np.nan
        rvs_cd[0] = np.nan
        unc_cd[0] = np.nan
    else:
        spec_start = 0
        for inight in range(gpars['n_nights']):
            spec_end = spec_start + gpars['n_obs_nights'][inight]
            rvs_single_night = rvs[spec_start:spec_end]
            w = 1 / rms[spec_start:spec_end]**2
            w = w / np.nansum(w)
            if gpars['n_obs_nights'][inight] > 1:
                rvs_cd[inight] = pcmath.weighted_mean(rvs_single_night, w)
                unc_cd[inight] = pcmath.weighted_stddev(rvs_single_night, w) / np.sqrt(gpars['n_obs_nights'][inight])
            else:
                rvs_cd[inight] = rvs_single_night[0]
                unc_cd[inight] = 0
            spec_start = spec_end

    # Cross correlation RVs
    rvs_xcorr = np.empty(gpars['n_spec'], dtype=np.float64)
    for ispec in range(gpars['n_spec']):
        best_ind = np.argmin(forward_models[ispec].cross_correlations[iter_num])
        rvs_xcorr[ispec] = forward_models[ispec].cross_correlation_vels[iter_num][best_ind]

    # Plot RVs and cross correlation RVs
    plt.figure(num=1, figsize=(gpars['rv_img_w'], gpars['rv_img_h']), dpi=gpars['dpi'])
    plt.plot(gpars['BJDS'] - gpars['BJDS_nightly'][0], rvs-np.median(rvs_cd), marker='.', linewidth=0, color=gpars['colors']['green'], alpha=0.7)
    plt.plot(gpars['BJDS'] - gpars['BJDS_nightly'][0], rvs_xcorr-np.median(rvs_xcorr), marker='.', linewidth=0, color='black')
    plt.errorbar(gpars['BJDS_nightly'] - gpars['BJDS_nightly'][0], rvs_cd-np.median(rvs_cd), yerr=unc_cd, marker='o', linewidth=0, elinewidth=1, color=gpars['colors']['light_blue'])
    plt.title(gpars['star_name'] + ' RVs Order ' + str(forward_models[0].order_num+1) + ' Iteration ' + str(iter_num+1))
    plt.xlabel('BJD - BJD0')
    plt.ylabel('RV [m/s]')
    plt.tight_layout()
    fname = gpars['run_output_path'] + 'Order' + str(forward_models[0].order_num+1) + os.sep + 'RVs' + os.sep + gpars['full_tag'] + '_rvs_ord' + str(forward_models[0].order_num+1) + '_iter' + str(iter_num+1) + '.png'
    plt.savefig(fname)
    plt.close()

    return rvs, rvs_cd, unc_cd, rvs_xcorr

def optimize_guess_parameters(forward_models, model_blueprints, templates_dict, gpars):

    # Handle the star in parallel.
    if gpars['do_init_guess']:
        cross_correlate_all(forward_models, templates_dict, None, gpars, update_from=True)
        for ispec in range(gpars['n_spec']):
            forward_models[ispec].modify(model_components={'star': True})
    else: # disable
        for ispec in range(gpars['n_spec']):
            forward_models[ispec].modify(model_components={'star': False})

    for ispec in range(gpars['n_spec']):

        # Figure out any parameters with locked parameters by checking any pars with min=max
        for par_name in forward_models[ispec].initial_parameters.keys():
            par = forward_models[ispec].initial_parameters[par_name]
            if par.minv == par.maxv:
                forward_models[ispec].initial_parameters[par_name].setv(vary=False)
                
        # Tellurics
        if 'tellurics' in forward_models[ispec].models_dict.keys():
            for jtell, tell in enumerate(forward_models[ispec].models_dict['tellurics'].species):
                max_range = np.nanmax(templates_dict['tellurics'][tell][:, 1]) - np.nanmin(templates_dict['tellurics'][tell][:, 1])
                if max_range < 0.02:
                    forward_models[ispec].modify(telluric_components={tell: False})
                
        # Delay any models with the delay keyword
        for model in forward_models[ispec].models_dict.keys():
            if hasattr(forward_models[ispec].models_dict[model], 'n_delay'):
                if forward_models[ispec].models_dict[model].n_delay > 0:
                    forward_models[ispec].modify(model_components={model: False})
        

# Solves and plots the forward model results. Needed for parallel processing.
def solver_wrapper(forward_model, iter_num, templates_dict, gpars):

    ti = time.time()
    
    args_to_pass = (forward_model, iter_num, templates_dict, gpars, None)

    names, gp, vlb, vub, vp, mcmc_scales = forward_model.initial_parameters.to_numpy(kind='all')
    vp = np.where(vp)[0].astype(int)

    # The call to the nelder mead solver
    result = pcsolver.simps(gp, rms_model, vlb, vub, vp, no_improv_break=3, args_to_pass=args_to_pass)

    forward_model.best_fit_pars[iter_num] = pcmodelcomponents.Parameters.from_numpy(names=names, values=result[0], minvs=vlb, maxvs=vub, varies=pcmath.mask_to_binary(vp, len(vlb)), mcmcscales=mcmc_scales)

    # Build the best fit forward model
    wave_grid_data, best_model = forward_model.build_full(forward_model.best_fit_pars[iter_num], templates_dict, gpars)
    forward_model.wavelength_solutions[:, iter_num] = wave_grid_data
    forward_model.models[:, iter_num] = best_model

    # Compute the residuals between the data and model, don't flag bad pixels here. Cropped pix are still nan
    forward_model.residuals[:, iter_num] = forward_model.data.flux - best_model

    if gpars['verbose']:
        forward_model.pretty_print(iter_num)
        print('RMS = %' + str(round(100*result[1], 5)))
        print('Function Calls = ' + str(result[2]))

    forward_model.opt[iter_num, :] = result[1:]
    
    if gpars['do_xcorr']:
        forward_model.cross_correlate(templates_dict, iter_num, gpars)
        compute_bisector_spans(forward_models, templates_dict, rvs, iter_num, gpars)

    tf = time.time()
    dt = str(round((tf - ti) / 60, 3))
    line = '    Fit Spectrum ' + str(forward_model.spec_num+1) + ' of ' + str(gpars['n_spec']) + ' in ' + dt + ' min'
    print(line, flush=True)

    # Output a plot
    forward_model.plot(iter_num, templates_dict, gpars)

    # Return new forward model object
    return forward_model

# Initialize the pipeline based on input_options file
def init_pipeline(user_input_options, user_model_blueprints):

    # Numpy warnings
    np.seterr(divide='ignore')
    warnings.filterwarnings('ignore')

    # Use a dictionary to store global pars.
    global_pars = {}

    # First the pipeline path to the dict
    global_pars['pipeline_path'] = os.path.dirname(os.path.realpath(__file__)) + os.sep
    
    # Load the config file and add to dict. Don't need to be worried about overwriting yet. 
    spec = importlib.util.spec_from_file_location('config', global_pars['pipeline_path'] + 'config.py')
    module_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_)
    config_parameters = module_.default_config
    for config_key in config_parameters:
        global_pars[config_key] = config_parameters[config_key]
    
    # Load in the default instrument settings and add to dict. Now want to overwrite
    spec = importlib.util.spec_from_file_location('parameters_' + user_input_options['instrument'].lower(), global_pars['pipeline_path'] + 'spectrographs' + os.sep + 'parameters_' + user_input_options['instrument'].lower() + '.py')
    module_ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_)
    instrument_parameters = module_.default_instrument_parameters
    model_blueprints = module_.default_model_blueprints
    for instrument_key in instrument_parameters:
        sub_keys = pcutils.find_all_items(global_pars, instrument_key)
        if len(sub_keys) == 0:
            global_pars[instrument_key] = instrument_parameters[instrument_key]
        else:
            pcutils.setInDict(global_pars, sub_keys[0][0], instrument_parameters[instrument_key])
    
    # Now do the same for the user input dict
    for user_key in user_input_options:
        sub_keys = pcutils.find_all_items(global_pars, user_key)
        if len(sub_keys) == 0:
            global_pars[user_key] = user_input_options[user_key]
        else:
            pcutils.setInDict(global_pars, sub_keys[0][0], user_input_options[user_key])
            
    # Now do something similar for the model blueprints dictionary
    for user_key in user_model_blueprints:
        # If the key is not in the dictionary, just add the model
        if user_key not in model_blueprints:
            model_blueprints[user_key] = user_model_blueprints[user_key]
        else:
            # Otherwise, we need to update the according subkey
            # For now, only a first lvl sub key is supported
            for user_sub_key in user_model_blueprints[user_key]:
                model_blueprints[user_key][user_sub_key] = user_model_blueprints[user_key][user_sub_key]
            
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

    # For now load in bary corr file before this code is run.
    # This is a workaround because astropy does not play well with the ARGO cluster
    global_pars['bary_corrs'] = None
    if global_pars['bary_corr_file'] is not None:
        global_pars['BJDS'], global_pars['bary_corrs'] = np.loadtxt(global_pars['data_input_path'] + global_pars['bary_corr_file'], delimiter=',', unpack=True)

    # Grab some relevant details of the observations that will be the same for all orders
    global_pars['data_filenames'] = np.atleast_1d(np.genfromtxt(global_pars['data_input_path'] + global_pars['filelist'], dtype='<U100'))
    global_pars['n_spec'] = len(global_pars['data_filenames'])
    data_all_first_order = []
    data_class_init = getattr(pcdata, 'SpecData' + global_pars['instrument'])
    for ispec in range(global_pars['n_spec']):
        print('Loading In Observation ' + str(ispec+1) + ' to determine order independent obs details ...', flush=True)
        data_all_first_order.append(data_class_init(global_pars['data_filenames'][ispec], global_pars['do_orders'][0], ispec, global_pars))
        
    # Sort by BJD, reset the BJDS and bary_corrs from the sorted data objects.
    # We also sort the data_filenames objects so that when read in later, they are in the correct order.
    global_pars['sorting_inds'] = pcutils.sort_data(data_all_first_order, global_pars)
    data_all_first_order = [data_all_first_order[global_pars['sorting_inds'][ispec]] for ispec in range(global_pars['n_spec'])]
    [setattr(data_all_first_order[ispec], 'spec_num', ispec) for ispec in range(global_pars['n_spec'])]
    
    # get the BJDS and barycorrs from the now sorted data.
    global_pars['BJDS'] = np.array([getattr(data_all_first_order[ispec], 'BJD') for ispec in range(global_pars['n_spec'])]).astype(np.float64)
    global_pars['bary_corrs'] = np.array([getattr(data_all_first_order[ispec], 'bary_corr') for ispec in range(global_pars['n_spec'])]).astype(np.float64)
    global_pars['data_filenames'] = global_pars['data_filenames'][global_pars['sorting_inds']]

    # Calculate some more parameters from the observations
    global_pars['BJDS_nightly'], global_pars['n_obs_nights'] = pcutils.get_nightly_jds(global_pars['BJDS'], global_pars)
    global_pars['n_nights'] = len(global_pars['n_obs_nights'])

    # Save the global parameters dictionary to the output directory
    np.savez(global_pars['run_output_path'] + 'global_parameters_dictionary.npz', global_pars)
    
    # Matplotlib backend
    if global_pars['n_threads'] > 1 or platform != 'darwin':
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
    print('N CORES USED: ' + str(global_pars['n_threads']), flush=True)

    return global_pars, model_blueprints, data_all_first_order

# Updates all spectra
def update_model_params(forward_models, iter_num, gpars):

    for ispec in range(gpars['n_spec']):

        # Pass the previous iterations best pars as starting points
        forward_models[ispec].initial_parameters = copy.deepcopy(forward_models[ispec].best_fit_pars[iter_num])

        # Stellar Template, after zeroth iteration
        if iter_num == 0 and not gpars['do_init_guess']:
            forward_models[ispec].modify(model_components={'star': True})
            forward_models[ispec].initial_parameters[forward_models[ispec].models_dict['star'].par_names[0]].setv(value=-1*forward_models[ispec].databary_corr)

        # Enable any models that were delayed
        for model in forward_models[ispec].models_dict.keys():
            if hasattr(forward_models[ispec].models_dict[model], 'n_delay'):
                if iter_num >= forward_models[ispec].models_dict[model].n_delay and not forward_models[ispec].models_dict[model].enabled:
                    forward_models[ispec].modify(model_components={model: True})

################################################################################################################
