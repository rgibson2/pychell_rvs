# Python built in modules
import copy
from collections import OrderedDict
import glob # File searching
import os # Making directories
import importlib.util # importing other modules from files
import warnings # ignore warnings
import time # Time the code
import multiprocessing as mp # parallelization on a single node
import sys # sys utils
from abc import ABC, abstractmethod # Abstract classes
from sys import platform # plotting backend
import pdb # debugging
stop = pdb.set_trace

# Graphics
import matplotlib # to set the backend
import matplotlib.pyplot as plt # Plotting
from matplotlib import cm

# Multiprocessing
from joblib import Parallel, delayed

# Science/math
import scipy
from scipy import constants as cs # cs.c = speed of light in m/s
from scipy.special import comb
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, prange

# User defined
import pychell_rvs.pychell_math as pcmath
import pychell_rvs.pychell_solver as pcsolver # nelder mead solver
import pychell_rvs.pychell_model_components as pcmodelcomponents
import pychell_rvs.pychell_target_functions as pctargetfuns
import pychell_rvs.pychell_utils as pcutils


# A useful wrapper to store all the forward model objects.
# This class is useful for defining things across forward models, like RVs.
class ForwardModels(list):
    
    def __init__(self, gpars, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Storage arrays for the RVs, saved each order
        self.rvs = np.empty(shape=(gpars['n_spec'], gpars['n_template_fits']), dtype=np.float64)
        self.rvs_nightly = np.empty(shape=(gpars['n_nights'], gpars['n_template_fits']), dtype=np.float64)
        self.rvs_unc_nightly = np.empty(shape=(gpars['n_nights'], gpars['n_template_fits']), dtype=np.float64)
        
        # Initialize xcorr analysis arrays
        if gpars['do_xcorr']:
            self.rvs_xcorr = np.empty(shape=(gpars['n_spec'], gpars['n_template_fits']), dtype=np.float64)
            self.bisector_spans = np.empty(shape=(gpars['n_spec'], gpars['n_template_fits']), dtype=np.float64)
        
    # Later on the individual RVs are typically combined in a user specific way
    # The co-added (nightly) RVs are mainly for output plots of individual orders for quick looks
    # This only combines single order RVs.
    def compute_nightly_rvs(self, iter_num, gpars):

        # The nightly RVs and error bars
        rvs_nightly = np.full(gpars['n_nights'], fill_value=np.nan)
        unc_nightly = np.full(gpars['n_nights'], fill_value=np.nan)

        # The best fit stellar RVs, remove the barycenter bias
        rvs = np.array([self[ispec].best_fit_pars[iter_num][self[ispec].models_dict['star'].par_names[0]].value + self[ispec].data.bary_corr for ispec in range(gpars['n_spec'])])
    
        # The RMS from the forward model fit
        rms = np.array([self[ispec].opt[iter_num][0] for ispec in range(gpars['n_spec'])])

        # Co-add to get nightly RVs
        # If only one spectrum and no initial guess, no rvs!
        if gpars['n_spec'] == 1 and not gpars['do_init_guess']:
            rvs[0] = np.nan
            rvs_nightly[0] = np.nan
            unc_nightly[0] = np.nan
        else:
            f = 0
            l = gpars['n_obs_nights'][0]
            for inight in range(gpars['n_nights']):
                rvs_single_night = rvs[f:l]
                w = 1 / rms[f:l]**2
                w = w / np.nansum(w)
                if gpars['n_obs_nights'][inight] > 1:
                    rvs_nightly[inight] = pcmath.weighted_mean(rvs_single_night, w)
                    unc_nightly[inight] = pcmath.weighted_stddev(rvs_single_night, w) / np.sqrt(gpars['n_obs_nights'][inight])
                else:
                    rvs_nightly[inight] = rvs_single_night[0]
                    unc_nightly[inight] = 0
                if inight < gpars['n_nights'] - 1:
                    f += gpars['n_obs_nights'][inight]
                    l += gpars['n_obs_nights'][inight+1]
                    
        # Pass to storage arrays
        self.rvs[:, iter_num] = rvs
        self.rvs_nightly[:, iter_num] = rvs_nightly
        self.rvs_unc_nightly[:, iter_num] = unc_nightly
        
    # Plots the RVs for this iteration
    def plot_rvs(self, iter_num, gpars):
        
        plt.figure(num=1, figsize=(gpars['rv_img_w'], gpars['rv_img_h']), dpi=gpars['dpi'])
        plt.plot(gpars['BJDS'] - gpars['BJDS_nightly'][0], self.rvs[:, iter_num]-np.nanmedian(self.rvs_nightly[:, iter_num]), marker='.', linewidth=0, color=gpars['colors']['green'], alpha=0.7)
    
        if gpars['do_xcorr']:
            plt.plot(gpars['BJDS'] - gpars['BJDS_nightly'][0], self.rvs_xcorr[:, iter_num]-np.nanmedian(self.rvs_xcorr[:, iter_num]), marker='.', linewidth=0, color='black')
        
        plt.errorbar(gpars['BJDS_nightly'] - gpars['BJDS_nightly'][0], self.rvs_nightly[:, iter_num]-np.nanmedian(self.rvs_nightly[:, iter_num]), yerr=self.rvs_unc_nightly[:, iter_num], marker='o', linewidth=0, elinewidth=1, color=gpars['colors']['light_blue'])
        plt.title(gpars['star_name'] + ' RVs Order ' + str(self[0].order_num+1) + ' Iteration ' + str(iter_num+1))
        plt.xlabel('BJD - BJD0')
        plt.ylabel('RV [m/s]')
        plt.tight_layout()
        fname = gpars['run_output_path'] + 'Order' + str(self[0].order_num+1) + os.sep + 'RVs' + os.sep + gpars['full_tag'] + '_rvs_ord' + str(self[0].order_num+1) + '_iter' + str(iter_num+1) + '.png'
        plt.savefig(fname)
        plt.close()
    
    # Saves the RVs. This is typically called after each iteration for diagnostics.
    def save_rvs(self, gpars):
        fname = gpars['run_output_path'] + 'Order' + str(self[0].order_num+1) + os.sep + 'RVs' + os.sep + gpars['full_tag'] + '_rvs_ord' + str(self[0].order_num+1) + '.npz'
        if gpars['do_xcorr']:
            bs = np.array([self[ispec].bisector_spans for ispec in range(gpars['n_spec'])]).astype(np.float64).T
            np.savez(fname, rvs=self.rvs, rvs_nightly=self.rvs_nightly, BJDS_nightly=gpars['BJDS_nightly'], BJDS=gpars['BJDS'], n_obs_nights=gpars['n_obs_nights'], rvs_unc_nightly=self.rvs_unc_nightly, rvsxcorr=self.rvs_xcorr, bisector_spans=bs)
        else:
            np.savez(fname, rvs=self.rvs, rvs_nightly=self.rvs_nightly, BJDS_nightly=gpars['BJDS_nightly'], BJDS=gpars['BJDS'], n_obs_nights=gpars['n_obs_nights'], rvs_unc_nightly=self.rvs_unc_nightly)


    # Updates all spectra according to best fit parameters and delay keywords
    def update_model_params(self, iter_num, gpars):

        for ispec in range(gpars['n_spec']):

            # Pass the previous iterations best pars as starting points
            self[ispec].initial_parameters = copy.deepcopy(self[ispec].best_fit_pars[iter_num])

            # Stellar Template, after zeroth iteration
            if iter_num == 0 and not gpars['do_init_guess']:
                self[ispec].modify(model_components={'star': True})
                self[ispec].initial_parameters[self[ispec].models_dict['star'].par_names[0]].setv(value=-1*self[ispec].data.bary_corr)

            # Enable any models that were delayed
            for model in self[ispec].models_dict.keys():
                if hasattr(self[ispec].models_dict[model], 'n_delay'):
                    if iter_num >= self[ispec].models_dict[model].n_delay and not self[ispec].models_dict[model].enabled:
                        self[ispec].modify(model_components={model: True})

    # Wrapper to fit all spectra
    def fit_spectra(self, iter_num, templates_dict, gpars):

        # Fit in Parallel
        stopwatch = pcutils.StopWatch()

        if gpars['n_cores'] > 1:

            # Construct the arguments
            iter_pass = []
            for spec_num in range(gpars['n_spec']):
                iter_pass.append((self[spec_num], iter_num, templates_dict, gpars))
            
            presults = Parallel(n_jobs=gpars['n_cores'], verbose=0, batch_size=1)(delayed(ForwardModels.solver_wrapper)(*iter_pass[ispec]) for ispec in range(gpars['n_spec']))

            # Sort of redundant
            for ispec in range(gpars['n_spec']):
                self[ispec] = presults[ispec]

        else:
            # Fit one at a time
            for ispec in range(gpars['n_spec']):
                print('    Performing Nelder-Mead Fit For Spectrum '  + str(ispec+1) + ' of ' + str(gpars['n_spec']), flush=True)
                self[ispec] = ForwardModels.solver_wrapper(self[ispec], iter_num, templates_dict, gpars)
            
        # Fit in Parallel
        print('Fitting Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)


    def optimize_guess_parameters(self, model_blueprints, templates_dict, gpars):

        # Handle the star in parallel.
        if gpars['do_init_guess']:
            self.cross_correlate_spectra(templates_dict, None, gpars)
            for ispec in range(gpars['n_spec']):
                self[ispec].modify(model_components={'star': True})
        else: # disable
            for ispec in range(gpars['n_spec']):
                self[ispec].modify(model_components={'star': False})

        for ispec in range(gpars['n_spec']):

            # Figure out any parameters with locked parameters by checking any pars with min=max
            for par_name in self[ispec].initial_parameters.keys():
                par = self[ispec].initial_parameters[par_name]
                if par.minv == par.maxv:
                    self[ispec].initial_parameters[par_name].setv(vary=False)
                    
            # Tellurics
            if 'tellurics' in self[ispec].models_dict.keys():
                for jtell, tell in enumerate(self[ispec].models_dict['tellurics'].species):
                    max_range = np.nanmax(templates_dict['tellurics'][tell][:, 1]) - np.nanmin(templates_dict['tellurics'][tell][:, 1])
                    if max_range < 0.02:
                        self[ispec].modify(telluric_components={tell: False})
                    
            # Delay any models with the delay keyword
            for model in self[ispec].models_dict.keys():
                if hasattr(self[ispec].models_dict[model], 'n_delay'):
                    if self[ispec].models_dict[model].n_delay > 0:
                        self[ispec].modify(model_components={model: False})

    
    def cross_correlate_spectra(self, templates_dict, iter_num, gpars):

        # Fit in Parallel
        stopwatch = pcutils.StopWatch()
        print('Cross Correlating Spectra ... ', flush=True)

        if gpars['n_cores'] > 1:

            # Construct the arguments
            iter_pass = []
            for ispec in range(gpars['n_spec']):
                iter_pass.append((self[ispec], templates_dict, iter_num, gpars))

            # Cross Correlate in Parallel
            presults = Parallel(n_jobs=gpars['n_cores'], verbose=0, batch_size=1)(delayed(ForwardModels.cc_wrapper)(*iter_pass[ispec]) for ispec in range(gpars['n_spec']))

            # Sort of redundant
            for ispec in range(gpars['n_spec']):
                self[ispec] = presults[ispec]
        else:
            for ispec in range(gpars['n_spec']):
                self[ispec] = ForwardModels.cc_wrapper(self[ispec], templates_dict, iter_num, gpars)
                
        print('Cross Correlation Finished in ' + str(round((stopwatch.time_since())/60, 3)) + ' min ', flush=True)

        
    @staticmethod
    def cc_wrapper(forward_model, templates_dict, iter_num, gpars):
        stopwatch = pcutils.StopWatch()
        forward_model.cross_correlate(templates_dict, iter_num, gpars)
        print('    Cross Correlated Spectrum ' + str(forward_model.spec_num+1) + ' of ' + str(gpars['n_spec']) + ' in ' + str(round((stopwatch.time_since()), 2)) + ' sec', flush=True)
        return forward_model


    # Wrapper for parallel processing. Solves and plots the forward model results. Also does xcorr if set.
    @staticmethod
    def solver_wrapper(forward_model, iter_num, templates_dict, gpars):

        stopwatch = pcutils.StopWatch()

        # Convert parameters to numpy. The optimization function must convert back to Parameter objects
        names, gp, vlb, vub, vp, mcmc_scales = forward_model.initial_parameters.to_numpy(kind='all')
        vp = np.where(vp)[0].astype(int)
        
        # Construct the extra arguments to pass to the target function
        args_to_pass = (forward_model, iter_num, templates_dict, gpars)
        
        # Get the target function
        target_fun = getattr(pctargetfuns, gpars['target_function'])

        # The call to the nelder mead solver
        opt_result = pcsolver.simps(gp, target_fun, vlb, vub, vp, no_improv_break=3, args_to_pass=args_to_pass)

        forward_model.best_fit_pars[iter_num] = pcmodelcomponents.Parameters.from_numpy(names=names, values=opt_result[0], minvs=vlb, maxvs=vub, varies=pcmath.mask_to_binary(vp, len(vlb)), mcmcscales=mcmc_scales)
        forward_model.opt[iter_num, :] = opt_result[1:]

        # Build the best fit forward model
        wave_grid_data, best_model = forward_model.build_full(forward_model.best_fit_pars[iter_num], templates_dict, gpars)
        forward_model.wavelength_solutions[:, iter_num] = wave_grid_data
        forward_model.models[:, iter_num] = best_model

        # Compute the residuals between the data and model, don't flag bad pixels here. Cropped may still be nan.
        forward_model.residuals[:, iter_num] = forward_model.data.flux - best_model

        # Print diagnostics if set
        if gpars['verbose_print']:
            print('RMS = %' + str(round(100*opt_result[1], 5)))
            print('Function Calls = ' + str(opt_result[2]))
            forward_model.pretty_print(iter_num)

        # Do a cross correlation analysis if set
        if gpars['do_xcorr']:
            print('    Cross Correlating Spectrum ' + str(forward_model.spec_num+1) + ' of ' + str(gpars['n_spec']), flush=True)
            forward_model.cross_correlate(templates_dict, iter_num, gpars)
            forward_model.bisector_spans[iter_num] = pcmath.compute_bisector_span(forward_model.cross_correlation_vels[iter_num], forward_model.cross_correlations[iter_num], gpars)
        
        print('    Fit Spectrum ' + str(forward_model.spec_num+1) + ' of ' + str(gpars['n_spec']) + ' in ' + str(round((stopwatch.time_since())/60, 2)) + ' min', flush=True)

        # Output a plot
        forward_model.plot(iter_num, templates_dict, gpars)

        # Return new forward model object since we possibly fit in parallel
        return forward_model


    # Calls the save model outputs for each array and forces a final overwrite of the RVs just in case (negligible time)
    def save_final_outputs(self, gpars):
        for ispec in range(gpars['n_spec']):
            self[ispec].save_model_outputs(gpars)
        
        self.save_rvs(gpars)

class ForwardModel(ABC):
    
    def __init__(self, spec_num, order_num, models_dict, data, initial_parameters, gpars):
        
        # Store the order and spec nums
        self.spec_num = spec_num
        self.order_num = order_num
    
        # Required Data Variable
        self.data = data # object
        
        # Parameters Object, updated each iteration.
        self.initial_parameters = initial_parameters
        
        # Dictionary to store model classes which define build methods and some model parameters
        # No templates are stored here.
        # Neither this or templates_dict can be empty. Otherwise there are no models.
        self.models_dict = models_dict
        
        # Storage arrays after each iteration
        
        # Stores the final RMS [0] and target function calls [1]
        self.opt = np.empty(shape=(gpars['n_template_fits']+gpars['ndi'], 2), dtype=np.float64)
        
        # Stores the best fit parameters (Parameter objects)
        self.best_fit_pars = np.empty(shape=(gpars['n_template_fits']+gpars['ndi']), dtype=pcmodelcomponents.Parameters)
        
        # Stores the wavelenth solutions (may just be copies if known a priori)
        self.wavelength_solutions = np.empty(shape=(gpars['n_data_pix'], gpars['n_template_fits']+gpars['ndi']), dtype=np.float64)
        
        # Stores the residuals
        self.residuals = np.empty(shape=(gpars['n_data_pix'], gpars['n_template_fits']+gpars['ndi']), dtype=np.float64)
        
        # Stores the best fit forward models (built from best_fit_pars)
        self.models = np.empty(shape=(gpars['n_data_pix'], gpars['n_template_fits']+gpars['ndi']), dtype=np.float64)
        
        if gpars['do_xcorr']:
            
            # Stores the cross correlation vels which correspond to the above cross_correlation arrays
            self.cross_correlation_vels = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.ndarray)
        
            # Stores full cross correlation functions (really brute force RMS minimzation through velocity space)
            self.cross_correlations = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.ndarray)
        
            # Stores the bisector spans
            self.bisector_spans = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.float64)
        
    # Must define a build_full method which returns wave, model_flux on the detector grid
    # Can also define other build methods that return modified forward models
    @abstractmethod
    def build_full(self, pars, templates_dict, *args, **kwargs):
        pass
        
    # Save outputs after last iteration. This method can be implemented or not and super can be called or not.
    def save_model_outputs(self, gpars):
        
        # Helpful strings
        ord_dir = gpars['run_output_path'] + 'Order' + str(self.order_num+1) + os.sep
        ord_spec = '_ord' + str(self.order_num+1) + '_spec' + str(self.spec_num+1)
        
        # Best fit parameters and opt arraye
        filename_opt = ord_dir + 'Opt' + os.sep + gpars['full_tag'] + '_opt' + ord_spec + '.npz'
        np.savez(filename_opt, best_fit_pars=self.best_fit_pars, opt=self.opt)
        
        # Data flux, flux_unc, badpix, best fit forward models, and residuals
        filename_data_models = ord_dir + 'Fits' + os.sep + gpars['full_tag'] + '_data_model' + ord_spec + '.npz'
        data_arr = np.array([self.data.flux, self.data.flux_unc, self.data.badpix]).T
        np.savez(filename_data_models, wavelength_solutions=self.wavelength_solutions, residuals=self.residuals, models=self.models, data=data_arr, obs_details=self.data.obs_details)
                
    # Prints the models and corresponding parameters after each fit if verbose_print=True
    def pretty_print(self, iter_num):
        # Loop over models
        for mname in self.models_dict.keys():
            # Print the model string
            print(self.models_dict[mname])
            # Sub loop over per model parameters
            for pname in self.models_dict[mname].par_names:
                print('    ', end='')
                print(self.best_fit_pars[iter_num][pname])
    
    # Plots the forward model after each iteration with other template as well if verbose_plot = True
    def plot(self, iter_num, templates_dict, gpars, save=False):
        
        # Extract the low res wave grid in proper units
        wave = self.wavelength_solutions[:, iter_num] * gpars['plot_wave_factor']
        pad = 1 * gpars['plot_wave_factor']
        
        # The best fit forward model for this iteration
        model = self.models[:, iter_num]
        
        # The residuals for this iteration
        residuals = self.residuals[:, iter_num]
        
        # The filename for the plot
        fname = gpars['run_output_path'] + 'Order' + str(self.order_num+1) + os.sep + 'Fits' + os.sep + gpars['full_tag'] + '_data_model_spec' + str(self.spec_num+1) + '_ord' + str(self.order_num+1) + '_iter' + str(iter_num+gpars['di']) + '.png'

        # Define some helpful indices
        f, l = gpars['crop_pix'][0], gpars['n_data_pix'] - gpars['crop_pix'][1]
        good = np.where(self.data.badpix == 1)[0]
        bad = np.where(self.data.badpix == 0)[0]
        bad_data_locs = np.argsort(np.abs(residuals[good]))[-1*gpars['flag_n_worst_pixels']:]
        use_pix = np.arange(good[0], good[-1]).astype(int)
        
        # Figure
        fig, ax = plt.subplots(1, 1, figsize=(gpars['spec_img_w'], gpars['spec_img_h']), dpi=gpars['dpi'])
        # Data
        ax.plot(wave[use_pix], self.data.flux[use_pix], color=gpars['colors']['blue'], linewidth=gpars['lw'])
        # Model
        ax.plot(wave[use_pix], model[use_pix], color=gpars['colors']['red'], linewidth=gpars['lw'])
        # Zero line
        ax.plot(wave[use_pix], np.zeros(wave[use_pix].size), color=gpars['colors']['purple'], linewidth=gpars['lw'], linestyle=':')
        # Residuals (all bad pixels will be zero here)
        ax.plot(wave[good], residuals[good], color=gpars['colors']['orange'], linewidth=gpars['lw'])
        # The worst N pixels that were flagged
        ax.plot(wave[good][bad_data_locs], residuals[good][bad_data_locs], color='darkred', marker='X', linewidth=0)
        
        # Plot the convolved low res templates for debugging 
        # Plots the star and tellurics by default. Plots gas cell if present.
        if gpars['verbose_plot']:
            
            #use = np.where((templates_dict['star'][:, 0] >= wave[f]*gpars['']) & (templates_dict['star'][:, 0] <= wave[l]*1E4))[0]
            
            pars = self.best_fit_pars[iter_num]
            lsf = self.models_dict['lsf'].build(pars=pars)
            
            # Extra zero line
            plt.plot(wave[use_pix], np.zeros(wave[use_pix].size) - 0.1, color=gpars['colors']['purple'], linewidth=gpars['lw'], linestyle=':', alpha=0.8)
            
            # Star
            star_flux_hr = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            star_convolved = self.models_dict['lsf'].convolve_flux(star_flux_hr, lsf=lsf)
            star_flux_lr = np.interp(wave / gpars['plot_wave_factor'], templates_dict['star'][:, 0], star_convolved, left=np.nan, right=np.nan)
            ax.plot(wave[use_pix], star_flux_lr[use_pix] - 1.1, label='Star', linewidth=gpars['lw'], color='deeppink', alpha=0.8)
            
            # Tellurics
            tellurics = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], templates_dict['star'][:, 0])
            tellurics_convolved = self.models_dict['lsf'].convolve_flux(tellurics, lsf=lsf)
            tell_flux_lr = np.interp(wave / gpars['plot_wave_factor'], templates_dict['star'][:, 0], tellurics_convolved, left=np.nan, right=np.nan)
            ax.plot(wave[use_pix], tell_flux_lr[use_pix] - 1.1, label='Tellurics', linewidth=gpars['lw'], color='indigo', alpha=0.8)
            
            # Gas Cell
            if 'gas_cell' in self.models_dict:
                gas_flux_hr = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], templates_dict['star'][:, 0])
                gas_cell_convolved = self.models_dict['lsf'].convolve_flux(gas_flux_hr, lsf=lsf)
                gas_flux_lr = np.interp(wave / gpars['plot_wave_factor'], templates_dict['star'][:, 0], gas_cell_convolved, left=np.nan, right=np.nan)
                ax.plot(wave[use_pix], gas_flux_lr[use_pix] - 1.1, label='Gas Cell', linewidth=gpars['lw'], color='green', alpha=0.8)

            plt.ylim(-1.1, 1.08)
            plt.legend(loc='lower right')
        else:
            plt.ylim(-0.2, 1.08)

        plt.xlim(wave[f] - pad, wave[l] + pad)
        plt.xlabel('Wavelength [' + gpars['plot_wave_unit'] + ']', fontsize=6)
        plt.ylabel('Data, Model, Residuals', fontsize=6)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        
        if save:
            fig.save(fname)
        else:
            return fig, ax
    
    
    # Estimate the best stellar shift via pseudo cross correlation through RMS minimization with possible weights.
    # If this is not enough, the user can implement their own cross_correlate function, but it must take these same args.
    def cross_correlate(self, templates_dict, iter_num, gpars):

        if iter_num is None:
            pars = copy.deepcopy(self.initial_parameters)
            vel_step = 100 # 100 m/s step interval if doing the first check
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000 * 225
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000 * 225
        else:
            pars = copy.deepcopy(self.best_fit_pars[iter_num])
            vel_step = 0.5 # 0.5 m/s step if parameters have been fit already.
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000 * 15
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000 * 15
            
        # Determine the shifts to compute against, +/- 50 km/sec
        vels = np.arange(vel_min, vel_max, vel_step)

        # Stores the rms as a function of velocity
        rms = np.empty(shape=vels.size, dtype=np.float64)
        
        # Weights for now are just bad pixels
        weights = np.copy(self.data.badpix)
        
        for i in range(vels.size):
            
            # Set the RV parameter to the current step
            pars[self.models_dict['star'].par_names[0]].setv(value=vels[i])
            
            # Build the model
            model_lr = self.build_full(pars, templates_dict, gpars)
            
            # Construct the RMS
            rms[i] = np.sqrt(np.nansum((self.data.flux - model_lr)**2 * weights / np.nansum(weights)))

        if iter_num is not None:
            self.cross_correlations[iter_num] = np.copy(rms)
            self.cross_correlation_vels[iter_num] = np.copy(vels + self.data.bary_corr)

        best_vel = vels[np.nanargmin(rms)] # Overall stellar RV assuming correct zero point
        best_rv = best_vel + self.data.bary_corr # Actual RV is bary corr corrected
            
        if iter_num is None:
            self.initial_parameters[self.models_dict['star'].par_names[0]].setv(value=best_vel)
    
    @abstractmethod
    def modify(self, *args, **kwargs):
        pass


class iSHELLForwardModel(ForwardModel):

    def __init__(self, spec_num, order_num, models_dict, data, initial_parameters, gpars):

        super().__init__(spec_num, order_num, models_dict, data, initial_parameters, gpars)

    def build_full(self, pars, templates_dict, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]

        # Star
        star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # Gas Cell
        gas = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        tell = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], final_hr_wave_grid)
        
        # AR fringing first pass
        fringing_1 = self.models_dict['fringing_first_pass'].build(pars, final_hr_wave_grid)
        
        # AR fringing second pass
        fringing_2 = self.models_dict['fringing_second_pass'].build(pars, final_hr_wave_grid)
        
        # Blaze Model
        blaze = self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux modulo blaze function and AR fringing
        raw_flux_pre_conv = blaze * gas * tell * fringing_1 * star

        # Convolve Model with LSF
        flux_post_conv = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Tack on the AR fringing and blaze
        final_hr_flux = flux_post_conv * fringing_2

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, final_hr_flux, left=final_hr_flux[0], right=final_hr_flux[-1])

        return wavelength_solution, model_lr
            
    # Possible options:
    # 1. Change parameters from vary to not vary (dict, keys are par names, vals are true or false)
    # 2. Enable a model component and its corresponding model (dict, keys are par names, vals are true or false)
    ###  Possible keys for model_dictionary:
    # 1. Any model name (gas_cell, tellurics, star, fringing_1, fringing_2, blaze, lsf).
    #       Vals are true, false
    # 3. Enable wave base (key=wave_base)
    #       Single, key. Val is true, false
    # 4. Enable wave splines (key=wave_splines)
    #       Single key. Val is true, false
    # 5. Enable blaze base (key=blaze_base)
    #       Single key. Val is true, false
    # 6. Enable blaze splines (key=blaze_splines)
    #       Single key. Val is true, false
    #####
    # 3. A telluric model component and its sub model (telluric_component, string)
    #       Vals are true, false for that component.
    def modify(self, par_names=None, model_components=None, telluric_components=None):
        
        # Par names
        if par_names is not None:
            for key in par_names:
                self.initial_parameters[key].setv(vary=par_names[key])
                    
        # Model names, also vary parameters
        if model_components is not None:
            for key in model_components:
                self.models_dict[key].modify(model_components[key])
                for pname in self.models_dict[key].par_names:
                    self.initial_parameters[pname].setv(vary=model_components[key])
                    
        # Tellurics. Make sure shift is also enabled.
        if telluric_components is not None:
            vel_tel_current_vary = np.copy(self.initial_parameters[self.models_dict['tellurics'].par_names[0]].vary)
            for key in telluric_components:
                self.models_dict['tellurics'].modify(key, telluric_components[key])
                species_i = self.models_dict['tellurics'].species.index(key)
                self.initial_parameters[self.models_dict['tellurics'].par_names[species_i + 1]].setv(vary=telluric_components[key])
            if np.all(~np.array(list(self.models_dict['tellurics'].species_enabled.values()))):
                self.initial_parameters[self.models_dict['tellurics'].par_names[0]].setv(vary=False)
            else:
                self.initial_parameters[self.models_dict['tellurics'].par_names[0]].setv(vary=vel_tel_current_vary)
            
class CHIRONForwardModel(ForwardModel):
    
    def __init__(self, spec_num, order_num, models_dict, data, initial_parameters, gpars):
        
        super().__init__(spec_num, order_num, models_dict, data, initial_parameters, gpars)

    def build_full(self, pars, templates_dict, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]

        # Star
        star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # Gas Cell
        gas = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        tell = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        blaze = self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux modulo blaze
        raw_flux_pre_conv = blaze * gas * tell * star

        # Convolve Model with LSF
        flux_hr_conv = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, flux_hr_conv, left=flux_hr_conv[0], right=flux_hr_conv[-1])
        
        return wavelength_solution, model_lr
    
    def modify(self, par_names=None, model_components=None, telluric_components=None):
        
        # Par names
        if par_names is not None:
            for key in par_names:
                self.initial_parameters[key].setv(vary=par_names[key])
                    
        # Model names, also vary parameters
        if model_components is not None:
            for key in model_components:
                self.models_dict[key].modify(model_components[key])
                for pname in self.models_dict[key].par_names:
                    self.initial_parameters[pname].setv(vary=model_components[key])
                    
        if telluric_components is not None:
            vel_tel_current_vary = np.copy(self.initial_parameters[self.models_dict['tellurics'].par_names[0]].vary)
            for key in telluric_components:
                self.models_dict['tellurics'].modify(key, telluric_components[key])
                species_i = self.models_dict['tellurics'].species.index(key)
                self.initial_parameters[self.models_dict['tellurics'].par_names[species_i + 1]].setv(vary=telluric_components[key])
            if np.all(~np.array(list(self.models_dict['tellurics'].species_enabled.values()))):
                self.initial_parameters[self.models_dict['tellurics'].par_names[0]].setv(vary=False)
            else:
                self.initial_parameters[self.models_dict['tellurics'].par_names[0]].setv(vary=vel_tel_current_vary)

class PARVIForwardModel(ForwardModel):
    
    def __init__(self, spec_num, order_num, models_dict, data, initial_parameters, gpars):
        
        super().__init__(spec_num, order_num, models_dict, data, initial_parameters, gpars)

    def build_full(self, pars, templates_dict, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]

        # Star
        star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        tell = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        blaze = self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux modulo blaze
        raw_flux_pre_conv = blaze * tell * star

        # Convolve Model with LSF
        flux_post_conv = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, flux_post_conv, left=flux_post_conv[0], right=flux_post_conv[-1])
        
        return wavelength_solution, model_lr
    
    def modify(self, par_names=None, model_components=None, telluric_components=None):
        
        # Par names
        if par_names is not None:
            for key in par_names:
                self.initial_parameters[key].setv(vary=par_names[key])
                    
        # Model names, also vary parameters
        if model_components is not None:
            for key in model_components:
                self.models_dict[key].modify(model_components[key])
                for pname in self.models_dict[key].par_names:
                    self.initial_parameters[pname].setv(vary=model_components[key])
                    
        if telluric_components is not None:
            vel_tel_current_vary = np.copy(self.initial_parameters[self.models_dict['tellurics'].par_names[0]].vary)
            for key in telluric_components:
                self.models_dict['tellurics'].modify(key, telluric_components[key])
                species_i = self.models_dict['tellurics'].species.index(key)
                self.initial_parameters[self.models_dict['tellurics'].par_names[species_i + 1]].setv(vary=telluric_components[key])
            if np.all(~np.array(list(self.models_dict['tellurics'].species_enabled.values()))):
                self.initial_parameters[self.models_dict['tellurics'].par_names[0]].setv(vary=False)
            else:
                self.initial_parameters[self.models_dict['tellurics'].par_names[0]].setv(vary=vel_tel_current_vary)