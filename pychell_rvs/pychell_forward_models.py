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
import corner

# Science/math
import scipy
from scipy import constants as cs # cs.c = speed of light in m/s
from scipy.special import comb
import emcee # not used yet
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation
import torch # Convex Optimization for templates, needs updated.

# llvm
from numba import njit, jit, prange

# User defined
import pychell_rvs.pychell_math as pcmath
import pychell_rvs.pychell_solver as pcsolver # nelder mead solver
import pychell_rvs.pychell_model_components as pcmodelcomponents

class ForwardModel(ABC):
    
    def __init__(self, spec_num, order_num, models_dict, data, initial_parameters, gpars):
    
        # Required Data Variables
        self.spec_num = spec_num # int
        self.order_num = order_num # int
        self.data = data # object
        
        # Parameters Object, updated each iteration.
        self.initial_parameters = initial_parameters
        
        # Dictionary to store model classes which define build methods and some model parameters
        # No templates are stored here.
        # Neither this or templates_dict can be empty. Otherwise there are no models.
        self.models_dict = models_dict
        
        # Storage arrays
        self.opt = np.empty(shape=(gpars['n_template_fits']+gpars['ndi'], 2), dtype=np.float64)
        self.best_fit_pars = np.empty(shape=(gpars['n_template_fits']+gpars['ndi']), dtype=pcmodelcomponents.Parameters)
        self.wavelength_solutions = np.empty(shape=(gpars['n_data_pix'], gpars['n_template_fits']+gpars['ndi']), dtype=np.float64)
        self.residuals = np.empty(shape=(gpars['n_data_pix'], gpars['n_template_fits']+gpars['ndi']), dtype=np.float64)
        self.models = np.empty(shape=(gpars['n_data_pix'], gpars['n_template_fits']+gpars['ndi']), dtype=np.float64)
        self.cross_correlations = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.ndarray)
        self.cross_correlation_vels = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.ndarray)
        self.bisectors = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.ndarray)
        self.bisector_spans = np.empty(gpars['n_template_fits']+gpars['ndi'], dtype=np.float64)
        
        self.residuals_post_compare = np.empty(gpars['n_data_pix'], dtype=np.float64)
        
        # MCMC results
        self.mcmc_best_fit_pars = None
        self.mcmc_wavelength_solution = None
        self.mcmc_model = None
        self.mcmc_residuals = None
        
    # Must define a build_full method
    # Can also define other build methods that return modified forward models
    @abstractmethod
    def build_full(self, pars, templates, *args, **kwargs):
        pass
        
    # Output after last iteration
    def save_final_outputs(self, gpars):
        ord_dir = gpars['run_output_path'] + 'Order' + str(self.order_num+1) + os.sep
        ord_spec = '_ord' + str(self.order_num+1) + '_spec' + str(self.spec_num+1)
        filename_opt = ord_dir + gpars['full_tag'] + '_opt' + ord_spec + '.npz'
        filename_data_models = ord_dir + gpars['full_tag'] + '_data_models' + ord_spec + '.npz'
        filename_cross_corrs = ord_dir + gpars['full_tag'] + '_cross_corrs' + ord_spec + '.npz'
        data_arr = np.array([self.data.flux, self.data.flux_unc, self.data.badpix]).T
        np.savez(filename_opt, best_fit_pars=self.best_fit_pars, opt=self.opt)
        np.savez(filename_data_models, wavelength_solutions=self.wavelength_solutions, residuals=self.residuals, models=self.models, data=data_arr, obs_details=self.data.obs_details)
        
    def extract_parameter_values(forward_models, par_name, iter_num, gpars):
        v = np.empty(gpars['n_spec'], dtype=np.float64)
        for ispec in range(gpars['n_spec']):
            v[ispec] = forward_models[ispec].best_fit_pars[iter_num][par_name].value
        return v
    
    def extract_rms(forward_models, iter_num, gpars):
        rms = np.empty(gpars['n_spec'], dtype=np.float64)
        for ispec in range(gpars['n_spec']):
            rms[ispec] = forward_models[ispec].opt[iter_num, 0]
        return rms
                
    def pretty_print(self, iter_num):
        # Loop over models
        for mname in self.models_dict.keys():
            # Print the model string
            print(self.models_dict[mname])
            # Sub loop over per model parameters
            for pname in self.models_dict[mname].par_names:
                print('    ', end='')
                print(self.best_fit_pars[iter_num][pname])
        
    @abstractmethod
    def modify(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def plot(self, *args, **kwargs):
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
        raw_flux_pre_conv = gas * tell * fringing_1 * star

        # Convolve Model with LSF
        flux_post_conv = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Tack on the AR fringing and blaze
        final_hr_flux = flux_post_conv * blaze * fringing_2

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, final_hr_flux, left=final_hr_flux[0], right=final_hr_flux[-1])

        return wavelength_solution, model_lr
    
    # Returns the flux pre-convolution without the star, blaze, and lsf
    def build_1(self, pars, templates_dict, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]
        
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

        # Total flux modulo star and blaze
        raw_flux_pre_conv = gas * tell * fringing_1
        
        lsf = self.models_dict['lsf'].build(pars)
        
        return raw_flux_pre_conv, blaze, lsf

    def plot(self, iter_num, templates_dict, gpars):

        # Extract the low res wave grid, model, and residuals
        wave = self.wavelength_solutions[:, iter_num] / 1E4
        model = self.models[:, iter_num]
        residuals = self.residuals[:, iter_num]
        fname = gpars['run_output_path'] + 'Order' + str(self.order_num+1) + os.sep + 'Fits' + os.sep + gpars['full_tag'] + '_data_model_spec' + str(self.spec_num+1) + '_ord' + str(self.order_num+1) + '_iter' + str(iter_num+gpars['di']) + '.png'
            
        # Define some helpful indices
        f, l = gpars['crop_pix'][0], gpars['n_data_pix'] - gpars['crop_pix'][1]
        good = np.where(self.data.badpix == 1)[0]
        bad = np.where(self.data.badpix == 0)[0]
        bad_data_locs = np.where((bad > f) & (bad < l))[0]
        use_pix = np.arange(good[0], good[-1]).astype(int)
        
        # Figure
        plt.figure(num=1, figsize=(gpars['spec_img_w'], gpars['spec_img_h']), dpi=gpars['dpi'])
        # Data
        plt.plot(wave[use_pix], self.data.flux[use_pix], color=gpars['colors']['blue'], linewidth=gpars['lw'])
        # Model
        plt.plot(wave[use_pix], model[use_pix], color=gpars['colors']['red'], linewidth=gpars['lw'])
        # Zero line
        plt.plot(wave[use_pix], np.zeros(wave[use_pix].size), color=gpars['colors']['purple'], linewidth=gpars['lw'])
        # Residuals (all bad pixels will be zero here)
        plt.plot(wave[good], residuals[good], color=gpars['colors']['orange'], linewidth=gpars['lw'])
        
        if gpars['verbose']:
            
            use = np.where((templates_dict['star'][:, 0] >= wave[f]*1E4) & (templates_dict['star'][:, 0] <= wave[l]*1E4))[0]
            
            pars = self.best_fit_pars[iter_num]
            lsf = self.models_dict['lsf'].build(pars=pars)
            
            # Gas Cell
            gas = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], templates_dict['star'][:, 0])
            gas_cell_convolved = self.models_dict['lsf'].convolve_flux(gas, lsf=lsf)
            plt.plot(templates_dict['star'][use, 0] / 1E4, gas_cell_convolved[use] - 1.1, label='Gas Cell', linewidth=gpars['lw'], color='green', alpha=0.8)
            
            # Star
            star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            star_convolved = self.models_dict['lsf'].convolve_flux(star, lsf=lsf)
            plt.plot(templates_dict['star'][use, 0] / 1E4, star_convolved[use] - 1.1, label='Star', linewidth=gpars['lw'], color='deeppink', alpha=0.6)
            
            # Tellurics
            tellurics = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], templates_dict['star'][:, 0])
            tellurics_convolved = self.models_dict['lsf'].convolve_flux(tellurics, lsf=lsf)
            plt.plot(templates_dict['star'][use, 0] / 1E4, tellurics_convolved[use] - 1.1, label='Tellurics', linewidth=gpars['lw'], color='indigo', alpha=0.8)
            
            plt.plot(wave[use_pix], np.zeros(wave[use_pix].size)-0.1, color=gpars['colors']['purple'], linewidth=gpars['lw'], linestyle=':', alpha=0.8)
                     
            plt.xlim(wave[f] - 5E-4, wave[l] + 15E-4)
            plt.ylim(-1.1, 1.08)
            plt.legend(loc='lower right')
        else:
            plt.xlim(wave[f] - 5E-4, wave[l] + 5E-4)
            plt.ylim(-0.2, 1.08)

        plt.xlim(wave[f] - 5E-4, wave[l] + 5E-4)
        plt.xlabel('Wavelength [Microns]', fontsize=6)
        plt.ylabel('Data, Model, Residuals', fontsize=6)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        

    # Output after last iteration
    def save_final_outputs(self, gpars):
        super().save_final_outputs(gpars)
    
    # Estimate the best stellar shift via pseudo cross correlation through RMS minimization
    # Since the stellar shift is all that's varied, and it's brute force, it's mathematically identical to cc.
    def cross_correlate(self, templates_dict, iter_num, gpars, update_from):
        
        print('Cross-Correlating Spectrum ' + str(self.spec_num + 1) + ' of ' + str(gpars['n_spec']), flush=True)

        if iter_num is None:
            pars = copy.deepcopy(self.initial_parameters)
            vel_step = 100 # 100 m/s step interval if doing the first check
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000 * 200
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000 * 200
            #vel_guess = self.models_dict['star'].abs_rv * 1E3 - self.data.obs_details['bary_corr']
            #vel_min = vel_guess - 10E3
            #vel_max = vel_guess + 10E3
        else:
            pars = copy.deepcopy(self.best_fit_pars[iter_num])
            vel_step = 0.5 # 0.5 m/s step if parameters have been fit already.
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000
            
        # Determine the shifts to compute against, +/- 50 km/sec
        vels = np.arange(vel_min, vel_max, vel_step)
        
        # Construct the high res model without the star
        model_raw_no_star_hr, blaze, lsf = self.build_1(pars, templates_dict, gpars)
        
        # Generate the data wave grid.
        wave_lr = self.models_dict['wavelength_solution'].build(pars)
        
        # Copy the data just in case
        data_cp = np.copy(self.data.flux)

        model_convolved_no_star_lr = np.interp(wave_lr, templates_dict['star'][:, 0], model_raw_no_star_hr, left=np.nan, right=np.nan)
        #compare_against = data_cp / model_convolved_no_star_lr
        rms = np.empty(shape=vels.size, dtype=np.float64)
        weights = np.copy(self.data.badpix)
        for i in range(vels.size):
            
            # Construct the shifted stellar template.
            pars[self.models_dict['star'].par_names[0]].setv(value=vels[i])
            star_flux_shifted = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            
            star_flux_shifted_lr = np.interp(wave_lr, templates_dict['star'][:, 0], star_flux_shifted, left=np.nan, right=np.nan)
            
            # Construct a full model to compute an RMS
            model_raw_full = star_flux_shifted * model_raw_no_star_hr
            model_convolved_full = self.models_dict['lsf'].convolve_flux(model_raw_full, lsf=lsf)
            model_convolved_full *= blaze
            model_lr = np.interp(wave_lr, templates_dict['star'][:, 0], model_convolved_full, left=np.nan, right=np.nan)
            
            rms[i] = np.sqrt(np.nansum((self.data.flux - model_lr)**2 * weights / np.nansum(weights)))


        if iter_num is not None:
            self.cross_correlations[iter_num] = np.copy(rms)
            self.cross_correlation_vels[iter_num] = np.copy(vels + self.data.bary_corr)

        best_vel = vels[np.nanargmin(rms)] # Overall stellar RV assuming correct zero point
        best_rv = best_vel + self.data.bary_corr # Actual RV is bary corr corrected
            
        if iter_num is None:
            vel = best_vel
            iter_num = -1
        else:
            vel = best_rv
        
        if update_from:
            self.initial_parameters[self.models_dict['star'].par_names[0]].setv(value=vel)
            
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
        raw_flux_pre_conv = gas * tell * star

        # Convolve Model with LSF
        flux_post_conv = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Tack on the blaze
        final_hr_flux = flux_post_conv * blaze

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, final_hr_flux, left=final_hr_flux[0], right=final_hr_flux[-1])
        
        return wavelength_solution, model_lr
        
    # Returns the flux pre-convolution without the star
    def build_1(self, pars, templates_dict, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]

        # Gas Cell
        gas = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], final_hr_wave_grid)
        
        # All tellurics
        tell = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        blaze = self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux modulo AR fringing and blaze
        raw_flux_pre_conv = gas * tell
        
        lsf = self.models_dict['lsf'].build(pars)
        
        return raw_flux_pre_conv, blaze, lsf

    def plot(self, iter_num, templates_dict, gpars):

        # Extract the low res wave grid, model, and residuals
        wave = self.wavelength_solutions[:, iter_num] / 10 # convert from angstromg to nm
        model = self.models[:, iter_num]
        residuals = self.residuals[:, iter_num]
        fname = gpars['run_output_path'] + 'Order' + str(self.order_num+1) + os.sep + 'Fits' + os.sep + gpars['full_tag'] + '_data_model_spec' + str(self.spec_num+1) + '_ord' + str(self.order_num+1) + '_iter' + str(iter_num+gpars['di']) + '.png'
            
        # Define some helpful indices
        f, l = gpars['crop_pix'][0], gpars['n_data_pix'] - gpars['crop_pix'][1]
        good = np.where(self.data.badpix == 1)[0]
        bad = np.where(self.data.badpix == 0)[0]
        bad_data_locs = np.where((bad > f) & (bad < l))[0]
        use_pix = np.arange(good[0], good[-1]).astype(int)
        
        # Figure
        plt.figure(num=1, figsize=(gpars['spec_img_w'], gpars['spec_img_h']), dpi=gpars['dpi'])
        # Data
        plt.plot(wave[use_pix], self.data.flux[use_pix], color=gpars['colors']['blue'], linewidth=gpars['lw'])
        # Model
        plt.plot(wave[use_pix], model[use_pix], color=gpars['colors']['red'], linewidth=gpars['lw'])
        # Zero line
        plt.plot(wave[use_pix], np.zeros(wave[use_pix].size), color=gpars['colors']['purple'], linewidth=gpars['lw'])
        # Residuals (all bad pixels will be zero here)
        plt.plot(wave[good], residuals[good], color=gpars['colors']['orange'], linewidth=gpars['lw'], zorder=1)
        
        if gpars['verbose']:
            
            pars = self.best_fit_pars[iter_num]
            lsf = self.models_dict['lsf'].build(pars=pars)
            
            # Gas Cell
            gas = self.models_dict['gas_cell'].build(pars, templates_dict['gas_cell'][:, 0], templates_dict['gas_cell'][:, 1], templates_dict['star'][:, 0])
            gas_cell_convolved = self.models_dict['lsf'].convolve_flux(gas, lsf=lsf)
            plt.plot(templates_dict['star'][:, 0] / 10, gas_cell_convolved - 1.1, label='Gas Cell', linewidth=gpars['lw'], color='green', alpha=0.5)
            
            # Star
            star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            star_convolved = self.models_dict['lsf'].convolve_flux(star, lsf=lsf)
            plt.plot(templates_dict['star'][:, 0] / 10, star_convolved - 1.1, label='Star', linewidth=gpars['lw'], color='deeppink', alpha=0.8)
            
            # Tellurics
            tellurics = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], templates_dict['star'][:, 0])
            tellurics_convolved = self.models_dict['lsf'].convolve_flux(tellurics, lsf=lsf)
            plt.plot(templates_dict['star'][:, 0] / 10, tellurics_convolved - 1.1, label='Tellurics', linewidth=gpars['lw'], color='indigo', alpha=0.8)
            
            plt.ylim(-1.1, 1.08)
            plt.legend(loc='lower right')
        else:
            plt.ylim(-0.2, 1.08)
            
        plt.xlim(wave[f] - 0.01, wave[l] + 0.01)
        plt.xlabel('Wavelength [nm]', fontsize=6)
        plt.ylabel('Data, Model, Residuals', fontsize=6)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    # Output after last iteration
    def save_final_outputs(self, gpars):
        super().save_final_outputs(gpars)
        
    # Estimate the best stellar shift via a cross correlation.
    # Idea: Build the best fit model and divide this out.
    # Then perform the xcorr between the data and template.
    def cross_correlate(self, templates_dict, iter_num, gpars, update_from):
        
        print('Cross-Correlating Spectrum ' + str(self.spec_num + 1) + ' of ' + str(gpars['n_spec']), flush=True)

        if iter_num is None:
            pars = copy.deepcopy(self.initial_parameters)
            vel_step = 500 # 100 m/s step interval if doing the first check
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000 * 200
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000 * 200
        else:
            pars = copy.deepcopy(self.best_fit_pars[iter_num])
            vel_step = 0.5 # 0.5 m/s step if parameters have been fit already.
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000
            
        # Determine the shifts to compute against, +/- 50 km/sec
        vels = np.arange(vel_min, vel_max, vel_step)
        
        # Construct the high res model without the star
        model_raw_no_star_hr, blaze, lsf = self.build_1(pars, templates_dict, gpars)
        
        # Copy the data just in case
        data_cp = np.copy(self.data.flux)
        
        # Generate the data wave grid.
        wave_lr = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)
        model_convolved_no_star_lr = np.interp(wave_lr, templates_dict['star'][:, 0], model_raw_no_star_hr, left=np.nan, right=np.nan)
        compare_against = data_cp / model_convolved_no_star_lr
        rms = np.empty(shape=vels.size, dtype=np.float64)
        for i in range(vels.size):
            
            # Construct the shifted stellar template.
            pars[self.models_dict['star'].par_names[0]].setv(value=vels[i])
            star_flux_shifted = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            
            # Also get the shifted wave anyway
            wave_shifted = templates_dict['star'][:, 0] * np.exp(vels[i] / cs.c)
            
            # Convolve the star and interpolate onto low res grid alone for a cross correlation
            star_shifted_convolved = self.models_dict['lsf'].convolve_flux(star_flux_shifted, lsf=lsf)
            
            # Also construct a full model to compute an RMS
            model_raw_full = star_flux_shifted * model_raw_no_star_hr
            model_convolved_full = self.models_dict['lsf'].convolve_flux(model_raw_full, lsf=lsf)
            model_convolved_full *= blaze
            model_lr = np.interp(wave_lr, templates_dict['star'][:, 0], model_convolved_full, left=np.nan, right=np.nan)
            
            rms[i] = np.sqrt(np.nansum((self.data.flux - model_lr)**2 * self.data.badpix) / np.nansum(self.data.badpix))
        
        if iter_num is not None:
            self.cross_correlations[iter_num] = np.copy(rms)
            self.cross_correlation_vels[iter_num] = np.copy(vels + self.data.bary_corr)

        best_vel = vels[np.nanargmin(rms)]
        best_rv = best_vel + self.data.bary_corr # Actual RV is bary corr corrected

        if iter_num is None:
            vel = best_vel
            iter_num = -1
        else:
            vel = best_rv
        
        if update_from:
            self.initial_parameters[self.models_dict['star'].par_names[0]].setv(value=vel)
    
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
        raw_flux_pre_conv = tell * star

        # Convolve Model with LSF
        flux_post_conv = self.models_dict['lsf'].convolve_flux(raw_flux_pre_conv, pars=pars)

        # Tack on the blaze
        final_hr_flux = flux_post_conv * blaze

        # Generate the wavelength solution of the data
        wavelength_solution = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)

        # Interpolate high res model onto data grid
        model_lr = np.interp(wavelength_solution, final_hr_wave_grid, final_hr_flux, left=final_hr_flux[0], right=final_hr_flux[-1])
        
        return wavelength_solution, model_lr
        
    # Returns the flux pre-convolution without the star
    def build_1(self, pars, templates_dict, gpars):
        
        # The final high res wave grid for the model
        # Eventually linearly interpolated to the data grid (wavelength solution)
        final_hr_wave_grid = templates_dict['star'][:, 0]
        
        # All tellurics
        tell = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], final_hr_wave_grid)
        
        # Blaze Model
        blaze = self.models_dict['blaze'].build(pars, final_hr_wave_grid)

        # Total flux module blaze
        raw_flux_pre_conv = tell
        
        lsf = self.models_dict['lsf'].build(pars)
        
        return raw_flux_pre_conv, blaze, lsf

    def plot(self, iter_num, templates_dict, gpars):

        # Extract the low res wave grid, model, and residuals
        wave = self.wavelength_solutions[:, iter_num] / 1E4 # convert from angstroms to microns
        model = self.models[:, iter_num]
        residuals = self.residuals[:, iter_num]
        fname = gpars['run_output_path'] + 'Order' + str(self.order_num+1) + os.sep + 'Fits' + os.sep + gpars['full_tag'] + '_data_model_spec' + str(self.spec_num+1) + '_ord' + str(self.order_num+1) + '_iter' + str(iter_num+gpars['di']) + '.png'
            
        # Define some helpful indices
        f, l = gpars['crop_pix'][0], gpars['n_data_pix'] - gpars['crop_pix'][1]
        good = np.where(self.data.badpix == 1)[0]
        bad = np.where(self.data.badpix == 0)[0]
        bad_data_locs = np.where((bad > f) & (bad < l))[0]
        use_pix = np.arange(good[0], good[-1]).astype(int)
        
        # Figure
        plt.figure(num=1, figsize=(gpars['spec_img_w'], gpars['spec_img_h']), dpi=gpars['dpi'])
        # Data
        plt.plot(wave[use_pix], self.data.flux[use_pix], color=gpars['colors']['blue'], linewidth=gpars['lw'])
        # Model
        plt.plot(wave[use_pix], model[use_pix], color=gpars['colors']['red'], linewidth=gpars['lw'])
        # Zero line
        plt.plot(wave[use_pix], np.zeros(wave[use_pix].size), color=gpars['colors']['purple'], linewidth=gpars['lw'])
        # Residuals (all bad pixels will be zero here)
        plt.plot(wave[good], residuals[good], color=gpars['colors']['orange'], linewidth=gpars['lw'], zorder=1)
        
        # Flagged Pixels
        #plt.plot(wave[bad[bad_data_locs]], residuals[bad[bad_data_locs]], marker='X', linewidth=0, color=gpars['colors']['darkpink'], markersize='4')
        
        if gpars['verbose']:
            
            pars = self.best_fit_pars[iter_num]
            lsf = self.models_dict['lsf'].build(pars=pars)
            
            # Star
            star = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            star_convolved = self.models_dict['lsf'].convolve_flux(star, lsf=lsf)
            plt.plot(templates_dict['star'][:, 0] / 1E4, star_convolved - 1, label='Star', linewidth=gpars['lw'], color='deeppink', alpha=0.5)
            
            # Tellurics
            tellurics = self.models_dict['tellurics'].build(pars, templates_dict['tellurics'], templates_dict['star'][:, 0])
            tellurics_convolved = self.models_dict['lsf'].convolve_flux(tellurics, lsf=lsf)
            plt.plot(templates_dict['star'][:, 0] / 1E4, tellurics_convolved - 1, label='Telurics', linewidth=gpars['lw'], color='indigo', alpha=0.5)
            
            plt.legend(loc='lower right')
                     
            plt.ylim(-1, 1.08)
        else:
            plt.ylim(-0.2, 1.08)
            
        plt.xlim(wave[f] - 0.0001, wave[l] + 0.0001)
        plt.xlabel('Wavelength [microns]', fontsize=6)
        plt.ylabel('Data, Model, Residuals', fontsize=6)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    # Output after last iteration
    def save_final_outputs(self, gpars):
        super().save_final_outputs(gpars)
        
    # Estimate the best stellar shift via a cross correlation.
    # Idea: Build the best fit model and divide this out.
    # Then perform the xcorr between the data and template.
    def cross_correlate(self, templates_dict, iter_num, gpars, update_from):
        
        print('Cross-Correlating Spectrum ' + str(self.spec_num + 1) + ' of ' + str(gpars['n_spec']), flush=True)

        if iter_num is None:
            pars = copy.deepcopy(self.initial_parameters)
            vel_step = 500 # 100 m/s step interval if doing the first check
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000 * 200
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000 * 200
        else:
            pars = copy.deepcopy(self.best_fit_pars[iter_num])
            vel_step = 0.5 # 0.5 m/s step if parameters have been fit already.
            vel_min = pars[self.models_dict['star'].par_names[0]].value - 1000
            vel_max = pars[self.models_dict['star'].par_names[0]].value + 1000
            
        # Determine the shifts to compute against, +/- 50 km/sec
        vels = np.arange(vel_min, vel_max, vel_step)
        
        # Construct the high res model without the star
        model_raw_no_star_hr, blaze, lsf = self.build_1(pars, templates_dict, gpars)
        
        # Generate the wave grid
        wave_lr = self.models_dict['wavelength_solution'].build(pars, wave_grid=self.data.wave_grid)
        
        # Copy the data just in case
        data_cp = np.copy(self.data.flux)
        
        model_convolved_no_star_lr = np.interp(wave_lr, templates_dict['star'][:, 0], model_raw_no_star_hr, left=np.nan, right=np.nan)
        compare_against = data_cp / model_convolved_no_star_lr
        rms = np.empty(shape=vels.size, dtype=np.float64)
        for i in range(vels.size):
            
            # Construct the shifted stellar template.
            pars[self.models_dict['star'].par_names[0]].setv(value=vels[i])
            star_flux_shifted = self.models_dict['star'].build(pars, templates_dict['star'][:, 0], templates_dict['star'][:, 1], templates_dict['star'][:, 0])
            
            # Also get the shifted wave anyway
            wave_shifted = templates_dict['star'][:, 0] * np.exp(vels[i] / cs.c)
            
            # Convolve the star and interpolate onto low res grid alone for a cross correlation
            star_shifted_convolved = self.models_dict['lsf'].convolve_flux(star_flux_shifted, lsf=lsf)
            
            # Also construct a full model to compute an RMS
            model_raw_full = star_flux_shifted * model_raw_no_star_hr
            model_convolved_full = self.models_dict['lsf'].convolve_flux(model_raw_full, lsf=lsf)
            model_convolved_full *= blaze
            model_lr = np.interp(wave_lr, templates_dict['star'][:, 0], model_convolved_full, left=np.nan, right=np.nan)
            
            rms[i] = np.sqrt(np.nansum((self.data.flux - model_lr)**2 * self.data.badpix) / np.nansum(self.data.badpix))
        
        if iter_num is not None:
            self.cross_correlations[iter_num] = np.copy(rms)
            self.cross_correlation_vels[iter_num] = np.copy(vels + self.data.bary_corr)

        best_vel = vels[np.nanargmin(rms)]
        best_rv = best_vel + self.data.bary_corr # Actual RV is bary corr corrected

        if iter_num is None:
            vel = best_vel
            iter_num = -1
        else:
            vel = best_rv
        
        if update_from:
            self.initial_parameters[self.models_dict['star'].par_names[0]].setv(value=vel)
    
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

# PyTorch Stuff
class StarSolver(torch.nn.Module):

    def __init__(self, star_flux, lab_flux, raw_models_partial, waves_lr, waves_hr_star_shifted, weights, lsf, data, wave_hr_master, blaze):
        super(StarSolver, self).__init__()
        
        # Possible Parameters to optimize
        self.star_flux = star_flux # the stellar flux
        self.lab_flux = lab_flux # the lab frame flux
        
        self.blaze = blaze # n_data x n_spec
        self.raw_models_partial = raw_models_partial
        self.waves_lr = waves_lr
        self.waves_hr_star_shifted = waves_hr_star_shifted
        self.weights = weights
        self.data = data
        self.nx_data, self.n_spec = self.data.shape
        self.n_lsf = lsf.shape[0]
        self.lsf = torch.ones(1, 1, self.n_lsf, self.n_spec, dtype=torch.float64)
        self.lsf[0, 0, :, :] = lsf
        self.nx_pad1 = int(self.n_lsf / 2) - 1
        self.nx_pad2 = int(self.n_lsf / 2)
        self.wave_hr_master = wave_hr_master
        self.nx_model = self.wave_hr_master.size()[0]
        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self):
    
        models_lr = torch.empty((self.nx_data, self.n_spec), dtype=torch.float64)
        
        for ispec in range(self.n_spec):

            # Interpolate the stellar variable back to master grid
            star = self.Interp1d()(self.waves_hr_star_shifted[:, ispec], self.star_flux, self.wave_hr_master)
            
            # Convolution. Note: PyTorch convolution is a pain in the ass.
            # Also torch.cat seems to take up too much memory, so a workaround.
            model_p = torch.ones((1, 1, self.nx_model + self.nx_pad1 + self.nx_pad2), dtype=torch.float64)

            model_p[0, 0, self.nx_pad1:(-self.nx_pad2)] = star.flatten() * self.raw_models_partial[:, ispec] + self.lab_flux.flatten()
            self.conv.weight.data = self.lsf[:, :, :, ispec]
            model_new = self.conv(model_p).flatten()

            model_hr = self.blaze[:, ispec] * model_new

            # Interpolate onto data grid
            models_lr[:, ispec] = self.Interp1d()(self.wave_hr_master, model_hr, self.waves_lr[:, ispec])

        # Weighted RMS, effectively reduced chi2 depending on weights
        wdiffs2 = (models_lr - self.data)**2 * self.weights

        loss = torch.sqrt(torch.sum(wdiffs2) / torch.sum(self.weights))

        return loss

    class Interp1d(torch.autograd.Function):
        def __call__(self, x, y, xnew, out=None):
            return self.forward(x, y, xnew, out)

        def forward(ctx, x, y, xnew, out=None):

            # making the vectors at least 2D
            is_flat = {}
            require_grad = {}
            v = {}
            device = []
            for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
                assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                            'at most 2-D.'
                if len(vec.shape) == 1:
                    v[name] = vec[None, :]
                else:
                    v[name] = vec
                is_flat[name] = v[name].shape[0] == 1
                require_grad[name] = vec.requires_grad
                device = list(set(device + [str(vec.device)]))
            assert len(device) == 1, 'All parameters must be on the same device.'
            device = device[0]

            # Checking for the dimensions
            assert (v['x'].shape[1] == v['y'].shape[1]
                    and (
                         v['x'].shape[0] == v['y'].shape[0]
                         or v['x'].shape[0] == 1
                         or v['y'].shape[0] == 1
                        )
                    ), ("x and y must have the same number of columns, and either "
                        "the same number of row or one of them having only one "
                        "row.")

            reshaped_xnew = False
            if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
               and (v['xnew'].shape[0] > 1)):
                # if there is only one row for both x and y, there is no need to
                # loop over the rows of xnew because they will all have to face the
                # same interpolation problem. We should just stack them together to
                # call interp1d and put them back in place afterwards.
                original_xnew_shape = v['xnew'].shape
                v['xnew'] = v['xnew'].contiguous().view(1, -1)
                reshaped_xnew = True

            # identify the dimensions of output and check if the one provided is ok
            D = max(v['x'].shape[0], v['xnew'].shape[0])
            shape_ynew = (D, v['xnew'].shape[-1])
            if out is not None:
                if out.numel() != shape_ynew[0]*shape_ynew[1]:
                    # The output provided is of incorrect shape.
                    # Going for a new one
                    out = None
                else:
                    ynew = out.reshape(shape_ynew)
            if out is None:
                ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

            # moving everything to the desired device in case it was not there
            # already (not handling the case things do not fit entirely, user will
            # do it if required.)
            for name in v:
                v[name] = v[name].to(device)

            # calling searchsorted on the x values.
            #ind = ynew
            #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
            ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
            ind = torch.tensor(ind)
            # the `-1` is because searchsorted looks for the index where the values
            # must be inserted to preserve order. And we want the index of the
            # preceeding value.
            ind -= 1
            # we clamp the index, because the number of intervals is x.shape-1,
            # and the left neighbour should hence be at most number of intervals
            # -1, i.e. number of columns in x -2
            ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

            # helper function to select stuff according to the found indices.
            def sel(name):
                if is_flat[name]:
                    return v[name].contiguous().view(-1)[ind]
                return torch.gather(v[name], 1, ind)

            # activating gradient storing for everything now
            enable_grad = False
            saved_inputs = []
            for name in ['x', 'y', 'xnew']:
                if require_grad[name]:
                    enable_grad = True
                    saved_inputs += [v[name]]
                else:
                    saved_inputs += [None, ]
            # assuming x are sorted in the dimension 1, computing the slopes for
            # the segments
            is_flat['slopes'] = is_flat['x']
            # now we have found the indices of the neighbors, we start building the
            # output. Hence, we start also activating gradient tracking
            with torch.enable_grad() if enable_grad else contextlib.suppress():
                v['slopes'] = (
                        (v['y'][:, 1:]-v['y'][:, :-1])
                        /
                        (v['x'][:, 1:]-v['x'][:, :-1])
                    )

                # now build the linear interpolation
                ynew = sel('y') + sel('slopes')*(
                                        v['xnew'] - sel('x'))

                if reshaped_xnew:
                    ynew = ynew.view(original_xnew_shape)

            ctx.save_for_backward(ynew, *saved_inputs)
            return ynew

        @staticmethod
        def backward(ctx, grad_out):
            inputs = ctx.saved_tensors[1:]
            gradients = torch.autograd.grad(
                            ctx.saved_tensors[0],
                            [i for i in inputs if i is not None],
                            grad_out, retain_graph=True)
            result = [None, ] * 5
            pos = 0
            for index in range(len(inputs)):
                if inputs[index] is not None:
                    result[index] = gradients[pos]
                    pos += 1
            return (*result,)
        
        
class StarSolverWOBBLE(torch.nn.Module):

    def __init__(self, star_flux, raw_models_partial, waves_lr, waves_hr_star_shifted, weights, data, wave_hr_master, blaze):
        super(StarSolverWOBBLE, self).__init__()
        
        # Possible Parameters to optimize
        self.star_logflux = star_flux # the stellar flux
        self.telluric_logflux = telluric_flux # the lab frame flux
        
        self.logblaze = logblaze # n_data x n_spec
        self.raw_models_partial = raw_models_partial
        self.waves_lr = waves_lr
        self.waves_hr_star_shifted = waves_hr_star_shifted
        self.weights = weights
        self.data = data
        self.nx_data, self.n_spec = self.data.shape
        self.wave_hr_master = wave_hr_master
        self.nx_model = self.wave_hr_master.size()[0]

    def forward(self):
    
        models_lr = torch.empty((self.nx_data, self.n_spec), dtype=torch.float64)
        
        for ispec in range(self.n_spec):

            # Interpolate the stellar variable back to master grid
            star_logflux = self.Interp1d()(self.waves_hr_star_shifted[:, ispec], self.star_logflux, self.wave_hr_master)
            
            # Interpolate the telluric flux onto the master grid
            tell_logflux = self.Interp1d()(self.waves_hr_tell_shifted[:, ispec], self.tell_depth * self.self.telluric_logflux, self.wave_hr_master)
            
            model_new = star_logflux.flatten() + tell_logflux.flatten()

            # Interpolate onto data grid
            models_lr[:, ispec] = self.Interp1d()(self.wave_hr_master, model_hr, self.waves_lr[:, ispec])

        # Weighted RMS
        wdiffs2 = (models_lr - self.data)**2 * self.weights

        loss = torch.sqrt(torch.sum(wdiffs2) / torch.sum(self.weights))

        return loss

    class Interp1d(torch.autograd.Function):
        def __call__(self, x, y, xnew, out=None):
            return self.forward(x, y, xnew, out)

        def forward(ctx, x, y, xnew, out=None):

            # making the vectors at least 2D
            is_flat = {}
            require_grad = {}
            v = {}
            device = []
            for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
                assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                            'at most 2-D.'
                if len(vec.shape) == 1:
                    v[name] = vec[None, :]
                else:
                    v[name] = vec
                is_flat[name] = v[name].shape[0] == 1
                require_grad[name] = vec.requires_grad
                device = list(set(device + [str(vec.device)]))
            assert len(device) == 1, 'All parameters must be on the same device.'
            device = device[0]

            # Checking for the dimensions
            assert (v['x'].shape[1] == v['y'].shape[1]
                    and (
                         v['x'].shape[0] == v['y'].shape[0]
                         or v['x'].shape[0] == 1
                         or v['y'].shape[0] == 1
                        )
                    ), ("x and y must have the same number of columns, and either "
                        "the same number of row or one of them having only one "
                        "row.")

            reshaped_xnew = False
            if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
               and (v['xnew'].shape[0] > 1)):
                # if there is only one row for both x and y, there is no need to
                # loop over the rows of xnew because they will all have to face the
                # same interpolation problem. We should just stack them together to
                # call interp1d and put them back in place afterwards.
                original_xnew_shape = v['xnew'].shape
                v['xnew'] = v['xnew'].contiguous().view(1, -1)
                reshaped_xnew = True

            # identify the dimensions of output and check if the one provided is ok
            D = max(v['x'].shape[0], v['xnew'].shape[0])
            shape_ynew = (D, v['xnew'].shape[-1])
            if out is not None:
                if out.numel() != shape_ynew[0]*shape_ynew[1]:
                    # The output provided is of incorrect shape.
                    # Going for a new one
                    out = None
                else:
                    ynew = out.reshape(shape_ynew)
            if out is None:
                ynew = torch.zeros(*shape_ynew, dtype=y.dtype, device=device)

            # moving everything to the desired device in case it was not there
            # already (not handling the case things do not fit entirely, user will
            # do it if required.)
            for name in v:
                v[name] = v[name].to(device)

            # calling searchsorted on the x values.
            #ind = ynew
            #searchsorted(v['x'].contiguous(), v['xnew'].contiguous(), ind)
            ind = np.searchsorted(v['x'].contiguous().numpy().flatten(), v['xnew'].contiguous().numpy().flatten())
            ind = torch.tensor(ind)
            # the `-1` is because searchsorted looks for the index where the values
            # must be inserted to preserve order. And we want the index of the
            # preceeding value.
            ind -= 1
            # we clamp the index, because the number of intervals is x.shape-1,
            # and the left neighbour should hence be at most number of intervals
            # -1, i.e. number of columns in x -2
            ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1).long()

            # helper function to select stuff according to the found indices.
            def sel(name):
                if is_flat[name]:
                    return v[name].contiguous().view(-1)[ind]
                return torch.gather(v[name], 1, ind)

            # activating gradient storing for everything now
            enable_grad = False
            saved_inputs = []
            for name in ['x', 'y', 'xnew']:
                if require_grad[name]:
                    enable_grad = True
                    saved_inputs += [v[name]]
                else:
                    saved_inputs += [None, ]
            # assuming x are sorted in the dimension 1, computing the slopes for
            # the segments
            is_flat['slopes'] = is_flat['x']
            # now we have found the indices of the neighbors, we start building the
            # output. Hence, we start also activating gradient tracking
            with torch.enable_grad() if enable_grad else contextlib.suppress():
                v['slopes'] = (
                        (v['y'][:, 1:]-v['y'][:, :-1])
                        /
                        (v['x'][:, 1:]-v['x'][:, :-1])
                    )

                # now build the linear interpolation
                ynew = sel('y') + sel('slopes')*(
                                        v['xnew'] - sel('x'))

                if reshaped_xnew:
                    ynew = ynew.view(original_xnew_shape)

            ctx.save_for_backward(ynew, *saved_inputs)
            return ynew

        @staticmethod
        def backward(ctx, grad_out):
            inputs = ctx.saved_tensors[1:]
            gradients = torch.autograd.grad(
                            ctx.saved_tensors[0],
                            [i for i in inputs if i is not None],
                            grad_out, retain_graph=True)
            result = [None, ] * 5
            pos = 0
            for index in range(len(inputs)):
                if inputs[index] is not None:
                    result[index] = gradients[pos]
                    pos += 1
            return (*result,)
        