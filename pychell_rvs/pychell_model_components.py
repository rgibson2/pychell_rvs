# Python built in modules
from collections import OrderedDict
from abc import ABC, abstractmethod # Abstract classes
import pdb # debugging
stop = pdb.set_trace

# Science/math
import scipy
from scipy import constants as cs # cs.c = speed of light in m/s
from scipy.special import comb
import numpy as np # Math, Arrays
import scipy.interpolate # Cubic interpolation, Akima interpolation

# llvm
from numba import njit, jit, jitclass
import numba

# User defined
import pychell_rvs.pychell_math as pcmath
import pychell_rvs.pychell_solver as pcsolver # nelder mead solver

class SpectralComponent(ABC):
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def build(self, pars, *args, **kwargs):
        pass
    
    @abstractmethod
    def build_fake(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def modify(self, *args, **kwargs):
        pass
    
    
class GasCellModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = False
        if blueprint['input_file'] is not None:
            self.enabled = True
            self.input_file = blueprint['input_file']
        self.name = blueprint['name']
        self.base_par_names = ['_shift', '_depth']
        self.par_names = [self.name + s for s in self.base_par_names]
    
    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            wave = wave + pars[self.par_names[0]].value # NOTE: Gas shift is additive in Angstroms
            flux = flux ** pars[self.par_names[1]].value
            return np.interp(wave_final, wave, flux, left=flux[0], right=flux[-1])
        else:
            return self.build_fake(wave_final.size)
        
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def modify(self, v):
        self.enabled = v
        
    def load_template(self, gpars):
        print('Loading in Gas Cell Template ...', flush=True)
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > gpars['wave_left'] - 1) & (wave < gpars['wave_right'] + 1))[0]
        template = np.array([wave[good], flux[good]]).T
        return template
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['shift'][1], minv=blueprint['shift'][0], maxv=blueprint['shift'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=blueprint['depth'][1], minv=blueprint['depth'][0], maxv=blueprint['depth'][2], mcmcscale=0.001))
        return pars
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
    
class GasCellModelOrderDependent(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = False
        if blueprint['input_file'] is not None:
            self.enabled = True
            self.input_file = blueprint['input_file']
        self.name = blueprint['name']
        self.base_par_names = ['_shift', '_depth']
        self.par_names = [self.name + s for s in self.base_par_names]
        self.order_num = order_num
    
    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            wave = wave + pars[self.par_names[0]].value # NOTE: Gas shift is additive in Angstroms
            flux = flux ** pars[self.par_names[1]].value
            return np.interp(wave_final, wave, flux, left=flux[0], right=flux[-1])
        else:
            return self.build_fake(wave_final.size)
        
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def modify(self, v):
        self.enabled = v
        
    def load_template(self, gpars):
        print('Loading in Gas Cell Template ...', flush=True)
        template = np.load(self.input_file)
        wave, flux = template['wave'], template['flux']
        good = np.where((wave > gpars['wave_left'] - 1) & (wave < gpars['wave_right'] + 1))[0]
        template = np.array([wave[good], flux[good]]).T
        return template
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        shift = blueprint['shifts'][self.order_num]
        depth = blueprint['depth']
        pars.append(Parameter(name=self.par_names[0], value=shift, minv=shift - blueprint['shift_range'][0], maxv=shift + blueprint['shift_range'][1], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=depth[1], minv=depth[0], maxv=depth[2], mcmcscale=0.001))
        return pars
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'    
    
    
    
    
    
    

    
class LSFHermiteModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.name = blueprint['name']
        self.hermdeg = blueprint['hermdeg']
        self.compress = blueprint['compress']
        self.nx_model = gpars['n_model_pix']
        self.dl = gpars['dl']
        self.nx = int(self.nx_model / self.compress)
        self.x = np.arange(-(int(self.nx / 2)-1), int(self.nx / 2)+1, 1) * self.dl
        self.base_par_names = ['_width']
        for k in range(self.hermdeg):
            self.base_par_names.append('_a' + str(k+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        self.enabled = True
        
    def build(self, pars):
        width = pars[self.par_names[0]].value
        herm = pcmath.hermfun(self.x / width, self.hermdeg)
        if self.hermdeg == 0: # just a Gaussian
            lsf = herm
        else:
            lsf = herm[:, 0]
        for i in range(self.hermdeg):
            lsf += pars[self.par_names[i+1]].value * herm[:, i+1]
        lsf /= np.nansum(lsf)
        return lsf
    
    def convolve_flux(self, raw_flux, pars=None, lsf=None):
        if lsf is None and pars is None:
            sys.exit("ERROR: Cannot construct LSF with no parameters")
        elif lsf is None:
            lsf = self.build(pars)
        padded_flux = np.pad(raw_flux, pad_width=(int(self.nx/2-1), int(self.nx/2)), mode='constant', constant_values=(raw_flux[0], raw_flux[-1]))
        convolved_flux = np.convolve(padded_flux, lsf, 'valid')
        return convolved_flux
    
    def modify(self, v):
        self.enabled = v
        
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['width'][1], minv=blueprint['width'][0], maxv=blueprint['width'][2], mcmcscale=0.1))
        for i in range(self.hermdeg):
            pars.append(Parameter(name=self.par_names[i+1], value=blueprint['ak'][1], minv=blueprint['ak'][0], maxv=blueprint['ak'][2], mcmcscale=0.001))
        return pars
            
    def build_fake(self):
        delta = np.zeros(self.nx, dtype=float)
        delta[int(self.nx / 2)] = 1.0
        return delta
    
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'

class ResidualBlazeModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.n_splines = blueprint['n_splines']
        self.blaze_wave_estimate = blueprint['blaze_wavelengths'][order_num]
        self.name = blueprint['name']
        self.enabled = True
        self.base_par_names = ['_base_quad', '_base_lin', '_base_zero']
        if self.n_splines > 0:
            self.spline_set_points = np.linspace(gpars['wave_left'], gpars['wave_right'], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
    def build(self, pars, wave_final):
        blaze_base_pars = np.array([pars[self.par_names[0]].value, pars[self.par_names[1]].value, pars[self.par_names[2]].value])
        blaze_base = np.polyval(blaze_base_pars, wave_final - self.blaze_wave_estimate)
        if self.n_splines == 0:
            return blaze_base
        else:
            splines = np.empty(self.n_splines+1, dtype=np.float64)
            for i in range(self.n_splines+1):
                splines[i] = pars[self.par_names[i+3]].value
            blaze_spline = scipy.interpolate.CubicSpline(self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
            return blaze_base + blaze_spline
        
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def modify(self, v):
        self.enabled = v
        
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['base_quad'][1], minv=blueprint['base_quad'][0], maxv=blueprint['base_quad'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=blueprint['base_lin'][1], minv=blueprint['base_lin'][0], maxv=blueprint['base_lin'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[2], value=blueprint['base_zero'][1], minv=blueprint['base_zero'][0], maxv=blueprint['base_zero'][2], mcmcscale=0.1))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                pars.append(Parameter(name=self.par_names[i+3], value=blueprint['spline'][1], minv=blueprint['spline'][0], maxv=blueprint['spline'][2], mcmcscale=0.001))
        return pars
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
    
class FullBlazeModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.n_splines = blueprint['n_splines']
        self.blaze_wave_estimate = blueprint['blaze_wavelengths'][order_num]
        self.name = blueprint['name']
        self.enabled = True
        self.base_par_names = ['_base_amp', '_base_b', '_base_c', '_base_d']

        if self.n_splines > 0:
            self.spline_set_points = np.linspace(gpars['wave_left'], gpars['wave_right'], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
    def build(self, pars, wave_final):
        amp = pars[self.par_names[0]].value
        b = pars[self.par_names[1]].value
        c = pars[self.par_names[2]].value
        d = pars[self.par_names[3]].value
        lam_b = self.blaze_wave_estimate + c
        blaze_base = amp * np.abs(np.sinc(b * (wave_final - lam_b)))**(2 * d)
        if self.n_splines == 0:
            return blaze_base
        else:
            splines = np.empty(self.n_splines+1, dtype=np.float64)
            for i in range(self.n_splines+1):
                splines[i] = pars[self.par_names[i+4]].value
            blaze_spline = scipy.interpolate.CubicSpline(self.spline_set_points, splines, extrapolate=True, bc_type='not-a-knot')(wave_final)
            return blaze_base + blaze_spline
        
    def build_fake(self, n):
        return np.ones(n, dtype=float)
            
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['base_amp'][1], minv=blueprint['base_amp'][0], maxv=blueprint['base_amp'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=blueprint['base_b'][1], minv=blueprint['base_b'][0], maxv=blueprint['base_b'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[2], value=blueprint['base_c'][1], minv=blueprint['base_c'][0], maxv=blueprint['base_c'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[3], value=blueprint['base_d'][1], minv=blueprint['base_d'][0], maxv=blueprint['base_d'][2], mcmcscale=0.1))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                pars.append(Parameter(name=self.par_names[i+4], value=blueprint['spline'][1], minv=blueprint['spline'][0], maxv=blueprint['spline'][2], mcmcscale=0.001))
        return pars
    
    def modify(self, v):
        self.enabled = v
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
        
# Quadratic + Splines
class WaveSolModelFull(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.name = blueprint['name']
        self.n_splines = blueprint['n_splines']
        self.base_pixel_set_points = np.array(blueprint['base_pixel_set_points'])
        self.base_wave_zero_points = np.array([blueprint['base_set_point_1'][order_num], blueprint['base_set_point_2'][order_num], blueprint['base_set_point_3'][order_num]])
        self.nx = gpars['n_data_pix']
        self.base_par_names = ['_wave_lagrange_1', '_wave_lagrange_2', '_wave_lagrange_3']
        if self.n_splines > 0:
            self.spline_pixel_set_points = np.linspace(gpars['pix_left'], gpars['pix_right'], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
    def build(self, pars, *args, **kwargs):
        pixel_grid = np.arange(self.nx)
        self.enabled = True
        base_wave_set_points = np.array([pars[self.par_names[0]].value, pars[self.par_names[1]].value, pars[self.par_names[2]].value]) + self.base_wave_zero_points
        base_coeffs = pcmath.poly_coeffs(self.base_pixel_set_points, base_wave_set_points)
        wave_base = np.polyval(base_coeffs, pixel_grid)
        if self.n_splines == 0:
            return wave_base
        else:
            splines = np.empty(self.n_splines+1, dtype=np.float64)
            for i in range(self.n_splines + 1):
                splines[i] = pars[self.par_names[i+3]].value
            wave_spline = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, splines, bc_type='not-a-knot', extrapolate=True)(pixel_grid)
            return wave_base + wave_spline
    
    # Should never be called. Need to implement!
    def build_fake(self):
        pass
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['base'][1], minv=blueprint['base'][0], maxv=blueprint['base'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=blueprint['base'][1], minv=blueprint['base'][0], maxv=blueprint['base'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[2], value=blueprint['base'][1], minv=blueprint['base'][0], maxv=blueprint['base'][2], mcmcscale=0.1))
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                pars.append(Parameter(name=self.par_names[i+3], value=blueprint['spline'][1], minv=blueprint['spline'][0], maxv=blueprint['spline'][2], mcmcscale=0.001))
        return pars
    
    def modify(self):
        pass
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'


# Starts from a known wavelength solution plus spline offset
# For CHIRON, we start with the ThAr wls and add splines since we have a gas cell.
# For PARVI, we are provided the exact wls (so no splines)
class WaveModelKnown(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = True
        self.n_splines = blueprint['n_splines']
        self.name = blueprint['name']
        self.nx = gpars['n_data_pix']
        self.base_par_names = []
        if self.n_splines > 0:
            self.spline_pixel_set_points = np.linspace(gpars['pix_left'], gpars['pix_right'], num=self.n_splines + 1)
            for i in range(self.n_splines+1):
                self.base_par_names.append('_wave_spline_' + str(i+1))
        self.par_names = [self.name + s for s in self.base_par_names]
        
    def build(self, pars, wave_grid):
        if self.n_splines == 0:
            return wave_grid
        else:
            pixel_grid = np.arange(self.nx)
            splines = np.empty(self.n_splines+1, dtype=np.float64)
            for i in range(self.n_splines + 1):
                splines[i] = pars[self.par_names[i]].value
            wave_spline = scipy.interpolate.CubicSpline(self.spline_pixel_set_points, splines, bc_type='not-a-knot', extrapolate=True)(pixel_grid)
            return wave_grid + wave_spline
    
    def build_fake(self):
        pass
    
    def modify(self):
        pass
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        if self.n_splines > 0:
            for i in range(self.n_splines + 1):
                pars.append(Parameter(name=self.par_names[i], value=blueprint['spline'][1], minv=blueprint['spline'][0], maxv=blueprint['spline'][2], mcmcscale=0.001))
        return pars
    
    
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
    
    
# Starts from a known wavelength solution plus spline offset
# For CHIRON, we start with the ThAr wls and add splines since we have a gas cell.
# For PARVI, we are provided the exact wls (so no splines)
class LSFModelKnown(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = True
        self.name = blueprint['name']
        self.base_par_names = []
        self.par_names = []
        self.dl_original = blueprint['dl_original']
        self.dl = gpars['dl']
        
        self.nx_model = gpars['n_model_pix']
        self.nx = blueprint['n_lsf_pix']
        
        # NOTE: FIX THIS! It depends on how the known LSF grid was computed. Maybe add to blueprint
        # self.dl = blueprint['dl] # Then remove above line
        self.x = np.arange(-(int(self.nx / 2)-1), int(self.nx / 2)+1, 1) * self.dl
        
    def build(self, pars, lsf):
        return lsf
    
    def build_fake(self):
        pass
    
    def convolve_flux(self, raw_flux, lsf):
        padded_flux = np.pad(raw_flux, pad_width=(int(self.nx/2-1), int(self.nx/2)), mode='constant', constant_values=(raw_flux[0], raw_flux[-1]))
        convolved_flux = np.convolve(padded_flux, lsf, 'valid')
        return convolved_flux
    
    def modify(self):
        pass
    
    def initialize_parameters(self, blueprint, gpars):
        return []
    
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
        
class StarModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = gpars['do_init_guess']
        self.name = blueprint['name']
        self.base_par_names = ['_vel']
        self.input_file = blueprint['input_file']
        self.par_names = [self.name + s for s in self.base_par_names]
    
    def build(self, pars, wave, flux, wave_final):
        if self.enabled:
            wave_shifted = wave * np.exp(pars[self.par_names[0]].value / cs.c)
            return np.interp(wave_final, wave_shifted, flux, left=np.nan, right=np.nan)
        else:
            return self.build_fake(wave_final.size)
    
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['vel'][1], minv=blueprint['vel'][0], maxv=blueprint['vel'][2], mcmcscale=0.1))
        return pars
        
    def load_template(self, gpars):
        pad = 15 # Angstroms for wlsol error (.5) and bc (4) and stellar absolute RV if init_guess=1, (10)
        if gpars['do_init_guess']:
            print('Loading in Synthetic Stellar Template', flush=True)
            template_raw = np.load(self.input_file)
            wave_init, flux_init = template_raw['wave'], template_raw['flux']
            good = np.where((wave_init > gpars['wave_left']-pad) & (wave_init < gpars['wave_right']+pad))[0]
            wave, flux = wave_init[good], flux_init[good]
            wave_star = np.linspace(wave[0], wave[-1], num=gpars['n_model_pix'])
            interp_fun = scipy.interpolate.CubicSpline(wave, flux, extrapolate=False, bc_type='not-a-knot')
            flux_star = interp_fun(wave_star)
            template = np.array([wave_star, flux_star]).T
        else:
            wave_star = np.linspace(gpars['wave_left']-pad, gpars['wave_right']+pad, num=gpars['n_model_pix'])
            template = np.array([wave_star, np.ones(gpars['n_model_pix'])]).T
        return template
    
    def modify(self, v):
        self.enabled = v
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'
        
class TelluricModelTAPAS(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        
        self.name = blueprint['name']
        self.base_par_names = ['_vel']
        
        if len(blueprint['components'].keys()) == 0:
            self.species = []
            self.enabled = False
        else:
            self.enabled = True
            self.species = list(blueprint['components'].keys())
            self.n_species = len(self.species)
            self.species_enabled = OrderedDict()
            self.species_input_files = OrderedDict()
            self.species_airmass_correlation = OrderedDict()
            for itell in range(self.n_species):
                self.species_input_files[self.species[itell]] = blueprint['components'][self.species[itell]]['input_file']
                self.species_enabled[self.species[itell]] = True
                self.base_par_names.append('_' + self.species[itell] + '_depth')
        self.par_names = [self.name + s for s in self.base_par_names]
        
    # Telluric templates is a dictionary of templates.
    # Keys are the telluric names, values are nx * 2 arrays with columns wave, mean_flux
    def build(self, pars, templates, wave_final):
        if self.enabled:
            flux = np.ones(wave_final.size, dtype=np.float64)
            for i in range(self.n_species):
                if self.species_enabled[self.species[i]]:
                    flux *= self.build_single_species(pars, templates, self.species[i], i, wave_final)
            return flux
        else:
            return self.build_fake(wave_final.size)
        
    def build_single_species(self, pars, templates, single_species, species_i, wave_final):
        shift = pars[self.par_names[0]].value
        depth = pars[self.par_names[species_i + 1]].value
        wave, flux = templates[single_species][:, 0], templates[single_species][:, 1]
        flux = flux ** depth
        wave_shifted = wave * np.exp(shift / cs.c)
        return np.interp(wave_final, wave_shifted, flux, left=flux[0], right=flux[-1])
    
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['vel'][1], minv=blueprint['vel'][0], maxv=blueprint['vel'][2], mcmcscale=0.1))
        for i in range(self.n_species):
            pars.append(Parameter(name=self.par_names[i+1], value=blueprint['components'][self.species[i]]['depth'][1], minv=blueprint['components'][self.species[i]]['depth'][0], maxv=blueprint['components'][self.species[i]]['depth'][2], mcmcscale=0.1))
        return pars
    
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def modify(self, component, v):
        self.species_enabled[component] = v
        if not np.any(list(self.species_enabled.values())) and not v:
            self.enabled = False
            
    def load_template(self, gpars):
        templates = OrderedDict()
        for i in range(self.n_species):
            print('Loading in Telluric Template For ' + self.species[i], flush=True)
            template = np.load(self.species_input_files[self.species[i]])
            wave, flux = template['wave'], template['flux']
            good = np.where((wave > gpars['wave_left'] - 1) & (wave < gpars['wave_right'] + 1))[0]
            wave, flux = wave[good], flux[good]
            templates[self.species[i]] = np.array([wave, flux]).T
        return templates
        
    def __repr__(self):
        ss = ' Model Name: ' + self.name + ', Species: ['
        for tell in self.species_enabled:
            if self.species_enabled[tell]:
                ss += tell + ': Active, '
            else:
                ss += tell + ': Deactive, '
        ss = ss[0:-2]
        return ss + ']'
    
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
        
    
class ComplexFringingModel(SpectralComponent):
    
    def __init__(self, order_num, blueprint, gpars):
        self.enabled = True
        self.name = blueprint['name']
        self.n_delay = blueprint['n_delay']
        self.k2 = 136
        self.beta_0 = 71.5709691870135 * np.pi / 180
        self.theta_b = 71.567952760064 * np.pi / 180 # For an ~R3 grating, pretty close!
        self.lam0_zero = blueprint['fringing_2_reflection'][order_num]
        self.lam2_zero = blueprint['fringing_2_set_point'][order_num]
        self.base_par_names = ['_amp', '_lam0', '_lam2', '_phase', '_tilt']
        self.par_names = [self.name + s for s in self.base_par_names]
        
    def build(self, pars, wave_final):
        if self.enabled:
            amp = pars[self.par_names[0]].value
            lam0 = pars[self.par_names[1]].value + self.lam0_zero
            lam2 = pars[self.par_names[2]].value + self.lam2_zero
            phase = pars[self.par_names[3]].value
            tilt = pars[self.par_names[4]].value
            
            D = (self.k2 * np.pi) / (np.cos(self.beta_0 - np.arcsin(lam2 / lam0 * (np.sin(self.beta_0) + np.sin(self.theta_b)) - np.sin(self.theta_b))) / lam2 - 1.0 / lam2)
            delta = D * (np.cos(self.beta_0 - np.arcsin(wave_final / lam0 * (np.sin(self.beta_0) + np.sin(self.theta_b)) - np.sin(self.theta_b))) / wave_final - 1.0 / wave_final)
            
            y = (1 / tilt) * np.arctan(tilt * np.sin(delta + phase) / (1 - tilt * np.cos(delta + phase)))
    
            # Normalize
            ymin = np.min(y)
            ymax = np.max(y)
            
            yrange = ymax - ymin
            ymid = ymin + yrange / 2
            ynorm = (y - ymid) / yrange
            fringing = amp * ynorm + 1

            if np.any(~np.isfinite(fringing)):
                fringing = self.build_fake(wave_final.size)
            return fringing
        else:
            return self.build_fake(wave_final.size)
    
    def build_fake(self, n):
        return np.ones(n, dtype=float)
    
    def modify(self, v):
        self.enabled = v
        
    def initialize_parameters(self, blueprint, gpars):
        pars = []
        pars.append(Parameter(name=self.par_names[0], value=blueprint['amp'][1], minv=blueprint['amp'][0], maxv=blueprint['amp'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[1], value=blueprint['lam0'][1], minv=blueprint['lam0'][0], maxv=blueprint['lam0'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[2], value=blueprint['lam2'][1], minv=blueprint['lam2'][0], maxv=blueprint['lam2'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[3], value=blueprint['phase'][1], minv=blueprint['phase'][0], maxv=blueprint['phase'][2], mcmcscale=0.1))
        pars.append(Parameter(name=self.par_names[4], value=blueprint['tilt'][1], minv=blueprint['tilt'][0], maxv=blueprint['tilt'][2], mcmcscale=0.1))
        return pars
        
    def __repr__(self):
        return ' Model Name: ' + self.name + ' [Active: ' + str(self.enabled) + ']'

#@jitclass([('name', numba.types.string),
#           ('value', numba.types.float64),
#           ('minv', numba.types.float64),
#           ('maxv', numba.types.float64),
#           ('vary', numba.types.boolean),
#           ('mcmcscale', numba.types.float64)])
class Parameter:

    def __init__(self, name, value, minv=-np.inf, maxv=np.inf, vary=True, mcmcscale=0.5):
        
        self.name = name
        self.value = value
        self.minv = minv
        self.maxv = maxv
        self.vary = vary
        self.mcmcscale = mcmcscale

    def __repr__(self):
        if self.vary:
            return '(Parameter)  Name: ' + self.name + ' | Value: ' + str(self.value) + ' | Bounds: [' + str(self.minv) + ', ' + str(self.maxv) + ']' + ' | MCMC scale: ' + str(self.mcmcscale)
        else:
            return '(Parameter)  Name: ' + self.name + ' | Value: ' + str(self.value) + ' (Locked) | Bounds: [' + str(self.minv) + ', ' + str(self.maxv) + ']' + ' | MCMC scale: ' + str(self.mcmcscale)
    
    def setv(self, value=None, minv=None, maxv=None, vary=None, mcmcscale=None):
        if value is not None:
            self.value = value
        if minv is not None:
            self.minv = minv
        if maxv is not None:
            self.maxv = maxv
        if vary is not None:
            self.vary = vary
        if mcmcscale is not None:
            self.mcmcscale = mcmcscale

class Parameters(OrderedDict):

    def __init__(self):
        super().__init__()

    def add_from_values(self, name, value, minv=-np.inf, maxv=np.inf, vary=True, mcmcscale=0.5):
        if name in self.keys():
            self[name].name = name
            self[name].value = value
            self[name].minv = minv
            self[name].maxv = maxv
            self[name].vary = vary
            self[name].mcmcscale = mcmcscale
        else:
            self[name] = Parameter(name=name, value=value, minv=minv, maxv=maxv, vary=vary, mcmcscale=mcmcscale)
            
            
    def add_from_parameter(self, parameter):
        self[parameter.name] = parameter

    def to_numpy(self, kind='all'):
        
        n = len(self)
        if kind == 'all':
            vals = np.empty(n, dtype=np.float64)
            brute_steps = np.empty(n, dtype=np.float64)
            min_vals = np.empty(n, dtype=np.float64)
            max_vals = np.empty(n, dtype=np.float64)
            vary = np.empty(n, dtype=bool)
            names = np.empty(n, dtype='<U50')
            mcmcscales = np.empty(n, dtype=np.float64)
            for i, key in enumerate(self):
                names[i] = self[key].name
                vals[i] = self[key].value
                min_vals[i] = self[key].minv
                max_vals[i] = self[key].maxv
                vary[i] = self[key].vary
                mcmcscales[i] = self[key].mcmcscale
            return names, vals, min_vals, max_vals, vary, mcmcscales
        elif kind == 'values':
            vals = np.empty(n, dtype=np.float64)
            for i in range(n):
                vals[i] = self[key].value
            return vals
        elif kind == 'varies':
            vals = np.empty(n, dtype=bool)
            for i in range(n):
                vals[i] = self[key].vary
            return vals
        elif kind == 'mcmcscale':
            vals = np.empty(n, dtype=np.float64)
            for i in range(n):
                vals[i] = self[key].mcmcscale
            return vals
        
    def from_numpy(names, values, minvs=None, maxvs=None, varies=None, mcmcscales=None):
        p = Parameters()
        n = values.size
        if varies is None:
            varies = np.ones(n).astype(bool)
        if minvs is None:
            minvs = np.full(n, fill_value=-np.inf)
        if maxvs is None:
            maxvs = np.full(n, fill_value=np.inf)
        if mcmcscales is None:
            mcmcscales = np.full(n, fill_value=0.5)
        for i in range(n):
            p.add_from_values(names[i], values[i], minv=minvs[i], maxv=maxvs[i], vary=varies[i], mcmcscale=mcmcscales[i])
        return p
            
    def pretty_print(self):
        for key in self.keys():
            print(self[key])