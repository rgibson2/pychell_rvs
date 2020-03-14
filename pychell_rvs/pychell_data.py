# Python built in modules
from collections import OrderedDict
from abc import ABC, abstractmethod # Abstract classes
import glob # File searching
import sys # sys utils
from barycorrpy import get_BC_vel # BC velocity correction
from barycorrpy.utc_tdb import JDUTC_to_BJDTDB
import pdb # debugging
stop = pdb.set_trace

# Science/math
import numpy as np # Math, Arrays

# LLVM
from numba import jit, njit
import numba

# Astropy
from astropy.time import Time
from astropy.io import fits

# User defined/pip modules
import pychell_rvs.pychell_math as pcmath # mathy equations

class SpecData(ABC):

    def __init__(self, input_file, order_num, spec_num, gpars):
        
        # Store the input file, spec, and order num
        self.input_file = input_file
        self.order_num = order_num
        self.spec_num = spec_num
        
        # Parse the observation details for this order (probably order independent)
        self.parse(gpars)
        
        # Force bad cropped pix to be zero just in case
        self.badpix[0:gpars['crop_pix'][0]] = 0
        self.badpix[-gpars['crop_pix'][1]:] = 0
        
    @abstractmethod
    def parse(self, gpars):
        pass

class SpecDataCHIRON(SpecData):
    
    def __init__(self, input_file, order_num, spec_num, gpars):
        
        super().__init__(input_file, order_num, spec_num, gpars)
        
    def parse(self, gpars):
        
        # Load the flux, flux unc, and bad pix arrays
        #fname = glob.glob(gpars['data_input_path'] + self.input_file[:-4] + '_ord' + str(self.order_num+1) + '.npz')[0]
        #data_ = np.load(fname, allow_pickle=True)
        fits_data = fits.open(gpars['data_input_path'] + self.input_file)[0]
        fits_data.verify('fix')
        
        self.wave_grid, self.flux = fits_data.data[self.order_num, :, 0].astype(np.float64), fits_data.data[self.order_num, :, 1].astype(np.float64)
        self.flux /= pcmath.weighted_median(self.flux, med_val=0.98)
        
        # For CHIRON, generate a dumby uncertainty grid and a bad pix array that will be updated or used
        self.flux_unc = np.zeros(gpars['n_data_pix'], dtype=np.float64) + 1E-3
        self.badpix = np.ones(gpars['n_data_pix'], dtype=np.float64)
        
        # Observation Details, silly indexing for a zero dimensional array
        # obs_details is kept in memory and saved later for sanity.
        self.obs_details = dict(fits_data.header)
        
        # Extract specific keys from the observation details used in the code.
        # NOTE: Figure out where exp meter info is.
        self.SNR = float(self.obs_details['EMAVGSQ'])
        self.JD = Time(self.obs_details['DATE'].replace('T', ' '), scale='utc').jd + float(self.obs_details['EXPTIME']) / (2 * 3600 * 24)
        
        # Calculate barycenter velocities
        if gpars['bary_corr_file'] is None and gpars['bary_corrs'] is None:
            self.bary_corr = get_BC_vel(JDUTC=self.JD, starname=gpars['star_name'].replace('_', ' '), obsname=gpars['observatory'])[0][0]
            self.BJD = JDUTC_to_BJDTDB(JDUTC=self.JD, starname=gpars['star_name'].replace('_', ' '), obsname=gpars['observatory'])[0][0]
        else:
            self.bary_corr = gpars['bary_corrs'][self.spec_num]
            self.BJD = gpars['BJDS'][self.spec_num]


class SpecDataiSHELL(SpecData):

    def __init__(self, input_file, order_num, spec_num, gpars):
        
        # Call the super class
        super().__init__(input_file, order_num, spec_num, gpars)
        
    def parse(self, gpars):
        
        # Load the flux, flux unc, and bad pix arrays
        
        fits_data = fits.open(gpars['data_input_path'] + self.input_file)[0]
        fits_data.verify('fix')
        
        self.flux, self.flux_unc, self.badpix = fits_data.data[self.order_num, :, 0].astype(np.float64), fits_data.data[self.order_num, :, 1].astype(np.float64), fits_data.data[self.order_num, :, 2].astype(np.float64)
        
        # Flip the data so wavelength is increasing
        self.flux = self.flux[::-1]
        self.badpix = self.badpix[::-1]
        self.flux_unc = self.flux_unc[::-1]
        
        # Observation Details, silly indexing for a zero dimensional array
        # obs_details is kept in memory and saved later for sanity.
        self.obs_details = dict(fits_data.header)
        
        # Extract specific keys from the observation details used in the code.
        self.SNR = float(self.obs_details['SNR'])
        self.JD = float(self.obs_details['TCS_UTC']) + 2400000.5 + float(self.obs_details['ITIME']) / (2 * 3600 * 24)
        
        # Calculate barycenter velocities
        if gpars['bary_corr_file'] is None and gpars['bary_corrs'] is None:
            self.bary_corr = get_BC_vel(JDUTC=self.JD, starname=gpars['star_name'].replace('_', ' '), obsname=gpars['observatory'])[0][0]
            self.BJD = JDUTC_to_BJDTDB(JDUTC=self.JD, starname=gpars['star_name'].replace('_', ' '), obsname=gpars['observatory'])[0][0]
        else:
            self.bary_corr = gpars['bary_corrs'][self.spec_num]
            self.BJD = gpars['BJDS'][self.spec_num]

#class SpecDataPARVI(SpecData):
    
#    def __init__(self, input_file, order_num, spec_num, gpars, init_obs_details=False, init_data=False):
        
#        super().__init__(input_file, order_num, spec_num, gpars)
        
        
    # Below is code to parse the PARVI data for what I think is the preferred data format.
    # Uncomment and tweak at will!
    # def parse(self, gpars):
        
        # Load the flux, flux unc, and bad pix arrays. Also load the known wavelength grid for a starting point
        # fits_data = fits.open(gpars['data_input_path'] + self.input_file)[0]
        #self.wave_grid, self.flux, self.flux_unc = fits_data.data[self.order_num, :, 0].astype(np.float64), fits_data.data[self.order_num, :, 5].astype(np.float64), fits_data.data[self.order_num, :, 6].astype(np.float64)
        
        # Normalize according to 98th percentile in flux
        #continuum = pcmath.weighted_median(self.flux, med_val=0.98)
        #self.flux /= continuum
        #self.flux_unc /= continuum
        
        # Create bad pix array, further cropped later
        #self.badpix = np.ones(gpars['n_data_pix'], dtype=np.float64)
        # bad = np.where(self.flux)
        #if bad.size > 0:
        #    self.badpix[bad] = 0
        
        # Convert wavelength grid to Angstroms, required!
        #self.wave_grid *= 10
        
        #self.JD = float(fits_data.header['JD'])
        
        #if gpars['bary_corr_file'] is None and gpars['bary_corrs'] is None:
        #    self.bary_corr = get_BC_vel(JDUTC=self.JD, starname=gpars['star_name'].replace('_', ' '), obsname=gpars['observatory'])[0][0]
        #    self.BJD = JDUTC_to_BJDTDB(JDUTC=self.JD, starname=gpars['star_name'].replace('_', ' '), obsname=gpars['observatory'])[0][0]
        #else:
        #    self.bary_corr = gpars['bary_corrs'][self.spec_num]
        #    self.BJD = gpars['BJDS'][self.spec_num]
        