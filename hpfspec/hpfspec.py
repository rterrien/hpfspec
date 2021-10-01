from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io
import scipy.interpolate
import os
import crosscorr
from collections import OrderedDict
from . import stats_help
from . import utils
from . import spec_help
from . import rotbroad_help
from . import target
from . import fitting_utils
from . import model_utils
DIRNAME = os.path.dirname(__file__)
PATH_FLAT_DEBLAZED = os.path.join(DIRNAME,"data/hpf/flats/alphabright_fcu_sept18_deblazed.fits")
PATH_FLAT_BLAZED = os.path.join(DIRNAME,"data/hpf/flats/alphabright_fcu_sept18.fits")
PATH_TELLMASK = os.path.join(DIRNAME,"data/masks/telluric/telfit_telmask_conv17_thres0.995_with17area.dat")
PATH_SKYMASK = os.path.join(DIRNAME,"data/masks/sky/HPF_SkyEmmissionLineWavlMask_broadened_11111_Compressed.txt")
PATH_CCF_MASK = crosscorr.mask.HPFGJ699MASK
PATH_WAVELENGTH = os.path.join(DIRNAME,"data/hpf/wavelength_solution/LFC_wavecal_scifiber_v2.fits")
PATH_TARGETS = target.PATH_TARGETS

class HPFSpectrum(object):
    """
    Yet another HPF Spectrum object. Can work with deblazed spectra.
    
    EXAMPLE:
        H = HPFSpectrum(fitsfiles[1])
        H.plot_order(14,deblazed=True)
    """
    path_flat_deblazed = PATH_FLAT_DEBLAZED
    path_flat_blazed = PATH_FLAT_BLAZED
    path_tellmask = PATH_TELLMASK
    path_skymask = PATH_SKYMASK
    path_ccf_mask = PATH_CCF_MASK
    path_wavelength_solution = PATH_WAVELENGTH
    
    def __init__(self,filename,targetname='',deblaze=True,ccf_redshift=True,tell_err_factor=1.,sky_err_factor=1.,sky_scaling_factor=1.0,
                auto_renorm=True):
        self.filename = filename
        self.basename = filename.split(os.sep)[-1]
        self.sky_scaling_factor = sky_scaling_factor
        
        # Read science frame
        self.hdu = astropy.io.fits.open(filename)
        self.header = self.hdu[0].header
        self.exptime = self.header["EXPLNDR"]
        self.object = self.header["OBJECT"]
        try: self.qprog = self.header["QPROG"]
        except Exception as e: self.qprog = np.nan
        midpoint_keywords = ['JD_FW{}'.format(i) for i in range(28)]
        self.jd_midpoint = np.median(np.array([self.header[i] for i in midpoint_keywords]))
        
        # Read Flat
        self.hdu_flat = astropy.io.fits.open(self.path_flat_deblazed)
        self.header_flat = self.hdu_flat[0].header
        self.flat_sci = self.hdu_flat[1].data
        self.flat_sky = self.hdu_flat[2].data
        
        self.e_sci = np.sqrt(self.hdu[4].data)*self.exptime
        self.e_sky = np.sqrt(self.hdu[5].data)*self.exptime*self.sky_scaling_factor
        self.e_cal = np.sqrt(self.hdu[6].data)*self.exptime
        self.e = np.sqrt(self.hdu[4].data + self.hdu[5].data)*self.exptime

        self.f_sky = (self.hdu[2].data*self.exptime/self.flat_sky)*self.sky_scaling_factor
        self._f_sky = self.hdu[2].data*self.exptime
        
        self._f_sci = self.hdu[1].data*self.exptime
        self.f_sci = self.hdu[1].data*self.exptime/self.flat_sci
        self.f = self.f_sci - self.f_sky

        # Read in wavelength
        try:
            self.w = self.hdu[7].data
            self.drift_corrected = True
        except Exception as e:
            print('Defaulting to fixed wavelength')
            self.w = astropy.io.fits.getdata(self.path_wavelength_solution)
            self.drift_corrected = False

        # Inflate errors around tellurics and sky emission lines
        mt = self.get_telluric_mask()
        ms = self.get_sky_mask()
        if tell_err_factor == sky_err_factor:
            mm = mt | ms
            self.e[mm] *= tell_err_factor
        else:
            self.e[mt] *= tell_err_factor
            self.e[ms] *= sky_err_factor

        self.sn5 = np.nanmedian(self.f[5]/self.e[5])
        self.sn6 = np.nanmedian(self.f[6]/self.e[6])
        self.sn14 = np.nanmedian(self.f[14]/self.e[14])
        self.sn15 = np.nanmedian(self.f[15]/self.e[15])
        self.sn16 = np.nanmedian(self.f[16]/self.e[16])
        self.sn17 = np.nanmedian(self.f[17]/self.e[17])
        self.sn18 = np.nanmedian(self.f[18]/self.e[18])
        self.sn = self.f/self.e
        
        if targetname=='':
            targetname = self.object
        self.target = target.Target(targetname)
        self.bjd, self.berv = self.target.calc_barycentric_velocity(self.jd_midpoint,'McDonald Observatory')

        print('Barycentric shifting')
        self.rv = 0.
        if ccf_redshift:
            v = np.linspace(-125,125,1501)
            _, rabs = self.rvabs_for_orders(v,orders=[5],plot=False)
            self.rv = np.median(rabs)
            self.redshift(rv=self.rv)

        if deblaze:
            self.deblaze()
        #self.hdu.close()

        if auto_renorm:
            self.renormalize_simple()


    def __repr__(self):
        return 'HPFSpec({},sn18={:0.1f})'.format(self.object,self.sn18)

    def get_telluric_mask(self,w=None,o=None):
        """
        Return telluric mask interpolated onto a given grid.
        
        INPUT:
            w - wavelength grid to interpolate on
            o - 
            
        OUTPUT:
        
        EXAMPLE:
        """
        if w is None:
            w = self.w
        mask = np.genfromtxt(self.path_tellmask)
        m = scipy.interpolate.interp1d(mask[:,0],mask[:,1])(w) > 0.01
        if o is None:
            return m
        else:
            m[o]

    def get_sky_mask(self,w=None,o=None):
        """
        Return sky mask interpolated onto a given grid.
        
        INPUT:
            w - wavelength grid to interpolate on
            o - 
            
        OUTPUT:
        
        EXAMPLE:
        """
        if w is None:
            w = self.w
        mask = np.genfromtxt(self.path_skymask)
        m = scipy.interpolate.interp1d(mask[:,0],mask[:,1],fill_value="extrapolate")(w) > 0.01
        if o is None:
            return m
        else:
            m[o]

    def calculate_ccf_for_orders(self,v,orders=[3,4,5,6,14,15,16,17,18],plot=True):
        """
        Calculate CCF for given orders

        INPUT:
            v - velocities

        EXAMPLE:
            H0 = astropylib.hpfspec.HPFSpectrum(df[df.name=="G_9-40"].filename.values[0])
            v = np.linspace(-0,25,161)
            H0.calculate_ccf_for_orders(v)

        NOTES: Calculates on barycentric shifted (NOT ABS RV SHIFTED) and undeblazed version
        """

        self.M = crosscorr.mask.Mask(self.path_ccf_mask)
        w = spec_help.redshift(self.w,vo=self.berv,ve=0.)
        self.ccf = crosscorr.calculate_ccf_for_hpf_orders(w,self.f,v,self.M,berv=0.,orders=orders,plot=plot)
        return self.ccf

    def rvabs_for_orders(self,v,orders,v2_width=25.0,plot=True, ax=None, bx=None, verbose=True, n_points=40):
        """
        Calculate absolute RV for different orders using two iterations (course + fine fitting Gaussian)

        INPUT:
            

        OUTPUT:
            rv1 - from 1st course iteration
            rv2 - from 2nd fine iteration

        EXAMPLE:
            v = np.linspace(-125,125,161)
            H0.rvabs_for_orders(v,orders=[4,5,6])

        NOTES: Calculates on barycentric shifted (NOT ABS RV SHIFTED) and undeblazed version
        """
        if (self.object == 'HD_24238') or (self.object == 'GJ_3507') or (self.object == 'LHS_3353') or (self.object == 'LSPM_J0255+2652E'):
            orders = [3]
        self.M = crosscorr.mask.Mask(self.path_ccf_mask)
        w = spec_help.redshift(self.w,vo=self.berv,ve=0.)
        rv1, rv2 = spec_help.rvabs_for_orders(w,self.f,orders,v,self.M,v2_width,plot,ax,bx,verbose,n_points)
        return rv1, rv2

    def resample_order(self,ww,p=None,vsini=None,shifted=True):
        """
        Resample order to a given grid. Useful when comparing spectra and calculating chi2

        NOTES:
            dt = 0.04/4 = 0.01 for HPF
        """
        
        if shifted: w = self.w_shifted
        else: w = self.w
            
        m = (w> ww.min()-2.)&(w<ww.max()+2.)
        w = w[m]
        f = self.f_debl[m]
        e = self.e_debl[m]
        m = np.isfinite(f)
        w = w[m]
        f = f[m]
        e = e[m]
        
        ff = scipy.interpolate.interp1d(w,f,kind='linear')(ww)
        ee = scipy.interpolate.interp1d(w,e,kind='linear')(ww)

        if p is not None:
            print('Applying Chebychev polynomial',p)
            ff*=np.polynomial.chebyshev.chebval(ww,p)
            ee*=np.polynomial.chebyshev.chebval(ww,p)
        if vsini is not None:
            print('Applying vsini: {}km/s'.format(vsini))
            ff = rotbroad_help.broaden(ww,ff,vsini)
        return ff, ee
            
    def deblaze(self):
        """
        Deblaze spectrum, make available with self.f_debl
        """
        hdu = astropy.io.fits.open(self.path_flat_blazed)
        self.f_sci_debl = self.hdu[1].data*self.exptime/hdu[1].data
        self.f_sky_debl = self.hdu[2].data*self.exptime/hdu[2].data
        self.f_debl = self.f_sci_debl-self.f_sky_debl*self.sky_scaling_factor
        for i in range(28): 
            self.f_debl[i] = self.f_debl[i]/np.nanmedian(self.f_debl[i])
        self.e_debl = self.f_debl/self.sn

    def renormalize_simple(self,which='f_debl'):
        """
        Use a simple renormalization technique to flatten the orders.
        Maxfilter -> Gaussfilter
        """
        if which not in ['f_debl','f_sci_debl','f_sky_debl','f_sci','f_cal','f_sky']:
            raise(ValueError)
        fluxToCorrect = getattr(self,which)
        fluxCorrected = np.full_like(fluxToCorrect,np.nan)
        trends = np.full_like(fluxToCorrect,np.nan)
        for oi in range(28):
            flux = fluxToCorrect[oi,:]
            corrected, trend = spec_help.detrend_maxfilter_gaussian(flux,n_max=300,n_gauss=500)
            fluxCorrected[oi,:] = corrected
            trends[oi,:] = trend
        if hasattr(self,'trends_removed'):
            self.trends_removed[which] = trends
        else:
            self.trends_removed = OrderedDict()
            self.trends_removed[which] = trends
        setattr(self,which,fluxCorrected)
            
    def redshift(self,berv=None,rv=None):
        """
        Redshift spectrum correcting for both berv and rv

        INPUT:
            berv in km/s
            rv in km/s
        """
        if berv is None:
            berv = self.berv
        if rv is None:
            rv = self.target.rv
        print('berv={},rv={}'.format(berv,rv))
        self.w_shifted = spec_help.redshift(self.w,vo=berv,ve=rv)
        self.rv = rv

    def rotbroad(self,ww,vsini,eps=0.6,plot=False):
        """
        Broaden with vsini
        """
        ff, ee = self.resample_order(ww)
        _f = rotbroad_help.broaden(ww,ff,vsini,u1=eps)
        return _f
        
    def plot_order(self,o,deblazed=False,shifted=False,ax=None,color=None,plot_shaded=True,alpha=1.):
        """
        Plot spectrum deblazed or not
        
        EXAMPLE:
            
        """
        mask_tell = np.genfromtxt(self.path_tellmask)
        mask_sky = np.genfromtxt(self.path_skymask)
        mt = self.get_telluric_mask()
        ms = self.get_sky_mask()

        if ax is None:
            fig, ax = plt.subplots(dpi=200)
        if deblazed:
            self.deblaze()
            f = self.f_debl[o]
            e = self.e_debl[o]
            f_mt = mask_tell[:,1]
            f_ms = mask_sky[:,1]
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        else:
            f = self.f[o]
            e = self.e[o]
            f_mt = mask_tell[:,1]*np.nanmax(f)
            f_ms = mask_sky[:,1]*np.nanmax(f)
            if shifted:
                w = self.w_shifted[o]
                w_mt = spec_help.redshift(mask_tell[:,0],vo=self.berv,ve=self.rv)
                w_ms = spec_help.redshift(mask_sky[:,0],vo=self.berv,ve=self.rv)
            else:
                w = self.w[o]
                w_mt = mask_tell[:,0]
                w_ms = mask_sky[:,0]
        ax.errorbar(w,f,e,marker='o',lw=0.5,capsize=2,mew=0.5,elinewidth=0.5,markersize=2,color=color,alpha=alpha)
        if plot_shaded:
            ax.plot(w[mt[o]],f[mt[o]],lw=0,marker='.',markersize=2,color='blue')
            ax.plot(w[ms[o]],f[ms[o]],lw=0,marker='.',markersize=2,color='red')
            ax.fill_between(w_mt,f_mt,color='blue',alpha=0.1)
            ax.fill_between(w_ms,f_ms,color='red',alpha=0.1)

        utils.ax_apply_settings(ax)
        ax.set_title('{}, {}, order={}, SN18={:0.2f}\nBJD={}, BERV={:0.5f}km/s'.format(self.object,
                                                                                       self.basename,o,self.sn18,self.bjd,self.berv))
        ax.set_xlabel('Wavelength [A]')
        ax.set_ylabel('Flux')
        ax.set_xlim(np.nanmin(self.w[o]),np.nanmax(self.w[o]))

    def find_peaks_order(self,oi,fl=None,w=None,prominence=0.1,width=(0,8),
                         pixel_to_wl_interpolation_kind='cubic',fill_value=np.nan,
                         fit_width_kms=None):
        """ Find peaks in a spectral order
        
        Use the scipy.signal.find_peaks routine to locate lines in a spectral order.
        Defaults to using f_debl and stellar rest frame wavelengths.
        Presently the precision is only pixel-level, so interpolation is overkill.
        
        Parameters
        ----------
        oi : {int}
            Order index
        fl : {ndarray}, optional
            1D array of fluxes. if provided, oi is ignored
        w : {ndarray}, optional
            1D array of wavelengths [ang]. if provided, oi is ignored
        prominence : {float}, optional
            Height above surroundings. Argument to scipy.signal.find_peaks (the default is 0.1)
        width : {tuple}, optional
            Bounds on peak width. Argument to scipy.signal.find_peaks (the default is (0,8))
        pixel_to_wl_interpolation_kind : {str}, optional
            Interpolation to use for converting pixels to wavelength (the default is 'cubic')
        fill_value : {number}, optional
            Fill value in interpolation (the default is np.nan)
        fit_width_kms : {float}, optional
            Fit the lines to find a more precise centroid. [km/s]
            Skip this by not providing a number.
        """
        if fl is None:
            fl = self.f_debl[oi]
        if w is None:
            w = self.w_shifted[oi]
        # Find pixel centers and interpolate to wavelength values
        pixel_peaks = scipy.signal.find_peaks(-fl,prominence=prominence,width=width)[0] # propertes in [1]
        xx = np.arange(2048)
        wl_peaks = scipy.interpolate.interp1d(xx,w,kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,bounds_error=False)(pixel_peaks)

        # If fit is not requested (i.e. fit_width_kms is None), just return pixel centers
        if fit_width_kms is None:
            return(wl_peaks)
        
        # Fit the centers using simple assumptions
        fitted_centers = []
        dwl_pix = np.nanmedian(np.diff(w))
        dv_pix = dwl_pix / np.nanmedian(w) * 3e5
        fit_width_pix = fit_width_kms / dv_pix
        for pi,wi in zip(pixel_peaks,wl_peaks):
            fitout = fitting_utils.fitProfile(w,fl,pi,fit_width_pix,func='fgauss_const',p0=[wi,-0.1,1.,1.])
            fitted_centers.append(fitout['centroid'])
        fitted_centers = np.array(fitted_centers)
        return(fitted_centers)

    def fit_peaks_order(self,oi,wl_peaks,fl=None,w=None,
                         pixel_to_wl_interpolation_kind='cubic',fill_value=np.nan,
                         fit_width_kms=8.):
        """ Fit peaks in a spectral order
        
        If you already have peaks roughly located in wavelength, use this routine to
        fit their locations more precisely.
        
        Parameters
        ----------
        oi : {int}
            Order index
        wl_peaks : [list]
            List of peak wavelengths in angstroms
        fl : {ndarray}, optional
            1D array of fluxes. if provided, oi is ignored
        w : {ndarray}, optional
            1D array of wavelengths [ang]. if provided, oi is ignored
        pixel_to_wl_interpolation_kind : {str}, optional
            Interpolation to use for converting pixels to wavelength (the default is 'cubic')
        fill_value : {number}, optional
            Fill value in interpolation (the default is np.nan)
        fit_width_kms : {float}, optional
            Fit the lines to find a more precise centroid. [km/s]
            Skip this by not providing a number.
        """
        if fl is None:
            fl = self.f_debl[oi]
        if w is None:
            w = self.w_shifted[oi]
        # Find pixel centers and interpolate to wavelength values
        #pixel_peaks = scipy.signal.find_peaks(-fl,prominence=prominence,width=width)[0] # propertes in [1]
        xx = np.arange(2048)
        pixel_peaks = scipy.interpolate.interp1d(w,xx,kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,bounds_error=False)(wl_peaks)
        #wl_peaks = scipy.interpolate.interp1d(xx,w,kind=pixel_to_wl_interpolation_kind,fill_value=fill_value,bounds_error=False)(pixel_peaks)

        # If fit is not requested (i.e. fit_width_kms is None), just return pixel centers
        if fit_width_kms is None:
            return(wl_peaks)
        
        # Fit the centers using simple assumptions
        fitted_centers = []
        dwl_pix = np.nanmedian(np.diff(w))
        dv_pix = dwl_pix / np.nanmedian(w) * 3e5
        fit_width_pix = fit_width_kms / dv_pix
        for pi,wi in zip(pixel_peaks,wl_peaks):
            fitout = fitting_utils.fitProfile(w,fl,pi,fit_width_pix,func='fgauss_const',p0=[wi,-0.1,1.,1.])
            fitted_centers.append(fitout['centroid'])
        fitted_centers = np.array(fitted_centers)
        return(fitted_centers)
    
    def measure_ew(self,lower=None,upper=None,feature=None,w=None,fl=None,diag=False,const_continuum_regions=[],
                   slope_continuum_regions=[]):
        if ((lower is None) or (upper is None)) and (feature is None):
            raise ValueError
        if feature is not None:
            lower = feature.lower.value
            upper = feature.upper.value

        if fl is None:
            fl = self.f_debl
        if w is None:
            w = self.w_shifted
        norders, npix = np.shape(fl)

        for oi in range(norders):
            wls = w[oi]
            omin, omax = np.nanmin(wls), np.nanmax(wls)
            if (lower > omin) and (upper < omax):
                o_use = oi
                #print('Using oi {}'.format(oi))
                break
        if o_use is None:
            raise ValueError('Feature not entirely in orders')
        fl_use = fl[o_use]
        wl_use = w[o_use]

        if len(slope_continuum_regions) > 0:
            print('Renormalizing to slope')
            ci_use = []
            for clim in slope_continuum_regions:
                lower_lim = clim[0]
                upper_lim = clim[1]
                tmp_i = np.nonzero((wl_use >= lower_lim) & (wl_use <= upper_lim))[0]
                if len(tmp_i) > 0:
                    for j in tmp_i:
                        ci_use.append(j)
                else:
                    print('No suitable pixels found for {:.2f} to {:.2f}'.format(lower_lim,upper_lim))
            ww_fit = wl_use[ci_use]
            ff_fit = fl_use[ci_use]
            pp = np.polyfit(ww_fit,ff_fit,1)
            norm = np.polyval(pp,wl_use)
            fl_use = fl_use / norm
        elif len(const_continuum_regions) > 0:
            print('Renormalizing to constant value')
            ci_use = []
            for clim in const_continuum_regions:
                lower_lim = clim[0]
                upper_lim = clim[1]
                tmp_i = np.nonzero((wl_use >= lower_lim) & (wl_use <= upper_lim))[0]
                if len(tmp_i) > 0:
                    for j in tmp_i:
                        ci_use.append(j)
                else:
                    print('No suitable pixels found for {:.2f} to {:.2f}'.format(lower_lim,upper_lim))
            if len(ci_use) > 0:
                new_norm = astropy.stats.biweight_location(fl_use[ci_use])
                print('New norm: {:.3}'.format(new_norm))
                fl_use = fl_use / new_norm
            else:
                print('No renorm pixels found, skipping')

        out = spec_help.calculate_ew(wl_use,fl_use,lower,upper)

        if not diag:
            return(out)
        else:
            inds = np.nonzero( (wl_use > lower) & (wl_use < upper) )
            outdict = {'ew':out,
                       'order':o_use,
                       'inds':inds}
            return(outdict)

    def which_order(self,wavelength):
        ''' Say which order a wavelength falls in
        
        Parameters
        ----------
        wavelength : float
            wavelength to query
        '''
        wmins = np.nanmin(self.w_shifted,axis=1)
        wmaxs = np.nanmax(self.w_shifted,axis=1)
        for oi in range(28):
            if (wavelength > wmins[oi]) and (wavelength < wmaxs[oi]):
                return(oi)
        return(None)

    def jitter_spectrum(self):
        ''' Jitter the spectrum by the given variance
        
        Jitter each pixel's slope value by the pipeline-reported variance.
        Make sure to do this on a copy of the original spectrum - the 
        flux values will be permanently changed for this object.

        The object then re-does the ingestion of the spectrum (flattening, deblazing)
        '''
        # self.header = inp[0].header.copy()
        # self.sci_slope = inp[1].data.copy()
        # self.sky_slope = inp[2].data.copy()
        # self.cal_slope = inp[3].data.copy()
        # self.sci_variance = inp[4].data.copy()
        # self.sky_variance = inp[5].data.copy()
        # self.cal_variance = inp[6].data.copy()
        # self.sci_wave = inp[7].data.copy()
        # self.sky_wave = inp[8].data.copy()
        # self.cal_wave = inp[9].data.copy()
        # if keepsciHDU:
        #     self.hdu = inp
        # else:
        #     inp.close()
        shape = np.shape(self.sci_slope)
        jitter_sci = default_rng().normal(0,np.sqrt(self.sci_variance),shape)
        jitter_cal = default_rng().normal(0,np.sqrt(self.cal_variance),shape)
        jitter_sky = default_rng().normal(0,np.sqrt(self.sky_variance),shape)
        self.sci_slope = self.sci_slope + jitter_sci
        self.cal_slope = self.cal_slope + jitter_cal
        self.sky_slope = self.sky_slope + jitter_sky

        # Turn slopes into fluxes
        self.sci_err = np.sqrt(self.sci_variance)*self.exptime
        self.sky_err = np.sqrt(self.sky_variance)*self.exptime
        self.cal_err = np.sqrt(self.cal_variance)*self.exptime

        self.sci_and_sky_err = np.sqrt(self.sci_variance + self.sky_variance)*self.exptime

        self.f_sci = self.sci_slope / self.flat_sci_slope * self.exptime
        self.f_sky = self.sky_slope / self.flat_sky_slope * self.exptime #* self.SKY_SCALING_FACTOR
        self.f_cal = self.cal_slope / self.flat_cal_slope * self.exptime

        self.f_sci_sky = self.f_sci - self.f_sky

        self.sn18 = self.snr_order_median(18)


        self.deblaze(norm_percentile_per_order=80.)


        print('Spectrum Jittered (automatically deblazed)')


class HPFSpecList(object):

    def __init__(self,splist=None,filelist=None,tell_err_factor=1.,sky_err_factor=1.,targetname=''):
        if splist is not None:
            self.splist = splist
        else:
            self.splist = [HPFSpectrum(i,tell_err_factor=tell_err_factor,sky_err_factor=sky_err_factor,targetname=targetname) for i in filelist]

    @property
    def sn18(self):
        return [sp.sn18 for sp in self.splist]

    @property
    def filenames(self):
        return [sp.filename for sp in self.splist]

    @property
    def objects(self):
        return [sp.object for sp in self.splist]

    @property
    def exptimes(self):
        return [sp.exptime for sp in self.splist]

    @property
    def qprog(self):
        return [sp.qprog for sp in self.splist]

    @property
    def rv(self):
        return [sp.rv for sp in self.splist]

    @property
    def df(self):
        d = pd.DataFrame(zip(self.objects,self.filenames,self.exptimes,self.sn18,self.qprog,self.rv),columns=['OBJECT_ID','filename','exptime','sn18','qprog','rv'])
        return d



#def chi2spectra(ww,H1,H2,rv1=None,rv2=None,plot=False,verbose=False):
#    """
#    
#    EXAMPLE:
#        H1 = HPFSpectrum(df[df.name=='G_9-40'].filename.values[0])
#        H2 = HPFSpectrum(df[df.name=='AD_Leo'].filename.values[0])
#
#        wmin = 10280.
#        wmax = 10380.
#        ww = np.arange(wmin,wmax,0.01)
#        chi2spectra(ww,H1,H2,rv1=14.51,plot=True)
#        
#    EXAMPLE loop through chi2 rv space:
#        wmin = 10280.
#        wmax = 10380.
#        ww = np.arange(wmin,wmax,0.01)
#
#        H1 = HPFSpectrum(df[df.name=='G_9-40'].filename.values[0])
#        H2 = HPFSpectrum(df[df.name=='AD_Leo'].filename.values[0])
#
#        chis = []
#        rvs = np.linspace(14,15,200)
#
#        for i, rv in enumerate(rvs):
#            chi = chi2spectra(ww,H1,H2,rv1=rv,plot=False)
#            chis.append(chi)
#            print(i,rv,chi)
#    """
#    H1.deblaze()
#    H1.redshift(rv=rv1)
#    ff1, ee1 = H1.resample_order(ww)
#
#    H2.deblaze()
#    H2.redshift(rv=rv2)
#    ff2, ee2 = H2.resample_order(ww)
#    
#    chi2 = stats.chi2(ff1-ff2,np.sqrt(ee1**2.+ee2**2.),verbose=verbose)
#
#    if plot:
#        fig, (ax,bx) = plt.subplots(dpi=200,nrows=2,sharex=True,gridspec_kw={'height_ratios':[4,2]})
#        if rv1 is None: rv1 = H1.rv
#        if rv2 is None: rv2 = H2.rv
#        ax.plot(ww,ff1,lw=1,color='black',label="{}, rv={:0.2f}km/s".format(H1.object,rv1))
#        ax.plot(ww,ff2,lw=1,color='crimson',label="{}, rv={:0.2f}km/s".format(H2.object,rv2))
#        bx.errorbar(ww,ff1-ff2,ee1+ee2,elinewidth=1,marker='o',markersize=2,lw=0.,color='crimson')
#        fig.subplots_adjust(hspace=0.05)
#        [utils.ax_apply_settings(xx,ticksize=10) for xx in (ax,bx)]
#        bx.set_xlabel('Wavelength [A]')
#        ax.set_ylabel('Flux')
#        bx.set_ylabel('Residuals')
#        ax.set_title('{} vs {}: $\chi^2=${:0.3f}'.format(H1.object,H2.object,chi2))
#        ax.legend(loc='upper right',fontsize=8,bbox_to_anchor=(1.4,1.))
#        
#    return chi2
