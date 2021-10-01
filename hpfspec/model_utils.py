import numpy as np
import astropy.convolution.kernels
from PyAstronomy import pyasl
import spectres
from hpfspec2 import spec_help

def read_btsettl(fname, minl, maxl,upsample=1.):
    ''' read in a BT-Settl model spectrum file
    
    Open, sort, and store in memory a model spectrum from the BT-Settl grid.
    
    Parameters
    ----------
    fname : string
        Path to spectrum (in ascii format)
    minl : float
        Min wavelength to read, in angstroms
    maxl : float
        Max wavelength to read, in angstroms
    upsample : float, optional
        factor by which to linearly upsample (the default is 1.)

    Returns
    -------
    2-element tuple
        wavelengths [angstroms], fluxes (float)
    '''
    ww1 = []
    ff1 = []

    with open(fname, 'r') as infile:
        # Test if number can be converted, may need to replace
        # D with E and/or split differently
        try:
            line1 = float(infile.readline().split()[0])
            worked = True
        except: # bare except; lazy!
            worked = False
        if worked:
            for line in infile:
                ll = line.split()
                ww1.append(float(ll[0].replace('D','E')))
                ff1.append(float(ll[1].replace('D','E')))
        else:
            print('splitting manually')
            for line in infile:
                ww1.append(float(line[0:13]))
                ff1.append(float(line[13:25].replace('D','E')))
                #print float(line[7:13]), float(line[13:25].replace('D','E'))
    print(ww1[0], ww1[-1], len(ww1))

    # convert the accumulated values into float arrays of correct quantities
    # fluxes are left in a logarithmic/offset format, so must be adjusted
    ww2 = np.array(ww1)
    ff2 = 10. ** (np.array(ff1) - 8.)

    # arrays are not sorted by wavelength, so must be re-sorted
    sorti = np.argsort(ww2)
    ww3 = ww2[sorti]
    ff3 = ff2[sorti]

    # select the region of interest
    print(np.amin(ww2), np.amax(ww2), ww2[10], len(ww2))
    rangei = np.nonzero((ww3 > minl) & (ww3 < maxl))[0]
    ww4 = ww3[rangei]
    ff4 = ff3[rangei]

    # resample if desired
    #ww, ff = resample(ww4, ff4, upsample=upsample)
    ww, ff = spec_help.resample_to_median_sampling(ww4, ff4, kind='cubic',upsample_factor=upsample)
    #resample_to_median_sampling(x,y,e=None,kind='FluxConservingSpectRes',fill_value=np.nan,upsample_factor=1.):

    return ww, ff

def resolution_kernel(resol,dwl_pix,wl,kernwidth=5,discretize_mode='center'):
    ''' Gaussian kernel generation
    
    Make a Gaussian kernel that can be used to approximate the PSF of a spectrograph.
    
    Parameters
    ----------
    resol : float
        Resolution (where R = lambda / Gaussian_FWHM)
    dwl_pix : float
        Change in wavelength per pixel (in same units as wl)
    wl : float
        Wavelength of kernel (in same units as dwl_pix)
    kernwidth : float, optional
        Width of the kernel in units of the FWHM (the default is 5, which [default_description])
    discretize_mode : str, optional
        how to discretize the gaussian (see Gaussian1DKernel docs) 
        (the default is 'center')

    Returns
    -------
    Float array
        Kernel that can be convolved with high-res spectrum
    '''
    # find the width of the kernel in wavelength units
    width_wl = wl / resol
    # find the width of the kernel in pixel units
    width_pix = width_wl / dwl_pix
    # calculate needed number of pixels
    arrsize = np.round(kernwidth * width_pix)
    # make sure length of kernel is odd
    if arrsize % 2 == 0:
        arrsize += 1
    # generate the kernel
    gk = astropy.convolution.kernels.Gaussian1DKernel(width_pix/2.35,mode=discretize_mode,x_size=arrsize)
    # normalize the kernel
    gk2 = gk.array / sum(gk.array)
    return gk2

def broaden_make_hpflike_fluxes(model_wl, model_fl, hpf_wcal, epsilon=0.6, vsini=0., edgehandeling='firstlast'):
    ''' Make rotationally-broadened HPF-like flux array from model
    
    Artificially broaden a spectrum with given vsini then
    convert and resample a model spectrum into a 28x2048 array
    similar to an HPF spectrum.
    
    Parameters
    ----------
    model_wl : float array
        Input wavelengths [angstroms]
    model_fl : float array
        Input fluxes
    hpf_wcal : float array (28 x 2048)
        Desired wavelength array [Angstroms]
    epsilon : float
        Linear limb-darkening coefficient (0-1)  
    vsini : float
        Projected rotational velocity [km/s]
    edgeHandling : string, {“firstlast”, “None”}
        The method used to handle edge effects.
    
    Returns
    -------
    Broadened float array (28 x 2048)
        Array of HPF-like fluxes of the broadend spectrum

    '''
   
    #broaden the spectrum and return new flux:
    if vsini > 0.:
        broad_fl = pyasl.rotBroad(model_wl, model_fl, epsilon, vsini, edgehandeling)
    else:
        broad_fl = model_fl

    # Hard-code the HPF parameters for the moment:
    res = 53000.
    discretize_mode = 'integrate'
    resol_kernwidth = 10.

    # Set the output array
    fl_final = hpf_wcal.copy()

    # Loop over all the orders
    for i in range(28):
        # Find appropriate wavelength range so we don't have to convolve
        # over the full array
        # use a slightly larger range than each order extent, so that we can
        # avoid edge effects in the convolution
        wmin = hpf_wcal[i,0]
        wmax = hpf_wcal[i,-1]
        buffer_frac = 0.1 # percent of order to include on edge
        buffer_angstrom = (wmax - wmin) * buffer_frac
        ii = np.nonzero( (model_wl > (wmin-buffer_angstrom)) & (model_wl < (wmax+buffer_angstrom)) )[0]
        wl_this = model_wl[ii]
        fl_this = broad_fl[ii] #changed model_fl to broad_fl here, I think the only place where I needed to change to broadened flux

        # generate the appropriate kernel for this order
        dwl_per_pixel = np.abs(wl_this[1] - wl_this[0])
        wlmid = np.median(wl_this)
        resol_kern = resolution_kernel(res, dwl_per_pixel, wlmid, discretize_mode=discretize_mode,
                                       kernwidth=resol_kernwidth)

        # convolve with the kernel
        fl_mod = np.convolve(fl_this, resol_kern,mode='same')
        wl_use = hpf_wcal[i,:]
        
        # resample the spectrum, the unused edges will be cut off here
        fl_out = spectres.spectres(wl_use,wl_this,fl_mod,verbose=False,fill=np.nan)
        fl_final[i,:] = fl_out

    return fl_final
