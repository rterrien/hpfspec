from __future__ import print_function
import numpy as np
import scipy.interpolate
import scipy.optimize
import copy
import scipy.special

def fgauss(x, center, sigma, amp):
    """A Gaussian function.

    This is a standard Gaussian function.

    Parameters
    ----------
    x : (N,) ndarray
        Independent variable for the Gaussian
    center : float
        Mean for the Gaussian
    sigma : float
        Standard deviation (sigma) for the Gaussian
    amp : float
        Amplitude of the Gaussian
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    return(amp * np.exp(-((x - center) / sigma) ** 2.))


def fgauss_const(x, center, sigma, amp, offset):
    """Gaussian + offset function.

    This is a Gaussian with a constant offset.

    Parameters
    ----------
    x : (N,) ndarray
        Independent variable for the Gaussian
    center : float
        Mean for the Gaussian
    sigma : float
        Standard deviation (sigma) for the Gaussian
    amp : float
        Amplitude of the Gaussian
    offset : float
        Offset for the Gaussian
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    offset = float(offset)
    return(float(amp) * np.exp(-((x - center) / sigma) ** 2.) + offset)

def fgauss_from_1(x, center, sigma, amp):
    """Gaussian + offset function.

    This is a Gaussian with a constant offset.

    Parameters
    ----------
    x : (N,) ndarray
        Independent variable for the Gaussian
    center : float
        Mean for the Gaussian
    sigma : float
        Standard deviation (sigma) for the Gaussian
    amp : float
        Amplitude of the Gaussian
    offset : float
        Offset for the Gaussian
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    offset = 1.
    return(float(amp) * np.exp(-((x - center) / sigma) ** 2.) + offset)

def fvoigt_from_1(x, center, sigma, amp, gamma):
    """ 1 - Voigt function for fitting spectral lines.
    
    Note that the Voigt function gives a Cauchy dist if sigma=0,
    and a gaussian of gamma=0. 

    Parameters
    ----------
    x : float array
        Independent variable for the function
    center : float
        Center of the function
    sigma : float
        Sigma (width) for gaussian component
    amp : float
        Amplitude for the function
    gamma : Half-width half-max for the Cuachy component
        
    """
    
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    gamma = float(gamma)
    offset = 1.

    x_translated = x - center

    return(offset - amp * scipy.special.voigt_profile(x_translated, sigma, gamma))


def fgauss_line(x, center, sigma, amp, offset, slope):
    """Gaussian + line function.

    This is a Gaussian with a linear offset.

    Parameters
    ----------
    x : (N,) ndarray
        Independent variable for the Gaussian
    center : float
        Mean for the Gaussian
    sigma : float
        Standard deviation (sigma) for the Gaussian
    amp : float
        Amplitude of the Gaussian
    offset : float
        Offset for the Gaussian linear offset (y-intercept)
    slope : float
        Slope for the Gaussian linear offset
    """
    center = float(center)
    sigma = float(sigma)
    amp = float(amp)
    offset = float(offset)
    slope = float(slope)
    return(float(amp) * np.exp(-((x - center) / sigma) ** 2.) + offset + x * slope)

def fitProfile(inp_x, inp_y, fit_center_in, fit_width=8, sigma=None,
               func='fgauss_const', return_residuals=False,p0=None,bounds=(-np.inf,np.inf),
               curve_fit_kw={}):
    """Perform a least-squares fit to a CCF.

    Parameters
    ----------
    inp_x : ndarray
        x-values of line to be fit (full array; subset is
        taken based on fit width)
    inp_y : ndarray
        y-valeus of line to be fit (full array; subset is
        taken based on fit width)
    fit_center_in : float
        Index value of estimated location of line center;
        used to select region for fitting
    fit_width : {int}, optional
        Half-width of fitting window. (the default is 8)
    sigma : {float}, optional
        The standard error for each x/y value in the fit.
        (the default is None, which implies an unweighted fit)
    func : {'fgauss','fgauss_const','fgauss_line','fgauss_from_1'} , optional
        The function to use for the fit. (the default is 'fgauss')
    return_residuals : {bool}, optional
        Output the fit residuals (the default is False)
    p0 : list of first-guess coefficients. The fit can be quite sensitive to these
        choices.
    bounds : Directly sent to scipy.optimize.curve_fit()

    Raises
    ------
    ValueError
        [description]

    Returns
    -------
    dict
        {'centroid': fitted centroid
        'e_centroid': std error of fitted gaussian centroid (covar diagonals)
        'sigma': fitted sigma of gaussian
        'e_sigma': std error of fitted sigma of gaussian (covar diagonals)
        'nanflag': are there NaNs present
        'pcov': covariance array - direct output of optimize.curve_fit
        'popt': parameter array - direct output of optimize.curve_fit
        'function_used': function used for fitting
        'tot_counts_in_line': simple sum of y-values in used line region
    """

    # select out the region to fit
    # this will be only consistent to +- integer pixels
    fit_center = copy.copy(fit_center_in)
    xx_index = np.arange(len(inp_x))
    assert len(inp_x) == len(inp_y)
    
    j1 = int(np.round(np.amax([0, fit_center - fit_width])))
    j2 = int(round(np.amin([np.amax(xx_index), fit_center + fit_width])))

    # define sub-arrays to fit
    sub_x1 = inp_x[j1:j2]
    sub_y1 = inp_y[j1:j2]

    tot_counts_in_line = float(np.nansum(sub_y1))

    # normalize the sub-array
    scale_value = np.nanmax(sub_y1)
    sub_y_norm1 = sub_y1 / scale_value

    # select out the finite elements
    ii_good = np.isfinite(sub_y_norm1)
    sub_x = sub_x1[ii_good]
    sub_y_norm = sub_y_norm1[ii_good]
    if sigma is not None:
        sub_sigma1 = sigma[j1:j2]
        ii_good = np.sfinite(sub_y_norm1) & (np.isfinite(sub_sigma1))
        sub_sigma = sub_sigma1[ii_good]
        sub_y_norm = sub_y_norm1[ii_good]
    else:
        sub_sigma = None

    # note whether any NaNs were present
    if len(sub_x) == len(sub_x1):
        nanflag = False
    else:
        nanflag = True

    # set up initial parameter guesses, function names, and bounds. 
    # initial guess assumes that the gaussian is centered at the middle of the input array
    # the sigma is "1" in x units
    # the amplitude is -0.1.
    # for the functions with an additional constant and line, the constant defaults to 1.
    if func == 'fgauss':
        if p0 is None:
            p0 = (np.mean(sub_x), 5., -0.5)
        use_function = fgauss
    elif func == 'fgauss_const':
        if p0 is None:
            p0 = (np.mean(sub_x),1., -np.ptp(sub_y_norm), np.nanmedian(sub_y_norm))
        use_function = fgauss_const
    elif func == 'fgauss_line':
        if p0 is None:
            p0 = (np.mean(sub_x), 1., -0.5, 1., 0.)
        use_function = fgauss_line
    elif func == 'fgauss_from_1':
        if p0 is None:
            p0 = (np.mean(sub_x),1., -np.ptp(sub_y_norm))
        use_function = fgauss_from_1
    elif func == 'fvoigt_from_1':
        if p0 is None:
            p0 = (np.mean(sub_x),1., np.ptp(sub_y_norm),1.)
        use_function = fvoigt_from_1
    else:
        raise ValueError

    # perform the least squares fit
    try:
        popt, pcov = scipy.optimize.curve_fit(use_function,
                                              sub_x,
                                              sub_y_norm,
                                              p0=p0,
                                              sigma=sub_sigma,
                                              maxfev=10000,
                                              bounds=bounds,
                                              **curve_fit_kw)

        # Pull out fit results
        # fitted values (0 is the centroid, 1 is the sigma, 2 is the amp)
        # lists used to facilitate json recording downstream
        errs = np.diag(pcov)
        centroid = popt[0]
        centroid_error = np.sqrt(errs[0])
        width = popt[1]
        width_error = np.sqrt(errs[1])
        fit_successful = True
        pcov_list = pcov.tolist()
        popt_list = popt.tolist()

    except RuntimeError:
        errs = np.NaN
        centroid = np.NaN
        centroid_error = np.NaN
        width = np.NaN
        width_error = np.NaN
        fit_successful = False
        pcov_list = []
        popt_list = []

    if np.isnan(centroid_error) or np.isnan(centroid):
        fit_successful = False

    # build the returned dictionary
    retval = {'centroid': centroid,
              'e_centroid': centroid_error,
              'sigma': width,
              'e_sigma': width_error,
              'nanflag': nanflag,
              'pcov': pcov_list,
              'popt': popt_list,
              'indices_used': (j1, j2),
              'function_used': func,
              'tot_counts_in_line': tot_counts_in_line,
              'fit_successful': fit_successful,
              'scale_value':scale_value}

    # since residual array can be large, optionally include it
    if return_residuals:
        if fit_successful:
            predicted = use_function(sub_x, *popt)
            residuals = (predicted - sub_y_norm).tolist()
        else:
            residuals = np.NaN
        retval['residuals'] = residuals

    #return(retval['popt'][0], retval['popt'][1], retval['popt'][2], retval)
    return(retval)
    