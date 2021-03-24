import pycwt
import numpy as np
import xarray as xr
import scipy
import pandas as pd
import copy
from scipy.ndimage.filters import gaussian_filter
import pyfocs
import pycwt as wavelet
from pycwt.helpers import find


def standardize(s, detrend=True, standardize=True, remove_mean=False):
    '''
    Helper function for pre-processing data, specifically for wavelet analysis

    INPUTS:
        s - numpy array of shape (n,) to be normalized
        detrend - Linearly detrend s
        standardize - divide by the standard deviation
        remove_mean - remove the mean of s. Exclusive with detrend.

    OUTPUTS:
        snorm - numpy array of shape (n,)
    '''

    # Derive the variance prior to any detrending
    std = s.std()
    smean = s.mean()

    if detrend and remove_mean:
        raise ValueError('Only standardize by either removing secular trend or mean, not both.')

    # Remove the trend if requested
    if detrend:
        arbitrary_x = np.arange(0, s.size)
        p = np.polyfit(arbitrary_x, s, 1)
        snorm = s - np.polyval(p, arbitrary_x)
    else:
        snorm = s

    if remove_mean:
        snorm = snorm - smean

    # Standardize by the standard deviation
    if standardize:
        snorm = (snorm / std)

    return snorm


def wavelet_power(
    signal,
    dx,
    x,
    mother,
    octaves=None,
    scales_to_avg=None,
    glbl_power_var_scaling=True,
    norm_kwargs={'detrend': True, 'standardize': True},
    variance=None,
    alpha=None,
    rectify=True,
):
    '''
    Helper function for using the pycwt library since the steps in this
    function need to be completed for each realization of a CWT,

    Note: this function is based on the example from the pycwt library
    (https://pycwt.readthedocs.io/en/latest/tutorial.html)

    The resulting spectra are rectified following Liu 2007
    '''

    # If we do not normalize by the std, we need to find the variance
    if not norm_kwargs['standardize'] and variance is None:
        std = np.std(signal)
        var = std ** 2  # Variance
    # For strongly non-stationary vectors, estimating the (white noise)
    # variance from the data itself is poorly defined. This option allows
    # the user to pass a pre-determined variance.

    elif variance is not None:
        var = variance
    # If the data were standardized, the variance should be 1 by definition
    # (assuming normally distributed processes)
    else:
        var = 1.

    # Standardize/detrend/demean the input signal
    signal_norm = standardize(signal, **norm_kwargs)
    N = signal_norm.size

    # Wavelet properties
    # Starting scale is twice the resolution
    s0 = 2 * dx
    # Default wavelet values yields 85 periods
    if octaves is None:
        # Twelve sub-octaves per octaves
        dj = 1 / 12
        # Seven powers of two with dj sub-octaves
        J = 7 / dj
    else:
        # Number of sub-octaves per octaves
        dj = octaves[0]
        # Number of powers of two with dj sub-octaves
        J = octaves[1]

    # Lag-1 autocorrelation for red noise-based significance testing
#     try:
    alpha, _, _ = wavelet.ar1(signal_norm)
#     except:
# Try to remove the trend, as that is often the issue
# Finally, just skip this step and the significance testing when requested
# or if the second try fails

    # Perform the wavelet transform using the parameters defined above.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        signal_norm, dx, dj, s0, J, mother)

    # Calculate the normalized wavelet and Fourier power spectra,
    # as well as the Fourier equivalent periods for each wavelet scale.
    power = np.abs(wave) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # Power spectra significance test, where the ratio power / sig95 > 1.
    signif, fft_theor = wavelet.significance(var, dx, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # Rectify the power spectrum according to Liu et al. (2007)[2]
    if rectify:
        power /= scales[:, None]

    # Then, we calculate the global wavelet spectrum and determine its significance level.
    glbl_power = power.mean(axis=1)
    if glbl_power_var_scaling:
        glbl_power = glbl_power * var
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dx, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)
    if rectify:
        glbl_signif = glbl_signif / scales

    if scales_to_avg is not None:
        Cdelta = mother.cdelta
        num_intervals = np.shape(scales_to_avg)[0]
        scale_avg = np.ones((len(x), num_intervals))
        scale_avg_signif = np.ones(num_intervals)

        for nint in np.arange(num_intervals):
            # Calculate the scale average between two periods and the significance level.
            sel = find((period >= scales_to_avg[nint][0]) & (period < scales_to_avg[nint][1]))


            # In the pycwt example and TC98 eq. 24 the scale-avg power is scale rectified.
            # However, the rectification occurs above in the calculation of power. Do not repeat here.
            if rectify:
                scale_avg[:, nint] = dj * dx / Cdelta * power[sel, :].sum(axis=0)
            # If no rectification is requested, then for consistency with TC98
            # we return the rectified power here.
            elif not rectify:
                # The example code is just included here as a reference
                # scale_avg = (scales * np.ones((N, 1))).transpose()
                # scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
                # TC98 eq. 24 does not include the variance as in the tutorial.
                scale_avg[:, nint] = dj * dx / Cdelta * power[sel, :].sum(axis=0) / scales[:, None]


            try:
                scale_avg_signif[nint], _ = wavelet.significance(
                    var, dx, scales, 2, alpha,
                    significance_level=0.95,
                    dof=[scales[sel[0]], scales[sel[-1]]],
                    wavelet=mother
                )
            except IndexError:
                # One of the scales to average was outside the range of the
                # CWT's scales. Return no significance level for this averaging
                # interval and move to the next one.
                scale_avg_signif[nint] = np.nan
                continue

    else:
        scale_avg = None
        scale_avg_signif = None

    return (signal_norm, period, coi, power,
            glbl_signif, glbl_power, sig95,
            scale_avg, scale_avg_signif, scales)


def coi_scale_avg(coi, scales):
    '''
    Returns the upper and lower coi bounds for scale averaged wavelet power.

    INPUTS:
        coi - np.array (or similar) indicating the location
              of the coi
        scales - np.array (or similar) of the cwt scales.

    RETURNS:
        min_index, max_index - indices of the coi scales for
            the longest/shortest scale in the averaging interval
    '''
    mindex1 = np.argmin(np.abs(coi[:len(coi)//2] - np.min(scales)))
    mindex2 = np.argmin(np.abs(coi[len(coi)//2:] - np.min(scales)))
    min_index = [mindex1, mindex2 + len(coi)//2]

    maxdex1 = np.argmin(np.abs(coi[:len(coi)//2] - np.max(scales)))
    maxdex2 = np.argmin(np.abs(coi[len(coi)//2:] - np.max(scales)))
    max_index = [maxdex1, maxdex2 + len(coi)//2]

    return min_index, max_index


def wavelet_coherent(
    s1,
    s2,
    dx,
    dj,
    s0,
    J,
    mother,
    norm_kwargs={'detrend': True, 'standardize': True},
    ):
    '''
    Calculate coherence between two vectors for a given wavelet. Currently
    only the Morlet wavelet is implemented. Returns the wavelet linear coherence
    (also known as coherency and wavelet coherency transform), the cross-wavelet
    transform, and the phase angle between signals.

    INPUTS:
        s1 - 1D numpy array or similar object of length n. Signal one to
            perform the wavelet linear coherence with against with s2. Assumed
            to be pre-formatted by the `standardaize` function. s1 and s2 must
            be the same size.
        s2 - 1D numpy array or similar object of length n. Signal two to
            perform the wavelet linear coherence with against with s1. Assumed
            to be pre-formatted by the `standardaize` function. s1 and s2 must
            be the same size.
        dx - scalar or float. Spacing between elements in s1 and s2. s1 and s2
            are assumed to have equal spacing.
        dj - float, number of suboctaves per octave expressed as a fraction.
            Default value is 1 / 12 (12 suboctaves per octave)
        s0 -
        J - float, number of octaves expressed as a power of 2 (e.g., the
            default value of 7 / dj means 7 powers of 2 octaves.
        mother - pycwt wavelet object, 'Morlet' is the only valid selection as
            a result of requiring an analytic expression for smoothing the
            wavelet.

    RETURNS:
        WCT - same type as s1 and s2 of length n, the wavelet coherence
            transform.
        aWCT - same type as s1 and s2 of length n, phase angle between s1 and
            s2.
        W12 - same type as s1 and s2 of length n, the cross-wavelet transform.
            Power is rectified following Veleda et al., 2012.
        period - numpy array of length p, fourier mode inverse frequency.
        coi - numpy array  of length n, cone of influence
        angle - phase angle in degrees
        w1 - same type as s1 (n by p),  CWT for s1
        w2 - same type as s1 (n by p),  CWT for s2
    '''
    assert mother.name == 'Morlet', "XWT requires smoothing, which is only available to the Morlet mother."
    wavelet_obj = wavelet.wavelet._check_parameter_wavelet(mother)

    assert np.size(s1) == np.size(s2), 's1 and s2 must be the same size.'

    # s1 and s2 MUST be the same size
    assert s1.shape == s2.shape, "Input signals must share the exact same shape."

    s1_norm = standardize(s1, **norm_kwargs)
    s2_norm = standardize(s2, **norm_kwargs)

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=mother)
    W1, sj, freq, coi, _, _ = wavelet.cwt(s1, dx, dj, s0, J, mother)
    W2, _, _, _, _, _ = wavelet.cwt(s2, dx, dj, s0, J, mother)

    # We need a 2D matrix for the math that follows
    scales = np.atleast_2d(sj)

    # Perform the cross-wavelet transform
    W12 = (W1 / (scales.T)**(1/2)) * ((W2 / (scales.T)**(1/2)).conj())

    # Coherence

    # Smooth the wavelet spectra before truncating.
    # @ this is terrible variable usage and needs to be fixed.
    if mother.name == 'Morlet':
        sW1 = wavelet_obj.smooth(np.abs(W1) ** 2, dx, dj, sj)
        sW2 = wavelet_obj.smooth(np.abs(W2) ** 2, dx, dj, sj)
        sW12 = wavelet_obj.smooth(W12, dx, dj, sj)
    WCT = np.abs(sW12) ** 2 / (sW1 * sW2)
    aWCT = np.angle(W12)
    # @ fix this incorrect angle conversion.
    angle = (0.5 * np.pi - aWCT) * 180 / np.pi

    # Rectify the XWT after calculating coherence
    # @ If the individual wavelets are scale rectified, then this step is...
    # redundant? Introduces an incorrect scaling? Unclear, but should be
    # addressed.
    # W12 = W12 / scales.T

    # @ Return the wavelet scales as well
    # @ better names to reflect fourier vs wavelet frequency/scale
    scales = np.squeeze(scales)

    return (WCT, aWCT, W12, 1 / freq, coi, angle, s1, s2)
