# @ clean-up pycwt vs wavelet calls
import pycwt
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
import pycwt as wavelet
from pycwt.helpers import find
from tqdm import tqdm
import pandas as pd
import copy


def standardize(
    s,
    detrend=True,
    standardize=True,
    remove_mean=False,
    bandpass_filter=False,
    bandpass_kwargs=None,
):
    '''
    Helper function for pre-processing data, specifically for wavelet analysis

    Parameters
    ----------
        s : numpy array of shape (n,) to be normalized
        detrend : Linearly detrend s
        standardize : divide by the standard deviation
        remove_mean : remove the mean of s. Exclusive with detrend.
        bandpass_filter : band pass the data. NOTE: At the moment I just implement
            a high-pass filter. This needs to be updated in the future to be a band
            pass instead.

    Returns
    ----------
        snorm : numpy array of shape (n,)
    '''

    # @ Turn this into a decorator in the future for extra sexy python code
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    # Logic checking
    if detrend and remove_mean:
        raise ValueError('Only standardize by either removing secular trend or mean, not both.')
    if (detrend and bandpass_filter) or (remove_mean and bandpass_filter):
        raise ValueError(
            'Standardizing can only take one of the following as True:'
            ' remove_mean, detrend, bandpass_filer.'
        )
    if bandpass_filter and bandpass_kwargs is None:
        raise ValueError(
            'When using the bandpass filter the bandpass_kwargs must be supplied.'
        )

    # Derive the variance prior to any treatment
    smean = s.mean()
        
    # Remove the trend if requested
    if detrend:
        arbitrary_x = np.arange(0, s.size)
        p = np.polyfit(arbitrary_x, s, 1)
        snorm = s - np.polyval(p, arbitrary_x)
    else:
        snorm = s

    if remove_mean:
        snorm = snorm - smean
        
    if bandpass_filter:
        bandpass_kwargs.setdefault('order', 5)
        cutoff = bandpass_kwargs['cutoff']
        order = bandpass_kwargs['order']
        fs = bandpass_kwargs['fs']
        snorm = butter_highpass_filter(s, cutoff, fs, order=order)
        
    # Standardize by the variance only after the above
    # data treatment.
    std = snorm.std()
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
    rectify=True,
    significance_test=True,
    sig_kwargs={},
):
    '''
    Helper function for using the pycwt library since the steps in this
    function need to be completed for each realization of a CWT,

    Note: this function is based on the example from the pycwt library
    (https://pycwt.readthedocs.io/en/latest/tutorial.html)

    The resulting spectra are rectified following Liu 2007

    Parameters
    ----------
        signal : ndarray
        dx : float
        x : ndarray
        mother : pycwt wavelet object
        octaves : tuple, optional
        scales_to_avg : list, optional
        glbl_power_var_scaling : boolean, optional
        norm_kwargs : dict
        variance : float, optional
        rectify : boolean, optional
        significance_test : boolean, optional
        sig_kwargs : dict, optional

    Reutnrs
    ----------
    '''

    norm_kwargs.setdefault('standardize', True)
    norm_kwargs.setdefault('detrend', True)

    # If we do not normalize by the std, we need to find the variance
    if not norm_kwargs['standardize'] and variance is None:
        std = np.std(signal)
        var = std ** 2
    # For strongly non-stationary vectors, estimating the (white noise)
    # variance from the data itself is poorly defined. This option allows
    # the user to pass a pre-determined variance.
    elif variance is not None:
        var = variance
    # If the data were standardized, the variance should be 1 by definition
    # (assuming normally distributed processes)
    else:
        var = 1.

    if significance_test and 'sig_lvl' in sig_kwargs:
        sig_lvl = sig_kwargs['sig_lvl']
    else:
        sig_lvl = 0.95

    # This is hacky just to not break all of my scripts
    # by changing the number of outputs.
    if significance_test and 'local_sig_type' in sig_kwargs:
        local_sig_type = sig_kwargs['local_sig_type']
    else:
        local_sig_type = 'index'

    # Standardize/detrend/de-mean the input signal
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

    # Perform the wavelet transform using the parameters defined above.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        signal_norm, dx, dj, s0, J, mother)

    # Calculate the normalized wavelet and Fourier power spectra,
    # as well as the Fourier equivalent periods for each wavelet scale.
    # Note that this power is not normalized as in TC98 equation 8, the factor
    # of dt^(1/2) is missing.
    power = np.abs(wave) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # Do the significance testing for both the local and global spectra
    if significance_test:

        if 'alpha' not in sig_kwargs:
            # Lag-1 autocorrelation for red noise-based significance testing
            alpha, _, _ = wavelet.ar1(signal_norm)
        else:
            alpha = sig_kwargs['alpha']

        # Local power spectra significance test, where the ratio power / signif > 1.
        signif, fft_theor = wavelet.significance(var, dx, scales, 0, alpha,
                                                 significance_level=sig_lvl,
                                                 wavelet=mother)
        sig_ind = np.ones([1, N]) * signif[:, None]
        # The default behavior is to return a matrix that can be directly used
        # to draw a single contour following the pycwt_plot_helper functions.
        # The other option is to just return the local significance levels
        # as a matrix with dimensions equal to 'wave'.
        if local_sig_type == 'index':
            sig_ind = power / sig_ind

    # Rectify the power spectrum according to Liu et al. (2007)[2]
    if rectify:
        power /= scales[:, None]

    # Calculate the global wavelet spectrum
    glbl_power = power.mean(axis=1)
    if glbl_power_var_scaling:
        glbl_power = glbl_power / var

    # Do the significance testing for the global spectra
    if significance_test:

        # Global power spectra significance test. Note: this variable  is
        # different than the local significance variable. Here the global
        # spectra is reported directly.
        dof = N - scales  # Correction for padding at edges
        glbl_signif, tmp = wavelet.significance(var, dx, scales, 1, alpha,
                                                significance_level=sig_lvl,
                                                dof=dof, wavelet=mother)
    # Just return nans in the significance variables as they won't break
    # e.g. plotting routines.
    else:
        glbl_signif = np.ones_like(glbl_power) * np.nan
        sig_ind = np.ones_like(power) * np.nan

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

            if significance_test:
                try:
                    scale_avg_signif[nint], _ = wavelet.significance(
                        var, dx, scales, 2, alpha,
                        significance_level=sig_lvl,
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
                scale_avg_signif[nint] = np.nan

    else:
        scale_avg = None
        scale_avg_signif = None

    # @ Rename period and scales to better reflect that period is the inverse
    # fourier frequencies and that scales are the inverse wavelet frequencies.
    return (signal_norm, period, coi, power,
            glbl_signif, glbl_power, sig_ind,
            scale_avg, scale_avg_signif, scales)


def coi_scale_avg(coi, scales):
    '''
    Returns the upper and lower coi bounds for scale averaged wavelet power.

    Parameters
    ----------
        coi - np.array (or similar)
            location  of the coi in period space
        scales - np.array (or similar)
            The cwt scales that label the coi

    Returns
    ----------
        min_index : list
            indices of the coi scales for the shortest scale in
            the scale averaging interval
        max_index : list
            indices of the coi scales for the shortest scale in
            the scale averaging interval
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

    Parameters
    ----------
        s1 : 1D numpy array or similar object of length N.
            Signal one to perform the wavelet linear coherence with s2. Assumed
            to be pre-formatted by the `standardaize` function. s1 and s2 must
            be the same size.
        s2 : 1D numpy array or similar object of length N.
            Signal two to perform the wavelet linear coherence with s1. Assumed
            to be pre-formatted by the `standardaize` function. s1 and s2 must
            be the same size.
        dx : scalar or float. Spacing between elements in s1 and s2. s1 and s2
            are assumed to have equal spacing.
        dj : float, number of suboctaves per octave expressed as a fraction.
            Default value is 1 / 12 (12 suboctaves per octave)
        s0 : Smallest scale of the wavelet (if unsure = 2 * dt)
        J : float, number of octaves expressed as a power of 2 (e.g., the
            default value of 7 / dj means 7 powers of 2 octaves.
        mother : pycwt wavelet object, 'Morlet' is the only valid selection as
            a result of requiring an analytic expression for smoothing the
            wavelet.

    Returns
    ----------
        WCT : same type as s1 and s2, size PxN
            The biwavelet linear coherence transform.
        aWCT : same type as s1 and s2, size of PxN
            phase angle between s1 and s2.
        W12 : same type as s1 and s2, size PxN
            The cross-wavelet transform power, unrectified.
        W12_corr : the same type as s1 and s2, size PxN
            Cross-wavelet power, rectified following Veleda et al., 2012 and
            the R biwavelet package.
        period : numpy array, length P
            Fourier mode inverse frequencies.
        coi : numpy array  of length n, cone of influence
        angle : phase angle in degrees
        w1 : same type as s1, size PxN
            CWT for s1
        w2 : same type as s1, size PxN
            CWT for s2
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
    W1, sj, freq, coi, _, _ = wavelet.cwt(s1_norm, dx, dj, s0, J, mother)
    W2, _, _, _, _, _ = wavelet.cwt(s2_norm, dx, dj, s0, J, mother)

    # We need a 2D matrix for the math that follows
    scales = np.atleast_2d(sj)
    periods = np.atleast_2d(1 / freq)

    # Perform the cross-wavelet transform
    W12 = W1 * W2.conj()
    # Here I follow the R biwavelet package for the implementation of the
    # scale rectification. Note that the R package normalizes by the largest
    # wavelet scale. I choose to not include that scaling factor here.
    # W12_corr = W1 * W2.conj() * np.max(periods) / periods.T
    W12_corr = W1 * W2.conj() / periods.T

    # Coherence

    # Smooth the wavelet spectra before truncating.
    if mother.name == 'Morlet':
        sW1 = wavelet_obj.smooth((np.abs(W1) ** 2 / scales.T), dx, dj, sj)
        sW2 = wavelet_obj.smooth((np.abs(W2) ** 2 / scales.T), dx, dj, sj)
        sW12 = wavelet_obj.smooth((W12 / scales.T), dx, dj, sj)
    WCT = np.abs(sW12) ** 2 / (sW1 * sW2)
    aWCT = np.angle(W12)
    # @ fix this incorrect angle conversion.
    angle = (0.5 * np.pi - aWCT) * 180 / np.pi

    # @ better names to reflect fourier vs wavelet frequency/scale
    scales = np.squeeze(scales)

    return (WCT, aWCT, W12, W12_corr, 1 / freq, coi, angle, sW1, sW2)


def wct_mc_sig(
    wavelet,
    J,
    dj,
    dt,
    s0,
    sfunc_args1=[],
    sfunc_args2=[],
    sfunc_kwargs1={},
    sfunc_kwargs2={},
    mc_count=60,
    slen=None,
    sig_lvl=0.95,
    sfunc=None,
):
    '''
    Parameters
    ----------
        wavelet : pycwt wavelet object class
        J : int
            Wavelet's maximum scale.
        dj : float
            Number of suboctaves / number of octaves for the wavelet.
        dt : float
            Spacing of the time series in time. It is recommended to use the
            spacing of the data being tested for consistenct.
        s0 : float
            Minimum resolvable scale for the wavelet. Recommended value is
            2 * dt
        sfunc_args1 : list, optional
            positional arguments for sfunc for time series one.
        sfunc_args2 : list
            positional arguments for sfunc for time series two.
        sfunc_kwargs1 : dictionary, optional
        sfunc_kwargs2 : dictionary, optional
        mc_count : int, optional
        slen : int, optional
        sig_lvl : float, optional
        sfunc : function handle
            sfunc is used to generate the synthetic data
            for the Monte-Carlo simulation. The default function is
            pycwt.rednoise(). Function must accept `N` as the first argument
            and return an array of length `N`.

    Returns
    -------
        coh : ndarray of floats
            Coherence from each Monte-Carlo draw

    '''

    if slen is None:
        # Choose N so that largest scale has at least
        # some part outside the COI
        slen = s0 * (2 ** (J * dj)) / dt

    # Assign the length of the synthetic signal
    N = slen

    # Assign the function for generating synthetic data
    if sfunc is None:
        # @ Replace with a functional red noise generator
        sfunc = pcywt.rednoise

    # Peak the details of the cwt output for a single realization
    # of the noise function.
    noise1 = sfunc(N, *sfunc_args1, **sfunc_kwargs1)

    # Check that sfunc returns an array with the necessary properties
    if not len(noise1) == N:
        raise ValueError('sfunc must return data of length N')

    nW1, sj, freq, coi, _, _ = pycwt.cwt(
        noise1, dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet
    )

    period = np.ones([1, N]) / freq[:, None]
    coi = np.ones([J + 1, 1]) * coi[None, :]
    outsidecoi = (period <= coi)
    scales = np.ones([1, N]) * sj[:, None]
    sig_ind = np.zeros(J + 1)
    maxscale = find(outsidecoi.any(axis=1))[-1]
    sig_ind[outsidecoi.any(axis=1)] = np.nan

    coh = np.ma.zeros([J + 1, N, mc_count])

    # Displays progress bar with tqdm
    for n_mc in tqdm(range(mc_count)):#, disable=not progress):
        # Generates a synthetic signal using the provided function and parameters

        noise1 = sfunc(N, *sfunc_args1, **sfunc_kwargs1)
        noise2 = sfunc(N, *sfunc_args2, **sfunc_kwargs2)

        # Calculate the cross wavelet transform of both red-noise signals
        kwargs = dict(dt=dt, dj=dj, s0=s0, J=J, wavelet=wavelet)
        nW1, sj, freq, coi, _, _ = pycwt.cwt(noise1, **kwargs)
        nW2, sj, freq, coi, _, _ = pycwt.cwt(noise2, **kwargs)
        nW12 = nW1 * nW2.conj()

        # Smooth wavelet wavelet transforms and calculate wavelet coherence
        # between both signals.

        S1 = wavelet.smooth(np.abs(nW1) ** 2 / scales, dt, dj, sj)
        S2 = wavelet.smooth(np.abs(nW2) ** 2 / scales, dt, dj, sj)
        S12 = wavelet.smooth(nW12 / scales, dt, dj, sj)

        R2 = np.ma.array(np.abs(S12) ** 2 / (S1 * S2), mask=~outsidecoi)
        coh[:, :, n_mc] = R2

    period = period[:, 0]

    return coh, period, scales, coi


def coi_where(period, coi, data):
    '''
    Finds where the period by num samples array is outside the coi.
    Useful for creating masked numpy arrays.

    INPUTS:
    period - N array, the wavelet periods
    coi - M array, the coi location for each sample. M should
        be the number of samples.
    data - N x M array, the data to broadcast the COI mask to.

    RETURNS:
    outside_coi - N by M array of booleans. Is True where
        the wavelet is effected by the coi.

    EXAMPLE:
        outside_coi = pycwt_stat_helpers.coi_where(period, coi, WCT)
        masked_WCT = np.ma.array(WCT, mask=~outside_coi)
    '''

    coi_matrix = np.ones(np.shape(data)) * coi
    period_matrix = (np.ones(np.shape(data)).T * period).T

    outside_coi = period_matrix < coi_matrix

    return outside_coi


def ar1_generator(N, alpha, noise):
    '''
    Generates a Markov chain lag-1 process through a brute force approach.

    Parameters
    -------

    Returns
    -------

    '''

    y = np.zeros((3 * N,))
    # Simulate a longer period and just return the last N point
    # to remove edge effects.
    for t in range(1, 3 * N):
        y[t] = alpha * y[t - 1] + np.random.normal(scale=noise)

    return y[-N:]


def nan_sequences(signal, dx, dim='time', units=None):
    '''
    Describes properties of NaN blocks, giving their indices, slices, and
    length.

    Parameters
    -------

    Returns
    -------

    '''

    # Derive the
    if dim == 'time':
        dt_type_test = pd.Timedelta(seconds=1)
        if units is None:
            raise ValueError(
                'When operating along the time dimension, the units keyword'\
                ' must be provided.'
            )
        units_kwarg = {units: dx}
        dx_td = pd.Timedelta(**units_kwarg)

    # First, find the indices for the nans
    # The drop=True makes the first loop easy, but the
    # indices are now no longer referenced to the original
    # signal dataset
    nanind = signal.where(np.isnan(signal), drop=True)

    # If no NaNs are present return empty lists
    if nanind.size == 0:
        nan_seq_len = []
        nan_seq_slc = []
        nan_seq_ind = []
        return nan_seq_ind, nan_seq_slc, nan_seq_len

    ind_beg = [0]
    ind_end = []
    for nt2, t2 in enumerate(nanind[dim][1:]):
        if dim == 'time':
            t1 = pd.Timestamp(nanind[dim].values[nt2])
            t2 = pd.Timestamp(t2.values)
        else:
            t1 = nanind[dim].values[nt2]
            t2 = t2.values
        # Maybe this extra step of checking for time is unnecessary
        # if we overwrite dx with the Timedelta value if dim=='time'.
        # I don't have time right now to check.
        if (dim == 'time') and (t2 - t1 > dx_td):
            ind_end.append(nt2)
            ind_beg.append(nt2 + 1)
        elif (not dim == 'time') and (t2 - t1 > dx):
            ind_end.append(nt2)
            ind_beg.append(nt2 + 1)
        
    ind_end.append(-1)

    # 3 steps:
    # 1) Convert back to the original signal's indices
    # 2) Find the duration of a nan sequence
    # 3) Give the slice (in time coordinates) for the nan block
    nan_seq_len = []
    nan_seq_slc = []
    nan_seq_ind = []

    for b, e in zip(ind_beg, ind_end):
        # Duration of the nan sequence in units of time
        if dim=='time':
            t1 = pd.Timestamp(nanind.time.values[b])
            t2 = pd.Timestamp(nanind.time.values[e])
        else:
            # Duration of the nan sequence in units of dx
            t1 = nanind[dim].values[b]
            t2 = nanind[dim].values[e]

        # Slices of nan blocks
        nan_seq_slc.append(slice(t1, t2))

        # The indices of nan blocks
        # (index beg of block, index end of block)
        nan_seq_ind.append(
            [
                np.flatnonzero(signal[dim] == t1)[0],
                np.flatnonzero(signal[dim] == t2)[0],
            ]
        )

        tdiff = np.flatnonzero(signal[dim] == t2)[0] - np.flatnonzero(signal[dim] == t1)[0]
        nan_seq_len.append(tdiff)

    return nan_seq_ind, nan_seq_slc, nan_seq_len


def nan_coi(nan_seq_ind, nan_seq_len, coi, period, signal, dx):
    '''
    Rebuilds the COI to have a new, scale-aware COI around nan gaps

    Parameters
    -------

    Returns
    -------

    '''
    if (not nan_seq_ind) or (not nan_seq_len):
        return coi

    # We need two copies of the data.
    # One is just the original coi. We will manipulate it
    # at the very end when we ammend it with the nan-coi gaps
    coi_mask = copy.deepcopy(coi)
    # And now we need just half of the coi
    coi_half = copy.deepcopy(coi[0:len(coi)//2])
    # Remove anything with a length longer than the resolvable
    coi_half[coi_half > np.max(period)] = np.nan

    coi_half_nanless = copy.deepcopy(coi_half)
    coi_half_nanless = coi_half_nanless[np.flatnonzero(~np.isnan(coi_half_nanless))]
    coi_half_nanless_len = len(coi_half_nanless)

    nan_mask = np.zeros_like(signal) * np.nan

    for ns, (n1, n2) in enumerate(nan_seq_ind):
        nan_mask[n1:n2 + 1] = 0

        coi_fill_len = np.min(
            (
                nan_seq_len[ns] + 1,
                coi_half_nanless_len,
                n1
            )
        )
        coi_nanseq_beg = n1 - coi_fill_len
        fill_values = coi_half_nanless[coi_fill_len - 1::-1]
        fill_values[fill_values > nan_seq_len[ns] * dx] = nan_seq_len[ns] * dx
        nan_mask[coi_nanseq_beg:n1] = fill_values

        coi_fill_len = np.min(
            (
                nan_seq_len[ns] + 1,
                coi_half_nanless_len,
                len(signal) - (n2 + 1)
            )
        )
        coi_nanseq_end = n2 + coi_fill_len
        fill_values = coi_half_nanless[0:coi_fill_len]
        fill_values[fill_values > nan_seq_len[ns] * dx] = nan_seq_len[ns] * dx
        nan_mask[n2:coi_nanseq_end] = fill_values

    # Don't overwrite the original COI from edge effects.
    nan_mask[nan_mask > coi] = coi[nan_mask > coi]

    # Fill in all the places without nans with the original COI
    coi_mask[~np.isnan(nan_mask)] = nan_mask[~np.isnan(nan_mask)]

    return(coi_mask)
