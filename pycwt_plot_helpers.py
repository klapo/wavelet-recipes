# import plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import dates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.dates import DateFormatter
from matplotlib import ticker
from matplotlib import colors
import itertools
import string
import numpy as np
from SRON import SRON


def plot_periodogram(
    power,
    x,
    dx,
    period,
    coi=None,
    sig95=None,
    ax=None,
    levels=None,
    log_power=True,
    cmap='viridis'
):
    '''
    Helper function for plotting the results of a CWT.

    INPUTS:

    RETURNS:
        None
    '''

    # Normalized wavelet power spectrum and significance level contour
    # lines and cone of influece hatched area. Note that period scale is logarithmic.

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if levels is None and log_power:
        uq = np.ceil(np.log10(np.quantile(power, 0.95)))
        lq = np.ceil(np.log10(np.quantile(power, 0.1)))
        if uq == lq:
            lq = lq - 0.5
            uq = uq + 0.5
        if np.abs(uq - lq) <= 1:
            dq = 0.1
        elif np.abs(uq - lq) <= 2:
            dq = 0.25
        else:
            dq = 0.5

        # Determine the log-space levels. Include the upper
        # quantile limit.
        lev_exp = np.arange(
            lq,
            uq + dq,
            dq,
        )
        levels = 10. ** lev_exp
    elif levels is None and not log_power:
        uq = np.quantile(power, 0.95)
        lq = np.quantile(power, 0.1)

    if log_power:
        cf = ax.contourf(
            x,
            np.log2(period),
            power,
            levels,
            norm=colors.LogNorm(),
            extend='both',
            cmap=cmap,
        )

    else:
        cf = ax.contourf(
            x,
            np.log2(period),
            power,
            levels,
            extend='both',
            cmap=cmap,
        )

    extent = [x.min(), x.max(), min(period), max(period)]
    if sig95 is not None:
        ax.contour(
            x, np.log2(period), sig95, [-99, 1],
            colors='k', linewidths=2, extent=extent)
    if coi is not None:
        ax.fill(
            np.concatenate(
                [x, x[-1:] + dx, x[-1:] + dx,
                 x[:1] - dx, x[:1] - dx]),
            np.concatenate(
                [
                    np.log2(coi),
                    [1e-9],
                    np.log2(period[-1:]),
                    np.log2(period[-1:]),
                    [1e-9]
                ]
            ),
            'k', alpha=0.3, hatch='x')

    Yticks = 2 ** np.arange(
        np.ceil(np.log2(period.min())),
        np.ceil(np.log2(period.max())))

    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels(Yticks)
    ax.set_ylim(np.log2([period.min(), period.max()]))
    ax.set_xlim(x.min(), x.max())

    return cf


def plot_wv_power(
    signal_norm=None,
    x=None,
    dx=None,
    coi=None,
    period=None,
    power=None,
    glbl_signif=None,
    glbl_power=None,
    sig95=None,
    scale_avg=None,
    scale_avg_signif=None,
    log_power=True,
    levels=None,
    fig_kwargs=None,
    fig_labels=None,
    include_colorbar=True,
    coherence=False,
    cmap='viridis',
    linecolors=None
):

    # Empty allocation of keyword dictionaries
    if fig_labels is None:
        fig_labels = dict()
    if fig_kwargs is None:
        fig_kwargs = dict()

    # Figure labels
    if 'label' in fig_labels:
        label = fig_labels['label']
    else:
        label = ''
    if 'mother' in fig_labels:
        mother = fig_labels['mother']
    else:
        mother = ''
    if 'avg_scales' in fig_labels:
        avg_scales = fig_labels['avg_scales']
    else:
        avg_scales = ''
    if 'scale_units' in fig_labels:
        scale_units = fig_labels['scale_units']
    else:
        scale_units = ''
    if 'units' in fig_labels:
        units = fig_labels['units']
    else:
        units = ''
    if 'xlabel' in fig_labels:
        xlabel = fig_labels['xlabel']
    else:
        xlabel = ''
    if 'title' in fig_labels:
        title = fig_labels['title']
    else:
        title = ''

    # Build figure
    if 'figsize' not in fig_kwargs:
        fig_kwargs['figsize'] = (10, 8)
    fig = plt.figure(**fig_kwargs)

    widths = [2, 0.5]
    if scale_avg is not None:
        nrows = 3
        heights = [1, 2.5, 1]
    else:
        nrows = 2
        heights = [1, 2.5]
    spec = fig.add_gridspec(ncols=2,
                            nrows=nrows,
                            width_ratios=widths,
                            height_ratios=heights,
                            hspace=0.15, wspace=0.15,
                            )
    # First, initiate all subplots
    if scale_avg is not None:
        dax = fig.add_subplot(spec[2, 0])
        ax = fig.add_subplot(spec[0, 0], sharex=dax)
        bax = fig.add_subplot(spec[1, 0], sharex=dax)
        cax = fig.add_subplot(spec[1, 1], sharey=bax)
    else:
        ax = fig.add_subplot(spec[0, 0])
        bax = fig.add_subplot(spec[1, 0], sharex=ax)
        cax = fig.add_subplot(spec[1, 1], sharey=bax)

    if include_colorbar:
        # Inset colorbar for the power map
        axins = inset_axes(
            bax,
            width="25%",  # width = 25% of parent_bbox width
            height="5%",  # height : 5%
            loc='upper right')

    fig.suptitle(title, y=0.95)

    # First sub-plot, the detrended and standardized data.
    # This means we have a cross-wavelet/coherence plot
    if len(np.shape(signal_norm)) > 1:
        if 'signal_labels' in fig_labels:
            slabels = fig_labels['signal_labels']
        else:
            slabels = [None, None]
        if np.size(slabels) < 2:
            slabels = [None, None]
        ax.plot(x, signal_norm[0], color=SRON(2)[0], linewidth=1.5, label=slabels[0])
        ax.plot(x, signal_norm[1], color=SRON(2)[1], linewidth=1.5, label=slabels[1])
        ax.legend()
    else:
        ax.plot(x, signal_norm, 'k', linewidth=1.5)
    ax.set_title('a) Detrended data', loc='left')
    ax.set_ylabel(r'{} [{}]'.format(label, units))
    plt.setp(ax.get_xticklabels(), visible=False)

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    cf = plot_periodogram(
        power,
        x,
        dx,
        period,
        coi=coi,
        sig95=sig95,
        ax=bax,
        levels=levels,
        log_power=log_power,
        cmap=cmap
    )

    if not coherence:
        bax.set_title('b) {} Power Spectrum ({})'.format(label, mother), loc='left')
    else:
        bax.set_title('b) {} Wavelet linear coherence ({})'.format(label, mother), loc='left')
    bax.set_ylabel('Period ({})'.format(scale_units))
    Yticks = 2 ** np.arange(
        np.ceil(np.log2(period.min())),
        np.ceil(np.log2(period.max())))
    bax.set_yticks(np.log2(Yticks))
    bax.set_yticklabels(Yticks)
    if scale_avg is None:
        bax.set_xlim([x.min(), x.max()])
        bax.set_xlabel(xlabel)
    else:
        plt.setp(bax.get_xticklabels(), visible=False)

    if include_colorbar:
        if log_power:
            formatter = ticker.LogFormatterSciNotation(10, labelOnlyBase=True)
        else:
            formatter = None
        cb = plt.colorbar(
            cf,
            cax=axins,
            orientation="horizontal",
            format=formatter,
        )
        if not coherence:
            cb.set_label(r'Power ({}^2)'.format(units), color='w')
        else:
            cb.set_label('Coherence', color='w')
        cb.ax.xaxis.set_minor_locator(ticker.NullLocator())
        cb.ax.xaxis.set_tick_params(color='w')
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color='w')

    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra. Note that period scale is logarithmic.
    if glbl_signif is not None:
        cax.plot(glbl_signif, np.log2(period), 'k--', label='95% sig')
    else:
        glbl_signif = np.array([0])

    if glbl_power is None:
        glbl_power = power.mean(axis=1)
    cax.plot(glbl_power, np.log2(period), 'k-', linewidth=1.5, label='Mean power')
    cax.set_title('c) Global Wavelet Spectrum', loc='left')
    if not coherence:
        cax.set_xlim([0, np.max([glbl_power.max(), glbl_signif.max()])])
        cax.set_xlabel(r'Power [({})^2]'.format(units))
    elif coherence:
        cax.set_xlim([0, 1])
        cax.set_xlabel('Coherence (-)')
    cax.set_ylim(np.log2([period.min(), period.max()]))
    cax.set_yticks(np.log2(Yticks))
    cax.set_yticklabels(Yticks)
    plt.setp(cax.get_yticklabels(), visible=False)

    # Fourth sub-plot, the scale averaged wavelet spectrum.
    if scale_avg is not None:

        # Line colors for each scale averaging
        num_intervals = np.shape(scale_avg)[1]
        if linecolors is None:
            colors = SRON(num_intervals)
        else:
            colors = linecolors

        for nint in np.arange(num_intervals):
            label = '{bs}{p_units} to {ts}{p_units}'.format(
                bs=avg_scales[nint][0],
                ts=avg_scales[nint][1],
                p_units=scale_units
            )

            if scale_avg_signif is not None:
                dax.axhline(
                    scale_avg_signif[nint],
                    color=colors[nint],
                    linestyle='--',
                    linewidth=1.
                )

            dax.plot(
                x,
                scale_avg[:, nint],
                color=colors[nint],
                linewidth=1.5,
                label=label
            )

        if not coherence:
            dax.set_title('d) Scale-averaged power', loc='left')
            dax.set_ylabel(r'Average variance [{}]'.format(units))
        else:
            dax.set_title('d) Scale-averaged coherence', loc='left')
            dax.set_ylabel(r'Average coherence')
            dax.set_ylim([0, 1])
        dax.legend(loc='lower right')
        dax.set_xlabel(xlabel)
        dax.set_xlim([x.min(), x.max()])
