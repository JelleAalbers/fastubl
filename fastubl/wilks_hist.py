from multihist import Hist1d

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

import fastubl

__all__ = ['wilks_hist']


default_percentiles = (
    (50, '50%'),
    (90, '90%'),
    (100 * (1 - stats.norm.cdf(-2)), '2$\sigma$'),
    (100 * (1 - stats.norm.cdf(-3)), '3$\sigma$'))

theory_colors = dict(wilks='darkorange',
                     chernoff='seagreen')

def wilks_hist(result, bins=None,
               show_percentiles=None,
               signed=False,
               show_theory=('wilks',)):
    if show_percentiles is None:
        show_percentiles = default_percentiles
    if not show_percentiles:
        show_percentiles = tuple()

    if isinstance(show_theory, str):
        show_theory = (show_theory,)

    if bins is None:
        if signed:
            bins = np.linspace(-15, 15, 100)
        else:
            bins = np.linspace(-1, 15, 100)

    h = Hist1d(result, bins=bins)
    x = h.bin_centers
    y = h.histogram
    plt.fill_between(x, y - y ** 0.5, y + y ** 0.5,
                     color='b', label='Simulation',
                     alpha=0.4, step='mid', linewidth=0)
    plt.plot(x, y, drawstyle='steps-mid', color='b', linewidth=0.5)

    wilks_y = np.diff(fastubl.wilks_t_cdf(bins, abs=not signed)) * h.n
    chernoff_y0 = (lookup(0, x, wilks_y) + h.n) / 2

    if 'wilks' in show_theory:
        plt.plot(x,
                 wilks_y,
                 color=theory_colors['wilks'], label='Wilks')
    if 'chernoff' in show_theory:
        plt.plot(x,
                 wilks_y / 2,
                 color=theory_colors['chernoff'], label='Chernoff')
        plt.scatter(0, chernoff_y0,
                    marker='.', color=theory_colors['chernoff'])

    plt.yscale('log')
    plt.ylabel("Toys / bin")
    plt.ylim(0.8, None)
    plt.gca().yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter('%g'))

    plt.xlabel("-2 $\log ( L({\mu_s}^{*}) / L(\hat{\mu_s}) )$")
    plt.xlim(h.bin_edges[0], h.bin_edges[-1])

    plt.legend(loc='upper right')

    ax = plt.gca()
    t1 = ax.transData
    t2 = ax.transAxes.inverted()

    def data_to_axes(x, y):
        return t2.transform(t1.transform((x, y)))

    def pc_line(x, y, label=None, color='b', alpha=0.8):
        plt.axvline(x,
                    ymax=data_to_axes(x, y)[1],
                    color=color, alpha=alpha, linewidth=0.5)
        if label:
            plt.text(x + 0.15, .9, label,
                     rotation=90,
                     horizontalalignment='left',
                     verticalalignment='bottom')

    for pc, label in show_percentiles:
        x = np.percentile(result, pc)
        y = h.lookup(x)
        pc_line(x, y, label=label, color='k', alpha=1)

        if 'wilks' in show_theory:
            x = fastubl.wilks_t_ppf(pc / 100, abs=not signed)
            y = lookup(x, h.bin_centers, wilks_y)
            pc_line(x, y, color=theory_colors['wilks'])

        if 'chernoff' in show_theory:
            if pc <= 50:
                x = 0
                y = chernoff_y0
            else:
                x = fastubl.wilks_t_ppf(1 - 2 * (1 - pc/100), abs=not signed)
                y = lookup(x, h.bin_centers, wilks_y) / 2
            pc_line(x, y, color=theory_colors['chernoff'])


def lookup(x, xp, yp):
    # TODO: use searchsorted instead?
    return yp[np.argmin(np.abs(x - xp))]
