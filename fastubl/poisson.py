import numpy as np
from multihist import poisson_central_interval
from scipy import stats, optimize

import fastubl
export, __all__ = fastubl.exporter()


@export
def poisson_ul(n, mu_bg=0, cl=fastubl.DEFAULT_CL):
    """Upper limit on mu_signal, from observing n events
    where mu_bg background events were expected

    NB: can be negative if mu_bg large enough.
    It's your responsibility to clip to 0...
    """
    return stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg


@export
class Poisson(fastubl.StatisticalProcedure):

    def compute_intervals(self, r,
                          kind=fastubl.DEFAULT_KIND,
                          cl=fastubl.DEFAULT_CL):
        n = r['present'].sum(axis=1)
        mu_bg = np.sum(self.true_mu[1:])
        if kind == 'upper':
            return np.zeros(r['n_trials']), poisson_ul(n, mu_bg, cl=cl)
        elif kind == 'central':
            return [x - mu_bg
                    for x in poisson_central_interval(n, cl=cl)]
        raise NotImplementedError(kind)


@export
class OptimalCutPoisson(fastubl.StatisticalProcedure):
    """Poisson upper limit, computed on a pre-determined restricted interval,
    which gives optimal no-signal 90% upper limits.
    """
    interval: np.ndarray
    fraction_in_interval: np.ndarray

    def __init__(self, *args,
                 optimize_for_cl=fastubl.DEFAULT_CL,
                 interval_guess=(0.1, 1.),
                 **kwargs):
        super().__init__(*args, **kwargs)

        if np.sum(self.true_mu[1:]) == 0:
            # No background: include all events
            self.interval = np.array([-float('inf'), float('inf')])
        else:
            # Find interval maximizing mean exclusion limit
            optresult = optimize.minimize(
                self.mean_sensitivity,
                interval_guess,
                args=(0, optimize_for_cl))
            if not optresult.success:
                print(
                    f"Optimization failed after {optresult.nfev} iterations! "
                    f"Current value: {optresult.fun}; "
                    f"message: {optresult.message}.\n")
            self.interval = optresult.x
        self.fraction_in_interval = self.compute_fraction_in_interval(
            self.interval)

    def mean_sensitivity(self,
                         interval=(-np.float('inf'), np.float('inf')),
                         mu_s_true=None,
                         cl=fastubl.DEFAULT_CL):
        """Return expected upper limit on mu_s

        """
        # Compute expected events in interval
        if mu_s_true is None:
            mu_s_true = self.true_mu[0]
        frac = self.compute_fraction_in_interval(interval)
        mu_bg = (self.true_mu[1:] * frac[1:]).sum()
        mu = mu_s_true * frac[0] + mu_bg
        if mu == 0.:
            # This will give a floating point warning in the optimizer...
            # oh well.
            return float('inf')

        # Sum over a *very* generous range of ns.
        # Restricting this too much would make the function discontinuous,
        # and thus hard to optimize.
        # Note the +1, since ppfs both give 0 if mu very small.
        small_alpha = (1 - cl) / 1000
        n = np.arange(stats.poisson(mu).ppf(small_alpha),
                      stats.poisson(mu).ppf(1 - small_alpha) + 1)

        return np.sum(
            stats.poisson(mu).pmf(n) *
            poisson_ul(n, mu_bg, cl=cl)) / frac[0]

    def compute_fraction_in_interval(self, interval):
        """Return fraction of events expected in (left, right) interval
        for each of the model's distributions
        """
        return np.array([d.cdf(interval[1]) - d.cdf(interval[0])
                         for d in self.dists])

    def compute_intervals(self, r,
                          kind=fastubl.DEFAULT_KIND,
                          cl=fastubl.DEFAULT_CL):
        # Count observed events inside the interval
        mask = (r['present']
                & (self.interval[0] <= r['x_obs'])
                & (r['x_obs'] < self.interval[1]))
        n = mask.sum(axis=1)

        # Expected BG in interval
        mu_bg = np.sum(self.true_mu[1:] * self.fraction_in_interval[1:])

        # Signal fraction in interval
        fs = self.fraction_in_interval[0]

        if kind == 'upper':
            return (np.zeros(r['n_trials']),
                    poisson_ul(n, mu_bg, cl=cl) / fs)
        elif kind == 'central':
            return [(x - mu_bg) / fs
                    for x in poisson_central_interval(n, cl=cl)]
        raise NotImplementedError(kind)
