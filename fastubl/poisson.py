import numpy as np
from multihist import poisson_central_interval
from tqdm import tqdm
from scipy import stats, optimize

import fastubl
export, __all__ = fastubl.exporter()


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
                 method='optimize',
                 domain=(0., 1.),
                 **kwargs):
        super().__init__(*args, **kwargs)

        if np.sum(self.true_mu[1:]) == 0:
            # No background: include full range
            self.interval = np.array([-float('inf'), float('inf')])

        elif method == 'bruteforce':
            # Find interval maximizing mean exclusion limit by bruteforce
            x = np.linspace(*domain, num=101)
            left = np.repeat(x, len(x)).ravel()
            right = np.tile(x, len(x)).ravel()
            valid = right > left
            left, right = left[valid], right[valid]
            max_i = np.argmin([
                self.mean_sensitivity((l, r), cl=optimize_for_cl)
                for l, r in tqdm(zip(left, right),
                                 total=len(left),
                                 desc='Bruteforce-optimizing domain')])
            self.interval = np.asarray([left[max_i], right[max_i]])

        elif method =='optimize':
            # Find it by optimization
            # Have you sacrificed to the optimizer gods today?
            guess = np.asarray(domain)
            guess[0] += np.diff(guess)/4
            guess[1] -= np.diff(guess)/4
            optresult = optimize.minimize(
                self.mean_sensitivity,
                x0=guess,
                bounds=(domain, domain))
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
        small_alpha = (1 - cl) / 1e4
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



@export
class SacrificialPoisson(fastubl.StatisticalProcedure):
    """Poisson upper limit computed on an interval optimized on training data,
    which is excluded from the inference.
    """
    def __init__(self, *args,
                 sacrifice_f=0.3,
                 optimize_for_cl=fastubl.DEFAULT_CL,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sacrifice_f = sacrifice_f
        self.guide = fastubl.PoissonGuide(optimize_for_cl)

    def compute_intervals(self, r,
                          kind=fastubl.DEFAULT_KIND,
                          cl=fastubl.DEFAULT_CL):
        # Peel off training data
        # TODO: should be deterministic, e.g. set seed based on something..?
        is_training = np.random.rand(*r['present'].shape) < self.sacrifice_f

        # Find best Poisson limit on training data
        best_poisson = self.guide(
            x_obs=r['x_obs'],
            present=r['present'] & is_training,
            p_obs=r['p_obs'],
            dists=self.dists,
            bg_mus=self.true_mu[1:] * self.sacrifice_f)

        # Compute Poisson limit on test data in the same interval
        left, right = best_poisson['interval_bounds']
        n = (r['present']
             & ~is_training
             & (left[:, np.newaxis] < r['x_obs'])
             & (r['x_obs'] < right[:, np.newaxis])).sum(axis=1)

        f_sig = best_poisson['acceptance'][:, 0] * (1 - self.sacrifice_f)
        mu_bg = (self.true_mu[np.newaxis,1:] * best_poisson['acceptance'][:,1:]).sum(axis=1) \
                * (1 - self.sacrifice_f)

        if kind == 'upper':
            return (np.zeros(r['n_trials']),
                    fastubl.poisson_ul(n, mu_bg, cl=cl)/f_sig)
        elif kind == 'central':
            return [(x - mu_bg)/f_sig
                    for x in poisson_central_interval(n, cl=cl)]
        raise NotImplementedError(kind)


@export
def poisson_ul(n, mu_bg=0, cl=fastubl.DEFAULT_CL):
    """Upper limit on mu_signal, from observing n events
    where mu_bg background events were expected

    NB: can be negative if mu_bg large enough.
    It's your responsibility to clip to 0...
    """
    return stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg
