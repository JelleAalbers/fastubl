import numpy as np
from scipy import stats

import fastubl

export, __all__ = fastubl.exporter()


@export
class PoissonSeeker(fastubl.NeymanConstruction):
    """Use P(more events in interval), in interval with lowest Poisson upper
    limit.
    """
    def __init__(self, *args, optimize_for_cl=0.9, **kwargs):
        self.optimize_for_cl = optimize_for_cl
        super().__init__(*args, **kwargs)

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)
        if 'best_poisson' not in r:
            r['best_poisson'] = best_poisson_limit(
                r['x_obs'],
                r['present'],
                dists=self.dists,
                bg_mus=self.true_mu[1:],
                cl=self.optimize_for_cl)

        bp = r['best_poisson']
        pmax = stats.poisson(bp['f_sig'] * mu_null + bp['mu_bg'])\
                .sf(bp['n_observed'])

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))


@export
def best_poisson_limit(x_obs, present, dists, bg_mus, cl=fastubl.DEFAULT_CL):
    """Find best Poisson limit among intervals

    :param x: (trial_i, event_j)
    :param present: (trial_i, event_j)
    :param dists: signal and background distributions
    :param bg_mus: expected events for backgrounds

    :return: (p, (li, ri), (lx, rx)), each n_trials arrays of:
        p: p of containing more events than expected
        li, ri: left, right event indices (inclusive) of intervals
            0 = start of domain (e.g. -float('inf')
            1 = at first event
            ...
        lx, rx: left, right x_obs boundaries of intervals
    """
    assert len(bg_mus) == len(dists) - 1
    n_events = present.sum(axis=1)
    nmax = n_events.max()

    # Map fake events to inf, where they won't affect the method.
    # Sort by ascending x values, then add fake events at (-inf, inf)
    # (This is similar to 'Yellin normalize' in yellin.py)
    x = np.where(present, x_obs, float('inf'))
    x = np.sort(x, axis=-1)
    n_trials = x.shape[0]
    x = np.concatenate([-np.ones((n_trials, 1)) * float('inf'),
                        x,
                        np.ones((n_trials, 1)) * float('inf')],
                       axis=1)

    # Get CDF * mu for the backgrounds at each of the observed events.
    # this is a (trial_i, event_j) matrix, like x
    if not len(bg_mus):
        cdf_mu_bgs = np.zeros_like(x)
    else:
        cdf_mu_bgs = np.stack([
            dist.cdf(x) * mu
            for dist, mu in zip(dists[1:], bg_mus)],
            axis=2).sum(axis=2)

    # Lookup signal cdf at the events
    cdf_sig = dists[0].cdf(x)

    # Poisson upper limits for all event counts we need consider
    poisson_uls_flat = fastubl.poisson_ul(np.arange(nmax+1))

    # Get matrices of inclusive (left, right) indices of all intervals.
    # With the addition of two 'events' at the bounds, we have nmax+2 valid
    # indices to consider.
    # left indices change quickly, right indices repeat before changing.
    left, right = np.meshgrid(np.arange(nmax+2), np.arange(nmax+2))
    left, right = left.ravel(), right.ravel()

    # Events observed inside interval; negative for invalid intervals.
    n_observed = right - left - 1

    # Get expected background events inside all intervals
    mu_bg = cdf_mu_bgs[:, right]  - cdf_mu_bgs[:, left]
    f_sig = cdf_sig[:, right] - cdf_sig[:, left]

    # Obtained Poisson upper limit

    poisson_uls = np.where(
        n_observed >= 0,
        (poisson_uls_flat[n_observed.clip(0, None)] - mu_bg) \
            # Note: clip to poisson_uls_flat[0] to avoid
            # attraction to background underfluctuations
            .clip(poisson_uls_flat[0], None) / f_sig,
        float('inf'))

    # Find lowest Poisson UL
    i = np.argmin(poisson_uls, axis=1)
    li, ri = left.ravel()[i], right.ravel()[i]
    lookup = fastubl.lookup_axis1
    lx, rx = lookup(x, li), lookup(x, ri)
    return dict(poisson_ul=lookup(poisson_uls, i),
                interval_indices=(li, ri),
                interval_bounds=(lx, rx),
                n_observed=n_observed[i],
                mu_bg=lookup(mu_bg, i),
                f_sig=lookup(f_sig, i))
