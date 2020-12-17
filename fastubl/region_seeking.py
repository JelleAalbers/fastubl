from contextlib import contextmanager

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

        mu_bg = np.sum(r['best_poisson']['acceptance'][:,1:]
                       * np.array(self.true_mu[1:])[np.newaxis, :],
                       axis=1)
        mu_sig = mu_null * r['best_poisson']['acceptance'][:,0]

        bp = r['best_poisson']
        pmax = stats.poisson(mu_sig + mu_bg).sf(bp['n_observed'])

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))


@export
class PoissonGuidedLikelihood(fastubl.UnbinnedLikelihoodExact):
    """Likelihood inside interval found by Poisson seeker
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

            # TODO: compute new mus and pobs here, to save time?

        # Mask out data, except in the interval
        intervals = r['best_poisson']['interval_bounds']
        in_interval = ((intervals[0][:,np.newaxis] < r['x_obs'])
                       & (r['x_obs'] < intervals[1][:,np.newaxis]))

        new_r = {
            **r,
            'present': r['present'] & in_interval,
            'acceptance': r['best_poisson']['acceptance']}
        result = super().statistic(new_r, mu_null)
        # Add any new keys computed by the statistic (e.g. mu_best)
        for k in new_r.keys():
            if k not in r:
                r[k] = new_r[k]
        return result


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
    bg_mus = np.asarray(bg_mus)
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

    # Poisson upper limits for all event counts we need consider
    poisson_uls_flat = fastubl.poisson_ul(np.arange(nmax+1))

    # Get matrices of inclusive (left, right) indices of all intervals.
    # With the addition of two 'events' at the bounds, we have nmax+2 valid
    # indices to consider.
    # left indices change quickly, right indices repeat before changing.
    left, right = np.meshgrid(np.arange(nmax+2), np.arange(nmax+2))
    left, right = left.ravel(), right.ravel()

    # Events observed inside interval; negative for invalid intervals.
    n_observed = right - left - 1                 # (indices)

    # Find acceptances (fraction of surviving events) of each interval
    # (trial, event, source)
    _cdfs = np.stack([dist.cdf(x) for dist in dists], axis=2)
    acceptance = _cdfs[:, right, :] - _cdfs[:, left, :]
    assert len(acceptance.shape) == 3

    if bg_mus:
        # Get expected background events inside all intervals
        # (trial, interval)
        mu_bg = (bg_mus[np.newaxis, np.newaxis, :]
                 * acceptance[:, :, 1:]).sum(axis=2)
    else:
        mu_bg = np.zeros((n_trials, len(left)))

    # Obtain Poisson upper limits; (trial, interval) array
    poisson_uls = np.where(
        n_observed >= 0,
        (poisson_uls_flat[n_observed.clip(0, None)] - mu_bg) \
            # Note: clip to poisson_uls_flat[0] to avoid
            # attraction to background underfluctuations
            .clip(poisson_uls_flat[0], None) / acceptance[:,:,0],
        float('inf'))

    # Find lowest Poisson UL. This is still an (n_trials,) array!
    i = np.argmin(poisson_uls, axis=1)

    # Interval/event indices for each trial; (n_trials,) arrays
    li, ri = left[i], right[i]

    # Get (trials, sources) array of acceptances of the interval
    lookup = fastubl.lookup_axis1

    return dict(poisson_ul=lookup(poisson_uls, i),
                interval_indices=(li, ri),
                interval_bounds=(lookup(x, li), lookup(x, ri)),
                n_observed=n_observed[i],
                acceptance=lookup(acceptance, i))
