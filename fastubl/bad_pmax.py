import numpy as np
from scipy import stats

import fastubl

export, __all__ = fastubl.exporter()


@export
class PMaxBG(fastubl.NeymanConstruction):
    """Use p(more events) in the interval with largest underfluctuation

    Performs poorly! It selects intervals with large underfluctuations
    far away from the signal

    (Also slow, has to consider all intervals every mu_null)
    """

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)

        pmax, *_ = sparsest_interval(
            r['x_obs'],
            r['present'],
            dists=self.dists,
            mus=np.concatenate([[mu_null], self.true_mu[1:]]))

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))


@export
def sparsest_interval(x_obs, present, dists, mus):
    """Find interval that was most likely to contain more events
    than observed.

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
    assert len(mus) == len(dists)
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

    # Get CDF * mu at each of the observed events.
    # this is a (trial_i, event_j) matrix, like x
    cdf_mu = np.sum(np.stack([
        dist.cdf(x) * mu
        for dist, mu in zip(dists, mus)],
        axis=2), axis=2)

    # Get matrices of inclusive (left, right) indices of all intervals.
    # With the addition of two 'events' at the bounds, we have nmax+2 valid
    # indices to consider.
    # left indices change quickly, right indices repeat before changing.
    left, right = np.meshgrid(np.arange(nmax+2), np.arange(nmax+2))
    left, right = left.ravel(), right.ravel()

    # Events observed inside interval; negative for invalid intervals.
    n_observed = right - left - 1

    # Get expected events inside all intervals
    mu = cdf_mu[:, right]  - cdf_mu[:, left]

    # TODO: much of the above can be cached ... but would that actually help?

    # P of observing more events inside the interval
    p_more_events = np.where(
        n_observed >= 0,
        stats.poisson(mu).sf(n_observed),
        -1)

    # Find highest p_more_events
    i = np.argmax(p_more_events, axis=1)
    li, ri = left.ravel()[i], right.ravel()[i]
    lookup = fastubl.lookup_axis1
    lx, rx = lookup(x, li), lookup(x, ri)
    return lookup(p_more_events, i), (li, ri), (lx, rx)
