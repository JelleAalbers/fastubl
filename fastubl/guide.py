import numpy as np

import fastubl

export, __all__ = fastubl.exporter()


@export
class Guide:

    def guide(self, n_observed, mu_bg, acceptance):
        raise NotImplementedError

    def __call__(self, x_obs, present, dists, bg_mus, cl=fastubl.DEFAULT_CL):
        """Find best interval in the data that maximizes a guide function

        :param x_obs: (trial_i, event_j)
        :param present: (trial_i, event_j)
        :param dists: signal and background distributions
        :param bg_mus: expected events for backgrounds

        :return: Dict with
            guide_results: Maximum value of the guide function, (n_trials) array
            n_observed: Events counted in intervals, (n_trials) array
            interval_indices: (left, right) event indices (inclusive) of intervals
                each (n_trials) arrays.
                0 = start of domain (e.g. -float('inf')
                1 = at first event
                ...
            interval_bounds: left, right x_obs boundaries of intervals
                each (n_trials) arrays
            acceptance: (n_trials, n_sources) array of interval acceptance
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

        # Get matrices of inclusive (left, right) indices of all intervals.
        # With the addition of two 'events' at the bounds, we have nmax+2 valid
        # indices to consider.
        # left indices change quickly, right indices repeat before changing.
        left, right = np.meshgrid(np.arange(nmax + 2), np.arange(nmax + 2))
        left, right = left.ravel(), right.ravel()

        # Events observed inside interval; negative for invalid intervals.
        n_observed = right - left - 1  # (indices)

        # Find acceptances (fraction of surviving events) of each interval
        # (trial, event, source)
        _cdfs = np.stack([dist.cdf(x) for dist in dists], axis=2)
        acceptance = _cdfs[:, right, :] - _cdfs[:, left, :]
        assert len(acceptance.shape) == 3

        if len(bg_mus):  # It's a numpy array, can't do 'if bg_mus'
            # Get expected background events inside all intervals
            # (trial, interval)
            mu_bg = (bg_mus[np.newaxis, np.newaxis, :]
                     * acceptance[:, :, 1:]).sum(axis=2)
        else:
            mu_bg = np.zeros((n_trials, len(left)))

        # Obtain Poisson upper limits; (trial, interval) array
        guide_results = np.where(
            n_observed >= 0,
            self.guide(n_observed, mu_bg, acceptance),
            float('inf'))

        # Find lowest Poisson UL. This is still an (n_trials,) array!
        i = np.argmin(guide_results, axis=1)

        # Interval/event indices for each trial; (n_trials,) arrays
        li, ri = left[i], right[i]

        # Get (trials, sources) array of acceptances of the interval
        lookup = fastubl.lookup_axis1

        return dict(guide_results=lookup(guide_results, i),
                    interval_indices=(li, ri),
                    interval_bounds=(lookup(x, li), lookup(x, ri)),
                    n_observed=n_observed[i],
                    acceptance=lookup(acceptance, i))


@export
class PoissonGuide(Guide):
    def __init__(self):
        # Cache of Poisson ULs
        self.poisson_uls_flat = fastubl.poisson_ul(np.arange(1000))

    def guide(self, n_observed, mu_bg, acceptance):
        n_observed = n_observed.clip(0, None)
        n_limit = self.poisson_uls_flat[n_observed] - mu_bg
        # Note: clip to poisson_uls_flat[0] to avoid
        # attraction to background underfluctuations
        n_limit = n_limit.clip(self.poisson_uls_flat[0], None)
        return n_limit / acceptance[:, :, 0]


# class LikelihoodGuide(Guide):
#
#     def __init__(self, ll):
#         self.ll = ll
#
#