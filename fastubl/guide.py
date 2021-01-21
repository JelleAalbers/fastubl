import numpy as np
from scipy import stats
from scipy.special import xlogy

import fastubl

export, __all__ = fastubl.exporter()
__all__ += ['DEFAULT_MU_REFERENCE']

DEFAULT_MU_REFERENCE = 10.
lookup = fastubl.lookup_axis1

@export
class Guide:

    def guide(self, left, right, n_observed, p_obs, present, bg_mus, acceptance):
        raise NotImplementedError

    def __call__(self, x_obs, p_obs, present, dists, bg_mus, domain):
        """Find best interval in the data that minimizes a guide function

        :param x_obs: (trial_i, event_j)
        :param p_obs (trial_i, event_j, source_k)
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
        assert x_obs.shape == present.shape
        assert len(bg_mus) == len(dists) - 1
        bg_mus = np.asarray(bg_mus)
        n_events = present.sum(axis=1)
        nmax = n_events.max()

        x, present, p_obs = fastubl.endpoints(x_obs, present, p_obs, domain)
        n_endpoints = nmax + 2
        left, right = fastubl.interval_indices(n_endpoints)

        # Intervals end just before the events, so -1 here, not +1!
        n_observed = right - left - 1  # (indices)

        # Find acceptances (fraction of surviving events) of each interval
        # (trial, interval, source)
        acceptance = fastubl.interval_acceptances(x, left, right, dists)

        # Obtain guide results; (trial, interval) array
        guide_results = self.guide(
            left=left,
            right=right,
            n_observed=n_observed,
            p_obs=p_obs,
            present=present,
            bg_mus=bg_mus,
            acceptance=acceptance)

        # Find lowest guide value. This is still an (n_trials,) array!
        i = np.argmin(guide_results, axis=1)

        # Interval/event indices for each trial; (n_trials,) arrays
        li, ri = left[i], right[i]

        return dict(interval_indices=(li, ri),
                    n_observed=n_observed[i],
                    guide_results=lookup(guide_results, i),
                    interval_bounds=(lookup(x, li), lookup(x, ri)),
                    # Get (trials, sources) array of acceptances of the interval
                    acceptance=lookup(acceptance, i))


@export
class PoissonGuide(Guide):
    """Guide to the interval that would give the best Poisson upper limit.
    """
    def __init__(self,
                 pcl_sigma=None,
                 optimize_for_cl=fastubl.DEFAULT_CL):
        self.optimize_for_cl = optimize_for_cl
        self.pcl_sigma = pcl_sigma
        # Lookup array of Poisson ULs
        self.poisson_uls_flat = fastubl.poisson_ul(np.arange(1000),
                                                   cl=self.optimize_for_cl)

    def guide(self, left, right, n_observed, p_obs, present, bg_mus, acceptance):
        n_trials, _, _ = acceptance.shape

        # Compute background expectation in each interval
        if len(bg_mus):  # It's a numpy array, can't do 'if bg_mus'
            # Get expected background events inside all intervals
            # (trial, interval)
            mu_bg = (bg_mus[np.newaxis, np.newaxis, :]
                     * acceptance[:, :, 1:]).sum(axis=2)
        else:
            mu_bg = np.zeros((n_trials, len(left)))

        assert n_observed.min() == 0
        n_limit = self.poisson_uls_flat[n_observed] - mu_bg

        # Handle background underfluctuations
        if self.pcl_sigma is None:
            # Clip to poisson_uls_flat[0]
            n_limit = n_limit.clip(self.poisson_uls_flat[0], None)
        else:
            # PCL, with Gaussian approx to poisson
            power_constraint = (
                (n_observed - mu_bg).clip(0, None)
                + np.sqrt(mu_bg) * stats.norm.ppf(self.optimize_for_cl))
            n_limit = n_limit.clip(power_constraint, None)

        with np.errstate(all='ignore'):
            # 0 acceptance -> no limit on mu (infinity)
            return np.where(acceptance[:, :, 0] == 0,
                            float('inf'),
                            n_limit / acceptance[:, :, 0])


@export
class LikelihoodGuide(Guide):
    """Guide to the interval that would most favor low over high
    signal hypotheses in a likelihood-ratio test.
    """

    def __init__(self, mu_low, mu_high):
        # TODO: mu_low and mu_high should be in Neyman hash!
        self.mu_low, self.mu_high = mu_low, mu_high

    def guide(self, left, right, n_observed, p_obs, present, bg_mus, acceptance):
        # Compute - log L(mu_low)/L(mu_high) for each interval
        #   0: fit equally well
        #   +: favors mu_high
        #   -: favors mu_low -> Good, we want this interval!

        # Compute differential rate (p * mu) for each event and source
        # Note this does not depend on the interval -- cuts reduce mu
        # and increase p_obs.
        # (trial, event, source) array
        dr_mu_low = (p_obs * np.concatenate([[self.mu_low], bg_mus]).reshape((1, 1, -1))).sum(axis=2)
        dr_mu_high = (p_obs * np.concatenate([[self.mu_high], bg_mus]).reshape((1, 1, -1))).sum(axis=2)

        # Inner term of log L = sum_events log(dr mu=0) - sum_events log(dr)
        # = sum_events [log(dr mu=0) - log(dr)]
        # Start with cumulative sum in brackets...
        # (trials, events) array
        cumsum_inner = np.cumsum(
            xlogy(present, dr_mu_low) - xlogy(present, dr_mu_high),
            axis=1)

        # ... then use interval indices to get only sum over included events
        # right <= left "intervals" are already removed
        # right = left + 1 means no events in the interval -> 0
        # (trials, intervals) array
        inner = cumsum_inner[:,right] - cumsum_inner[:,left + 1]

        result = -(
            # -mu term: background is the same under both hypotheses
            ((-self.mu_low) - (-self.mu_high)) * acceptance[:,:,0]
            + inner)
        return result
