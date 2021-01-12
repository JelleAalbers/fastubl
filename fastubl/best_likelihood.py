import numpy as np
from scipy import special, stats

import fastubl

export, __all__ = fastubl.exporter()


@export
class BestZech(fastubl.NeymanConstruction):

    def statistic(self, r, mu_null):
        if 'p_fewer_0' not in r:
            r['p_fewer_0'] = self.p_fewer(r, 0)

        # p(fewer events): high value = excess, so take min over intervals
        score = self.p_fewer(r, mu_null) / r['p_fewer_0']
        best_i = np.argmin(score, axis=1)

        trials = np.arange(r['n_trials'])
        r['interval'] = (
            r['all_intervals']['left'][trials, best_i],
            r['all_intervals']['right'][trials, best_i],
            r['all_intervals']['n_observed'][best_i])

        return score[trials, best_i]

    def p_fewer(self, r, mu_signal):
        x_obs, p_obs, present = r['x_obs'], r['p_obs'], r['present']
        assert x_obs.shape == present.shape
        n_trials, n_max_events, n_sources = p_obs.shape

        if 'all_intervals' not in r:
            x_endpoints = fastubl.endpoints(x_obs, present, p_obs, self.domain, only_x=True)
            n_endpoints = n_max_events + 2

            left, right = interval_indices(n_endpoints)
            acceptance = interval_acceptances(x_endpoints, left, right, self.dists)
            r['all_intervals'] = dict(left_i=left,
                                      left=x_endpoints[:, left],
                                      right=x_endpoints[:, right],
                                      right_i=right,
                                      # If right 1 bigger than left, 0 observed
                                      n_observed=right - left - 1,
                                      acceptance=acceptance)

        # TODO: this should be factored out, it's done elsewhere
        mu = self.true_mu.copy()
        mu[0] = mu_signal

        # Get (trials, intervals) array of total mu
        mu_total = (mu[None,None,:] * r['all_intervals']['acceptance']).sum(axis=2)

        return stats.poisson(mu_total).cdf(r['all_intervals']['n_observed'][None,:])


@export
class BestLikelihood(fastubl.NeymanConstruction):

    def statistic(self, r, mu_null):
        mu_i = np.searchsorted(self.mu_s_grid, mu_null)
        assert self.mu_s_grid[mu_i] == mu_null, "Can only test mus on grid"

        if 'itv_ll' not in r:
            self.compute_ll_grid(r)

        # Compute (trial, interval) array of ts
        # itv_ll: (trial, interval, mu) array
        # itv_mu_best and itv_ll_best: (trial, interval) arrays
        ts = (-2
              * np.sign(r['itv_mu_best'] - mu_null)
              * r['itv_llr'][:,:,mu_i])

        # Low t -> strong deficit. Want minimum t across intervals
        return ts.min(axis=1)

    def compute_ll_grid(self, r):
        """Computes log L(mu_s) in each interval, for all mu_s in grid
        """
        x_obs, p_obs, present = r['x_obs'], r['p_obs'], r['present']
        assert x_obs.shape == present.shape
        n_trials, n_max_events, n_sources = p_obs.shape

        x_obs, p_obs, present = fastubl.endpoints(x_obs, present, p_obs, self.domain)
        n_endpoints = n_max_events + 2

        left, right = interval_indices(n_endpoints)
        n_intervals = left.size
        acceptance = interval_acceptances(x_obs, left, right, self.dists)

        # Compute grid of mus for all hypotheses (mu_i, source)
        # Add a few hypotheses higher than self.mu_s_grid;
        # without this t for final mu would often be 0
        # TODO: Do we need as many as 10? We're looking at sparse intervals
        mu_s_grid = np.concatenate([
            self.mu_s_grid,
            self.mu_s_grid[-1] * np.geomspace(1.1, 10, 10)])
        n_mus = mu_s_grid.size
        mu_grid = np.concatenate([
            mu_s_grid[:,None],
            np.tile(self.true_mu[1:], n_mus).reshape(n_mus, n_sources - 1)],
            axis=1)
        assert mu_grid.shape == (n_mus, n_sources)

        # Compute total mu per interval
        # (trial, interval, mu_i)
        mu_tot = np.sum(mu_grid[None,None,:,:] * acceptance[:,:,None,:],
                        axis=-1)
        assert mu_tot.shape == (n_trials, n_intervals, n_mus)

        # Compute differential rate sum_sources (p * mu) for each event and mu
        # Note this does not depend on the interval -- cuts reduce mu
        # and increase p_obs.
        # (trial, event, mu_i) array
        #       Inside sum: (trial, event, mu_i, source)
        dr_mu = (p_obs[:,:,None,:] * mu_grid[None,None,:,:]).sum(axis=-1)
        assert dr_mu.shape == (n_trials, n_endpoints, n_mus)

        # Inner term of log L = sum_events log(sum_sources dr)
        # Start with cumulative sum over events...
        # (trial, event, mu_i)
        cumsum_inner = np.cumsum(
            special.xlogy(present[:,:,None], dr_mu),
            axis=1)
        del dr_mu
        assert cumsum_inner.shape == (n_trials, n_endpoints, n_mus)

        # ... then use interval indices to get only sum over included events
        # right <= left "intervals" are already removed
        # right = left + 1 means no events in the interval -> 0
        # (trial, interval, mu_i) array
        r['itv_ll'] = -mu_tot + (
                cumsum_inner[:,right,:] - cumsum_inner[:,left + 1,:])
        del cumsum_inner
        assert r['itv_ll'].shape == (n_trials, n_intervals, n_mus)

        # Find best mu in each interval
        # (trial, interval) array
        # TODO: fancy indexing might beat doing min twice
        r['itv_mu_best'] = mu_s_grid[np.argmax(r['itv_ll'], axis=2)]
        r['itv_ll_best'] = np.max(r['itv_ll'], axis=2)

        r['itv_llr'] = r['itv_ll'] - r['itv_ll_best'][:,:,None]

        # For diagnosis/inspection
        r['left'], r['right'] = left, right



@export
def interval_indices(n_endpoints):
    """Return (left, right) index array of all valid intervals
    :param n_endpoints: number of 'bin edges' / interval endpoints
    :return: tuple of integer ndarrays, right > left
    """
    # left indices change quickly, right indices repeat before changing.
    left, right = np.meshgrid(np.arange(n_endpoints, dtype=np.int),
                              np.arange(n_endpoints, dtype=np.int))
    left, right = left.ravel(), right.ravel()
    valid = right > left
    return left[valid], right[valid]


@export
def interval_acceptances(x_obs_endpoints, left, right, dists):
    """Return (n_trials, n_intervals, n_sources) array of acceptances
    (fraction of surviving events) in each interval
    """
    _cdfs = np.stack([dist.cdf(x_obs_endpoints) for dist in dists], axis=2)
    acceptance = _cdfs[:, right, :] - _cdfs[:, left, :]
    assert len(acceptance.shape) == 3
    return acceptance
