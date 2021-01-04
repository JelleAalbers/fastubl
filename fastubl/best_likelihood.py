import numpy as np
from scipy import special

import fastubl

export, __all__ = fastubl.exporter()


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
        ts = -2 * np.sign(r['itv_mu_best'] - mu_null) * \
             (r['itv_ll'][:,:,mu_i] - r['itv_ll_best'])

        # Low t -> strong deficit. Want minimum t across intervals
        return ts.min(axis=1)

    def compute_ll_grid(self, r):
        """Computes log L(mu_s) in each interval, for all mu_s in grid
        """
        x_obs, p_obs, present = r['x_obs'], r['p_obs'], r['present']
        assert x_obs.shape == present.shape
        n_events = present.sum(axis=1)
        n_sources = p_obs.shape[2]

        # (Code below is similar to 'Yellin normalize' in yellin.py)

        # Map fake events to inf, where they won't affect the method.
        x = np.where(present, x_obs, float('inf'))
        # Sort by ascending x values
        x, p_obs, present = fastubl.sort_all_by_axis1(x, p_obs, present)
        # add fake events at(-inf, inf)
        n_trials = x.shape[0]
        x = np.concatenate([-np.ones((n_trials, 1)) * float('inf'),
                            x,
                            np.ones((n_trials, 1)) * float('inf')],
                           axis=1)
        n_endpoints = n_events.max() + 2

        # Add p_obs and present for fake events
        p_obs = np.concatenate([np.ones((n_trials, 1, n_sources)),
                                p_obs,
                                np.ones((n_trials, 1, n_sources))],
                               axis=1)
        present = np.concatenate([np.zeros((n_trials, 1), dtype=np.bool_),
                                  present,
                                  np.zeros((n_trials, 1), dtype=np.bool_)],
                                 axis=1)

        # Get matrices of (left, right) indices of all intervals.
        # With the addition of two 'events' at the bounds, we have nmax+2 valid
        # indices to consider.
        # left indices change quickly, right indices repeat before changing.
        left, right = np.meshgrid(np.arange(n_endpoints),
                                  np.arange(n_endpoints))
        left, right = left.ravel(), right.ravel()
        valid = right > left
        left, right = left[valid], right[valid]
        n_intervals = left.size

        # Find acceptances (fraction of surviving events) of each interval
        # (trial, interval, source)
        _cdfs = np.stack([dist.cdf(x) for dist in self.dists], axis=2)
        acceptance = _cdfs[:, right, :] - _cdfs[:, left, :]
        assert len(acceptance.shape) == 3

        # Compute grid of mus for all hypotheses (mu_i, source)
        # Add one hypothesis beyond self.mu_s_grid;
        # without this t for final mu would often be 0
        mu_s_grid = np.concatenate([self.mu_s_grid, [self.mu_s_grid[-1] + 1]])
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
        # TODO: fancy indexing might beat doing min twice
        r['itv_mu_best'] = mu_s_grid[np.argmin(r['itv_ll'], axis=2)]
        r['itv_ll_best'] = np.min(r['itv_ll'], axis=2)

        # For diagnosis/inspection
        r['left'], r['right'] = left, right
