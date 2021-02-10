import numba
import numpy as np
from scipy import special, stats
from tqdm import tqdm

import fastubl

export, __all__ = fastubl.exporter()


@export
class DeficitHawk(fastubl.NeymanConstruction):

    def score_regions(self, r, mu_null):
        raise NotImplementedError

    def region_details(self, r, mu_null):
        best_i = np.min(self.score_regions(r, mu_null),
                        axis=1)
        return self._region_details(r, best_i)

    def statistic(self, r, mu_null):
        return np.min(self.score_regions(r, mu_null),
                      axis=1)

    def _consider_background(self):
        return len(self.dists) > 1


@export
class IntervalHawk(DeficitHawk):
    k_largest_only = False

    interval_info_keys = 'left right n_observed acceptance is_valid'.split()

    def score_intervals(self, r, interval_info, mu_null):
        raise NotImplementedError

    def score_regions(self, r, mu_null):
        itv_info = self.interval_info(r)
        ts = self.score_intervals(r, itv_info, mu_null)
        # Never pick invalid intervals
        return np.where(itv_info['is_valid'], ts, np.inf)

    def _region_details(self, r, best_i):
        itv_info = self.interval_info(r)
        return {
            k: itv_info[np.arange(r['n_trials']), best_i, ...]
            for k in self.interval_info_keys}

    def interval_info(self, r):
        if 'all_intervals' not in r:
            r['all_intervals'] = self.all_intervals(r)
        itv_info = r['all_intervals']

        if self._consider_background():
            # We have to consider all intervals
            return itv_info

        # We only need to consider
        # {itv with most expected signal among those with k events observed,
        #  foreach k}
        if 'k_largest_intervals' not in r:
            _sizes, indices = self.largest_k(
                sizes=r['all_intervals']['acceptance'][:,:,0],
                n_observed=r['all_intervals']['n_observed'],
                is_valid=r['all_intervals']['is_valid'])
            trials = np.arange(r['n_trials'])[:,None] * np.ones((1, indices.shape[1]))
            r['k_largest_intervals'] = {k: itv_info[k][trials.astype(np.int), indices, ...]
                                        for k in self.interval_info_keys}
            # Copy other keys unchanged, e.g. x_endpoints
            r['k_largest_intervals'].update({k: v for k, v in itv_info.items()
                                             if k not in self.interval_info_keys})
        return r['k_largest_intervals']

    def all_intervals(self, r):
        itv_info = dict()

        itv_info['x_endpoints'], itv_info['present_endpoints'], itv_info['p_obs_endpoints'] \
            = fastubl.endpoints(r['x_obs'], r['present'], r['p_obs'], self.domain)
        n_trials, n_endpoints_max = itv_info['x_endpoints'].shape
        left_i, right_i = fastubl.interval_indices(n_endpoints_max)
        n_intervals = left_i.size
        n_observed = right_i - left_i - 1  # 0 observed if right = left + 1

        # Intervals may include 'fake' events mapped to right domain boundary.
        # Make sure to include right domain boundary only once as a possible
        # endpoint: endpoint index n_events + 1
        n_events = r['present'].sum(axis=1)
        is_valid = right_i[None, :] <= n_events[:, None] + 1

        left = itv_info['x_endpoints'][:, left_i]
        right = itv_info['x_endpoints'][:, right_i]
        n_observed = n_observed[None, :] * np.ones(left.shape, dtype=np.int)
        assert left.shape == right.shape == n_observed.shape == (n_trials, n_intervals)

        # (0, 1) is indeed marked as valid only once per trial:
        assert ((left[is_valid] == 0.) & (right[is_valid] == 1.)).sum() \
               == r['n_trials']

        itv_info['left'] = left
        itv_info['right'] = right
        itv_info['left_i'] = left_i
        itv_info['right_i'] = right_i
        itv_info['n_observed'] = n_observed
        itv_info['is_valid'] = is_valid
        itv_info['acceptance'] = fastubl.interval_acceptances(
            itv_info['x_endpoints'], left_i, right_i, self.dists)
        return itv_info

    @staticmethod
    @numba.njit
    def largest_k(sizes, n_observed, is_valid):
        """Return size and index of largest size interval containing different
        numbers of observed events.

        Returns (sizes, indices), each an (trials, max_n + 1) array,
        where max_n is the largest number of observed events across all trials
        and intervals.

        :param sizes: (trials, n_intervals), some "size" value to maximize,
        must be > 0 in all intervals, and 1 in the full domain.
        :param n_observed: (trials, n_intervals) observed number of events
        :param is_valid: (trials, n_intervals) mark if interval is valid
        """
        n_trials, n_intervals = sizes.shape
        n_k = n_observed.max() + 1

        largest_k = np.zeros(n_trials, dtype=np.int64)
        result_size = np.zeros((n_trials, n_k), dtype=np.float64)
        result_index = np.zeros((n_trials, n_k), dtype=np.int64)
        found_one = np.zeros((n_trials, n_k), dtype=np.bool_)

        for trial_i in range(n_trials):
            for interval_i in range(n_intervals):
                if not is_valid[trial_i, interval_i]:
                    continue
                size = sizes[trial_i, interval_i]
                k = n_observed[trial_i, interval_i]
                if not found_one[trial_i, k] or size > result_size[trial_i, k]:
                    result_size[trial_i, k] = size
                    result_index[trial_i, k] = interval_i
                    found_one[trial_i, k] = True
                    largest_k[trial_i] = max(k, largest_k[trial_i])

            # For k > n_total, set size = 1index equal to k = n_total
            for k in range(largest_k[trial_i], n_k):
                result_index[trial_i, k] = result_index[trial_i, largest_k[trial_i]]
                result_size[trial_i, k] = 1.

        return result_size, result_index


@export
class YellinPMax(IntervalHawk):

    def score_intervals(self, r, itv_info, mu_null):
        # P(observe <= N)
        mu = self.mu_all(mu_null)[None,None,:] * itv_info['acceptance']
        return stats.poisson.cdf(itv_info['n_observed'], mu=mu)


@export
class NaivePMax(IntervalHawk):
    k_largest_only = True

    def score_intervals(self, r, itv_info, mu_null):
        # P(observe <= N of *signal* events)
        mu = mu_null * itv_info['acceptance'][:,:,0]
        return stats.poisson.cdf(itv_info['n_observed'], mu=mu)


@export
class YellinOptItv(IntervalHawk):

    # It would be very inefficient to store this as a 3d array;
    # it would be ~90% 1s for a reasonable mu_s_grid.
    sizes_mc : list  # (mu_i) -> (max_n, mc_trials) arrays

    extra_cache_attributes = ('sizes_mc',)

    def do_neyman_construction(self):
        # TODO: compute P(Cn | n) instead, combine with Poisson for P(Cn | mu)?
        # -- only useful for Vanilla Yellin, not naive variation
        self.sizes_mc = []
        for mu_i, mu_s in enumerate(tqdm(
                self.mu_s_grid,
                desc='MC for interval size lookup table')):
            r = self.toy_data(self.trials_per_s, mu_s_true=mu_s)

            itv_info = self.interval_info(r)
            sizes = (self.mu_all(mu_s)
                     * itv_info['acceptance'][None,None,:])
            sizes, _ = self.largest_k(sizes=sizes,
                                      n_observed=itv_info['n_observed'],
                                      is_valid=itv_info['is_valid'])

            sizes = sizes.T    # (trials, n_in_interval)   -> (n_in, trials)
            sizes.sort(axis=-1)
            self.sizes_mc.append(sizes)

        super().do_neyman_construction()

    def score_intervals(self, r, interval_info, mu_null):
        mu_i = self.get_mu_i(mu_null)
        sizes = interval_info['acceptance'][:,:,0]

        # TODO: max_n isn really max_n + 1, badly named...
        n_trials, max_n = sizes.shape
        max_n = min(max_n, self.sizes_mc[mu_i].shape[0] - 1)
        p_smaller = np.zeros((n_trials, max_n))
        # TODO: faster way? np.searchsorted has no axis argument...
        # TODO: maybe numba helps?
        for n in range(max_n):
            # P(size_n < observed| mu)
            # i.e. P that the n-interval is 'smaller' (has fewer events expected)
            # Excess -> small intervals -> small ps
            p = np.searchsorted(self.sizes_mc[mu_i][n], sizes[:,n]) / self.trials_per_s
            p_smaller[:, n] = p.clip(0, 1)

        # NB: using -p_smaller, so we can take minimum
        return -p_smaller


@export
class LikelihoodIntervalHawk(IntervalHawk):
    batch_size = 50

    def score_intervals(self, r, itv_info, mu_null):
        if not self._consider_background():
            mu = mu_null * itv_info['acceptance'][...,0]
            n = itv_info['n_observed']

            # We know mu_best = n, and details of the diffrates cancel
            loglr = -(mu - n) + n * np.log(mu) - special.xlogy(n, n)
            ts = -2 * loglr * np.sign(n - mu)
            return ts

        if 'llr' not in itv_info:
            self.compute_ll_grid(r, itv_info)

        mu_i = self.get_mu_i(mu_null)

        # Compute (trial, interval) array of ts.
        # itv_ll: (trial, interval, mu) array
        # itv_mu_best: (trial, interval) array
        return (
                -2
                * np.sign(itv_info['mu_best'] - mu_null)
                * itv_info['llr'][:, :, mu_i])

    def compute_ll_grid(self, r, itv_info):
        n_trials, n_max_events, n_sources = r['p_obs'].shape
        _, n_intervals = itv_info['left'].shape
        _, n_endpoints = itv_info['x_endpoints'].shape

        # Compute grid of mus for all hypotheses (mu_i, source)
        # Add a few hypotheses higher than self.mu_s_grid;
        # without this t for final mu would often be 0
        # TODO: Do we need as many as 10? We're looking at sparse intervals
        mu_s_grid = np.concatenate([
            self.mu_s_grid,
            self.mu_s_grid[-1] * np.geomspace(1.1, 10, 10)])
        n_mus = mu_s_grid.size
        if len(self.true_mu) > 1:
            mu_grid = np.concatenate([
                mu_s_grid[:,None],
                np.tile(self.true_mu[1:], n_mus).reshape(n_mus, n_sources - 1)],
                axis=1)
        else:
            mu_grid = mu_s_grid.reshape(n_mus, 1)
        assert mu_grid.shape == (n_mus, n_sources)

        # Compute total mu per interval
        # (trial, interval, mu_i)
        #       Inside sum: both (trial, event, mu_i, source) arrays
        mu_tot = np.sum(mu_grid[None,None,:,:]
                        * r['all_intervals']['acceptance'][:,:,None,:],
                        axis=-1)
        assert mu_tot.shape == (n_trials, n_intervals, n_mus)

        # Compute differential rate sum_sources (p * mu) for each event and mu
        # Note this does not depend on the interval -- cuts reduce mu
        # and increase p_obs.
        # (trial, event, mu_i) array
        #       Inside sum: both (trial, event, mu_i, source) arrays
        dr_mu = (r['all_intervals']['p_obs_endpoints'][:,:,None,:]
                 * mu_grid[None,None,:,:]
                 ).sum(axis=-1)
        assert dr_mu.shape == (n_trials, n_endpoints, n_mus)

        # Inner term of log L = sum_events log(sum_sources dr)
        # Start with cumulative sum over events...
        # (trial, event, mu_i)
        cumsum_inner = np.cumsum(
            special.xlogy(itv_info['present_endpoints'][:,:,None], dr_mu),
            axis=1)
        del dr_mu
        assert cumsum_inner.shape == (n_trials, n_endpoints, n_mus)

        # ... then use interval indices to get only sum over included events
        # all intervals have right > left (see all_intervals)
        # right = left + 1 means no events in the interval -> 0
        # Right and left event themselves must never be included
        # (trial, interval, mu_i) array
        itv_info['ll'] = -mu_tot + (
                cumsum_inner[:,itv_info['right_i'] - 1,:]
                - cumsum_inner[:,itv_info['left_i'],:])
        del cumsum_inner
        assert itv_info['ll'].shape == (n_trials, n_intervals, n_mus)

        # Find best mu in each interval
        # (trial, interval) array
        # TODO: fancy indexing might beat doing min twice
        itv_info['mu_best'] = mu_s_grid[np.argmax(itv_info['ll'], axis=2)]
        itv_info['ll_best'] = np.max(itv_info['ll'], axis=2)

        itv_info['llr'] = itv_info['ll'] - itv_info['ll_best'][:,:,None]
