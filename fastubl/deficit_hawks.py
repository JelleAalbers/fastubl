import numba
import numpy as np
from scipy import stats
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

    def get_mu_i(self, mu_null):
        mu_i = np.searchsorted(self.mu_s_grid, mu_null)
        assert self.mu_s_grid[mu_i] == mu_null, "mu_s must be on grid"
        return mu_i


@export
class IntervalHawk(DeficitHawk):
    k_largest_only = False

    interval_info_keys = 'left right n_observed acceptance'.split()

    def score_intervals(self, interval_info, mu_null):
        raise NotImplementedError

    def score_regions(self, r, mu_null):
        return self.score_intervals(self.interval_info(r), mu_null)

    def _region_details(self, r, best_i):
        itv_info = self.interval_info(r)
        return {
            k: itv_info[np.arange(r['n_trials']), best_i, ...]
            for k in self.interval_info_keys}

    def interval_info(self, r):
        if 'all_intervals' not in r:
            r['all_intervals'] = self.all_intervals(r)
        itv_info = r['all_intervals']

        if self.n_sources > 1 or not self.k_largest_only:
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

        x_endpoints, itv_info['present_endpoints'], itv_info['p_obs_endpoints'] \
            = fastubl.endpoints(r['x_obs'], r['present'], r['p_obs'], self.domain)
        n_trials, n_endpoints_max = x_endpoints.shape
        left_i, right_i = fastubl.interval_indices(n_endpoints_max)
        n_observed = right_i - left_i - 1  # 0 observed if right = left + 1

        # Intervals may include 'fake' events mapped to right domain boundary.
        # Make sure to include right domain boundary only once as a possible
        # endpoint: endpoint index n_events + 1
        n_events = r['present'].sum(axis=1)
        is_valid = right_i[None, :] <= n_events[:, None] + 1

        left, right = x_endpoints[:, left_i], x_endpoints[:, right_i]
        n_observed = n_observed[None, :] * np.ones(left.shape, dtype=np.int)
        assert left.shape == right.shape == n_observed.shape

        # (0, 1) is indeed marked as valid only once per trial:
        assert ((left[is_valid] == 0.) & (right[is_valid] == 1.)).sum() \
               == r['n_trials']

        itv_info['left'] = left
        itv_info['right'] = right
        itv_info['left_i'] = left_i
        itv_info['right_i'] = right_i
        itv_info['n_observed'] = n_observed
        itv_info['is_valid'] = is_valid
        itv_info['acceptance'] = fastubl.interval_acceptances(x_endpoints, left_i, right_i, self.dists)
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

    def score_intervals(self, itv_info, mu_null):
        # P(observe <= N)
        mu = self.mu_all(mu_null)[None,None,:] * itv_info['acceptance']
        return stats.poisson.cdf(itv_info['n_observed'], mu=mu)


@export
class NaivePMax(IntervalHawk):
    k_largest_only = True

    def score_intervals(self, itv_info, mu_null):
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

    def score_intervals(self, interval_info, mu_null):
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
