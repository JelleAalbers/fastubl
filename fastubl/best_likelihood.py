import numba
import numpy as np
from scipy import special, stats
from tqdm import tqdm

import fastubl

export, __all__ = fastubl.exporter()


@export
class BestZech(fastubl.NeymanConstruction):
    lee_correction = None

    def do_neyman_construction(self):
        if self.lee_correction != 'p_per_n':
            return super().do_neyman_construction()

        # List of list of arrays, (mu_i, n_events, trial_i)
        # best CLs scores among intervals containing n_events (if present)
        self.cls_n_mc = []
        for mu_i, mu_s in enumerate(tqdm(
                self.mu_s_grid,
                desc='MC for CLs(n) lookup table')):

            r = self.toy_data(self.trials_per_s, mu_s_true=mu_s)
            n_max = int(r['present'].sum(axis=1).max())

            score = self.p_fewer(r, mu_s) / self.p_fewer(r, 0)
            best_cls, _ = self.get_min_per_n(
                score=score,
                n=r['all_intervals']['n_observed'],
                max_n=n_max,
                is_valid=r['all_intervals']['is_valid'].astype(np.int))
            assert best_cls.shape == (n_max + 1, r['n_trials'])
            self.cls_n_mc.append([np.sort(x[np.isfinite(x)])
                                  for x in best_cls])

        return super().do_neyman_construction()

    def get_min_per_n(self, score, n, max_n, is_valid):
        """Return (max_n+1, n_trials) array"""
        n_trials, n_intervals = score.shape
        assert n.shape == is_valid.shape == score.shape

        result_i = np.ones((max_n + 1, n_trials), dtype=np.int)
        result = np.ones((max_n + 1, n_trials), dtype=np.float) * np.nan

        self._fill_min_per_n_result(score, n, max_n, is_valid, result, result_i)
        return result, result_i

    @staticmethod
    @numba.njit
    def _fill_min_per_n_result(score, n, max_n, is_valid, result, result_i):
        n_trials, n_intervals = score.shape
        for trial_i in range(n_trials):
            for interval_i in range(n_intervals):
                if not is_valid[trial_i, interval_i]:
                    continue
                _n = n[trial_i, interval_i]
                if _n > max_n:
                    # Beyond max_n we needed to consider
                    # (if we'd proceed, we get a buffer overflow
                    #  which numba does not check...)
                    continue
                x = score[trial_i, interval_i]
                if np.isnan(result[_n, trial_i]) or x < result[_n, trial_i]:
                    result[_n, trial_i] = x
                    result_i[_n, trial_i] = interval_i

    def statistic(self, r, mu_null):
        if 'p_fewer_0' not in r:
            r['p_fewer_0'] = self.p_fewer(r, 0)
        trials = np.arange(r['n_trials'])
        # Careful with .sum() here, true_mu[1:] might be empty.
        # If you add before sum, broadcasting makes the
        # addition vanish...
        total_mu = (mu_null + self.true_mu[1:].sum())

        # p(fewer events): high value = excess, so we want to minimize score
        score = self.p_fewer(r, mu_null) / r['p_fewer_0']
        score[~r['all_intervals']['is_valid']] = float('inf')

        if self.lee_correction == 'p_per_n':
            # Very similar to Optimum interval here
            # Any way top reduce duplication?
            mu_i = np.searchsorted(self.mu_s_grid, mu_null)
            max_n = len(self.cls_n_mc[mu_i]) - 1
            cls_n, min_is = self.get_min_per_n(
                score=score,
                n=r['all_intervals']['n_observed'],
                max_n=max_n,
                is_valid=r['all_intervals']['is_valid'].astype(np.int))

            # TODO: ? CLS: Lowest -> Least probability to be <=
            ps = np.ones((r['n_trials'], max_n + 1)) * float('inf')
            # TODO: faster way? np.searchsorted has no axis argument...
            n_events = r['present'].sum(axis=1)
            for n in range(ps.shape[1]):
                # Do not consider n larger than numbers seen in 80% of MC
                # trials. We're after sparse intervals: these clearly aren't.
                # (and the estimate under (a) below would be unreliable)
                mc_scores = self.cls_n_mc[mu_i][n]
                if len(mc_scores) < self.trials_per_s * 0.2:
                    break

                # P of having < n events in the full domain
                # (so there won't be an n-events-containing interval)
                p_fewer = stats.poisson(total_mu).cdf(n - 1)

                # Probability of
                # (a) seeing a lower min-score (and >= n events) or
                # (b) having < n events in the domain
                p = p_fewer + (1 - p_fewer) * np.searchsorted(mc_scores, cls_n[n,:]) / len(mc_scores)

                # Never pick n > total observed events
                # (differs each trial, so couldn't just limit for loop)
                ps[:, n] = np.where(n > n_events,
                                    float('inf'),
                                    p)
            # TODO: temp
            r['cls_n'] = cls_n
            r['p_higher'] = ps

            best_i = min_is[np.argmin(ps, axis=1), trials]

        elif self.lee_correction == 'approx':
            n = r['all_intervals']['n_observed']
            score /= (
                ((1 + n)/(1 + total_mu)).clip(0, 1)
                * stats.poisson(total_mu).sf(n - 1))
            best_i = np.argmin(score, axis=1)

        else:
            best_i = np.argmin(score, axis=1)

        r['interval'] = (
            r['all_intervals']['left'][trials, best_i],
            r['all_intervals']['right'][trials, best_i],
            r['all_intervals']['n_observed'][trials, best_i])

        return score[trials, best_i]

    def p_fewer(self, r, mu_signal):
        x_obs, p_obs, present = r['x_obs'], r['p_obs'], r['present']
        assert x_obs.shape == present.shape

        if 'all_intervals' not in r:
            r['all_intervals'] = all_intervals(r, self.domain, self.dists)

        # TODO: this should be factored out, it's done elsewhere
        mu = self.true_mu.copy()
        mu[0] = mu_signal

        # Get (trials, intervals) array of total mu
        mu_total = (mu[None,None,:] * r['all_intervals']['acceptance']).sum(axis=2)

        return stats.poisson(mu_total).cdf(r['all_intervals']['n_observed'])


@export
class BestZechLEE(BestZech):
    lee_correction = 'p_per_n'
    extra_cache_attributes = ('cls_n_mc',)


@export
class BestZechLEEApprox(BestZech):
    lee_correction = 'approx'


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


@export
def all_intervals(r, domain, dists=None):
    x_endpoints = fastubl.endpoints(
        r['x_obs'], r['present'], r['p_obs'], domain, only_x=True)
    n_trials, n_endpoints_max = x_endpoints.shape
    left_i, right_i = fastubl.interval_indices(n_endpoints_max)
    n_observed = right_i - left_i - 1  # 0 observed if right = left + 1

    # Intervals may include 'fake' events mapped to right endpoint -> 1
    # Make sure to include '1' only once as an endpoint:
    # first occurrence is at endpoint index last_event + 2 = n_event + 1
    n_events = r['present'].sum(axis=1)
    is_valid = right_i[None, :] <= n_events[:, None] + 1

    left, right = x_endpoints[:, left_i], x_endpoints[:, right_i]
    n_observed = n_observed[None, :] * np.ones(left.shape, dtype=np.int)
    assert left.shape == right.shape == n_observed.shape

    # (0, 1) is indeed marked as valid only once per trial:
    assert ((left[is_valid] == 0.) & (right[is_valid] == 1.)).sum() \
           == r['n_trials']

    result = dict(left=left,
                  right=right,
                  left_i=left_i,
                  right_i=right_i,
                  n_observed=n_observed,
                  is_valid=is_valid)

    if dists is not None:
        result['acceptance'] = interval_acceptances(x_endpoints, left_i, right_i, dists)
    return result
