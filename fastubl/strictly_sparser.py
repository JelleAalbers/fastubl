# import boost_histogram as bh
from copy import deepcopy
from multihist import Histdd

import numpy as np
from scipy import stats
from tqdm import tqdm

import fastubl

export, __all__ = fastubl.utils.exporter()

@export
class StrictlySparserInterval(fastubl.NeymanConstruction):

    max_hist_gb = 2
    n_domain_bins = 30
    sizes_hist: Histdd

    extra_cache_attributes = ('itv_hist', 'itv_hist_orig')

    def extra_hash_dict(self):
        return dict(domain_bins=self.n_domain_bins)

    def do_neyman_construction(self):
        max_n = stats.poisson(self.mu_s_grid[-1]).ppf(.99) + 5

        nbytes = 8 * self.n_domain_bins**2 * self.mu_s_grid.size * (max_n + 1)
        if nbytes > 1e9 * self.max_hist_gb:
            raise ValueError(f"Data would exceed {self.max_hist_gb} GB, "
                             f"increase max_hist_gb or use fewer bins.")

        self.itv_hist = Histdd(dimensions=(
            ('mu_i', np.arange(0, self.mu_s_grid.size) - 0.5),
            ('left', np.linspace(0, 1, self.n_domain_bins + 1)),
            ('right', np.linspace(0, 1, self.n_domain_bins + 1)),
            ('n', np.arange(0, max_n + 1) - 0.5)))

        for mu_i, mu_s in enumerate(tqdm(
                self.mu_s_grid,
                desc='MC for interval lookup hist')):
            # TODO: x5 for now. Really needed or overkill?
            r = self.toy_data(self.trials_per_s * 5, mu_s_true=mu_s)
            left, right, n, is_valid = self.intervals(r)
            # indexing with is_valid wil auto-ravel
            left, right, n, = left[is_valid], right[is_valid], n[is_valid]
            self.itv_hist.add(np.ones_like(left) * mu_i, left, right, n)

        self.itv_hist_orig = deepcopy(self.itv_hist)   # TEMP! for debugging

        x = self.itv_hist.histogram.astype(np.float)
        x /= self.trials_per_s

        # (<= left, right, <= n)
        np.cumsum(x, axis=1, out=x)
        np.cumsum(x, axis=3, out=x)

        # (<= left, >= right, <= n)
        # Note flip should return view, big histogram is not copied
        np.cumsum(x[:,:,::-1,:], axis=2, out=x)
        self.itv_hist.histogram = np.flip(x, axis=2)

        super().do_neyman_construction()

    def statistic(self, r, mu_null):
        mu_i = np.argmin(np.abs(mu_null - self.mu_s_grid))
        sparser = self.e_sparser(r, mu_i)
        if mu_i == 0:
            return np.min(sparser, axis=1)

        # Use CLS-like scoring to avoid picking
        # intervals with large background underfluctuations
        if 'e_sparser_0' not in r:
            r['e_sparser_0'] = self.e_sparser(r, 0)
        sparser = np.where(np.isfinite(sparser) & np.isfinite(r['e_sparser_0']),
                           sparser / r['e_sparser_0'],
                           np.inf)
        return np.min(sparser, axis=1)

    def e_sparser(self, r, mu_i):
        if 'intervals' not in r:
            r['intervals'] = self.intervals(r)
        left, right, n, is_valid = r['intervals']

        # E(number of strictly sparser intervals)
        # Lookup doesn't preserve shape, so ravel and reshape it is...
        sparser = self.itv_hist.lookup(
            np.ones(left.size) * mu_i,
            left.ravel(), right.ravel(), n.ravel()).reshape(left.shape)
        return np.where(is_valid, sparser, float('inf'))

    def intervals(self, r):
        # TODO: actually don't need cdf transform at all!
        x = fastubl.yellin_normalize(r['x_obs'], r['present'],
                                     cdf=self.dists[0].cdf)
        n_trials, n_endpoints_max = x.shape
        left_i, right_i = fastubl.interval_indices(n_endpoints_max)
        n_observed = right_i - left_i - 1  # 0 observed if right = left + 1

        # Intervals may include 'fake' events mapped to right endpoint -> 1
        # Make sure to include '1' only once as an endpoint:
        # first occurrence is at endpoint index last_event + 2 = n_event + 1
        n_events = r['present'].sum(axis=1)
        is_valid = right_i[None,:] <= n_events[:,None] + 1

        left, right = x[:,left_i], x[:,right_i]
        n_observed = n_observed[None, :] * np.ones(left.shape, dtype=np.int)
        assert left.shape == right.shape == n_observed.shape

        # (0, 1) is indeed included only once per trial:
        assert ((left[is_valid] == 0.) & (right[is_valid] == 1.)).sum() \
               == r['n_trials']
        return left, right, n_observed, is_valid
