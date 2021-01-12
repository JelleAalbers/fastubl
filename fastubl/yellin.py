from functools import partial

import numpy as np
from scipy import stats
from tqdm import tqdm

import fastubl


export, __all__ = fastubl.utils.exporter()


class YellinMethod(fastubl.NeymanConstruction):
    # This statistic is much cheaper than likelihood ratios.
    # You need at least 100_000 trials on a dense mu grid
    # to get something like Yellin's figure 2.
    # (std of cmax is about .25 (at high mu))
    default_trials = 20_000

    include_background = False

    # TODO: use fact that Neyman construction doesn't depend on dist shapes
    # for include_background = True, have to subtract mu_b if using the same
    # Neyman table. (check this is correct first)
    # In optitv, make sure sizes_mc table uses total events as index

    def statistic(self, r, mu_null):
        raise NotImplementedError

    def mu_all(self, mu_s):
        """Return (n_sources) array of expected events
        :param mu_s: Expected signal events, float
        """
        if mu_s is None:
            mu_s = self.true_mu[0]
        if len(self.dists) == 1:
            return np.array([mu_s, ])
        else:
            return np.concatenate([[mu_s], self.true_mu[1:]])

    def sum_cdf(self, x, mu_null):
        """Return cdf of all sources combined
        :param x: Observed x, array (arbitrary shape)
        :param mu_null: Hypothesized expected signal events
        """
        mu_all = self.mu_all(mu_null)
        sum_cdf = np.stack(
            [mu * dist.cdf(x)
             for mu, dist in zip(mu_all, self.dists)],
            axis=-1).sum(axis=-1) / mu_all.sum()
        return sum_cdf

    def get_k_largest(self, r, mu_null=None):
        """Return (sizes, skip indices) of k-largest intervals
        :param r: result dict
        :param mu_null: Expected signal events under test
        :return: (sizes, skips), both nd arrays. See fastubl.k_largest for details.
        """
        if self.include_background and len(self.dists) > 1:
            # Use the sum of the signal and background
            # This changes shape depending on mu_null.
            cdf = partial(self.sum_cdf, mu_null=mu_null)
            xn = yellin_normalize(r['x_obs'], r['present'], cdf)
            return k_largest(xn)

        else:
            # Without considering backgrounds, the interval sizes
            # are the same for all mu_null.
            # Cache them in a key in r.
            if 'k_largest' not in r:
                cdf = self.dists[0].cdf
                xn = yellin_normalize(r['x_obs'], r['present'], cdf)
                r['k_largest'] = k_largest(xn)

            return r['k_largest']

    def recover_intervals(self, r, skips, n_in_interval):
        return fastubl.recover_intervals(r, skips, n_in_interval, self.domain)


@export
class PMax(YellinMethod):

    def statistic(self, r, mu_null):
        sizes, skips = self.get_k_largest(r, mu_null)

        # (n_trials, events_observed): P(more events in largest iterval)
        p_more = self.p_more_events(sizes, mu_null)

        # Find best interval
        # (among maximal intervals of different observed count)
        best_n = np.argmax(p_more, axis=1)   # (n_trials,) array

        # For debugging / characterization, recover interval
        r['interval'] = self.recover_intervals(
            r,
            skips=skips[np.arange(r['n_trials']), best_n],
            n_in_interval=best_n)

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        p_more_max = p_more[np.arange(r['n_trials']), best_n]
        return -np.log(np.maximum(p_more_max, 1.e-9))

    def p_more_events(self, sizes, mu_null):
        total_events = (self.mu_all(mu_null).sum()
                        if self.include_background
                        else mu_null)

        # P(more events in random interval of size)
        n_in_interval = np.arange(sizes.shape[1])[np.newaxis, :]
        return stats.poisson(sizes * total_events).sf(n_in_interval)



@export
class PMaxYellin(PMax):
    include_background = True


@export
class YMin(YellinMethod):

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)

        sizes, skips = self.get_k_largest(r, mu_null)
        n_in_interval = np.arange(sizes.shape[1])[np.newaxis, :]

        y = (n_in_interval - sizes * mu_null)/np.sqrt(sizes * mu_null)

        # Stronger deficits -> lower n | higher size -> lower y_min
        best_n = np.argmin(y, axis=1)

        r['interval'] = self.recover_intervals(
            r,
            skips=skips[np.arange(r['n_trials']), best_n],
            n_in_interval=best_n)

        return y[np.arange(r['n_trials']), best_n]


@export
class YMinYellin(YMin):
    include_background = True


@export
class OptItv(YellinMethod):

    # It would be very inefficient to store this as a 3d array;
    # it would be ~90% 1s for a reasonable mu_s_grid.
    sizes_mc : list  # (mu_i) -> (max_n, mc_trials) arrays

    extra_cache_attributes = ('sizes_mc',)

    def do_neyman_construction(self):
        # TODO: compute P(Cn | n) instead, combine with Poisson for P(Cn | mu)?
        # -- only useful for Vanilla Yellin, not Neyman variation
        self.sizes_mc = []
        for mu_i, mu_s in enumerate(tqdm(
                self.mu_s_grid,
                desc='MC for interval size lookup table')):
            r = self.toy_data(self.trials_per_s, mu_s_true=mu_s)
            sizes, _ = self.get_k_largest(r, mu_null=mu_s)
            sizes = sizes.T    # (trials, n_in_interval)   -> (n_in, trials)
            sizes.sort(axis=-1)
            self.sizes_mc.append(sizes)

        super().do_neyman_construction()

    def statistic(self, r, mu_null):
        sizes, skips = self.get_k_largest(r, mu_null)

        mu_i = np.searchsorted(self.mu_s_grid, mu_null)
        n_trials, max_n = sizes.shape
        max_n = min(max_n, self.sizes_mc[mu_i].shape[0] - 1)

        # Compute P(size_n < observed| mu)
        cdf_size = np.zeros((n_trials, max_n))
        # TODO: faster way? np.searchsorted has no axis argument...
        # TODO: maybe numba helps?
        for n in range(max_n):
            # P(size_n < observed| mu)
            # i.e. P that the n-interval is 'smaller' (has fewer events expected)
            # Excess -> small intervals -> small ps
            p = np.searchsorted(self.sizes_mc[mu_i][n], sizes[:,n]) / self.trials_per_s
            cdf_size[:, n] = p.clip(0, 1)

        # Find optimum interval n (for this mu)
        # highest cdf_size -> least indication of excess
        best_n = np.argmax(cdf_size, axis=1)

        r['interval'] = self.recover_intervals(
            r,
            skips=skips[np.arange(r['n_trials']), best_n],
            n_in_interval=best_n)

        # (We have to flip sign for our Neyman code, just like pmax)
        return -cdf_size[np.arange(r['n_trials']), best_n]


@export
class OptItvYellin(OptItv):
    include_background = True


@export
def k_largest(xn, max_n=None):
    """Return (sizes, skip_events) of largest intervals with different event count
    Both are (n_trials, max_n) arrays.

    Here size is the expected number of events inside the interval,
    and skip_events is the number of observed events left of the start of the interval.
    The index in sizes/skip_events denotes the event count in the interval.

    For example, sizes[0] gives the expected number of events in the largest observed gap.

    :param xn: Input data, Yellin-normalized, with added 0/1 events.
        The last axis must be the data dimension, earlier axes can run over trials.
    :param max_n: Maximum event count inside interval to consider.
        Defaults to len(x) - 2, i.e. all events except the fake boundary events\

    """
    x = xn
    if len(x.shape) > 1:
        other_dims = list(x.shape[:-1])
    else:
        other_dims = []
    if max_n is None:
        max_n = x.shape[-1] - 2

    # Default values (size 1, start 0) apply to i > n - 1
    sizes = np.ones(other_dims + [max_n + 1])
    starts = np.zeros(other_dims + [max_n + 1], dtype=np.int)

    for i in range(sizes.shape[-1]):
        # Compute sizes of gaps with i events in them
        y = x[..., (i + 1):] - x[..., :-(i + 1)]
        # TODO: maybe clever indexing is faster than two max computations?
        starts[..., i] = np.argmax(y, axis=-1)
        sizes[..., i] = np.max(y, axis=-1)

    assert sizes.min() > 0, "Encountered edge case"

    return sizes, starts


@export
def yellin_normalize(x, present, cdf=None):
    x = np.asarray(x)

    # Put float('inf') in place of present events
    # -> They will be mapped to 1 after the CDF transform
    # where they will not affect Yellin-like methods
    x = np.where(present, x, float('inf'))

    x = np.sort(x, axis=-1)
    if cdf is not None:
        x = cdf(x)
    x = add_bounds(x)
    return x


@export
def add_bounds(x):
    """pad data along final axis by 0 and 1"""
    x = np.asarray(x)
    for q in [0, 1]:
        x = np.pad(
            x,
            [(0, 0)] * (len(x.shape) - 1) + [(1 - q, q)],
            mode='constant', constant_values=q)
    return x
