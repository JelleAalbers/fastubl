from functools import partial
from scipy import special
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
        """Return ((sizes, skip indices) of k-largest intervals, x_endpoints)
        :param r: result dict
        :param mu_null: Expected signal events under test
        :return: (sizes, skips), both nd arrays. See fastubl.k_largest for details.
        """
        if 'x_endpoints' not in r:
            # Needed for interval lookups later
            r['x_endpoints'] = fastubl.endpoints(
                r['x_obs'], r['present'], r['p_obs'], self.domain, only_x=True)

        if self.include_background and len(self.dists) > 1:
            # Use the sum of the signal and background
            # This changes shape depending on mu_null -> have to recompute
            # the endpoints
            return k_largest(self.sum_cdf(r['x_endpoints'], mu_null))

        else:
            # Without (considering) backgrounds, the interval sizes
            # are the same for all mu_null, and based on the x_endpoints
            # we already computed.
            if 'k_largest' not in r:
                r['k_largest'] = k_largest(self.dists[0].cdf(r['x_endpoints']))
            return r['k_largest']


@export
class PMax(YellinMethod):
    gaps_only = False

    def statistic(self, r, mu_null):
        (sizes, skips) = self.get_k_largest(r, mu_null)

        # (n_trials, events_observed): P(more events in largest iterval)
        p_more = self.p_more_events(sizes, mu_null)

        if self.gaps_only:
            best_n = 0
        else:
            # Find best interval
            # (among maximal intervals of different observed count)
            best_n = np.argmax(p_more, axis=1)   # (n_trials,) array

        trials = np.arange(r['n_trials'])
        sizes, skips = sizes[trials, best_n], skips[trials, best_n]

        # For debugging / characterization, recover interval
        r['interval'] = (
            r['x_endpoints'][trials, skips],
            r['x_endpoints'][trials, skips + best_n + 1],
            best_n)

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
        return stats.poisson.sf(n_in_interval, mu=sizes * total_events)


@export
class MaxGap(PMax):
    gaps_only = True


@export
class MaxGapYellin(MaxGap):
    include_background = True


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

        trials = np.arange(r['n_trials'])
        sizes, skips = sizes[trials, best_n], skips[trials, best_n]
        r['interval'] = (
            r['x_endpoints'][trials, skips],
            r['x_endpoints'][trials, skips + best_n + 1],
            best_n)

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
    cls = False

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

        # Compute P(size_n < observed| mu)
        p_smaller = self.p_smaller(sizes, mu_i)

        # Find optimum interval n (for this mu)
        # highest p_smaller -> lowest p_bigger -> least indication of excess
        if self.cls and mu_i > 0:
            # Do NOT recompute k_largest: want to know how this exact
            # set of intervals looks under bg-only
            # TODO: This probably only makes sense for Neyman version, not Yellin
            p0 = self.p_smaller(sizes, 0)
            if p0.shape[1] < p_smaller.shape[1]:
                # We have some intervals with n_observed > anything ever
                # seen by background-only MC, causing shape mismatch
                p_smaller_0 = p_smaller * 0
                p_smaller_0[:, :p0.shape[1]] = p0
            else:
                p_smaller_0 = p0[:p_smaller.shape[0],:p_smaller.shape[1]]
            # TODO: check and handle div-zero, zero-div-zero...
            best_n = np.argmin((1 - p_smaller)
                               /(1 - p_smaller_0),
                               axis=1)
        else:
            best_n = np.argmax(p_smaller, axis=1)

        # For debugging / characterization, recover interval
        trials = np.arange(r['n_trials'])
        sizes, skips = sizes[trials, best_n], skips[trials, best_n]
        r['interval'] = (
            r['x_endpoints'][trials, skips],
            r['x_endpoints'][trials, skips + best_n + 1],
            best_n)

        # (We have to flip sign for our Neyman code, just like pmax)
        return -p_smaller[trials, best_n]

    def p_smaller(self, sizes, mu_i):
        # TODO: max_n isn't really n, badly named...
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
        return p_smaller


@export
class OptItvCLs(OptItv):
    cls = True


@export
class OptItvYellin(OptItv):
    include_background = True


@export
class BestLikelihoodNoBackground(YellinMethod):
    weighted = False

    def statistic(self, r, mu_null):
        sizes, skips = self.get_k_largest(r, mu_null)
        n = np.arange(sizes.shape[1])[np.newaxis, :]
        mu = sizes * mu_null

        loglr = -(mu - n) + n * np.log(mu) - special.xlogy(n, n)
        ts = -2 * loglr * np.sign(n - mu)

        if self.weighted:
            # Make smaller intervals less likely to be picked
            ts = self.p_t(ts, sizes)

        best_n = np.argmin(ts, axis=1)

        trials = np.arange(r['n_trials'])
        sizes, skips = sizes[trials, best_n], skips[trials, best_n]
        r['interval'] = (
            r['x_endpoints'][trials, skips],
            r['x_endpoints'][trials, skips + best_n + 1],
            best_n)

        return ts[np.arange(r['n_trials']), best_n]

    @staticmethod
    def p_t(t, a):
        # Clip ridiculously ts, preventing numerical errors below
        # (e.g. when testing super-high mu)
        t = t.clip(-30, 30)

        # Convert t to a p-value using large-sample limit
        # The large-sample limit can be very wrong, but that's ok:
        # it's just some monotonic transformation of the ts.
        p = 0.5 * (1 + np.sign(t) * stats.chi2.cdf(np.abs(t), df=1))

        # Cut out central region of size 1-a, map to 0.5
        # Scale remaining ps inward to 0.5 to fill gap
        left, right = a/2, 1 -(a/2)

        # Careful with piecewise etc., t=-0 can slip through if a=1...
        result = np.ones_like(t) * .5
        result = np.where(p < left, p/a, result)
        result = np.where(right < p, (p + a - 1)/a, result)
        return result


@export
class BestLikelihoodNoBackgroundYellin(BestLikelihoodNoBackground):
    include_background = True


@export
class BestLikelihoodNoBackgroundWeighted(BestLikelihoodNoBackground):
    weighted = True


@export
def k_largest(endpoints, max_n=None):
    """Return (sizes, skip_events) of largest intervals with different event count
    Both are (n_trials, max_n) arrays.

    Here size is the expected number of events inside the interval,
    and skip_events is the number of observed events left of the start of the interval.
    The index in sizes/skip_events denotes the event count in the interval.

    For example, sizes[0] gives the expected number of events in the largest observed gap.

    :param endpoints: CDF-mapped sorted endpoints to consider
        The last axis must be the data dimension, earlier axes can run over trials.
    :param max_n: Maximum event count inside interval to consider.
        Defaults to len(x) - 2, i.e. all events except the fake boundary events\

    """
    x = endpoints
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
