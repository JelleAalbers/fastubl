import numpy as np
from scipy import optimize, interpolate, special, stats
from tqdm import tqdm

import fastubl


export, __all__ = fastubl.utils.exporter()


def add_interval_sizes(r, cdf, clear=False):
    """

    :param r:
    :param cdf:
    :return:
    """
    if clear:
        if 'sizes' in r:
            del r['sizes']
        if 'skips' in r:
            del r['skips']
    if 'sizes' not in r:
        xn = fastubl.yellin_normalize(r['x_obs'], r['present'], cdf)
        r['sizes'], r['skips'] = fastubl.k_largest(xn)


@export
class MaxGap(fastubl.StatisticalProcedure):

    _limit_loglog_curves = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_loglog_curves = dict()

    def compute_intervals(self, r, *, kind, cl, **kwargs):
        g, _ = self.max_gap(r)

        if kind == 'upper':
            ul = 10 ** self.limit_loglog_curve(cl)(np.log10(g))
            # TODO: Why is Poisson clipping needed? Interpolation trouble?
            ul = np.clip(ul, fastubl.poisson_ul(0), None)
            return np.zeros(r['n_trials']), ul
        else:
            raise NotImplementedError(kind)

    def max_gap(self, r):
        """Return (gap sizes, skip_events) of maximum gap
        indices are in sorted version of r['x'], with non-present events removed
        """
        add_interval_sizes(r, self.dists[0].cdf)
        return r['sizes'][..., 0], r['skips'][..., 0]

    def limit_loglog_curve(self, cl):
        if cl not in self._limit_loglog_curves:
            # compute exact values for reasonable N
            mus = self.mu_s_grid
            critical_values = np.clip(
                [optimize.brentq(lambda x: self.p_maxgap(x, mu) - cl, 0, mus[-1])
                 for mu in mus],
                0, 1)
            # Use interpolation in the reasonable range, extrapolation outside
            self._limit_loglog_curves[cl] = interpolate.interp1d(
                np.log10(critical_values),
                np.log10(mus),
                fill_value='extrapolate')
        return self._limit_loglog_curves[cl]

    @staticmethod
    def p_maxgap(x, mu):
        """Probability of observing a smaller gap than x if mu events expected
        From Yellin's paper

        You will get overflow errors for mu > 1000
        """
        if x == 0:
            return 0
        if x == 1:
            # TODO HACK...
            x = 1 - 1e-10

        # Yellin's x is mu * our x
        x *= mu

        m = int(mu / x)

        # Yellin's formula gives zero division errors
        # I guess <- in his eq. 2 should really be < ?
        if mu - m * x == 0:
            m -= 1

        return sum([(k * x - mu) ** k
                    * np.exp(-k * x) / special.factorial(k)
                    * (1 + k / (mu - k * x))
                    for k in range(m + 1)])


@export
class PMax(fastubl.NeymanConstruction):

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)

        add_interval_sizes(r, self.dists[0].cdf)

        # P(more events in random interval of size):
        p_more_events = stats.poisson(mu_null * r['sizes']).sf(
            np.arange(r['sizes'].shape[1])[np.newaxis,:])
        pmax = p_more_events.max(axis=1)

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))


@export
class PMaxYellin(fastubl.NeymanConstruction):

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)

        if len(self.dists) > 1:
            mu_all = np.concatenate([[mu_null], self.true_mu[1:]])

            # Distribution of signal + background
            # This depends on mu_null!
            def sum_cdf(x):
                return np.stack([mu * dist.cdf(x)
                                 for mu, dist in zip(mu_all, self.dists)],
                                axis=-1).sum(axis=-1) / mu_all.sum()
            add_interval_sizes(r, sum_cdf, clear=True)

        else:
            # No known background: equivalent to Neyman pmax
            mu_all = np.array([mu_null,])
            add_interval_sizes(r, self.dists[0].cdf)

        # P(more events in random interval of size)
        # Note the mu is the *total* mu now
        p_more_events = stats.poisson(mu_all.sum() * r['sizes']).sf(
            np.arange(r['sizes'].shape[1])[np.newaxis,:])
        pmax = p_more_events.max(axis=1)

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))

    # def compute_intervals(self,
    #                       r,
    #                       cl=fastubl.DEFAULT_CL,
    #                       kind=fastubl.DEFAULT_KIND):
    #     lower, upper = self.compute_intervals(r, cl, kind)
    #     # Subtract the background mean
    #     lower -= se


@export
class OptItv(fastubl.NeymanConstruction):
    max_n: int
    sizes_mc : np.ndarray  # (mu, max_n, mc_trials)

    extra_cache_attributes = ('sizes_mc',)

    def do_neyman_construction(self):
        # Largest events per interval to consider
        max_n = 5 + int(stats.poisson(self.mu_s_grid[-1] + self.true_mu[1:].sum()).ppf(.9999))

        # Default is 1, since size largest interval with huge n is 1
        self.sizes_mc = np.ones((
            self.mu_s_grid.size,
            max_n,
            self.trials_per_s))

        # TODO: need more trials here?
        for mu_i, mu_s in enumerate(tqdm(
                self.mu_s_grid,
                desc='MC for interval size lookup table')):
            r = self.toy_data(self.trials_per_s, mu_s_true=mu_s)
            add_interval_sizes(r, self.dists[0].cdf)
            self.sizes_mc[mu_i,:r['sizes'].shape[1],:] = r['sizes'].T
        self.sizes_mc.sort(axis=2)

        super().do_neyman_construction()

    def statistic(self, r, mu_null):
        add_interval_sizes(r, self.dists[0].cdf)

        mu_i = np.searchsorted(self.mu_s_grid, mu_null)
        n_trials, max_n = r['sizes'].shape

        # We can't compute P for sizes we haven't MC'ed.
        # No matter how large we make max_n in the MC, higher ns
        # will occur if unknown backgrounds are large enough.
        max_n = min(max_n, self.sizes_mc.shape[1] - 1)

        # Compute P(size_n < observed| mu)
        cdf_size = np.zeros((n_trials, max_n))
        # TODO: faster way? np.searchsorted has no axis argument...
        # TODO: maybe numba helps?
        for n in range(max_n):
            # P(size_n < observed| mu)
            # i.e. P that the n-interval is 'smaller' (has fewer events expected)
            # Excess -> small intervals -> small ps
            p = np.searchsorted(self.sizes_mc[mu_i, n], r['sizes'][:,n]) / self.trials_per_s
            cdf_size[:, n] = p.clip(0, 1)

        # Find optimum interval n (for this mu)
        # optimum_n = np.argmax(cdf_size, axis=1)

        # highest cdf_size -> least indication of excess
        # (We have to flip sign for our Neyman code, just like c0 and pmax)
        return -np.max(cdf_size, axis=1)


@export
def k_largest(xn, max_n=None):
    """Return (sizes, skip_events) of largest intervals with different event count
    Both are (n_trials, max_n) arrays.

    Here size is the expected number of events inside the interval,
    and skip_events is the number of observed events left of the start of the interval.
    The index in sizes/skip_events denotes the event count in the interval.

    For example, sizes[0] gives the expected number of events in the largest observed gap.

    :param xm: Input data, Yellin-normalized, with added 0/1 events.
        The last axis must be the data dimension, earlier axes can run over trials.
    :param present: presence mask
    :param cdf: Expected CDF, default is standard uniform
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
