import numpy as np
from .utils import exporter
from .core import StatisticalProcedure

from scipy import optimize, interpolate, special

export, __all__ = exporter()


@export
class MaxGap(StatisticalProcedure):

    _limit_loglog_curves = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_loglog_curves = dict()

    def compute_intervals(self, r, *, kind, cl, **kwargs):
        g, _ = self.max_gap(r)

        if kind == 'upper':
            return np.zeros(r['n']), 10 ** self.limit_loglog_curve(cl)(np.log10(g))
        else:
            raise NotImplementedError(kind)

    def max_gap(self, r):
        """Return (gap sizes, skip_events) of maximum gap
        indices are in sorted version of r['x'], with non-present events removed
        """
        if r['x_obs'].size == 0:
            # No trials have events!
            return np.ones(r['n']), np.zeros(r['n'])
        # Find the largest gap in each trial
        # Non-present events are set to 1,
        # so they influence neither max gap size nor the skip_events
        x = np.where(r['present'], r['x_obs'], np.ones(r['x_obs'].shape))
        sizes, skips = k_largest(x, self.dists[0].cdf, max_n=0)
        return sizes[..., 0], skips[..., 0]

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


def k_largest(x, cdf=lambda x: x, max_n=None):
    """Return (sizes, skip_events) of largest intervals with different event count

    Here size is the expected number of events inside the interval,
    and skip_events is the number of observed events left of the start of the interval.
    The index in sizes/skip_events denotes the event count in the interval.

    For example, sizes[0] gives the expected number of events in the largest observed gap.

    :param x: Input data. The last axis must be the data dimension, earlier axes can run over trials.
    :param cdf: Expected CDF, default is standard uniform
    :param max_n: Maximum event count to consider. Defaults to len(x)
    """
    if len(x.shape) > 1:
        other_dims = list(x.shape[:-1])
    else:
        other_dims = []

    x = np.asarray(x)
    if max_n is None:
        max_n = x.shape[-1]
    x = np.sort(x, axis=-1)
    if cdf is not None:
        x = cdf(x)
    x = add_bounds(x)

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


def add_bounds(x):
    """pad data along final axis by 0 and 1"""
    x = np.asarray(x)
    for q in [0, 1]:
        x = np.pad(
            x,
            [(0, 0)] * (len(x.shape) - 1) + [(1 - q, q)],
            mode='constant', constant_values=q)
    return x
