import numpy as np
from scipy import optimize, interpolate, special, stats
import fastubl


export, __all__ = fastubl.utils.exporter()


@export
class MaxGap(fastubl.StatisticalProcedure):

    _limit_loglog_curves = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._limit_loglog_curves = dict()

    def compute_intervals(self, r, *, kind, cl, **kwargs):
        g, _ = self.max_gap(r)

        if kind == 'upper':
            return (np.zeros(r['n_trials']),
                    10 ** self.limit_loglog_curve(cl)(np.log10(g)))
        else:
            raise NotImplementedError(kind)

    def max_gap(self, r):
        """Return (gap sizes, skip_events) of maximum gap
        indices are in sorted version of r['x'], with non-present events removed
        """
        xn = fastubl.yellin_normalize(r['x_obs'],
                                    r['present'],
                                    cdf=self.dists[0].cdf)
        sizes, skips = fastubl.k_largest(xn, max_n=0)
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


@export
class PMax(fastubl.NeymanConstruction):

    def statistic(self, r, mu_null=None):
        # NB: using -log(pmax)

        if 'sizes' not in r:
            xn = fastubl.yellin_normalize(r['x_obs'],
                                          r['present'],
                                          cdf=self.dists[0].cdf)
            # These are (trial_i, events_in_interval_j) matrices
            r['sizes'], r['skips'] = fastubl.k_largest(xn)

        if mu_null is None:
            # TODO: ugly to repeat every time
            mu_null = self.true_mu[0]

        # P(more events in random interval of size):
        p_more_events = stats.poisson(mu_null * r['sizes']).sf(
            np.arange(r['sizes'].shape[1])[np.newaxis,:])
        pmax = p_more_events.max(axis=1)

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))
