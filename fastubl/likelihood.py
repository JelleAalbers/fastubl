import numpy as np
from .core import StatisticalProcedure, DEFAULT_MU_S_GRID, DEFAULT_BATCH_SIZE
from .utils import exporter

from scipy import optimize, stats, special

export, __all__ = exporter()


@export
class UnbinnedLikelihood(StatisticalProcedure):

    def __init__(self, *args, guess=None, mu_s_grid=None, **kwargs):
        super().__init__(*args, **kwargs)
        if mu_s_grid is None:
            mu_s_grid = DEFAULT_MU_S_GRID.copy()
        self.mu_s_grid = mu_s_grid
        self.guess = guess

    def compute_intervals(self,
                          r,
                          kind='central',
                          cl=0.9,
                          mu_s=None,
                          critical_ts=None,
                          guess=None):
        """Return n_trials (upper, lower) inclusive confidence interval bounds.
        :param mu_s: Signal hypotheses to test
        :param critical_ts: Critical values of the test statistic distribution,
        array of same length as mu_s.
        If not given, will use asymptotic 90th percentile values.
        :param kind: 'central' for two-sided intervals (ordered by likelihood),
        'upper' or 'lower' for one-sided limits.

        Other arguments are as for toy_llrs.
        """
        if mu_s is None:
            mu_s = self.mu_s_grid
        if mu_s is None:
            raise ValueError("Must specify grid of mu_s")

        if critical_ts is None:
            # I hope this is right... Test coverage please!!
            if kind == 'central':
                critical_ts = stats.chi2(1).ppf(cl)
            else:
                critical_ts = stats.chi2(1).ppf(2 * cl - 1)
            critical_ts = np.ones(len(mu_s)) * critical_ts

        # Find the best-fit, if we haven't already
        if not 'mu_hat' in r:
            self.optimize_batch(r, guess)

        # (signal hypothesis, trial index) matrix
        is_included = np.zeros((len(mu_s), r['n']), dtype=np.int)

        # For each signal strength, see if it is in the interval
        # TODO: use optimizer instead, or at least offer the option
        # TODO: duplicate LL computations with the ones inside optimize
        for s_i, ms in enumerate(mu_s):
            t = -2 * (self.ll(ms * np.ones(r['n']), r['p_obs'],
                              r['present'])
                      - r['ll_best'])
            if kind == 'upper':
                t[r['mu_hat'] > ms] = 0
            elif kind == 'lower':
                t[r['mu_hat'] < ms] = 0
            is_included[s_i, :] = t <= critical_ts[s_i]

        # 0 is lower, 1 is upper
        intervals = [None, None]
        for side in (0, 1):
            # Get a decreasing/increasing sequence for lower/upper limits
            x = 1 + np.arange(len(mu_s), dtype=np.int)
            if side == 0:
                x = 2 * len(mu_s) - x
            # Zero excluded models. Limit is at highest number.
            x = x[:, np.newaxis] * is_included
            intervals[side] = mu_s[np.argmax(x, axis=0)]
        return intervals

    def ll(self, mu_signal, p_obs, present, gradient=False):
        """Return array of log likelihoods for toys at mu_signal

        :param mu_signal: (trial_i,) hypothesized signal means at trial_i
        :param p_obs: (trial_i, event_i, source_i) event pdfs, see make_toys
        :param present: (trial_i, event_i) presence matrix, see make_toys
        :param gradient: If True, instead return (ll, grad) tuple
        of arrays. Second element is derivative of ll with respect
        to mu_signal.
        """
        assert p_obs.shape[2] == self.n_sources
        mus = self._stack_mus(mu_signal)  # -> (trial_i, source_i)

        inner_term = np.sum(p_obs * mus[:, np.newaxis, :],
                            axis=2)  # -> (trial_i, event_i)

        ll = (-mus.sum(axis=1) + np.sum(
            special.xlogy(present, inner_term),
            axis=1))
        if not gradient:
            return ll

        grad = (-1 + np.sum(
            np.nan_to_num(present * p_obs[:, :, 0] / inner_term),
            axis=1))
        return ll, grad

    def optimize(self, p_obs, present, guess=None):
        """Return (best-fit signal rate mu_hat,
                   likelihood at mu_hat)
        :param guess: guess for signal rate
        Other parameters are as for ll
        """
        batch_size = len(p_obs)

        def objective(mu_signal):
            ll, grad = self.ll(mu_signal, p_obs, present, gradient=True)
            return - 2 * np.sum(ll), -2 * grad

        if guess is None:
            # A guess very close to the bound is often bad
            # so let's have the guess be 1 at least
            guess = max(1, self.true_mu[0])
        guess = guess * np.ones(len(p_obs))

        minimize_opts = dict(
            x0=guess,
            bounds=[(0, None)] * batch_size,
            method='TNC')
        optresult = optimize.minimize(objective, **minimize_opts, jac=True)

        if not optresult.success:
            # TODO: Still needed? Remove if not used after several tries.
            print("Optimization failure, retrying with finite diff gradients")

            def objective(mu_signal):
                return -2 * self.ll(mu_signal, p_obs, present, gradient=False)
            optresult = optimize.minimize(objective, **minimize_opts)

            if not optresult.success:
                raise ValueError(
                    f"Optimization failed after {optresult.nfev} iterations! "
                    f"Current value: {optresult.fun}; "
                    f"message: {optresult.message}")

        mu_hat = optresult.x
        return mu_hat, self.ll(mu_hat, p_obs, present)

    def optimize_batch(self, result, guess=None):
        result['mu_hat'], result['ll_best'] = \
            self.optimize(result['p_obs'], result['present'],
                          guess=guess)

    def toy_llrs(self, n_trials=int(2e4), batch_size=DEFAULT_BATCH_SIZE,
                 mu_null=None,
                 guess=None, progress=True, toy_maker=None):
        """Return bestfit and -2 LLR for n_trials toy MCs.

        Result is a 2-tuple of numpy arrays, each of length n_trials:
        (bestfit_mu, -2 * [ Log(L(bestfit) - L(mu_null) ]).

        :param batch_size: Number of toy MCs to optimize at once.
        :param mu_null: Null hypothesis to test. Default is true signal rate.
        n_trials will be ceiled to batch size.
        """
        bestfit = []
        result = []
        for r in self.iter_toys(
                n_trials,  batch_size,
                progress=progress, toy_maker=toy_maker):

            self.optimize_batch(r, guess=guess)

            if mu_null is None:
                mu_null = self.true_mu[0]
            mu_null = mu_null * np.ones(r['n'])

            ll_null = self.ll(mu_null, r['p_obs'], r['present'])
            bestfit.append(r['mu_hat'])
            result.append(-2 * (ll_null - r['ll_best']))

        return np.concatenate(bestfit), np.concatenate(result)
