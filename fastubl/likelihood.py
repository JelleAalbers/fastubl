import numpy as np
from scipy import optimize, stats, special

import fastubl

export, __all__ = fastubl.exporter()



@export
class UnbinnedLikelihoodBase:

    def statistic(self, r, mu_null=None):
        if mu_null is None:
            mu_null = self.true_mu[0]
        mu_null = mu_null * np.ones(r['n_trials'])
        ll_null = self.ll(mu_null, r['p_obs'], r['present'])

        if not 'mu_best' in r:
            r['mu_best'], r['ll_best'] = \
                self.optimize(r['p_obs'],
                              r['present'],
                              guess=np.clip(mu_null, 1, None))

        return -2 * (ll_null - r['ll_best']), r['mu_best']

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

        # This avoids floating-point warnings/errors
        # inner_term must always be multiplied by present!
        inner_term[~present] = 1

        ll = (-mus.sum(axis=1) + np.sum(
            special.xlogy(present, inner_term),
            axis=1))
        if not gradient:
            return ll

        grad = (-1 + np.sum(
            present * p_obs[:, :, 0] / inner_term,
            axis=1))
        if np.any(np.isnan(ll)) or np.any(np.isnan(grad)):
            raise RuntimeError("Bad stuff happening")
        if not np.all(np.isfinite(ll)) and np.all(np.isfinite(grad)):
            raise RuntimeError("Also bad stuff happening")
        return ll, grad

    def optimize(self, p_obs, present, guess=None):
        """Return (best-fit signal rate mu_best,
                   likelihood at mu_best)
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
            jac=True,
            bounds=[(self.mu_s_grid[0], self.mu_s_grid[-1])] * batch_size,)

        # Optimizers fail sometimes...
        for method in ('TNC', 'L-BFGS-B', 'Powell'):
            optresult = optimize.minimize(objective, **minimize_opts,
                                          method=method)
            if optresult.success:
                break
        else:
            raise ValueError(
                f"Optimization failed after {optresult.nfev} iterations! "
                f"Current value: {optresult.fun}; "
                f"message: {optresult.message}")

        mu_best = optresult.x
        return mu_best, self.ll(mu_best, p_obs, present)


@export
class UnbinnedLikelihoodExact(UnbinnedLikelihoodBase,
                              fastubl.NeymanConstruction):
    pass


@export
class UnbinnedLikelihoodWilks(UnbinnedLikelihoodBase,
                              fastubl.FittingStatistic):

    def critical_quantile(self,
                          cl=fastubl.DEFAULT_CL,
                          kind=fastubl.DEFAULT_KIND):
        if kind == 'central':
            critical_ts = stats.chi2(1).ppf(cl)
        else:
            critical_ts = stats.chi2(1).ppf(2 * cl - 1)
        return np.ones(len(self.mu_s_grid)) * critical_ts

