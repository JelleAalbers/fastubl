import numpy as np
from scipy import optimize, stats, special

import fastubl

export, __all__ = fastubl.exporter()



@export
class UnbinnedLikelihoodBase:

    def statistic(self, r, mu_null):
        mu_null = mu_null * np.ones(r['n_trials'])
        ll_null = self.ll(mu_null, r)

        if not 'mu_best' in r:
            r['mu_best'], r['ll_best'] = \
                self.optimize(r, guess=np.clip(mu_null, 1, None))

        lr = - 2 * (ll_null - r['ll_best'])

        # 'Sign' the likelihood ratio so deficits (mu_best < mu_null)
        # give negative ts, and we can use the usual percentile-based
        # interval setting code
        lr *= np.sign(r['mu_best'] - mu_null)
        return lr

    def ll(self, mu_signal, r, gradient=False):
        """Return array of log likelihoods for toys at mu_signal

        :param mu_signal: (trial_i,) hypothesized signal means at trial_i
        :param r: Results/data dict. Must contain:
            p_obs: (trial_i, event_i, source_i) event pdfs, see make_toys
            present: (trial_i, event_i) presence matrix, see make_toys

        :param gradient: If True, instead return (ll, grad) tuple
            of arrays. Second element is derivative of ll with respect
            to mu_signal.
        """
        p_obs, present = r['p_obs'], r['present']
        assert p_obs.shape[2] == self.n_sources
        mus, p_obs = self._correct_mus_pobs(mu_signal, r)  # -> (trial_i, source_i)

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

    def optimize(self, r, guess=None):
        """Return (best-fit signal rate mu_best,
                   likelihood at mu_best)
        :param guess: guess for signal rate
        Other parameters are as for ll
        """
        n_trials = r['n_trials']

        def objective(mu_signal):
            ll, grad = self.ll(mu_signal, r, gradient=True)
            return - 2 * np.sum(ll), -2 * grad

        if guess is None:
            # A guess very close to the bound is often bad
            # so let's have the guess be 1 at least
            guess = max(1, self.true_mu[0])
        guess = guess * np.ones(n_trials)

        minimize_opts = dict(
            x0=guess,
            jac=True,
            bounds=[(self.mu_s_grid[0], self.mu_s_grid[-1])] * n_trials,)

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
        return mu_best, self.ll(mu_best, r)


@export
class UnbinnedLikelihoodExact(UnbinnedLikelihoodBase,
                              fastubl.NeymanConstruction):
    pass


@export
class UnbinnedLikelihoodWilks(UnbinnedLikelihoodBase,
                              fastubl.RegularProcedure):

    def t_percentile(self,
                     cl=fastubl.DEFAULT_CL,
                     abs=False):
        if abs:
            # For unsigned LR
            critical_ts = stats.chi2(1).ppf(cl)
        else:
            # Signed LR
            # TODO: justify in a comment here
            critical_ts = stats.norm.ppf(cl)**2 * np.sign(cl - 0.5)
        return np.ones(len(self.mu_s_grid)) * critical_ts
