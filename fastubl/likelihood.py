import numpy as np
from scipy import optimize, stats, special

import fastubl

export, __all__ = fastubl.exporter()


@export
def wilks_t_ppf(cl, abs=False):
    if abs:
        # For unsigned LR
        return stats.chi2(1).ppf(cl)
    else:
        # Signed LR
        # TODO: justify in a comment here
        return stats.norm.ppf(cl) ** 2 * np.sign(cl - 0.5)


@export
def wilks_t_cdf(t, abs=False):
    if abs:
        return stats.chi2(1).cdf(t)
    else:
        return (1 - np.sign(t)) / 2 + np.sign(t) * stats.norm.cdf(np.abs(t) ** 0.5)


@export
class UnbinnedLikelihoodBase:

    def statistic(self, r, mu_null):
        mu_null = mu_null * np.ones(r['n_trials'])
        ll_null = self.ll(mu_null, r)

        if not 'mu_best' in r:
            r['mu_best'], r['ll_best'] = \
                self.optimize(r, guess=np.clip(mu_null, 1, None))

        ts = - 2 * (ll_null - r['ll_best'])

        # Choose the sign so that deficits (mu_best < mu_null)
        # give negative ts, and we can use the usual percentile-based
        # interval setting code
        ts *= np.sign(r['mu_best'] - mu_null)
        return ts

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
        mus = self._mu_array(mu_signal)   # (trials, sources)
        if 'acceptance' in r:
            mus = mus * r['acceptance']
            # p_obs increases for lower acceptances (PDF renormalizes)
            # (/0 non-finiteness is handled below, no need for warning)
            with np.errstate(all='ignore'):
                p_obs = p_obs / r['acceptance'][:, None, :]
            # If acceptances = 0, the zero division can cause odd results
            # -- but no events are possible, so we set p_obs = 0
            p_obs[~np.isfinite(p_obs)] = 0
        return log_likelihood(p_obs, present, mus,
                              weights=r.get('weights'),
                              gradient=gradient)

    def _mu_array(self, mu_s):
        """Return (trials, sources) array of expected events for all sources
        """
        assert isinstance(mu_s, np.ndarray)
        n_trials = mu_s.size
        mu_bg = (np.array(self.true_mu[1:])[np.newaxis, :]
                 * np.ones((n_trials, 1)))
        mu_s = mu_s.reshape(-1, 1)
        return np.concatenate([mu_s, mu_bg], axis=1)

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
            bounds=optimize.Bounds(np.ones(n_trials) * self.mu_s_grid[0],
                                   np.ones(n_trials) * self.mu_s_grid[-1],
                                   keep_feasible=True))

        # Optimizers fail sometimes, so try several methods
        # (not Powell: it does not use gradients, so it can't handle
        #  the large # of variables in our objective)
        for method in ('TNC', 'L-BFGS-B'):
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
class TemperedLikelihoodBase(UnbinnedLikelihoodBase):

    def ll(self, mu_signal, r, gradient=False):
        result = super().ll(mu_signal, r, gradient=gradient)

        _rand = r['random_number'] - 0.5
        extra = np.log(mu_signal) * _rand

        if gradient:
            ll, grad = result
            extra_grad = _rand / mu_signal
            return ll + extra, grad + extra_grad
        else:
            return result + extra


@export
class UnbinnedLikelihoodExact(UnbinnedLikelihoodBase,
                              fastubl.NeymanConstruction):
    pass


@export
class UnbinnedLikelihoodWilks(UnbinnedLikelihoodBase,
                              fastubl.RegularProcedure):

    def t_ppf(self, cl=fastubl.DEFAULT_CL, abs=False):
        return np.ones(self.mu_s_grid.size) * wilks_t_ppf(cl, abs)


@export
class TemperedLikelihoodExact(TemperedLikelihoodBase,
                              fastubl.NeymanConstruction):
    pass


@export
class TemperedLikelihoodWilks(TemperedLikelihoodBase,
                              fastubl.RegularProcedure):

    def t_ppf(self, cl=fastubl.DEFAULT_CL, abs=False):
        return np.ones(self.mu_s_grid.size) * wilks_t_ppf(cl, abs)


@export
def log_likelihood(p_obs, present, mus, weights=None, gradient=True):
    """Compute log likelihood

    :param p_obs: P(event | source), (trials, events, sources) array
    :param present: whether event is real, (trials, events) array
    :param mus: applicable mus, (trials, sources) array
    :param weights: (trials, events) array of weights for events.
        If not specified, assumed all one.
    :param gradient: If True, instead return (ll, grad) tuple
        of arrays. Second element is derivative of ll with respect
        to mu_signal.
    :return: log likelihood, (trials) array
    """
    n_trials, n_events, n_sources = p_obs.shape
    assert present.shape == (n_trials, n_events)
    assert mus.shape == (n_trials, n_sources)

    inner_term = np.sum(p_obs * mus[:, np.newaxis, :],
                        axis=2)  # -> (trial_i, event_i)

    # This avoids floating-point warnings/errors
    # inner_term must always be multiplied by present!
    inner_term[~present] = 1

    if weights is not None:
        # Weights enters algebraically just like present
        # (except that it's a float, not a bool)
        present = weights * present

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
