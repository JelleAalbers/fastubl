from functools import partial

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import xlogy
from tqdm import tqdm


__all__ = ['FastUnbinned']


class FastUnbinned:

    def __init__(self, true_mu, dists, labels=None):
        """Return fast unbinned likelihood computer.
        The first distribution is considered signal, others backgrounds.

        :param true_mu: Sequence of true expected number of events
        :param dists: Sequence of true distributions
        :param labels: Sequence of labels;
        """
        self.true_mu = true_mu
        self.dists = dists

        n_bgs = len(self.dists) - 1
        if n_bgs < 0:
            raise ValueError("Provide at least a signal distribution")

        if labels is None:
            self.labels = ['Signal']
            if n_bgs == 1:
                self.labels += ['Background']
            else:
                self.labels += ['Background %d' % i
                                for i in range(n_bgs)]
            self.labels = tuple(self.labels)

        self.n_sources = len(self.dists)

    def show_pdfs(self, domain):
        x = domain
        ysum = 0
        for i, dist in enumerate(self.dists):
            y = self.true_mu[i] * dist.pdf(x)
            ysum += y
            plt.plot(x, y, label=self.labels[i])
        plt.plot(x, ysum,
                 color='k', linestyle='--', label='Total')

        plt.legend(loc='best')
        plt.xlabel("Some observable x")
        plt.xlim(domain[0], domain[-1])
        plt.ylabel("d rate / dx")

    def make_toys(self, n_trials=400):
        """Return (p, x, present) for n_trials toy examples, where
            p: (trial_i, event_j, source_k), P(event_j from trial_i | source_k)
            x: (trial_i, event_j), Observed x-value of event_j in trial_i
            present: (trial_i, event_j), whether trial_i has an event_j
        """
        # Total and per-source event count per trial
        n_obs_per_source = np.zeros((self.n_sources, n_trials), np.int)
        for i, mu in enumerate(self.true_mu):
            n_obs_per_source[i] = stats.poisson(mu).rvs(n_trials)

        # Total events to simulate (over all trials) per source
        tot_per_source = n_obs_per_source.sum(axis=1)

        # Draw observed x values
        x_per_source = np.zeros((self.n_sources, max(tot_per_source)))
        for i, n in enumerate(tot_per_source):
            x_per_source[i, :n] = self.dists[i].rvs(n)

        # Split the events over toy datasets
        x_obs, present = _split_over_trials(n_obs_per_source, x_per_source)

        # Compute p(event | source)
        # Indexing: (trial_i, event_i, source_i)
        p_obs = np.zeros(list(x_obs.shape) + [self.n_sources])
        for i, dist in enumerate(self.dists):
            p_obs[:, :, i] = dist.pdf(x_obs)

        return p_obs, x_obs, present

    def _stack_mus(self, mu_s):
        return np.stack(
            [mu_s] + [mu * np.ones_like(mu_s)
                      for mu in self.true_mu[1:]],
            axis=1)

    @staticmethod
    def _inner_term(ps, mus):
        return np.sum(ps * mus[:, np.newaxis, :],
                      axis=2)

    def ll(self, mu_signal, p_obs, present):
        """Return array of log likelihoods for toys at mu_signal

        :param mu_signal: (trial_i,) hypothesized signal means at trial_i
        :param p_obs: (trial_i, event_i, source_i) event pdfs, see make_toys
        :param present: (trial_i, event_i) presence matrix, see make_toys
        """
        mus = self._stack_mus(mu_signal)  # (trial_i, source_i)

        return (
                -mus.sum(axis=1)
                + np.sum(
                    xlogy(present,
                          self._inner_term(p_obs, mus)),
                    axis=1))

    def gradient(self, mu_signal, p_obs, present):
        """Return -2 * gradient of log likelihood with respect to mu_s

        Parameters are as for ll
        """
        mus = self._stack_mus(mu_signal)

        return -2 * (
                -1
                + np.sum(
                    np.nan_to_num(
                        present
                        * p_obs[:, :, 0]
                        / self._inner_term(p_obs, mus)),
                    axis=1))

    def optimize(self, p_obs, present, guess):
        """Return (best-fit signal rate mu_hat,
                   likelihood at mu_hat)
        :param guess_mu: guess for signal rate
        Other parameters are as for ll
        """
        batch_size = len(p_obs)

        def objective(mu_signal):
            return - 2 * np.sum(self.ll(mu_signal, p_obs, present))

        optresult = minimize(
            objective,
            x0=guess,
            bounds=[(0, None)] * batch_size,
            jac=partial(self.gradient,
                        p_obs=p_obs, present=present))
        if not optresult.success:
            raise ValueError(
                f"Optimization failed after {optresult.nfev} iterations! "
                f"Current value: {optresult.fun}; "
                f"message: {optresult.message}")
        mu_hat = optresult.x
        return mu_hat, self.ll(mu_hat, p_obs, present)

    def toy_llrs(self, n_trials=int(2e4), batch_size=400, mu_null=None):
        """Return bestfit and -2 LLR for n_trials toy MCs.

        Result is a 2-tuple of numpy arrays, each of length n_trials:
        (bestfit_mu, -2 * [ Log(L(bestfit) - L(mu_null) ]).

        :param batch_size: Number of toy MCs to optimize at once.
        :param mu_null: Null hypothesis to test. Default is true signal rate.
        n_trials will be ceiled to batch size.
        """
        if mu_null is None:
            mu_null = self.true_mu[0]

        n_batches = n_trials // batch_size + 1
        mu_null = mu_null * np.ones(batch_size)

        bestfit = []
        result = []
        for _ in tqdm(range(n_batches)):
            p_obs, x_obs, present = self.make_toys(batch_size)

            ll_null = self.ll(mu_null, p_obs, present)

            mu_hat, ll_best = self.optimize(p_obs, present, guess=mu_null)
            bestfit.append(mu_hat)
            result.append(-2 * (ll_null - ll_best))

        return (np.concatenate(bestfit)[:n_trials],
                np.concatenate(result)[:n_trials])


@numba.njit
def _split_over_trials(n_obs_per_source, x_per_source):
    """Return (x, present) matrices of (n_trials, max_events_per_trial)
    giving the observed x values and whether the event is present.
    """
    n_sources, n_trials = n_obs_per_source.shape

    # Total observed events per trial
    n_observed = n_obs_per_source.sum(axis=0)

    x = np.zeros((n_trials, n_observed.max()))
    present = np.zeros(x.shape, dtype=np.bool_)

    # Track index in total counts per source
    # / how many events did we already give out
    i_per_source = np.zeros(n_sources, dtype=np.int64)

    for trial_i in range(n_trials):
        # Track index of event in trial
        event_i = 0

        for source_i in range(n_sources):
            for _ in range(n_obs_per_source[source_i, trial_i]):
                present[trial_i, event_i] = True
                x[trial_i, event_i] = x_per_source[source_i,
                                                   i_per_source[source_i]]
                i_per_source[source_i] += 1
                event_i += 1

    return x, present
