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

    def optimize(self, p_obs, present, guess=None):
        """Return (best-fit signal rate mu_hat,
                   likelihood at mu_hat)
        :param guess_mu: guess for signal rate
        Other parameters are as for ll
        """
        batch_size = len(p_obs)

        def objective(mu_signal):
            return - 2 * np.sum(self.ll(mu_signal, p_obs, present))

        if guess is None:
            # A guess very close to the bound is often bad
            # so let's have the guess be 1 at least
            guess = min(1, self.true_mu[0])
        guess = guess * np.ones(len(p_obs))

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

    def iter_toys(self, n_trials, batch_size, guess=None, progress=True):
        """Iterate over n_trials toy likelihoods in batches of size batch_size
        Each iteration yields a dictionary with the following keys:
          - p_obs, x_obs, present: see make_toys
          - mu_hat, ll_best, see optimize
          - n, the number of toy datasets in the batch.
        """
        n_batches = n_trials // batch_size + 1
        last_batch_size = n_trials % batch_size

        iter = range(n_batches)
        if progress:
            iter = tqdm(iter)

        for batch_i in iter:
            if batch_i == n_batches - 1 and last_batch_size != 0:
                batch_size = last_batch_size

            p_obs, x_obs, present = self.make_toys(batch_size)

            mu_hat, ll_best = self.optimize(p_obs, present, guess=guess)

            yield dict(p_obs=p_obs, x_obs=x_obs, present=present,
                       mu_hat=mu_hat, ll_best=ll_best, n=batch_size)

    def toy_llrs(self, n_trials=int(2e4), batch_size=400,
                 guess=None, progress=True):
        """Return bestfit and -2 LLR for n_trials toy MCs.

        Result is a 2-tuple of numpy arrays, each of length n_trials:
        (bestfit_mu, -2 * [ Log(L(bestfit) - L(mu_null) ]).

        :param batch_size: Number of toy MCs to optimize at once.
        :param mu_null: Null hypothesis to test. Default is true signal rate.
        n_trials will be ceiled to batch size.
        """
        bestfit = []
        result = []
        for r in self.iter_toys(n_trials, batch_size, guess, progress):
            mu_null = self.true_mu[0] * np.ones(r['n'])
            ll_null = self.ll(mu_null, r['p_obs'], r['present'])
            bestfit.append(r['mu_hat'])
            result.append(-2 * (ll_null - r['ll_best']))

        return (np.concatenate(bestfit), np.concatenate(result))

    def toy_intervals(self, mu_s, critical_ts=None, kind='central',
                      n_trials=int(2e4), batch_size=400, progress=True,
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
        if critical_ts is None:
            if kind == 'central':
                critical_ts = stats.chi2(1).ppf(0.9)
            else:
                critical_ts = stats.chi2(1).ppf(0.8)
            critical_ts = np.ones(len(mu_s)) * critical_ts

        intervals = [[], []]
        for r in self.iter_toys(n_trials, batch_size, guess, progress):
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
            for side in (0, 1):
                # Get a decreasing/increasing sequence for lower/upper limits
                x = 1 + np.arange(len(mu_s), dtype=np.int)
                if side == 0:
                    x = 2 * len(mu_s) - x
                # Zero excluded models. Limit is at highest number.
                x = x[:, np.newaxis] * is_included
                intervals[side].append(mu_s[np.argmax(x, axis=0)])

        return np.concatenate(intervals[0]), np.concatenate(intervals[1])


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
