import os
import pickle

import matplotlib.pyplot as plt
from multihist import poisson_central_interval
import numba
import numpy as np
from scipy import stats
from tqdm import tqdm

from .utils import exporter

export, __all__ = exporter()
__all__ += ['DEFAULT_BATCH_SIZE', 'DEFAULT_MU_S_GRID']

DEFAULT_BATCH_SIZE = 400
DEFAULT_MU_S_GRID = np.geomspace(0.1, 50, 100)
DEFAULT_CL = 0.9
DEFAULT_INTERVAL = 'upper'


@export
def poisson_ul(n, mu_bg, cl=DEFAULT_CL):
    """Upper limit on mu_signal, from observing n events
    where mu_bg background events were expected
    """
    return stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg


@export
class StatisticalProcedure:

    def __init__(self, true_mu, dists, labels=None, mu_s_grid=None):
        """Generic statistical procedure

        :param true_mu: Sequence of true expected number of events.
            The first is signal, any subsequent ones are backgrounds.
        :param dists: Sequence of true distributions
            The first is signal, any subsequent ones are backgrounds.
        :param labels: Sequence of labels;
        :param mu_s_grid: array of mu hypotheses used for testing
        """
        self.true_mu = true_mu
        self.dists = dists
        if mu_s_grid is None:
            mu_s_grid = DEFAULT_MU_S_GRID.copy()
        self.mu_s_grid = mu_s_grid

        n_bgs = len(self.dists) - 1
        if n_bgs < 0:
            raise ValueError("Provide at least a signal distribution")

        if labels is None:
            labels = ['Signal']
            if n_bgs == 1:
                labels += ['Background']
            else:
                labels += ['Background %d' % i
                           for i in range(n_bgs)]
        self.labels = tuple(labels)

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

    def compute_pdfs(self, x_obs):
        """Return p(event | source)
           x: (trial_i, event_j) array
           Returns: (trial_i, event_i, source_i array
        """
        p_obs = np.zeros(list(x_obs.shape) + [self.n_sources])
        for i, dist in enumerate(self.dists):
            p_obs[:, :, i] = dist.pdf(x_obs)
        return p_obs

    def make_toys(self, n_trials=DEFAULT_BATCH_SIZE, mu_s_true=None):
        """Return (x, present) for n_trials toy examples, where
            x: (trial_i, event_j), Observed x-value of event_j in trial_i
            present: (trial_i, event_j), whether trial_i has an event_j
            skip_compute: if True, return just (x, present), skipping p computation
            mus_true: Override the signal mu for the toy generation
        """
        # Total and per-source event count per trial
        n_obs_per_source = np.zeros((self.n_sources, n_trials), np.int)
        for i, mu in enumerate(self.true_mu):
            if i == 0 and mu_s_true is not None:
                mu = mu_s_true
            n_obs_per_source[i] = stats.poisson(mu).rvs(n_trials)

        # Total events to simulate (over all trials) per source
        tot_per_source = n_obs_per_source.sum(axis=1)

        # Draw observed x values
        x_per_source = np.zeros((self.n_sources, max(tot_per_source)))
        for i, n in enumerate(tot_per_source):
            x_per_source[i, :n] = self.dists[i].rvs(n)

        # Split the events over toy datasets
        x_obs, present = _split_over_trials(n_obs_per_source, x_per_source)
        return x_obs, present

    def compute_ps(self, x):
        """Compute P(event_j from trial_i | source_k)
            Returns (trial_i, event_j, source_k),
            x must have shape (n_trials, max_n_events)
        """
        p_obs = np.zeros(list(x.shape) + [self.n_sources])
        for i, dist in enumerate(self.dists):
            p_obs[:, :, i] = dist.pdf(x)
        return p_obs

    def _stack_mus(self, mu_s):
        return np.stack(
            [mu_s] + [mu * np.ones_like(mu_s)
                      for mu in self.true_mu[1:]],
            axis=1)

    def _wrap_toy_maker(self, toy_maker):
        if toy_maker is None:
            return self.make_toys
        assert isinstance(toy_maker, StatisticalProcedure)
        # Wrap the toy maker so it does not compute p(event|source),
        # but only simulates
        def wrapped(*args, **kwargs):
            kwargs['skip_compute'] = True
            x, present = toy_maker.make_toys(*args, **kwargs)
            return self.compute_ps(x), x, present
        return wrapped

    def iter_toys(self, n_trials,
                  batch_size=DEFAULT_BATCH_SIZE,
                  mu_s_true=None,
                  progress=True,
                  toy_maker=None):
        """Iterate over n_trials toy datasets in batches of size batch_size
        Each iteration yields a dictionary with the following keys:
          - x_obs, present: see make_toys
          - p_obs: see compute_ps
          - n, the number of toy datasets in the batch.

        Kwargs:
            - mu_s_true: True expected signal events, passed to make_toys
            - toy_maker: StatisticalProcedure to call make_toys for instead
            - progress: if True (default), show a progress bar
        """
        if toy_maker is None:
            toy_maker = self.make_toys
        else:
            assert isinstance(toy_maker, StatisticalProcedure)
            toy_maker = toy_maker.make_toys

        n_batches = n_trials // batch_size + 1
        last_batch_size = n_trials % batch_size

        iter = range(n_batches)
        if progress:
            iter = tqdm(iter)

        for batch_i in iter:
            if batch_i == n_batches - 1 and last_batch_size != 0:
                batch_size = last_batch_size

            x_obs, present = toy_maker(batch_size, mu_s_true=mu_s_true)

            yield dict(p_obs=self.compute_ps(x_obs),
                       x_obs=x_obs,
                       present=present,
                       n=batch_size)

    def toy_intervals(self,
                      *,
                      kind=DEFAULT_INTERVAL,
                      cl=DEFAULT_CL,
                      n_trials=int(2e4),
                      batch_size=DEFAULT_BATCH_SIZE,
                      progress=True,
                      mu_s_true=None,
                      toy_maker=None):
        """Return n_trials (upper, lower) inclusive confidence interval bounds.

        :param kind: 'central' for central intervals,
            'upper' or 'lower' for one-sided limits.
        :param cl: Confidence level
            Other arguments are as for iter_toys.
        """
        intervals = [[], []]
        for r in self.iter_toys(n_trials, batch_size,
                                mu_s_true=mu_s_true,
                                progress=progress,
                                toy_maker=toy_maker):
            lower, upper = self.compute_intervals(r, kind=kind, cl=cl)
            intervals[0].append(lower)
            intervals[1].append(upper)
        return np.concatenate(intervals[0]), np.concatenate(intervals[1])

    def compute_intervals(self, r, kind, cl):
        raise NotImplementedError


@export
class DummyProcedure(StatisticalProcedure):
    pass


@export
class Poisson(StatisticalProcedure):

    def compute_intervals(self, r, kind=DEFAULT_INTERVAL, cl=DEFAULT_CL):
        n = r['present'].sum(axis=1)
        mu_bg = np.sum(self.true_mu[1:])
        if kind == 'upper':
            return np.zeros(r['n']), poisson_ul(r['n'], mu_bg, cl=cl)
        elif kind == 'central':
            return [x - mu_bg
                    for x in poisson_central_interval(n, cl=cl)]
        raise NotImplementedError(kind)


class FittingStatistic(StatisticalProcedure):
    """A statistic that is:
      - A function of mu_s (as well as the data);
      - zero at a "best-fit" mu_s;
      - higher otherwise

    Negative log likelihood ratios behave like this.
    """

    def statistic(self, r, mu_null=None):
        """Return statistic evaluated at mu_null.
        """
        raise NotImplementedError

    def critical_quantile(self, cl=DEFAULT_CL, kind=DEFAULT_INTERVAL):
        raise NotImplementedError

    def compute_intervals(self, r, cl=DEFAULT_CL, kind=DEFAULT_INTERVAL):
        critical_ts = self.critical_quantile(cl=cl, kind=kind)

        # Compute statistic on data, for each possible mu_s
        # TODO: this is some overkill, perhaps
        # Maybe use optimizer instead, or at least offer the option..
        n_trials = r['x'].shape[0]
        n_mus = self.mu_s_grid.size
        ts = np.zeros(n_mus, n_trials)
        bestfits = np.zeros(n_mus, n_trials)
        for i, mu in enumerate(self.mu_s_grid):
            ts[i, :], bestfits[i, :] = self.statistic(r, mu)
        ts = self.zero_excessive_bestfits(ts, bestfits)

        # For each signal strength, see if it is in the interval
        is_included = ts > critical_ts[:, np.newaxis]

        intervals = [None, None]
        for side in [0, 1]:
            # Get a decreasing/increasing sequence for lower/upper limits
            x = 1 + np.arange(n_mus, dtype=np.int)
            if side == 0:
                x = 2 * len(self.mu_s_grid) - x
            # Zero excluded models, limit is at highest remaining number.
            x = x[:, np.newaxis] * is_included
            intervals[side] = self.mu_s_grid[np.argmax(x, axis=0)]

        return intervals

    def zero_excessive_bestfits(self, ts, mu_hat, kind=DEFAULT_INTERVAL):
        """Return ts with bestfits above/below the null hypothesis zeroed out
        for upper/lower limits.

        :param x: (mu_s, trial_i) array of statistic results
        :param kind: upper, lower, or central
        :return:
        """
        assert ts.shape[0] == self.mu_s_grid.size
        assert ts.shape == mu_hat.shape
        if kind == 'central':
            return ts
        assert kind in ('upper', 'lower')
        # For upper limits, bestfits > the true signal should be zeroed
        mask = ts > self.mu_s_grid[:, np.newaxis]
        if kind == 'lower':
            mask = True ^ mask
        return np.where(mask, 1, ts)

    def toy_statistics(
            self,
            n_trials=int(2e4),
            batch_size=DEFAULT_BATCH_SIZE,
            mu_s_true=None,
            mu_null=None,
            progress=True,
            toy_maker=None):
        """Return (test statistic, dict with arrays of bonus info) for n_trials toy MCs.
        :param batch_size: Number of toy MCs to optimize at once.
        :param mu_s_true: True signal rate.
        :param mu_null: Null hypothesis to test. Default is true signal rate.
        n_trials will be ceiled to batch size.
        """
        if mu_null is None:
            mu_null = self.true_mus[0]

        result = []
        bestfits = []
        for r in self.iter_toys(
                n_trials,
                batch_size,
                mu_s_true=mu_s_true,
                progress=progress,
                toy_maker=toy_maker):

            r = self.statistic(r, mu_null=mu_null)
            bestfits.append(r['mu_hat'])
            result.append(r)

        result = np.concatenate(result)
        bestfits = np.concatenate(bestfits)
        return result, bestfits


class NeymanConstruction(StatisticalProcedure):

    mc_results : np.ndarray   # (mu_s, trial i)
    mc_bestfits : np.ndarray  # (mu_s, trial i)

    def statistic(self, r, mu_null):
        raise NotImplementedError

    def __init__(self, *args, filename=None, trials_per_s=int(1000), **kwargs):
        super().__init__(*args, **kwargs)

        if filename is not None and os.path.exists(filename):
            with open(filename, mode='rb') as f:
                self.mc_results = pickle.load(f)
        else:
            self.mc_results = np.zeros(self.mu_s_grid.size, trials_per_s)
            self.mc_bestfits = np.zeros_like(self.mc_results)
            for i, mu_s in enumerate(tqdm(self.mu_s_grid)):
                self.mc_results[i], self.mc_bestfits = \
                    self.toy_statistics(trials_per_s,
                                        mu_null=mu_s)

        if filename is not None and not os.path.exists(filename):
            with open(filename, mode='wb') as f:
                pickle.dump(self.mc_results, f)

    def critical_quantile(self, cl, kind='upper'):
        """Return len(self.mu_s_grid) array of critical test statistic values"""
        mc_results = self.zero_excessive_bestfits(
            self.mc_results, self.mc_bestfits, kind=kind)
        return np.percentile(mc_results, cl * 100, axis=1)


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
