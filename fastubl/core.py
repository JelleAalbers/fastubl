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

    def iter_toys(self, n_trials, batch_size=DEFAULT_BATCH_SIZE,
                  progress=True, toy_maker=None):
        """Iterate over n_trials toy datasets in batches of size batch_size
        Each iteration yields a dictionary with the following keys:
          - x_obs, present: see make_toys
          - p_obs: see compute_ps
          - n, the number of toy datasets in the batch.

        Kwargs:
            - toy_maker: Function that produces toys, like make_toys
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

            x_obs, present = toy_maker(batch_size)

            yield dict(p_obs=self.compute_ps(x_obs),
                       x_obs=x_obs,
                       present=present,
                       n=batch_size)

    def toy_intervals(self,
                      *,
                      kind='upper',
                      cl=0.9,
                      n_trials=int(2e4),
                      batch_size=DEFAULT_BATCH_SIZE,
                      progress=True,
                      toy_maker=None,
                      **kwargs):
        """Return n_trials (upper, lower) inclusive confidence interval bounds.

        :param kind: 'central' for two-sided intervals (ordered by likelihood),
        'upper' or 'lower' for one-sided limits.
        Other arguments are as for toy_llrs.
        """
        intervals = [[], []]
        for r in self.iter_toys(n_trials, batch_size,
                                progress=progress, toy_maker=toy_maker):
            lower, upper = self.compute_intervals(r, kind=kind, cl=cl, **kwargs)
            intervals[0].append(lower)
            intervals[1].append(upper)
        return np.concatenate(intervals[0]), np.concatenate(intervals[1])

    def compute_intervals(self, r, *, kind, cl, **kwargs):
        raise NotImplementedError


@export
class DummyProcedure(StatisticalProcedure):
    pass


@export
class Poisson(StatisticalProcedure):

    def compute_intervals(self, r, *, kind, cl, **kwargs):
        n = r['present'].sum(axis=1)
        mu_bg = np.sum(self.true_mu[1:])
        if kind == 'upper':
            return np.zeros(r['n']), stats.chi2.ppf(cl, 2 * n + 2) / 2 - mu_bg
        elif kind == 'central':
            return [x - mu_bg
                    for x in poisson_central_interval(n, cl=cl)]
        raise NotImplementedError(kind)


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
