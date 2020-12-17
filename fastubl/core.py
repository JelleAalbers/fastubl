import os
import pickle

import matplotlib.pyplot as plt

import numba
import numpy as np
from scipy import stats
from tqdm import tqdm

import fastubl

export, __all__ = fastubl.exporter()
__all__ += ['DEFAULT_BATCH_SIZE',
            'DEFAULT_MU_S_GRID',
            'DEFAULT_CL',
            'DEFAULT_KIND']

DEFAULT_BATCH_SIZE = 400
DEFAULT_MU_S_GRID = np.geomspace(0.1, 50, 100)
DEFAULT_CL = 0.9
DEFAULT_KIND = 'upper'


@export
class StatisticalProcedure:

    def __init__(self,
                 signal, *backgrounds,
                 mu_s_grid=None):
        """Generic statistical procedure

        :param true_mu: Sequence of true expected number of events.
            The first is signal, any subsequent ones are backgrounds.
        :param dists: Sequence of true distributions
            The first is signal, any subsequent ones are backgrounds.
        :param labels: Sequence of labels;
        :param mu_s_grid: array of mu hypotheses used for testing
        """
        self.sources = sources = [signal] + list(backgrounds)
        for s in sources:
            assert isinstance(s, dict), "Sources must be specified as dicts"

        self.n_sources = len(sources)
        self.true_mu = np.array([s.get('mu', 0) for s in sources])
        self.dists = [getattr(stats, s['distribution'])(**s.get('params', {}))
                      for s in sources]
        self.labels =[s.get('label',
                            'Signal' if i == 0 else f'Background {i-1}')
                      for i, s in enumerate(sources)]

        if mu_s_grid is None:
            mu_s_grid = DEFAULT_MU_S_GRID.copy()
        self.mu_s_grid = mu_s_grid

    def show_pdfs(self, domain=None):
        if domain is None:
            domain = np.linspace(-0.05, 1.05, 1000)
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

    def _mu_array(self, mu_s):
        """Return (trials, sources) array of expected events for all sources
        """
        assert isinstance(mu_s, np.ndarray)
        n_trials = mu_s.size
        mu_bg = (np.array(self.true_mu[1:])[np.newaxis, :]
                 * np.ones((n_trials, 1)))
        mu_s = mu_s.reshape(-1, 1)
        return np.concatenate([mu_s, mu_bg], axis=1)

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
                  desc=None,
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
            iter = tqdm(iter, desc=desc)

        for batch_i in iter:
            if batch_i == n_batches - 1 and last_batch_size != 0:
                batch_size = last_batch_size

            x_obs, present = toy_maker(batch_size, mu_s_true=mu_s_true)

            yield dict(p_obs=self.compute_ps(x_obs),
                       x_obs=x_obs,
                       present=present,
                       n_trials=batch_size)

    def toy_intervals(self,
                      *,
                      kind=DEFAULT_KIND,
                      cl=DEFAULT_CL,
                      n_trials=int(2e4),
                      batch_size=DEFAULT_BATCH_SIZE,
                      progress=True,
                      desc=None,
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
                                desc=desc,
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
class RegularProcedure(StatisticalProcedure):
    """Procedure using a statistic
     - for which higher values indicate excesses,
       and lower values deficits.
     - possibly a function of mu, as well as the data.

    E.g. count, 1/gap, signed LLR
    """

    def statistic(self, r, mu_null):
        """Return statistic evaluated at mu_null
        """
        raise NotImplementedError

    def t_percentile(self, cl=DEFAULT_CL, abs=False):
        raise NotImplementedError

    def compute_intervals(self, r, cl=DEFAULT_CL, kind=DEFAULT_KIND):
        # Compute (n_mu, n_trials) array with statistics for all mu on data
        # TODO: this is some overkill, perhaps
        # Maybe use optimizer instead, or at least offer the option..
        ts = np.stack([self.statistic(r, mu)
                      for mu in self.mu_s_grid])

        # TODO: consider <= or =
        # Note: upper limit boundaries are *low* percentiles
        # of the t distribution! See Neyman belt construction diagram.
        kind = kind.lower()
        if kind == 'upper':
            is_included = ts >= self.t_percentile(1 - cl)[:, np.newaxis]
        elif kind == 'lower':
            is_included = ts <= self.t_percentile(cl)[:, np.newaxis]
        elif kind == 'central':
            new_cl = 1 - 0.5 * (1 - cl)   # Twice as close to 1
            is_included =   \
                (ts >= self.t_percentile(1 - new_cl)[:, np.newaxis]) \
                & (ts <= self.t_percentile(new_cl)[:, np.newaxis])
        elif kind in ('abs_unified', 'feldman_cousins', 'fc'):
            # Feldman cousins: for every mu, include lowest absolute t values
            # => boundary is a high percentile
            is_included = ts <= self.t_percentile(cl, abs=True)[:, np.newaxis]
        else:
            raise NotImplementedError(f"Unsupporterd kind '{kind}'")

        intervals = [None, None]
        n_mus = self.mu_s_grid.size
        for side in [0, 1]:
            # Get a decreasing/increasing sequence for lower/upper limits
            x = 1 + np.arange(n_mus, dtype=np.int)
            if side == 0:
                x = 2 * n_mus - x
            # Zero excluded models, limit is at highest remaining number
            x = x[:, np.newaxis] * is_included
            intervals[side] = self.mu_s_grid[np.argmax(x, axis=0)]

        # By default, we'd get the lowest/highest mu in the grid for non-central
        # intervals. Better to go all the way:
        # TODO: what about extreme results in central intervals?
        if kind == 'upper':
            intervals[0] *= 0
        elif kind == 'central':
            intervals[1] += float('inf')

        return intervals

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
        """
        if mu_s_true is None:
            mu_s_true = self.true_mus[0]
        if mu_null is None:
            mu_null = mu_s_true

        result = [
            self.statistic(r, mu_null=mu_null)
            for r in self.iter_toys(
                n_trials,
                batch_size,
                mu_s_true=mu_s_true,
                progress=progress,
                toy_maker=toy_maker)]
        result = np.concatenate(result)
        return result


@export
class NeymanConstruction(RegularProcedure):

    mc_results : np.ndarray = None   # (mu_s, trial i)

    def __init__(self, *args,
                 cache_folder='./fastubl_neyman_cache',
                 trials_per_s=int(5000),
                 cache=True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: include extra options with Neyman hash in child classes?
        self.hash = fastubl.deterministic_hash(dict(
            sources=self.sources,
            mu_s_grid=self.mu_s_grid))
        fn = os.path.join(
            cache_folder,
            f'{self.__class__.__name__}_{trials_per_s}_{self.hash}')
        if cache:
            os.makedirs(cache_folder, exist_ok=True)
            if os.path.exists(fn):
                with open(fn, mode='rb') as f:
                    self.mc_results = pickle.load(f)

        if self.mc_results is None:
            self.mc_results = np.zeros((self.mu_s_grid.size, trials_per_s))
            for i, mu_s in enumerate(tqdm(self.mu_s_grid,
                                          desc='MC for Neyman construction')):
                self.mc_results[i] = \
                    self.toy_statistics(trials_per_s,
                                        mu_s_true=mu_s,
                                        mu_null=mu_s,
                                        progress=False)

        if cache and not os.path.exists(fn):
            with open(fn, mode='wb') as f:
                pickle.dump(self.mc_results, f)

    def statistic(self, r, mu_null):
        """Return statistic evaluated at mu_null
        """
        raise NotImplementedError

    def t_percentile(self, cl=DEFAULT_CL, abs=False):
        """Return len(self.mu_s_grid) array of critical test statistic values"""
        x = self.mc_results
        if abs:
            x = np.abs(x)
        return np.percentile(x, cl * 100, axis=1)


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
