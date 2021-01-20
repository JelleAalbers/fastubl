import os
import pickle
import typing

try:
    import blosc
except ImportError:
    print("Blosc not installed, falling back to zlib")
    import zlib as blosc

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
DEFAULT_MU_S_GRID.flags.writeable = False   # Prevent accidental clobbering
DEFAULT_CL = 0.9
DEFAULT_KIND = 'upper'


@export
class StatisticalProcedure:
    random_unknown_background = 0
    random_unknown_kind = 'spike'

    def __init__(self,
                 signal, *backgrounds,
                 domain=(0., 1.),
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
        self.domain = domain

    def show_pdfs(self, x=None):
        if x is None:
            x = np.linspace(*self.domain, num=1000)
        ysum = 0
        for i, dist in enumerate(self.dists):
            y = self.true_mu[i] * dist.pdf(x)
            ysum += y
            plt.plot(x, y, label=self.labels[i])
        plt.plot(x, ysum,
                 color='k', linestyle='--', label='Total')

        plt.legend(loc='best')
        plt.xlabel("Some observable x")
        plt.xlim(x[0], x[-1])
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

        # Add random background to each trial, if requested
        if self.random_unknown_background != 0:
            # Draw events in (0, 1) at first
            n_random = stats.poisson(self.random_unknown_background).rvs(n_trials)
            x_random, present_random = _split_over_trials(
                n_random[None,:],
                np.random.rand(n_random.sum())[:,None])

            # Draw width and side (left/right) of background
            # independently each trial
            if self.random_unknown_kind == 'border_flat':
                width = np.random.rand(n_trials)[:,None]
                x_random = x_random * width
                is_flipped = np.random.randint(2, size=n_trials)[:,None]
                x_random = x_random * (1-is_flipped) + (1-x_random) * is_flipped

            elif self.random_unknown_kind == 'spike':
                x_random = 0 * x_random + np.random.rand(n_trials)[:,None]

            else:
                raise NotImplementedError(self.random_unknown_kind)

            # Transform from (0,1) to domain
            x_random = self.domain[0] + x_random * (self.domain[1] - self.domain[0])

            # Concatenate to x_obs and present
            x_obs = np.concatenate([x_obs, x_random], axis=1)
            present = np.concatenate([present, present_random], axis=1)

            # TODO: Should we sort so fake events are always on right side?
            # (this might have been assumed somewhere?)

        return x_obs, present

    def compute_ps(self, x):
        """Compute P(event_j from trial_i | source_k)
            Returns (trial_i, event_j, source_k),
            x must have shape (n_trials, max_n_events)
        """
        p_obs = np.zeros(list(x.shape) + [self.n_sources])
        for i, dist in enumerate(self.dists):
            p_obs[:, :, i] = dist.pdf(x)
        p_obs = np.nan_to_num(p_obs)   # Impossible events have p=0
        return p_obs

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

    def toy_data(self, n_trials, batch_size=None, mu_s_true=None, toy_maker=None):
        # TODO: make different defaults for different methods
        # And don't let this one behave weirdly
        if batch_size is None:
            batch_size = n_trials
        return list(self.iter_toys(n_trials,
                                   progress=False,
                                   toy_maker=toy_maker,
                                   batch_size=batch_size,
                                   mu_s_true=mu_s_true))[0]

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
        n_trials = int(n_trials)
        batch_size = int(batch_size)

        n_batches = int(np.ceil(n_trials / batch_size))
        last_batch_size = n_trials - (n_batches - 1) * batch_size

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
                       # Fixed random number per trial, several applications
                       random_number=np.random.rand(batch_size),
                       n_trials=batch_size)

    def toy_intervals(self,
                      *,
                      kind=DEFAULT_KIND,
                      cl=DEFAULT_CL,
                      n_trials=int(2e4),
                      batch_size=DEFAULT_BATCH_SIZE,
                      progress=False,
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

    def t_ppf(self, cl=DEFAULT_CL, abs=False):
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
            is_included = ts >= self.t_ppf(1 - cl)[:, np.newaxis]
        elif kind == 'lower':
            is_included = ts <= self.t_ppf(cl)[:, np.newaxis]
        elif kind == 'central':
            new_cl = 1 - 0.5 * (1 - cl)   # Twice as close to 1
            is_included =   \
                (ts >= self.t_ppf(1 - new_cl)[:, np.newaxis]) \
                & (ts <= self.t_ppf(new_cl)[:, np.newaxis])
        elif kind in ('abs_unified', 'feldman_cousins', 'fc'):
            # Feldman cousins: for every mu, include lowest absolute t values
            # => boundary is a high percentile
            # TODO: test!
            is_included = ts <= self.t_ppf(cl, abs=True)[:, np.newaxis]
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
            progress=False,
            toy_maker=None):
        """Return (test statistic) array for n_trials toy MCs.
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

    # TODO: store ppfs instead of raw values, to accommodate large n_trials

    mc_results : np.ndarray = None   # (mu_s, trial i)
    extra_cache_attributes = tuple()
    default_trials = 5000

    def __init__(self, *args,
                 cache_folder='./fastubl_neyman_cache',
                 trials_per_s=None,
                 cache=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if trials_per_s is None:
            trials_per_s = self.default_trials
        self.trials_per_s = trials_per_s

        self.hash = fastubl.deterministic_hash(dict(
            sources=self.sources,
            trials_per_s=self.trials_per_s,
            mu_s_grid=self.mu_s_grid,
            **self.extra_hash_dict()))
        fn = os.path.join(
            cache_folder,
            f'{self.__class__.__name__}_{trials_per_s}_{self.hash}')

        loaded_from_cache = False
        if cache:
            os.makedirs(cache_folder, exist_ok=True)
            if os.path.exists(fn):
                with open(fn, mode='rb') as f:
                    data = f.read()
                    try:
                        data = blosc.decompress(data)
                    except Exception:
                        # TODO: Old uncompressed pickles -- temp hack...
                        pass
                    stuff = pickle.loads(data)
                    # If any extra keys are stored in the pickle, ignore them
                    for k in self.cache_attributes():
                        setattr(self, k, stuff[k])
                loaded_from_cache = True

        if not loaded_from_cache:
            self.do_neyman_construction()

        if cache and not os.path.exists(fn):
            with open(fn, mode='wb') as f:
                to_cache = {
                    k: getattr(self, k)
                    for k in self.cache_attributes()}
                f.write(blosc.compress(pickle.dumps(to_cache)))

    def cache_attributes(self):
        return tuple(['mc_results'] + list(self.extra_cache_attributes))

    def do_neyman_construction(self):
        self.mc_results = np.zeros((self.mu_s_grid.size, self.trials_per_s),
                                   dtype=np.float32)
        for i, mu_s in enumerate(tqdm(self.mu_s_grid,
                                      desc='MC for Neyman construction')):
            self.mc_results[i] = \
                self.toy_statistics(self.trials_per_s,
                                    mu_s_true=mu_s,
                                    mu_null=mu_s,
                                    progress=False)
        # Sort, so percentile lookups are easier
        self.mc_results.sort(axis=1)

    def extra_hash_dict(self):
        return dict()

    def statistic(self, r, mu_null):
        """Return statistic evaluated at mu_null
        """
        raise NotImplementedError

    def t_ppf(self, quantile=DEFAULT_CL, abs=False):
        """Return len(self.mu_s_grid) array of critical test statistic values"""
        if abs:
            x = np.abs(self.mc_results)
            return np.percentile(x, quantile * 100, axis=1)
        else:
            # Mc results are already sorted
            return self.mc_results[:,np.round(quantile * self.trials_per_s).astype(np.int)]

    def t_cdf(self, t, mu_null):
        """Return array of P(t < ... | mu_null)"""
        mu_i = np.argmin(np.abs(self.mu_s_grid - mu_null))
        return np.searchsorted(self.mc_results[mu_i], t) / self.trials_per_s


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
