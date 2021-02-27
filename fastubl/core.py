import os
import pickle

try:
    import blosc
except ImportError:
    print("Blosc not installed, falling back to zlib")
    import zlib as blosc

import matplotlib.pyplot as plt
import numba
import numpy as np
from scipy import stats, interpolate
from tqdm import tqdm

import fastubl

export, __all__ = fastubl.exporter()
__all__ += ['DEFAULT_MU_S_GRID',
            'DEFAULT_CL',
            'DEFAULT_KIND',
            'DEFAULT_DOMAIN']


DEFAULT_CL = 0.9
DEFAULT_KIND = 'upper'
DEFAULT_DOMAIN = (0., 1.)

##
# Create the default mu_s_grid
##
# Start with 0.1 - 2, with 0.1 steps
_q = np.arange(0.1, 2.1, 0.1).tolist()
# Advance by 5%, or 0.25 * sigma, whichever is lower.
# Until 150, i.e. +5 sigma if true signal is 100.
# This way, we should get reasonable results for signals < 100
# even if there is some unknown background
while _q[-1] < 150:
    _q.append(min(
        _q[-1] + 0.25 * _q[-1]**0.5,
        _q[-1] * 1.05
        ))
# Round to one decimal, and at most three significant figures,
# so results don't appear unreasonably precise
#_q = [float('%.3g' % x) for x in _q]
DEFAULT_MU_S_GRID = np.unique(np.round([float('%.3g' % x) for x in _q], decimals=1))
DEFAULT_MU_S_GRID.flags.writeable = False   # Prevent accidental clobbering
del _q


@export
class StatisticalProcedure:
    batch_size = 1000
    random_unknown_background = 0
    random_unknown_kind = 'spike'
    aux_dimensions = 0

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
        self.true_mu = []
        self.dists = []
        for s in [signal] + list(backgrounds):
            assert isinstance(s, tuple), "Specify distributions as tuples"
            mu, dist = s
            assert isinstance(dist, stats._distn_infrastructure.rv_frozen), "PDF must be a scipy stats frozen dist"
            self.true_mu.append(mu)
            self.dists.append(dist)
        self.n_sources = len(self.dists)

        self.identifiers = [(d.dist.name, d.args, d.kwds)
                            for d in self.dists]

        if mu_s_grid is None:
            mu_s_grid = DEFAULT_MU_S_GRID.copy()
        self.mu_s_grid = mu_s_grid
        domains = set([dist.support() for dist in self.dists])
        if not len(domains) == 1:
            import warnings
            warnings.warn("Distributions have different supports: " + str(domains))
            self.domain = min([x[0] for x in domains]), max([x[1] for x in domains])
        else:
            self.domain = domains[0]

    def show_pdfs(self, x=None):
        if x is None:
            x = np.linspace(*self.domain, num=1000)

        ysum = 0
        for i, dist in enumerate(self.dists):
            if i == 0:
                label = 'Signal'
            elif i == 1 and len(self.dists) == 2:
                label = 'Background'
            else:
                label = f'Background {i}'
            y = self.true_mu[i] * dist.pdf(x)
            ysum += y
            plt.plot(x, y, label=label)

        plt.plot(x, ysum,
                 color='k', linestyle='--', label='Total')

        plt.legend(loc='best')
        plt.xlabel("Some observable x")
        plt.xlim(x[0], x[-1])
        plt.ylabel("d rate / dx")

    def get_mu_i(self, mu_null):
        mu_i = np.searchsorted(self.mu_s_grid, mu_null)
        assert self.mu_s_grid[mu_i] == mu_null, "mu_s must be on grid"
        return mu_i

    def mu_all(self, mu_s=None):
        """Return (n_sources) array of expected events
        :param mu_s: Expected signal events, float
        """
        if mu_s is None:
            mu_s = self.true_mu[0]
        if len(self.dists) == 1:
            return np.array([mu_s, ])
        else:
            return np.concatenate([[mu_s], self.true_mu[1:]])

    def sum_cdf(self, x, mu_s=None):
        """Compute cdf of all sources combined
        :param x: Observed x, array (arbitrary shape)
        :param mu_s: Hypothesized expected signal events
        """
        mu_all = self.mu_all(mu_s)
        sum_cdf = np.stack(
            [mu * dist.cdf(x)
             for mu, dist in zip(mu_all, self.dists)],
            axis=-1).sum(axis=-1) / mu_all.sum()
        return sum_cdf

    def compute_pdfs(self, x_obs):
        """Return p(event | source)
           x: (trial_i, event_j) array
           Returns: (trial_i, event_i, source_i array
        """
        p_obs = np.zeros(list(x_obs.shape) + [self.n_sources])
        for i, dist in enumerate(self.dists):
            p_obs[:, :, i] = dist.pdf(x_obs)
        return p_obs

    def make_toys(self, n_trials=None, mu_s_true=None):
        """Return (x, present) for n_trials toy examples, where
            x: (trial_i, event_j), Observed x-value of event_j in trial_i
            present: (trial_i, event_j), whether trial_i has an event_j
            skip_compute: if True, return just (x, present), skipping p computation
            mus_true: Override the signal mu for the toy generation
        """
        if n_trials is None:
            n_trials = self.batch_size

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

        # Draw aux dimensions (last dimension might be 0-length)
        aux_obs = np.random.rand(n_trials, x_obs.shape[1], self.aux_dimensions)

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

        return x_obs, present, aux_obs

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
                  batch_size=None,
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
        if batch_size is None:
            batch_size = self.batch_size
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

            x_obs, present, aux_obs = toy_maker(batch_size, mu_s_true=mu_s_true)

            yield dict(p_obs=self.compute_ps(x_obs),
                       x_obs=x_obs,
                       aux_obs=aux_obs,
                       present=present,
                       # Fixed random number per trial, several applications
                       random_number=np.random.rand(batch_size),
                       n_trials=batch_size)

    def toy_intervals(self,
                      *,
                      kind=DEFAULT_KIND,
                      cl=DEFAULT_CL,
                      n_trials=int(2e4),
                      batch_size=None,
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
        if batch_size is None:
            batch_size = self.batch_size
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

    def all_ts(self, r):
        """Return (n_trials, n_mu) array of statistics evaluated at all
        mu the grid
        :param r: Dictionary with data
        """
        return np.stack([self.statistic(r, mu)
                         for mu in self.mu_s_grid]).T

    def compute_intervals(self, r, cl=DEFAULT_CL, kind=DEFAULT_KIND):
        # Get upper and lower bounds of the 'Accept' region
        # TODO: account for floating-point errors
        high = np.inf * np.ones(self.mu_s_grid.size)
        low = -high
        kind = kind.lower()
        if kind == 'upper':
            # Note: upper limit boundaries are *low* percentiles
            # of the t distribution! See Neyman belt construction diagram.
            low = self.t_ppf(1 - cl)
        elif kind == 'lower':
            high = self.t_ppf(cl)
        elif kind == 'central':
            new_cl = 1 - 0.5 * (1 - cl)   # Twice as close to 1
            low = self.t_ppf(1 - new_cl)
            high = self.t_ppf(new_cl)
        elif kind in ('abs_unified', 'feldman_cousins', 'fc'):
            # Feldman cousins: for every mu, include lowest absolute t values
            # => boundary is a high percentile
            # TODO: test!
            high = self.t_ppf(cl, abs=True)
            low = -high
        else:
            raise NotImplementedError(f"Unsupported kind '{kind}'")

        ts = self.all_ts(r)

        # TODO: numbafy? probably not worth it.
        lower, upper = np.zeros((2, r['n_trials']))
        for i, t in enumerate(ts):
            lower[i] = fastubl.find_zero(
                self.mu_s_grid, high - t, last=True,
                fallback=(np.inf, 0))
            upper[i] = fastubl.find_zero(
                self.mu_s_grid, t - low, last=False,
                fallback=(0, np.inf))
        return lower, upper

    def toy_statistics(
            self,
            n_trials=int(2e4),
            batch_size=None,
            mu_s_true=None,
            mu_null=None,
            progress=False,
            toy_maker=None):
        """Return (test statistic) array for n_trials toy MCs.
        :param batch_size: Number of toy MCs to optimize at once.
        :param mu_s_true: True signal rate.
        :param mu_null: Null hypothesis to test. Default is true signal rate.
        """
        if batch_size is None:
            batch_size = self.batch_size
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

    # TODO: store ppfs instead of raw values, to accommodate large n_trials?

    mc_results : np.ndarray = None   # (mu_s, trial i)
    extra_cache_attributes = tuple()
    default_trials = 5000

    def __init__(self, *args,
                 cache_folder='./fastubl_neyman_cache',
                 trials_per_s=None,
                 cache=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_neyman_init()

        if trials_per_s is None:
            trials_per_s = self.default_trials
        self.trials_per_s = trials_per_s

        self.hash = fastubl.deterministic_hash(dict(
            identifiers=self.identifiers,
            background_mus=self.true_mu[1:],
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
                assert self.mc_results.dtype == np.float64, \
                    f"{fn} has imprecise floats, remove and redo Neyman construction"

        if not loaded_from_cache:
            self.do_neyman_construction()

        if cache and not os.path.exists(fn):
            with open(fn, mode='wb') as f:
                to_cache = {
                    k: getattr(self, k)
                    for k in self.cache_attributes()}
                f.write(blosc.compress(pickle.dumps(to_cache)))

    def pre_neyman_init(self):
        pass

    def cache_attributes(self):
        return tuple(['mc_results'] + list(self.extra_cache_attributes))

    def do_neyman_construction(self):
        self.mc_results = np.zeros((self.mu_s_grid.size, self.trials_per_s),
                                   dtype=np.float64)
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
        # First find the ppf on the Neyman grid
        if abs:
            x = np.abs(self.mc_results)
            t_neyman = np.percentile(x, quantile * 100, axis=1)
        else:
            # Mc results are already sorted
            t_neyman = self.mc_results[:,np.round(quantile * self.trials_per_s).astype(np.int)]

        # Upsample to the mu_s_grid
        return self._upsample_neyman(t_neyman)

    def t_cdf(self, t, mu_null):
        """Return array of P(t < ... | mu_null)"""
        mu_i = np.argmin(np.abs(self.mu_s_grid - mu_null))
        t_neyman = np.searchsorted(self.mc_results[mu_i], t) / self.trials_per_s
        return self._upsample_neyman(t_neyman)

    def _upsample_neyman(self, y):
        return interpolate.interp1d(
            self.mu_s_neyman_grid, y,
            bounds_error=False,
            fill_value=(y[0], y[-1]))(self.mu_s_grid)


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
