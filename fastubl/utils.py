from base64 import b32encode
from hashlib import sha1
import json

import numba
import numpy as np
from scipy import stats

def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
@numba.jit
def lookup_axis1(x, indices):
    """Take along different columns of x for each row

    :param x: (n_rows, n_cols, ...)
    :param indices: (n_rows) array of indices in axis 1 of x
    :returns: (n_rows, ...) array of results
    """
    assert indices.size == x.shape[0]

    # Array of shape of x, with axis=1 removed
    result = np.zeros((x.shape[0], *x.shape[2:]),
                      dtype=x.dtype)

    # Lookup a different value for each row
    for i, index in enumerate(indices):
        result[i, ...] = x[i, index, ...]
    return result


@export
def sort_all_by_axis1(x, *others):
    """Return (x sorted by axis 1, other arrays sorted in same order)

    :param x: 2d array

    Other arrays may have more dims than x,
    but the first two dimensions must match in shape
    """
    # Adapted from https://stackoverflow.com/questions/6155649
    sort_indices = np.argsort(x, axis=1)
    results = []
    for arr in [x] + list(others):
        # Expand sort_indices with ones until it matches the arr.shape
        si = sort_indices.reshape(tuple(
            list(sort_indices.shape)
            + [1] * (len(arr.shape)  - len(x.shape))))
        indices = np.indices(arr.shape)
        indices[1] = si
        results.append(arr[tuple(indices)])
    return tuple(results)


@export
def hashablize(obj):
    """Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    """
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, '__iter__'):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj


@export
class NumpyJSONEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types
    Edited from mpl3d: mpld3/_display.py
    """

    def default(self, obj):
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return [self.default(item) for item in iterable]
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@export
def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing
    a container hierarchy
    """
    hashable = hashablize(thing)
    jsonned = json.dumps(hashable, cls=NumpyJSONEncoder)
    digest = sha1(jsonned.encode('ascii')).digest()
    return b32encode(digest)[:length].decode('ascii').lower()


@export
def binom_interval(success, total, cl=0.6826895):
    """Confidence interval on binomial - using Jeffreys interval
    Code stolen from https://gist.github.com/paulgb/6627336
    Agrees with http://statpages.info/confint.html for binom_interval(1, 10)
    """
    # TODO: special case for success = 0 or = total? see wikipedia
    quantile = (1 - cl) / 2.
    lower = stats.beta.ppf(quantile, success, total - success + 1)
    upper = stats.beta.ppf(1 - quantile, success + 1, total - success)
    # If something went wrong with a limit calculation, report the trivial limit
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 1
    return lower, upper


@export
def endpoints(x_obs, present, p_obs, domain, only_x=False):
    n_trials, n_max_events, n_sources = p_obs.shape

    # Map fake events to right edge of domain
    x = np.where(present, x_obs, domain[1])
    # Sort by ascending x values
    x, present, p_obs = sort_all_by_axis1(x, present, p_obs)
    # add endpoint markers events at edges of domain
    x = np.concatenate([np.ones((n_trials, 1)) * domain[0],
                        x,
                        np.ones((n_trials, 1)) * domain[1]],
                       axis=1)

    if only_x:
        return x

    # Add p_obs and present for fake events
    p_obs = np.concatenate([np.ones((n_trials, 1, n_sources)),
                            p_obs,
                            np.ones((n_trials, 1, n_sources))],
                           axis=1)
    present = np.concatenate([np.zeros((n_trials, 1), dtype=np.bool_),
                              present,
                              np.zeros((n_trials, 1), dtype=np.bool_)],
                             axis=1)
    return x, present, p_obs


@export
def find_zero(x, y, last=False, fallback=None):
    """Return x at which y is 0, linearly interpolating y(x)

    :param last: if True, return last zero crossing. Otherwise returns first.
    :param fallback: return this value if there are no zero crossings.
       If not provided, ValueError is raised if no crossings are found.
       If a 2-tuple, return first element if y is always negative, second if
       y is always positive
    """
    x, y = np.asarray(x), np.asarray(y)

    # Indices at which the next y flips sign
    sign = np.sign(y)
    crossings = np.where(sign[:-1] != sign[1:])[0]

    if not len(crossings):
        if fallback is None:
            raise ValueError("No zero-crossing found")
        if isinstance(fallback, tuple):
            if np.min(sign) == 1:
                # y always positive
                return fallback[1]
            return fallback[0]
        return fallback

    # Interpolation between two points
    i = crossings[-1] if last else crossings[0]
    dx = x[i + 1] - x[i]
    dy = y[i + 1] - y[i]
    return x[i] - y[i] * dx / dy
