import numba
import numpy as np

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
