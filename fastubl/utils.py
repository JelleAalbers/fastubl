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
def lookup_axis1(x, indices):
    """Return values of x at indices along axis 1"""
    d = indices
    return np.take_along_axis(
        x,
        d.reshape(len(d), -1), axis=1
    ).reshape(d.shape)
