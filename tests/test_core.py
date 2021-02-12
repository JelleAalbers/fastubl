import numpy as np
import pytest
from scipy import stats

import fastubl

# Run tests over all statistical procedures
# (Omitted Neyman-construction ones for now, take time..)
@pytest.fixture(params=(
        "Poisson",
        "OptimalCutPoisson",
        "UnbinnedLikelihoodWilks",
        "MaxGap",
        "SacrificialPoisson"
))
def proc(request):
    bg_slope = 0.2
    return getattr(fastubl, request.param)(
        (0, stats.uniform()),
        (5, stats.truncexpon(scale=bg_slope, b=1 / bg_slope)))


def test_basics(proc):
    # Toy data generation
    x, present, aux = proc.make_toys(n_trials=10)
    assert x.shape == present.shape
    assert x.shape[0] == 10
    assert aux.shape[0] == 10
    assert aux.shape[2] == 0
    assert present.dtype == np.bool_

    # Toy interval generation
    lower, upper = proc.toy_intervals(n_trials=10,
                                      kind='upper',
                                      cl=0.9,
                                      progress=False)
    assert lower.shape == upper.shape == (10,)
    assert np.all(lower == 0)
