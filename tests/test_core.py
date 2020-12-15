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
))
def proc(request):
    slope = 0.2
    dist = stats.truncexpon(scale=slope, b=1 / slope)

    return getattr(fastubl, request.param)(
        true_mu=[0, 5],
        dists=[stats.uniform(), dist])


def test_basics(proc):
    # Toy data generation
    x, present = proc.make_toys(n_trials=10)
    assert x.shape == present.shape
    assert x.shape[0] == 10
    assert present.dtype == np.bool_

    # Toy interval generation
    lower, upper = proc.toy_intervals(n_trials=10,
                                      kind='upper',
                                      cl=0.9,
                                      progress=False)
    assert lower.shape == upper.shape == (10,)
    assert np.all(lower == 0)
