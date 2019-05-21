import numpy as np
from scipy import stats

from fastubl import *


def test_basics():
    max_bg_sigma = 5

    fu = FastUnbinned(
        true_mu=[5, 10],
        dists=[stats.norm(),
               stats.uniform(loc=-max_bg_sigma,
                             scale=2 * max_bg_sigma)])

    (bf, result) = fu.toy_llrs(100)

    # Length matches
    assert len(bf) == 100
    assert len(result) == 100

    # No Nans
    assert np.sum(np.isnan(bf)) == 0
    assert np.sum(np.isnan(result)) == 0

    # Best fits >= 0
    assert np.all(bf >= 0)
