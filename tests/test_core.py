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

    n_toys = 1000
    (bf, result) = fu.toy_llrs(n_trials=n_toys)

    # Length matches
    assert len(bf) == n_toys
    assert len(result) == n_toys

    # No Nans
    assert np.sum(np.isnan(bf)) == 0
    assert np.sum(np.isnan(result)) == 0

    # Best fits >= 0
    assert np.all(bf >= 0)


    signal_hyps = np.arange(10)
    lower, upper = fu.toy_intervals(signal_hyps, n_trials=n_toys)

    # Length matches
    assert len(lower) == n_toys
    assert len(upper) == n_toys

    # Only signal hypotheses in intervals
    assert np.all(np.in1d(lower, signal_hyps))
    assert np.all(np.in1d(upper, signal_hyps))

    # Lower limits never above upper limits
    assert np.all(lower <= upper)
