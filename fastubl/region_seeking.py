import numpy as np
from scipy import stats

import fastubl

export, __all__ = fastubl.exporter()


@export
class PoissonSeeker(fastubl.NeymanConstruction):
    """Use P(more events in interval), in interval with lowest Poisson upper
    limit.
    """
    def __init__(self, *args, optimize_for_cl=fastubl.DEFAULT_CL, **kwargs):
        self.guide = fastubl.PoissonGuide(optimize_for_cl)
        super().__init__(*args, **kwargs)

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)
        if 'best_poisson' not in r:
            r['best_poisson'] = self.guide(
                x_obs=r['x_obs'],
                present=r['present'],
                p_obs=r['p_obs'],
                dists=self.dists,
                bg_mus=self.true_mu[1:])

        mu_bg = np.sum(r['best_poisson']['acceptance'][:,1:]
                       * np.array(self.true_mu[1:])[np.newaxis, :],
                       axis=1)
        mu_sig = mu_null * r['best_poisson']['acceptance'][:,0]

        bp = r['best_poisson']
        pmax = stats.poisson(mu_sig + mu_bg).sf(bp['n_observed'])

        # Excesses give low pmax, so we need to invert (or sign-flip)
        # to use the regular interval setting code.
        # Logging seems nice anyway
        return -np.log(np.maximum(pmax, 1.e-9))



class GuidedLikelihoodBase(fastubl.UnbinnedLikelihoodExact):

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)
        if 'guide_result' not in r:
            r['guide_result'] = self.guide(
                x_obs=r['x_obs'],
                present=r['present'],
                p_obs=r['p_obs'],
                dists=self.dists,
                bg_mus=self.true_mu[1:])

        # Mask out data, except in the interval
        intervals = r['guide_result']['interval_bounds']
        in_interval = ((intervals[0][:,np.newaxis] < r['x_obs'])
                       & (r['x_obs'] < intervals[1][:,np.newaxis]))

        new_r = {
            **r,
            'present': r['present'] & in_interval,
            'acceptance': r['guide_result']['acceptance']}
        result = super().statistic(new_r, mu_null)
        # Add new keys computed by the statistic, e.g. mu_best
        for k in new_r.keys():
            if k not in r:
                r[k] = new_r[k]
        return result


@export
class PoissonGuidedLikelihood(GuidedLikelihoodBase):
    """Likelihood inside interval found by Poisson seeker
    """
    def __init__(self, *args, optimize_for_cl=fastubl.DEFAULT_CL, **kwargs):
        self.guide = fastubl.PoissonGuide(optimize_for_cl=optimize_for_cl)
        super().__init__(*args, **kwargs)


@export
class LikelihoodGuidedLikelihood(GuidedLikelihoodBase):
    def __init__(self,
                 *args,
                 mu_low=1., mu_high=10.,
                 **kwargs):
        self.guide = fastubl.LikelihoodGuide(mu_low=mu_low, mu_high=mu_high)
        super().__init__(*args, **kwargs)
