import numpy as np
from scipy import stats

import fastubl

export, __all__ = fastubl.exporter()


@export
class PoissonSeeker(fastubl.NeymanConstruction):
    """Return lowest Poisson limit among intervals
    """
    def __init__(self, *args,
                 optimize_for_cl=fastubl.DEFAULT_CL,
                 pcl_sigma=None,
                 **kwargs):
        self.optimize_for_cl = optimize_for_cl
        self.pcl_sigma = pcl_sigma
        self.guide = fastubl.PoissonGuide(
            optimize_for_cl=optimize_for_cl,
            pcl_sigma=pcl_sigma)
        super().__init__(*args, **kwargs)

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)
        if 'best_poisson' not in r:
            r['best_poisson'] = self.guide(
                x_obs=r['x_obs'],
                present=r['present'],
                p_obs=r['p_obs'],
                dists=self.dists,
                bg_mus=self.true_mu[1:],
                domain=self.domain)
            r['interval'] = (
                r['best_poisson']['interval_bounds'][0],
                r['best_poisson']['interval_bounds'][1],
                r['best_poisson']['n_observed'])

        return r['best_poisson']['guide_results']

    def extra_hash_dict(self):
        return dict(optimize_for_cl=self.optimize_for_cl,
                    pcl_sigma=self.pcl_sigma)


class GuidedLikelihoodBase(fastubl.UnbinnedLikelihoodExact):

    def statistic(self, r, mu_null):
        # NB: using -log(pmax)
        if 'guide_result' not in r:
            r['guide_result'] = self.guide(
                x_obs=r['x_obs'],
                present=r['present'],
                p_obs=r['p_obs'],
                dists=self.dists,
                bg_mus=self.true_mu[1:],
                domain=self.domain)

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
        self.optimize_for_cl = optimize_for_cl
        self.guide = fastubl.PoissonGuide(optimize_for_cl=optimize_for_cl)
        super().__init__(*args, **kwargs)

    def extra_hash_dict(self):
        return dict(optimize_for_cl=self.optimize_for_cl)


@export
class LikelihoodGuidedLikelihood(GuidedLikelihoodBase):
    def __init__(self,
                 *args,
                 mu_low=1., mu_high=40.,
                 **kwargs):
        self.mu_low, self.mu_high = mu_low, mu_high
        self.guide = fastubl.LikelihoodGuide(mu_low=mu_low, mu_high=mu_high)
        super().__init__(*args, **kwargs)

    def extra_hash_dict(self):
        return dict(mu_low=self.mu_low, mu_high=self.mu_high)

