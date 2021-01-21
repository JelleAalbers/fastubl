import numpy as np
import fastubl

export, __all__ = fastubl.exporter()


@export
class CombinedProcedure(fastubl.NeymanConstruction):

    procedures = tuple()

    def __init__(self, *args, **kwargs):
        self.procedures = [p_class(*args, **kwargs)
                           for p_class in self.procedures]
        super().__init__(*args, **kwargs)

    def statistic(self, r, mu_null):
        return np.min(
            [p.t_cdf(p.statistic(r, mu_null), mu_null)
             for p in self.procedures],
            axis=0)


@export
class OIWithLLR(CombinedProcedure):
    procedures = (fastubl.OptItv,
                  fastubl.UnbinnedLikelihoodExact)


@export
class OIYWithLLR(CombinedProcedure):
    procedures = (fastubl.OptItvYellin,
                  fastubl.UnbinnedLikelihoodExact)


@export
class PMaxWithLLR(CombinedProcedure):
    procedures = (fastubl.PMax,
                  fastubl.UnbinnedLikelihoodExact)


@export
class PMaxYWithLLR(CombinedProcedure):
    procedures = (fastubl.PMaxYellin,
                  fastubl.UnbinnedLikelihoodExact)

@export
class PMaxOILLR(CombinedProcedure):
    procedures = (fastubl.PMax,
                  fastubl.OptItv,
                  fastubl.UnbinnedLikelihoodExact)

@export
class BestZechLLR(CombinedProcedure):
    procedures = (fastubl.BestZech,
                  fastubl.UnbinnedLikelihoodExact)
