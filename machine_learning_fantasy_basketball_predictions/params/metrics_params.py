from typing import Optional, Tuple
from dataclasses import dataclass

from machine_learning.params.abstract_params import AbstractMetricParams, AbstractLossParams, \
    AbstractBinaryClassificationMetricParams
from machine_learning.registries import params_registry

@params_registry.register
@dataclass(kw_only=True)
class OptimizeTeamsLossParams(AbstractLossParams):
    name_for_logs = 'optimize_teams'


@params_registry.register
@dataclass
class PairwiseRankingLossParams(AbstractLossParams):
    margin: float = 0.0
    margin_is_pct_of_true_diffs: bool = True
    # If True, scales the gradient by the magnitude of the true difference.
    # (e.g., Missing a 50pt gap is penalized 50x more than a 1pt gap)
    use_true_diff_weights: bool = True

@params_registry.register
@dataclass
class BradleyTerryLossParams(AbstractLossParams):
    margin: float = 0.0
    margin_is_pct_of_true_diffs: bool = True
