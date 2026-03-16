from typing import Optional, Tuple
from dataclasses import dataclass

from machine_learning.params.abstract_params import AbstractMetricParams, AbstractLossParams, \
    AbstractBinaryClassificationMetricParams
from machine_learning.registries import params_registry

@params_registry.register
@dataclass
class MAEMetricParams(AbstractMetricParams):
    pass


@params_registry.register
@dataclass
class MAPEMetricParams(AbstractMetricParams):
    pass


@params_registry.register
@dataclass
class BinaryCrossEntropyParams(AbstractLossParams):
    logits: bool = True

    # when the true regression data is between these values, we'll change the true binary label to match the predicted
    # label
    middle_values: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.middle_values is not None and len(self.middle_values) != 2:
            raise ValueError('middle_values has to be a list of length 2!')


@params_registry.register
@dataclass
class ROCAUCParams(AbstractBinaryClassificationMetricParams):
    accumulate: bool = True


@params_registry.register
@dataclass
class BinaryClassificationMetricsParams(AbstractBinaryClassificationMetricParams):
    pass


@params_registry.register
@dataclass
class MSELossParams(AbstractLossParams):
    pass


@params_registry.register
@dataclass
class QuantileRegressionLossParams(AbstractLossParams):
    quantile: float = 0.5
