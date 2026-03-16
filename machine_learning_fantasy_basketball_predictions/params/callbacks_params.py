from dataclasses import dataclass
from machine_learning.params.callbacks_params import CollectTruePredictedOutputsParams
from machine_learning.registries import params_registry


@params_registry.register
@dataclass
class CollectTruePredictedOutputsParamsFantasyBasketballPredictions(CollectTruePredictedOutputsParams):
    pass