from dataclasses import dataclass
from machine_learning.params.machine_learning_dataset_params import MachineLearningDatasetSampleWeightsTabularParams, \
    MachineLearningDatasetTabularCategoricalEncodingParams
from machine_learning.registries import params_registry


@params_registry.register
@dataclass
class MachineLearningDatasetSampleWeightsTabularFantasyBasketballPredictionsParams(MachineLearningDatasetSampleWeightsTabularParams):
    pass


@params_registry.register
@dataclass
class MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams(MachineLearningDatasetTabularCategoricalEncodingParams):
    pass
