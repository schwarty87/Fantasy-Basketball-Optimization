import os
from typing import Optional, Dict, Type
from torch.utils.data import DataLoader

from machine_learning.enums import DataSplitType
from machine_learning.lightning_data_module import MachineLearningDataModule
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning_fantasy_basketball_predictions.constants import PROJECT_NAME
from machine_learning_fantasy_basketball_predictions.params.params import (
    MachineLearningParamsFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import (
    MachineLearningDatasetCombinedFantasyBasketballPredictions,
)


class FantasyBasketballDataModule(MachineLearningDataModule):
    """
    Fantasy Basketball specific LightningDataModule using
    MachineLearningDatasetCombinedFantasyBasketballPredictions.
    """

    def __init__(
        self,
        params: Optional[MachineLearningParamsFantasyBasketballPredictions] = None,
        model_dir: Optional[str] = None,
        predict_data_split_type: DataSplitType = DataSplitType.TRAIN,
        data_loaders: Optional[Dict[DataSplitType, DataLoader]] = None,
        project_name: str = PROJECT_NAME,
    ):
        super().__init__(
            params=params,
            project_name=project_name,
            model_dir=model_dir,
            predict_data_split_type=predict_data_split_type,
            data_loaders=data_loaders,
        )

    @staticmethod
    def _machine_learning_datasets_combined_class() -> Type[MachineLearningDatasetCombined]:
        return MachineLearningDatasetCombinedFantasyBasketballPredictions