import os
import logging
from typing import Optional, Dict, Type
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from machine_learning.enums import DataSplitType
from machine_learning.params.layers_params import EmbeddingsLayerParams
from machine_learning.params.params import MachineLearningParams
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined


class MachineLearningDataModule(pl.LightningDataModule):
    """
    Generic LightningDataModule encapsulating dataset loading, feature scaling,
    transformer saving/loading, and categorical embedding updates across data splits.
    """

    def __init__(
        self,
        params: MachineLearningParams,
        project_name: str,
        model_dir: Optional[str] = None,
        predict_data_split_type: DataSplitType = DataSplitType.TRAIN,
        data_loaders: Optional[Dict[DataSplitType, DataLoader]] = None,
    ):
        super().__init__()
        self.params = params
        self.project_name = project_name
        self.model_dir = model_dir
        self.predict_data_split_type = predict_data_split_type

        self.dataset_combined: Optional[MachineLearningDatasetCombined] = None
        self.data_loaders: Dict[DataSplitType, DataLoader] = data_loaders or {}

    @staticmethod
    def _machine_learning_datasets_combined_class() -> Type[MachineLearningDatasetCombined]:
        return MachineLearningDatasetCombined

    def setup(self, stage: Optional[str] = None):
        """Initialises datasets, transformers, and dataloaders if not already provided."""
        if self.data_loaders or self.dataset_combined is not None:
            return

        scalers: Optional[Dict[str, StandardScaler]] = None
        label_encoders: Optional[OrdinalEncoder] = None

        load_dir = None
        if self.params is not None and getattr(self.params, "continue_model_id", None) is not None and self.model_dir is not None:
            load_dir = os.path.join(os.path.dirname(self.model_dir), str(self.params.continue_model_id))
        elif self.model_dir is not None and os.path.isdir(os.path.join(self.model_dir, "scalers")):
            load_dir = self.model_dir

        dataset_cls = self._machine_learning_datasets_combined_class()
        if load_dir and os.path.isdir(load_dir):
            try:
                scalers, label_encoders = dataset_cls.load_transformers(load_dir)
                logging.info(f"Loaded transformers from {load_dir}")
            except Exception as e:
                logging.warning(f"Could not load transformers from {load_dir}: {e}")

        if self.params is not None:
            self.dataset_combined = dataset_cls(
                params=self.params.features_params,
                training_params=self.params.training_params,
                project_name=self.project_name,
                dataset_name_to_scalers=scalers,
                dataset_name_to_categorical_encoder=label_encoders,
            )

            batch_size = self.params.training_params.batch_size
            num_workers = self.params.training_params.dataloader_num_workers or 0
            self.data_loaders = self.dataset_combined.create_data_loaders(
                batch_size=batch_size, num_workers=num_workers
            )

            self.update_categorical_params()

    def update_categorical_params(self):
        """Discovers unique category counts and updates EmbeddingsLayerParams in the architecture."""
        if not self.data_loaders or self.params is None:
            return
        for p in self.params.architecture_params.layer_params_list:
            if isinstance(p, EmbeddingsLayerParams):
                try:
                    categorical_dataset = (
                        self._machine_learning_datasets_combined_class().get_single_dataset_from_data_loaders(
                            data_loaders=self.data_loaders,
                            data_split_type=DataSplitType.TRAIN,
                            feature_type=p.input_tensor_name,
                        )
                    )
                    num_unique = categorical_dataset.num_unique_values_per_feature()
                    update_dict = {"num_embeddings": num_unique.astype(int).tolist()}
                    p.update_after_loading_data(update_dict)
                except Exception as e:
                    logging.warning(f"Could not update categorical params for {p.input_tensor_name}: {e}")

    def save_transformers(self, save_dir: str):
        """Saves scalers and categorical encoders to the target directory."""
        if self.data_loaders:
            loader = self.data_loaders.get(DataSplitType.TRAIN, self.data_loaders)
            self._machine_learning_datasets_combined_class().save_transformers(
                data_loader=loader, save_dir=save_dir
            )

    def train_dataloader(self) -> DataLoader:
        if not self.data_loaders:
            self.setup(stage="fit")
        return self.data_loaders[DataSplitType.TRAIN]

    def val_dataloader(self) -> DataLoader:
        if not self.data_loaders:
            self.setup(stage="fit")
        return self.data_loaders[DataSplitType.VALIDATION]

    def test_dataloader(self) -> DataLoader:
        if not self.data_loaders:
            self.setup(stage="test")
        return self.data_loaders[DataSplitType.TEST]

    def predict_dataloader(self) -> DataLoader:
        if not self.data_loaders:
            self.setup(stage="predict")
        split_type = self.predict_data_split_type
        return self.data_loaders.get(split_type, self.data_loaders[DataSplitType.TRAIN])
