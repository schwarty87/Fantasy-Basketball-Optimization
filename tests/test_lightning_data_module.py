import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from machine_learning.enums import DataSplitType
from machine_learning.params.layers_params import EmbeddingsLayerParams
from run_all import create_params
from machine_learning_fantasy_basketball_predictions.lightning_data_module import (
    FantasyBasketballDataModule,
)


class TestFantasyBasketballDataModule(unittest.TestCase):
    def setUp(self):
        self.params = create_params()

    def test_init(self):
        """Tests standard initialization of the data module."""
        dm = FantasyBasketballDataModule(params=self.params, model_dir="/tmp/test_model")
        self.assertEqual(dm.params, self.params)
        self.assertEqual(dm.model_dir, "/tmp/test_model")
        self.assertEqual(dm.predict_data_split_type, DataSplitType.TRAIN)
        self.assertIsNone(dm.dataset_combined)
        self.assertEqual(dm.data_loaders, {})

    def test_reusing_provided_data_loaders(self):
        """Tests that passing existing data_loaders avoids re-running dataset setup."""
        mock_train_loader = DataLoader(TensorDataset(torch.randn(10, 2)))
        mock_val_loader = DataLoader(TensorDataset(torch.randn(5, 2)))
        mock_test_loader = DataLoader(TensorDataset(torch.randn(3, 2)))

        data_loaders = {
            DataSplitType.TRAIN: mock_train_loader,
            DataSplitType.VALIDATION: mock_val_loader,
            DataSplitType.TEST: mock_test_loader,
        }

        dm = FantasyBasketballDataModule(
            params=self.params,
            data_loaders=data_loaders,
        )

        # Calling setup should be a no-op when data_loaders are pre-populated
        dm.setup(stage="fit")
        self.assertIsNone(dm.dataset_combined)

        self.assertIs(dm.train_dataloader(), mock_train_loader)
        self.assertIs(dm.val_dataloader(), mock_val_loader)
        self.assertIs(dm.test_dataloader(), mock_test_loader)
        self.assertIs(dm.predict_dataloader(), mock_train_loader)

    def test_predict_dataloader_split_selection(self):
        """Tests that predict_dataloader respects predict_data_split_type."""
        mock_train_loader = DataLoader(TensorDataset(torch.randn(10, 2)))
        mock_val_loader = DataLoader(TensorDataset(torch.randn(5, 2)))

        data_loaders = {
            DataSplitType.TRAIN: mock_train_loader,
            DataSplitType.VALIDATION: mock_val_loader,
        }

        dm = FantasyBasketballDataModule(
            params=self.params,
            data_loaders=data_loaders,
            predict_data_split_type=DataSplitType.VALIDATION,
        )

        self.assertIs(dm.predict_dataloader(), mock_val_loader)

        # Switch to train
        dm.predict_data_split_type = DataSplitType.TRAIN
        self.assertIs(dm.predict_dataloader(), mock_train_loader)

    def test_update_categorical_params(self):
        """Tests that update_categorical_params populates num_embeddings on EmbeddingsLayerParams."""
        # Find an EmbeddingsLayerParams in the architecture
        embedding_params = [
            p
            for p in self.params.architecture_params.layer_params_list
            if isinstance(p, EmbeddingsLayerParams)
        ]
        self.assertGreater(len(embedding_params), 0)

        # Reset num_embeddings to None to simulate un-initialized state
        for p in embedding_params:
            p.num_embeddings = None

        mock_train_loader = MagicMock(spec=DataLoader)
        dm = FantasyBasketballDataModule(
            params=self.params,
            data_loaders={DataSplitType.TRAIN: mock_train_loader},
        )

        # Mock get_single_dataset_from_data_loaders to return an object with num_unique_values_per_feature
        mock_cat_dataset = MagicMock()
        mock_cat_dataset.num_unique_values_per_feature.return_value = np.array([50, 60, 70])

        with patch(
            "machine_learning.machine_learning_datasets.MachineLearningDatasetCombined.get_single_dataset_from_data_loaders",
            return_value=mock_cat_dataset,
        ):
            dm.update_categorical_params()

        # Check that num_embeddings was populated on embedding layers
        for p in embedding_params:
            self.assertIsNotNone(p.num_embeddings)
            self.assertEqual(p.num_embeddings, [50, 60, 70])

    def test_save_transformers(self):
        """Tests that save_transformers delegates to MachineLearningDatasetCombined."""
        mock_train_loader = DataLoader(TensorDataset(torch.randn(10, 2)))
        dm = FantasyBasketballDataModule(
            params=self.params,
            data_loaders={DataSplitType.TRAIN: mock_train_loader},
        )

        with patch(
            "machine_learning.machine_learning_datasets.MachineLearningDatasetCombined.save_transformers"
        ) as mock_save:
            dm.save_transformers(save_dir="/tmp/dummy_save_dir")
            mock_save.assert_called_once_with(
                data_loader=mock_train_loader, save_dir="/tmp/dummy_save_dir"
            )

    @patch("os.path.isdir")
    @patch("machine_learning_fantasy_basketball_predictions.machine_learning_datasets.MachineLearningDatasetCombinedFantasyBasketballPredictions.load_transformers")
    @patch("machine_learning_fantasy_basketball_predictions.machine_learning_datasets.MachineLearningDatasetCombinedFantasyBasketballPredictions.__init__", return_value=None)
    @patch("machine_learning_fantasy_basketball_predictions.machine_learning_datasets.MachineLearningDatasetCombinedFantasyBasketballPredictions.create_data_loaders", return_value={})
    def test_setup_loads_transformers(self, mock_create_dl, mock_ds_init, mock_load_trans, mock_isdir):
        """Tests that setup attempts to load transformers if scalers dir exists."""
        mock_isdir.return_value = True
        mock_load_trans.return_value = ({"scaler": MagicMock()}, MagicMock())

        dm = FantasyBasketballDataModule(
            params=self.params,
            model_dir="/tmp/existing_model",
        )
        dm.update_categorical_params = MagicMock()
        dm.setup(stage="fit")

        mock_load_trans.assert_called_once_with("/tmp/existing_model")
        mock_ds_init.assert_called_once()


if __name__ == "__main__":
    unittest.main()
