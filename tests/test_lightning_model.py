import os
import shutil
import tempfile
import datetime
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from sklearn.preprocessing import StandardScaler

from constants import Sites
from machine_learning.constants import METADATA
from machine_learning.params.machine_learning_dataset_params import (
    MachineLearningDatasetTabularScalingParams,
)
from run_all import create_params
from machine_learning_fantasy_basketball_predictions.lightning_model import (
    FantasyBasketballLightningModule,
    _convert_date_col_to_date,
)


class TestFantasyBasketballLightningModule(unittest.TestCase):
    def setUp(self):
        self.params = create_params()
        # Initialize num_embeddings for testing embedding layers without loading full dataset
        for p in self.params.architecture_params.layer_params_list:
            if hasattr(p, "embeddings_dims") and hasattr(p, "num_embeddings"):
                p.num_embeddings = [50] * len(p.embeddings_dims)

        self.module = FantasyBasketballLightningModule(params=self.params)

    def _create_synthetic_batch(self, num_games: int = 1) -> dict:
        """Creates a synthetic batch matching the 3D sample_per_game architecture."""
        batch = {}
        output_features = self.params.features_params.feature_name_to_machine_learning_dataset_params[
            "true_label"
        ].feature_names
        meta_features = self.params.features_params.feature_name_to_machine_learning_dataset_params[
            METADATA
        ].feature_names
        n_players_per_game = (5 + self.params.features_params.num_bench_players) * 2  # 22 players

        fp_map = self.params.features_params.feature_name_to_machine_learning_dataset_params
        tabular_dim = len(fp_map["tabular"].feature_names)
        lagged_dim = len(fp_map["lagged"].feature_names)
        pos_dim = len(fp_map["positional_encoding_features"].feature_names)
        emb_dim = len(fp_map["embedding_features"].feature_names)

        batch["tabular"] = torch.randn((num_games, n_players_per_game, tabular_dim), dtype=torch.float32)
        batch["lagged"] = torch.randn(
            (num_games, n_players_per_game, lagged_dim), dtype=torch.float32
        )
        batch["positional_encoding_features"] = torch.zeros(
            (num_games, n_players_per_game, pos_dim), dtype=torch.long
        )
        batch["embedding_features"] = torch.zeros(
            (num_games, n_players_per_game, emb_dim), dtype=torch.long
        )
        batch["true_label"] = torch.randn(
            (num_games, n_players_per_game, len(output_features)), dtype=torch.float32
        )
        batch["sample_weights"] = torch.ones(
            (num_games, n_players_per_game, len(output_features)), dtype=torch.float32
        )
        batch["keep_mask"] = torch.ones(
            (num_games, n_players_per_game, 1), dtype=torch.float32
        )
        batch["real_player_mask"] = torch.ones(
            (num_games, n_players_per_game, 1), dtype=torch.float32
        )

        # METADATA: ['DATE', 'split_type', 'PLAYER_ID', 'YAHOO_SALARY', 'YAHOO_POSITION', 'YAHOO_POINTS']
        meta = np.zeros((num_games, n_players_per_game, len(meta_features)), dtype=np.float32)
        meta[:, :, meta_features.index("DATE")] = 20000.0  # ~2024 in days from epoch
        
        # Give distinct player IDs, positions (1..5 = PG..C), and valid salaries (10..30)
        p_ids = np.arange(1, n_players_per_game + 1, dtype=np.float32)
        positions = np.array([1, 2, 3, 4, 5] * 4 + [1, 2], dtype=np.float32)  # 22 positions
        salaries = np.full((n_players_per_game,), 15.0, dtype=np.float32)

        for g in range(num_games):
            meta[g, :, meta_features.index("PLAYER_ID")] = p_ids + g * 100
            meta[g, :, meta_features.index("YAHOO_POSITION")] = positions
            meta[g, :, meta_features.index("YAHOO_SALARY")] = salaries
            meta[g, :, meta_features.index("YAHOO_POINTS")] = 25.0 + (p_ids % 10)
        
        batch[METADATA] = torch.tensor(meta, dtype=torch.float32)

        return batch

    def test_init_and_hyperparameters(self):
        """Tests that module initializes correctly and saves hyperparameters."""
        self.assertIsNotNone(self.module.model)
        self.assertIsNotNone(self.module.loss_function)
        self.assertIsNotNone(self.module.metrics)
        self.assertIn("features_params", self.module.hparams)
        self.assertIn("architecture_params", self.module.hparams)
        self.assertIn("training_params", self.module.hparams)

    def test_forward(self):
        """Tests that module forward pass executes without error."""
        batch = self._create_synthetic_batch(num_games=1)
        batch_inputs = {
            k: v
            for k, v in batch.items()
            if k in self.params.architecture_params.all_layer_input_tensor_names
        }
        outputs = self.module(batch_inputs)
        self.assertIn("outputs_pred", outputs)
        self.assertEqual(outputs["outputs_pred"].shape[0], 1)

    def test_training_step(self):
        """Tests training_step produces a valid scalar loss with gradients."""
        batch = self._create_synthetic_batch(num_games=1)
        self.module.log = MagicMock()

        loss = self.module.training_step(batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(torch.isfinite(loss))
        self.module.log.assert_called()

    def test_validation_step_and_epoch_end(self):
        """Tests validation_step returns a valid loss and epoch end logs loss components and metrics."""
        batch = self._create_synthetic_batch(num_games=1)
        self.module.log = MagicMock()

        val_loss = self.module.validation_step(batch, batch_idx=0)
        self.assertIsInstance(val_loss, torch.Tensor)

        # Test validation epoch end logs loss components and resets accumulators
        self.module.on_validation_epoch_end()
        self.module.log.assert_called()

    def test_predict_step(self):
        """Tests predict_step returns CPU tensors for key predictions and metadata."""
        batch = self._create_synthetic_batch(num_games=1)
        pred_dict = self.module.predict_step(batch, batch_idx=0)
        self.assertIn("outputs_pred", pred_dict)
        self.assertIn("true_label", pred_dict)
        self.assertIn(METADATA, pred_dict)
        self.assertFalse(pred_dict["outputs_pred"].requires_grad)

    def test_format_predictions_and_unscaling(self):
        """Tests that format_predictions unscales targets and formats into a DataFrame."""
        batch = self._create_synthetic_batch(num_games=1)
        pred_dict = self.module.predict_step(batch, batch_idx=0)

        # Mock target scaler
        mock_scaler = MagicMock(spec=StandardScaler)
        mock_scaler.inverse_transform.side_effect = lambda x: x * 10.0
        self.module._target_scaler = mock_scaler

        df = self.module.format_predictions([pred_dict])

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 22)
        self.assertIn("YAHOO_POINTS_pred", df.columns)
        self.assertIn("PLAYER_ID", df.columns)
        self.assertIn("DATE", df.columns)

        # Verify date conversion to datetime.date
        self.assertIsInstance(df["DATE"].iloc[0], datetime.date)
        mock_scaler.inverse_transform.assert_called()

    def test_convert_date_col_to_date_formats(self):
        """Tests _convert_date_col_to_date handles seconds, days, strings, and dates."""
        # 1. Epoch seconds
        s_sec = pd.Series([1750000000.0, 1750086400.0])
        res_sec = _convert_date_col_to_date(s_sec)
        self.assertIsInstance(res_sec.iloc[0], datetime.date)

        # 2. Epoch days
        s_days = pd.Series([20000.0, 20001.0])
        res_days = _convert_date_col_to_date(s_days)
        self.assertIsInstance(res_days.iloc[0], datetime.date)

        # 3. String dates
        s_str = pd.Series(["2026-09-15", "2026-09-16"])
        res_str = _convert_date_col_to_date(s_str)
        self.assertEqual(res_str.iloc[0], datetime.date(2026, 9, 15))

        # 4. Already datetime.date
        s_date = pd.Series([datetime.date(2026, 9, 15)])
        res_date = _convert_date_col_to_date(s_date)
        self.assertEqual(res_date.iloc[0], datetime.date(2026, 9, 15))

    def test_configure_optimizers(self):
        """Tests that configure_optimizers returns AdamW and ReduceLROnPlateau."""
        opt_dict = self.module.configure_optimizers()
        self.assertIn("optimizer", opt_dict)
        self.assertIn("lr_scheduler", opt_dict)
        self.assertIsInstance(opt_dict["optimizer"], torch.optim.AdamW)
        self.assertEqual(opt_dict["lr_scheduler"]["monitor"], "val/loss")

    def test_checkpoint_save_and_load(self):
        """Tests saving and loading module from a checkpoint restores parameters."""
        temp_dir = tempfile.mkdtemp()
        try:
            ckpt_path = os.path.join(temp_dir, "test_model.ckpt")

            trainer = pl.Trainer(
                default_root_dir=temp_dir,
                accelerator="cpu",
                max_epochs=1,
                logger=False,
                enable_checkpointing=True,
            )
            trainer.strategy.connect(self.module)
            trainer.save_checkpoint(ckpt_path)

            self.assertTrue(os.path.isfile(ckpt_path))

            # Reload module from checkpoint without explicitly passing params
            loaded_module = FantasyBasketballLightningModule.load_from_checkpoint(ckpt_path)
            self.assertIsNotNone(loaded_module.params)
            self.assertEqual(
                loaded_module.params.training_params.learning_rate,
                self.params.training_params.learning_rate,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_lightning_format_predictions_fails_fast_when_scaler_missing(self):
        """Verify FantasyBasketballLightningModule.format_predictions raises FileNotFoundError when scaler is missing."""
        mock_params = MagicMock()
        mock_params.features_params.feature_name_to_machine_learning_dataset_params = {
            "true_label": MachineLearningDatasetTabularScalingParams(feature_names=["YAHOO_POINTS"])
        }
        empty_scaler_dir = os.path.join(tempfile.mkdtemp(), "empty_scalers")
        os.makedirs(empty_scaler_dir, exist_ok=True)
        try:
            module = MagicMock(spec=FantasyBasketballLightningModule)
            module.params = mock_params
            module.scaler_dir = empty_scaler_dir
            module._target_scaler = None
            module._load_target_scaler = MagicMock(return_value=None)
            with self.assertRaises(FileNotFoundError) as cm:
                FantasyBasketballLightningModule.format_predictions(module, predict_batches=[])
            self.assertIn("Target scaler true_label.pkl not found", str(cm.exception))
        finally:
            shutil.rmtree(empty_scaler_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
