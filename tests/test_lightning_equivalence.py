import os
import copy
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint as PLModelCheckpoint,
    EarlyStopping as PLEarlyStopping,
    LearningRateMonitor,
)

from machine_learning.enums import DataSplitType, ModelMode
from machine_learning.params.callbacks_params import (
    ModelCheckpointParams,
    EarlyStoppingParams,
    ReduceLearningRateOnPlateauParams,
)
from machine_learning.run_model import RunModel
from machine_learning_fantasy_basketball_predictions.run_model import (
    RunModelFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.lightning_model import (
    FantasyBasketballLightningModule,
)
from run_all import create_params


class TestLightningEquivalence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.params = create_params()
        # Initialize num_embeddings for embedding layers
        for p in self.params.architecture_params.layer_params_list:
            if hasattr(p, "embeddings_dims") and hasattr(p, "num_embeddings"):
                p.num_embeddings = [50] * len(p.embeddings_dims)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    def test_build_lightning_callbacks_honors_params(
        self, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that _build_lightning_callbacks translates dataclass params to Lightning callbacks."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 1

        custom_params = create_params()
        custom_params.training_params.callbacks_params = (
            ModelCheckpointParams(value_to_track="eval_mae", track_min_or_max=False),
            EarlyStoppingParams(value_to_track="eval_mae", patience=5, track_min_or_max=False),
        )

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=custom_params, project_name="test_proj", use_lightning=True)
            callbacks = runner._build_lightning_callbacks()

        ckpt_cb = next(c for c in callbacks if isinstance(c, PLModelCheckpoint))
        self.assertEqual(ckpt_cb.monitor, "val/mae")
        self.assertEqual(ckpt_cb.mode, "max")

        es_cb = next(c for c in callbacks if isinstance(c, PLEarlyStopping))
        self.assertEqual(es_cb.monitor, "val/mae")
        self.assertEqual(es_cb.patience, 5)
        self.assertEqual(es_cb.mode, "max")

    def test_configure_optimizers_with_reduce_lr_plateau(self):
        """Tests that configure_optimizers attaches ReduceLROnPlateau scheduler when specified."""
        custom_params = copy.deepcopy(self.params)
        custom_params.training_params.callbacks_params = (
            ReduceLearningRateOnPlateauParams(
                value_to_track="eval_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
        )

        module = FantasyBasketballLightningModule(params=custom_params)
        opt_config = module.configure_optimizers()

        self.assertIsInstance(opt_config, dict)
        self.assertIn("optimizer", opt_config)
        self.assertIn("lr_scheduler", opt_config)
        self.assertEqual(opt_config["lr_scheduler"]["monitor"], "val/loss")
        self.assertEqual(opt_config["lr_scheduler"]["scheduler"].factor, 0.5)
        self.assertEqual(opt_config["lr_scheduler"]["scheduler"].patience, 3)

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    def test_eval_lightning_returns_matching_metric_dict(
        self, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that _eval_lightning returns a un-prefixed metrics dictionary like legacy eval."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 2

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=self.params, project_name="test_proj", use_lightning=True)

        mock_trainer = MagicMock()
        runner.trainer = mock_trainer
        runner.model = MagicMock()
        runner.model._last_epoch_eval_results = {
            "loss": 0.35,
            "MeanAbsoluteError_yahoo_points": 1.25,
        }
        runner.data_loaders = {DataSplitType.VALIDATION: MagicMock()}

        metrics = runner.eval(data_split_type=DataSplitType.VALIDATION)
        mock_trainer.validate.assert_called_once()
        self.assertEqual(metrics["loss"], 0.35)
        self.assertEqual(metrics["MeanAbsoluteError_yahoo_points"], 1.25)

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    def test_predict_lightning_returns_dataframe_with_split_type(
        self, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that _predict_lightning outputs a DataFrame with split_type column."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 3

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=self.params, project_name="test_proj", use_lightning=True)

        mock_trainer = MagicMock()
        mock_trainer.predict.return_value = []
        runner.trainer = mock_trainer
        runner.model = MagicMock()
        runner.model.format_predictions.return_value = pd.DataFrame({"YAHOO_POINTS_pred": [25.0, 30.0]})
        runner.data_loaders = {DataSplitType.TRAIN: MagicMock()}

        pred_df = runner.predict(data_split_type=DataSplitType.TRAIN)
        self.assertIn("split_type", pred_df.columns)
        self.assertEqual(pred_df["split_type"].iloc[0], DataSplitType.TRAIN.name)
        self.assertEqual(len(pred_df), 2)

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning.run_model.SummaryWriter")
    def test_save_hyperparameters_metrics_to_tensorboard_handles_lists(
        self, mock_summary_writer_cls, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that _save_hyperparameters_metrics_to_tensorboard filters out list metrics safely."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 4
        mock_writer = MagicMock()
        mock_summary_writer_cls.return_value = mock_writer

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=self.params, project_name="test_proj", use_lightning=True)

        eval_results = pd.DataFrame(
            {
                "loss": [0.4, 0.5, 0.6],
                "portfolio_value_5050_all": [[1.0, 2.0], [0.5, 1.5], [0.0]],
            },
            index=[s.name for s in DataSplitType.all_not_none_split_types()],
        )

        runner._save_hyperparameters_metrics_to_tensorboard(eval_results)
        self.assertEqual(mock_writer.add_hparams.call_count, 3)
        for call in mock_writer.add_hparams.call_args_list:
            _, kwargs = call
            # Ensure the list metric was filtered out so tensorboard doesn't crash
            self.assertNotIn("portfolio_value_5050_all", kwargs["metric_dict"])
            self.assertIn("loss", kwargs["metric_dict"])
            self.assertIn("split_type", kwargs["metric_dict"])

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.SummaryWriter")
    def test_run_full_training_session_equivalence(
        self, mock_summary_writer_cls, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that run_full_training_session in Fantasy Basketball executes the full lifecycle identically."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 10
        mock_writer = MagicMock()
        mock_summary_writer_cls.return_value = mock_writer

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner_lightning = RunModelFantasyBasketballPredictions(
                params=self.params, project_name="test_proj", use_lightning=True
            )
            runner_legacy = RunModelFantasyBasketballPredictions(
                params=self.params, project_name="test_proj", use_lightning=False
            )

        dummy_eval = pd.DataFrame(
            {"loss": [0.2, 0.3, 0.4]},
            index=[s.name for s in DataSplitType.all_not_none_split_types()],
        )
        dummy_pred = pd.DataFrame(
            {"DATE": [1], "PLAYER_ID": [1], "split_type": ["TRAIN"]}
        )

        for runner in (runner_lightning, runner_legacy):
            runner.train = MagicMock()
            runner.eval_all_data = MagicMock(return_value=dummy_eval.copy())
            runner.predict_all_data = MagicMock(return_value=dummy_pred.copy())
            runner._add_extra_metrics_to_eval_results = MagicMock(side_effect=lambda eval_results, predict_results: eval_results)
            runner._save_eval_results_to_feather = MagicMock()
            runner._save_hyperparameters_metrics_to_tensorboard = MagicMock()

            eval_res, pred_res = runner.run_full_training_session()

            runner.train.assert_called_once()
            runner.eval_all_data.assert_called_once()
            runner.predict_all_data.assert_called_once()
            runner._add_extra_metrics_to_eval_results.assert_called_once()
            runner._save_eval_results_to_feather.assert_called_once()
            pd.testing.assert_frame_equal(eval_res, dummy_eval)

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning.run_model.pl.Trainer")
    def test_base_run_model_train_lightning_delegation(
        self, mock_trainer_cls, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that base RunModel executes Lightning training when use_lightning=True."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 11

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=self.params, project_name="test_proj", use_lightning=True)

        runner.model = MagicMock()
        runner.data_loaders = {DataSplitType.TRAIN: MagicMock(), DataSplitType.VALIDATION: MagicMock()}
        mock_trainer_instance = mock_trainer_cls.return_value
        runner.train()
        mock_trainer_instance.fit.assert_called_once()

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning.run_model.pl.Trainer")
    def test_lightning_trainer_extended_kwargs(
        self, mock_trainer_cls, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that limit_train_batches, gradient_clip_val, fast_dev_run, and SWA are forwarded to Trainer."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 12

        custom_params = copy.deepcopy(self.params)
        custom_params.training_params.num_batches_per_epoch = 15
        custom_params.training_params.gradient_clip_val = 1.0
        custom_params.training_params.fast_dev_run = True
        custom_params.training_params.use_swa = True

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=custom_params, project_name="test_proj", use_lightning=True)

        runner.model = MagicMock()
        runner.data_loaders = {DataSplitType.TRAIN: MagicMock(), DataSplitType.VALIDATION: MagicMock()}
        runner.train()

        _, trainer_kwargs = mock_trainer_cls.call_args
        self.assertEqual(trainer_kwargs["limit_train_batches"], 15)
        self.assertEqual(trainer_kwargs["gradient_clip_val"], 1.0)
        self.assertTrue(trainer_kwargs["fast_dev_run"])
        callbacks = trainer_kwargs["callbacks"]
        self.assertTrue(any("StochasticWeightAveraging" in c.__class__.__name__ for c in callbacks))

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning.run_model.pl.Trainer")
    def test_lightning_unified_tensorboard_logger_dir(
        self, mock_trainer_cls, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that TensorBoardLogger is configured to save directly inside model_dir/tensorboard_logs."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 13

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=self.params, project_name="test_proj", use_lightning=True)

        runner.model = MagicMock()
        runner.data_loaders = {DataSplitType.TRAIN: MagicMock(), DataSplitType.VALIDATION: MagicMock()}
        runner.train()

        _, trainer_kwargs = mock_trainer_cls.call_args
        logger = trainer_kwargs["logger"]
        self.assertEqual(logger.save_dir, runner._model_dir)
        self.assertEqual(logger.name, "tensorboard_logs")

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning.run_model.pl.Trainer")
    @patch("machine_learning.run_model.pl.seed_everything")
    def test_lightning_seed_everything_invoked(
        self, mock_seed_everything, mock_trainer_cls, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that pl.seed_everything is called with training_params.seed."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 14

        custom_params = copy.deepcopy(self.params)
        custom_params.training_params.seed = 12345

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=custom_params, project_name="test_proj", use_lightning=True)

        runner.model = MagicMock()
        runner.data_loaders = {DataSplitType.TRAIN: MagicMock(), DataSplitType.VALIDATION: MagicMock()}
        runner.train()

        mock_seed_everything.assert_called_once_with(12345, workers=True)

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    def test_lightning_log_hyperparams_called(
        self, mock_get_next_model_id, mock_models_dir
    ):
        """Tests that _save_hyperparameters_metrics_to_tensorboard calls logger.log_hyperparams."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 15

        with patch("machine_learning.run_model.RunModel.create_data_loaders"), \
             patch("machine_learning.run_model.RunModel.update_categorical_params"), \
             patch("machine_learning.run_model.RunModel._create_model"):
            runner = RunModel(params=self.params, project_name="test_proj", use_lightning=True)

        mock_logger = MagicMock()
        runner.trainer = MagicMock()
        runner.trainer.logger = mock_logger

        eval_df = pd.DataFrame(
            {"loss": [0.42, 0.55], "mae": [1.2, 1.5]},
            index=[DataSplitType.TRAIN.name, DataSplitType.VALIDATION.name],
        )

        runner._save_hyperparameters_metrics_to_tensorboard(eval_results=eval_df)
        mock_logger.log_hyperparams.assert_called_once()
        call_kwargs = mock_logger.log_hyperparams.call_args.kwargs
        self.assertIn("learning_rate", call_kwargs["params"])
        self.assertIn("validation/loss", call_kwargs["metrics"])

    def test_lightning_track_grad_norm_hook(self):
        """Tests that on_before_optimizer_step computes and logs gradient norms when enabled."""
        custom_params = copy.deepcopy(self.params)
        custom_params.training_params.track_grad_norm = 2

        module = FantasyBasketballLightningModule(params=custom_params)
        module.log_dict = MagicMock()

        # Create dummy weights and gradients
        for p in module.model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)

        module.on_before_optimizer_step(optimizer=MagicMock())
        module.log_dict.assert_called_once()
        logged_dict = module.log_dict.call_args[0][0]
        self.assertTrue(any("grad" in k for k in logged_dict.keys()))

    def test_lightning_sanity_check_guard(self):
        """Tests that on_validation_epoch_end returns early and resets accumulators during sanity checking."""
        module = FantasyBasketballLightningModule(params=self.params)
        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = True
        module._trainer = mock_trainer

        module.loss_function.reset = MagicMock()
        module.metrics.reset = MagicMock()

        module.on_validation_epoch_end()
        self.assertEqual(module._last_epoch_eval_results, {})
        module.loss_function.reset.assert_called_once()
        module.metrics.reset.assert_called_once()

    def test_independent_train_and_val_loss_tracking(self):
        """Tests that train and val loss functions accumulate independently without cross-resetting."""
        module = FantasyBasketballLightningModule(params=self.params)
        self.assertIsNotNone(module.loss_function)
        self.assertIsNotNone(module.val_loss_function)
        self.assertIsNot(module.loss_function, module.val_loss_function)

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = False
        module._trainer = mock_trainer

        # Ensure validation epoch end resets val_loss_function and not training loss_function
        module.loss_function.reset = MagicMock()
        module.val_loss_function.reset = MagicMock()
        module.val_loss_function.summarize = MagicMock()
        module.val_loss_function.get_logs = MagicMock(return_value={"loss": 1.5, "mse": 1.0})

        module.on_validation_epoch_end()
        module.val_loss_function.reset.assert_called_once()
        module.loss_function.reset.assert_not_called()

        # Ensure train epoch end resets training loss_function and not val_loss_function
        module.loss_function.summarize = MagicMock()
        module.loss_function.get_logs = MagicMock(return_value={"loss": 2.0, "mse": 1.2})
        module.on_train_epoch_end()
        module.loss_function.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()

