import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from run_all import main, clean_empty_model_dirs
from machine_learning_fantasy_basketball_predictions.run_model import (
    RunModelFantasyBasketballPredictions,
)


class TestRunAll(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("run_all.clean_empty_model_dirs")
    @patch("run_all.define_logger")
    @patch("run_all.create_params")
    @patch("run_all.test_params")
    @patch("run_all.RunModelFantasyBasketballPredictions")
    @patch("run_all.models_dir")
    def test_main_runs_lightning_by_default(
        self,
        mock_models_dir,
        mock_run_model_cls,
        mock_test_params,
        mock_create_params,
        mock_define_logger,
        mock_clean_empty,
    ):
        """Tests that run_all.main() defaults to use_lightning=False and runs the full pipeline."""
        mock_models_dir.return_value = self.temp_dir
        mock_params = MagicMock()
        mock_create_params.return_value = mock_params

        mock_run_model_instance = MagicMock()
        mock_run_model_instance.model_id = 101
        mock_eval = pd.DataFrame({"col": [1]})
        mock_pred = MagicMock()
        mock_run_model_instance.run_full_training_session.return_value = (mock_eval, mock_pred)
        mock_run_model_cls.return_value = mock_run_model_instance

        eval_res, pred_res = main()

        mock_clean_empty.assert_called_once()
        mock_define_logger.assert_called_once()
        mock_create_params.assert_called_once()
        mock_test_params.assert_called_once_with(mock_params)
        mock_run_model_cls.assert_called_once_with(
            params=mock_params, description=None, use_lightning=False
        )
        mock_run_model_instance.run_full_training_session.assert_called_once()
        expected_feather = os.path.join(self.temp_dir, "101", "predict_results.feather")
        mock_pred.to_feather.assert_called_once_with(expected_feather)
        pd.testing.assert_frame_equal(eval_res, mock_eval)
        self.assertEqual(pred_res, mock_pred)

    @patch("run_all.clean_empty_model_dirs")
    @patch("run_all.define_logger")
    @patch("run_all.create_params")
    @patch("run_all.test_params")
    @patch("run_all.RunModelFantasyBasketballPredictions")
    @patch("run_all.models_dir")
    def test_main_runs_lightning_when_flag_true(
        self,
        mock_models_dir,
        mock_run_model_cls,
        mock_test_params,
        mock_create_params,
        mock_define_logger,
        mock_clean_empty,
    ):
        """Tests that run_all.main(use_lightning=True) passes use_lightning=True to runner."""
        mock_models_dir.return_value = self.temp_dir
        mock_params = MagicMock()
        mock_create_params.return_value = mock_params

        mock_run_model_instance = MagicMock()
        mock_run_model_instance.model_id = 102
        mock_eval = pd.DataFrame()
        mock_pred = MagicMock()
        mock_run_model_instance.run_full_training_session.return_value = (mock_eval, mock_pred)
        mock_run_model_cls.return_value = mock_run_model_instance

        main(use_lightning=True)

        mock_run_model_cls.assert_called_once_with(
            params=mock_params, description=None, use_lightning=True
        )

    def test_clean_empty_model_dirs(self):
        """Tests that clean_empty_model_dirs removes empty folders but keeps folders with models."""
        empty_dir = os.path.join(self.temp_dir, "1")
        pt_dir = os.path.join(self.temp_dir, "2")
        ckpt_dir = os.path.join(self.temp_dir, "3")

        os.makedirs(empty_dir, exist_ok=True)
        os.makedirs(pt_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        with open(os.path.join(pt_dir, "model.pt"), "w") as f:
            f.write("dummy")
        with open(os.path.join(ckpt_dir, "last.ckpt"), "w") as f:
            f.write("dummy")

        with patch("run_all.models_dir", return_value=self.temp_dir):
            clean_empty_model_dirs("test_proj")

        self.assertFalse(os.path.exists(empty_dir))
        self.assertTrue(os.path.exists(pt_dir))
        self.assertTrue(os.path.exists(ckpt_dir))

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.pl.Trainer")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.torch.save")
    def test_run_model_lightning_train_lifecycle(
        self,
        mock_torch_save,
        mock_trainer_cls,
        mock_module_cls,
        mock_datamodule_cls,
        mock_get_next_model_id,
        mock_models_dir,
    ):
        """Tests the internal Lightning train lifecycle inside RunModelFantasyBasketballPredictions."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 55

        mock_params = MagicMock()
        mock_params.continue_model_id = None
        mock_params.training_params.num_epochs = 12
        mock_params.training_params.gradient_accumulation_num_batches = 3

        mock_dm = MagicMock()
        mock_datamodule_cls.return_value = mock_dm

        mock_lm = MagicMock()
        mock_module_cls.return_value = mock_lm

        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)
        expected_model_dir = os.path.join(self.temp_dir, "55")
        self.assertEqual(runner._model_dir, expected_model_dir)

        runner.train()

        # DataModule lifecycle
        mock_dm.setup.assert_called_once_with(stage="fit")
        mock_dm.save_transformers.assert_called_once_with(save_dir=expected_model_dir)
        mock_params.save_to_json.assert_called_once_with(os.path.join(expected_model_dir, "params.json"))

        # Trainer args
        mock_trainer_cls.assert_called_once()
        _, trainer_kwargs = mock_trainer_cls.call_args
        self.assertEqual(trainer_kwargs["max_epochs"], 12)
        self.assertEqual(trainer_kwargs["accumulate_grad_batches"], 3)
        self.assertEqual(trainer_kwargs["devices"], 1)

        # Callbacks
        callbacks = trainer_kwargs["callbacks"]
        self.assertTrue(any(isinstance(c, ModelCheckpoint) for c in callbacks))
        self.assertTrue(any(isinstance(c, EarlyStopping) for c in callbacks))
        self.assertTrue(any(isinstance(c, LearningRateMonitor) for c in callbacks))

        # Logger
        self.assertIsInstance(trainer_kwargs["logger"], TensorBoardLogger)

        # Fit execution & legacy save
        mock_trainer.fit.assert_called_once_with(model=mock_lm, datamodule=mock_dm)
        mock_torch_save.assert_called_once_with(mock_lm.model, os.path.join(expected_model_dir, "model.pt"))

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.pl.Trainer")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.torch.cuda.is_available")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.torch.save")
    def test_precision_cuda_vs_cpu(
        self,
        mock_torch_save,
        mock_cuda_avail,
        mock_trainer_cls,
        mock_module_cls,
        mock_datamodule_cls,
        mock_get_next_model_id,
        mock_models_dir,
    ):
        """Tests that precision selects '16-mixed' on CUDA and '32-true' on CPU."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 1
        mock_params = MagicMock()
        mock_params.continue_model_id = None
        mock_params.training_params.num_epochs = 5
        mock_params.training_params.gradient_accumulation_num_batches = 1

        runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)

        # 1. CUDA
        mock_cuda_avail.return_value = True
        runner.train()
        _, kwargs_cuda = mock_trainer_cls.call_args
        self.assertEqual(kwargs_cuda["precision"], "16-mixed")

        # 2. CPU
        mock_cuda_avail.return_value = False
        runner.train()
        _, kwargs_cpu = mock_trainer_cls.call_args
        self.assertEqual(kwargs_cpu["precision"], "32-true")

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.pl.Trainer")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.torch.save")
    def test_legacy_save_exception_handled_gracefully(
        self,
        mock_torch_save,
        mock_trainer_cls,
        mock_module_cls,
        mock_datamodule_cls,
        mock_get_next_model_id,
        mock_models_dir,
    ):
        """Tests that failures saving legacy model.pt do not raise unhandled exceptions."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 99
        mock_params = MagicMock()
        mock_params.continue_model_id = None
        mock_params.training_params.num_epochs = 1
        mock_params.training_params.gradient_accumulation_num_batches = 1

        mock_torch_save.side_effect = RuntimeError("Disk full")

        runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)
        try:
            runner.train()
        except Exception as e:
            self.fail(f"train() raised an unexpected exception when legacy save failed: {e}")

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.pl.Trainer")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.torch.save")
    def test_run_model_resumes_from_checkpoint_when_continue_model_id_is_set(
        self,
        mock_torch_save,
        mock_trainer_cls,
        mock_module_cls,
        mock_datamodule_cls,
        mock_get_next_model_id,
        mock_models_dir,
    ):
        """Tests that specifying continue_model_id passes ckpt_path to trainer.fit()."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 88

        # Create dummy continue model dir with last.ckpt
        continue_dir = os.path.join(self.temp_dir, "prior_model")
        os.makedirs(continue_dir, exist_ok=True)
        ckpt_path = os.path.join(continue_dir, "last.ckpt")
        with open(ckpt_path, "w") as f:
            f.write("dummy_checkpoint")

        mock_params = MagicMock()
        mock_params.continue_model_id = "prior_model"
        mock_params.training_params.num_epochs = 10
        mock_params.training_params.gradient_accumulation_num_batches = 1

        mock_dm = MagicMock()
        mock_datamodule_cls.return_value = mock_dm
        mock_lm = MagicMock()
        mock_module_cls.return_value = mock_lm
        mock_trainer = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        with patch("torch.load", return_value={"state_dict": {}}):
            runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)
            runner.train()

        mock_trainer.fit.assert_called_once_with(
            model=mock_lm, datamodule=mock_dm, ckpt_path=ckpt_path
        )


if __name__ == "__main__":
    unittest.main()
