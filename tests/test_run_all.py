import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from run_all import (
    main,
    clean_empty_model_dirs,
    compare_dataclasses,
    create_architecture_params,
    create_features_params,
    create_params,
    test_params,
)
from machine_learning.architecture import ModelArchitecture
from machine_learning.constants import METADATA
from machine_learning.metrics.combined_metrics import LossesCombined
from machine_learning.params.metrics_params import QuantileRegressionLossParams
from machine_learning.params.callbacks_params import (
    ModelCheckpointParams,
    EarlyStoppingParams,
    ReduceLearningRateOnPlateauParams,
)
from machine_learning.params.layers_params import (
    ReshapeLayerParams,
    SplitAttentionCombineWrapperParams,
    TransformerEncoderPytorchParams,
)
from machine_learning_fantasy_basketball_predictions.lightning_model import (
    FantasyBasketballLightningModule,
)
from machine_learning_fantasy_basketball_predictions.params.layers_params import (
    CalculateFantasyPointsParams,
    NonCrossingQuantilesParams,
    RateScaleHeadParams,
    TeamAllocationParams,
    update_calculate_fantasy_points_params,
)
from machine_learning_fantasy_basketball_predictions.params.params import (
    MachineLearningParamsFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.run_model import (
    RunModelFantasyBasketballPredictions,
)
from dataclasses import dataclass


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

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule")
    def test_add_extra_metrics_handles_missing_contests_file(
        self,
        mock_module_cls,
        mock_datamodule_cls,
        mock_get_next_model_id,
        mock_models_dir,
    ):
        """Tests that missing contest file does not raise an exception and returns eval_results."""
        mock_models_dir.return_value = self.temp_dir
        mock_get_next_model_id.return_value = 99
        mock_params = MagicMock()
        mock_params.continue_model_id = None

        runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)
        eval_df = pd.DataFrame({"loss": [0.5]})
        pred_df = pd.DataFrame({"pred": [1.0]})

        with patch.object(runner, "_load_contests_df", side_effect=FileNotFoundError("not found")):
            res = runner._add_extra_metrics_to_eval_results(eval_df, pred_df)
            pd.testing.assert_frame_equal(res, eval_df)

    def test_compare_dataclasses_matching_and_mismatch(self):
        """Tests that compare_dataclasses verifies identical structures and detects mismatches."""
        @dataclass
        class SubConfig:
            lr: float
            name: str

        @dataclass
        class Config:
            sub: SubConfig
            active: bool

        cfg1 = Config(sub=SubConfig(lr=0.01, name="model"), active=True)
        cfg2 = Config(sub=SubConfig(lr=0.01, name="model"), active=True)
        cfg_mismatch = Config(sub=SubConfig(lr=0.05, name="model"), active=True)

        # Matching: should not raise
        compare_dataclasses(cfg1, cfg2)

        # Mismatch: should raise AssertionError
        with self.assertRaises(AssertionError):
            compare_dataclasses(cfg1, cfg_mismatch)

    @patch("run_all.clean_empty_model_dirs")
    @patch("run_all.define_logger")
    @patch("run_all.test_params")
    @patch("run_all.RunModelFantasyBasketballPredictions")
    @patch("run_all.models_dir")
    def test_main_fast_dev_run_flag_propagates(
        self,
        mock_models_dir,
        mock_run_model_cls,
        mock_test_params,
        mock_define_logger,
        mock_clean_empty,
    ):
        """Tests that main(params=params) preserves fast_dev_run when set."""
        mock_models_dir.return_value = self.temp_dir
        params = MagicMock()
        params.training_params.fast_dev_run = True

        mock_runner = MagicMock()
        mock_runner.model_id = 105
        mock_runner.run_full_training_session.return_value = (pd.DataFrame(), MagicMock())
        mock_run_model_cls.return_value = mock_runner

        main(use_lightning=True, params=params)
        self.assertTrue(params.training_params.fast_dev_run)
        mock_run_model_cls.assert_called_once_with(
            params=params, description=None, use_lightning=True
        )

    @patch("machine_learning.run_model.models_dir")
    @patch("machine_learning.run_model.get_next_model_id")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.pl.Trainer")
    @patch("machine_learning_fantasy_basketball_predictions.run_model.torch.save")
    def test_model_id_stable_across_train_and_eval_save(
        self,
        mock_torch_save,
        mock_trainer_cls,
        mock_module_cls,
        mock_datamodule_cls,
        mock_get_next_model_id,
        mock_models_dir,
    ):
        """model_id must not be re-derived in train(); eval results go into the same dir as the checkpoints."""
        mock_models_dir.return_value = self.temp_dir
        # Simulates the real behavior: once dir 7 exists, the next lookup would return 8
        mock_get_next_model_id.side_effect = [7, 8, 9]
        mock_params = MagicMock()
        mock_params.continue_model_id = None
        mock_params.training_params.num_epochs = 1
        mock_params.training_params.gradient_accumulation_num_batches = 1

        runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)
        runner.train()

        expected_dir = os.path.join(self.temp_dir, "7")
        self.assertEqual(runner.model_id, 7)
        self.assertEqual(runner._model_dir, expected_dir)
        self.assertEqual(mock_get_next_model_id.call_count, 1)

        runner._save_eval_results_to_feather(pd.DataFrame({"loss": [0.5]}, index=["validation"]))
        self.assertTrue(os.path.isfile(os.path.join(expected_dir, "eval_results.arrow")))

    def test_composable_feature_and_bench_flags_and_bench_permutation_equivariance(self):
        """Verify composable flags (--num-bench-players, --extra-tabular-features, --positional-encoding-feature-names, --embedding-feature-names) and bench equivariance."""
        # 1. Default params stay completely unchanged
        default_params = create_params()
        self.assertEqual(default_params.features_params.num_bench_players, 6)
        default_ds = default_params.features_params.feature_name_to_machine_learning_dataset_params
        self.assertNotIn('STARTER', default_ds['tabular'].feature_names)
        self.assertNotIn('SITE_POSITION', default_ds['tabular'].feature_names)
        self.assertEqual(
            default_ds['positional_encoding_features'].feature_names,
            ['HOME', 'START_POSITION_CALCULATED'],
        )

        # 2. Composable flags configure 30-player order-free grid cleanly
        custom_params = create_params(
            num_bench_players=10,
            extra_tabular_features=['STARTER', 'SITE_POSITION'],
            positional_encoding_feature_names=['HOME', 'SITE_POSITION_ROUNDED'],
            embedding_feature_names=['PLAYER_ID', 'TEAM_ID'],
        )
        self.assertEqual(custom_params.features_params.num_bench_players, 10)
        ds_map = custom_params.features_params.feature_name_to_machine_learning_dataset_params
        for ds in ds_map.values():
            self.assertNotIn('START_POSITION_CALCULATED', ds.feature_names)
            self.assertNotIn('EXPECTED_ORDER', ds.feature_names)
        self.assertIn('START_POSITION_CALCULATED', custom_params.features_params.all_feature_names_list)
        self.assertIn('SITE_POSITION', custom_params.features_params.all_feature_names_list)
        self.assertIn('STARTER', ds_map['tabular'].feature_names)
        self.assertIn('SITE_POSITION', ds_map['tabular'].feature_names)
        # Sorted alphabetically in __post_init__: ['HOME', 'SITE_POSITION_ROUNDED'] -> [1, 5]
        self.assertEqual(ds_map['positional_encoding_features'].feature_names, ['HOME', 'SITE_POSITION_ROUNDED'])
        pos_layer = next(
            lp for lp in custom_params.architecture_params.layer_params_list
            if getattr(lp, 'input_tensor_name', None) == 'positional_encoding_features'
        )
        self.assertEqual(pos_layer.embeddings_dims, [1, 5])
        test_params(custom_params)

        # 3. Also verify ['COURT_POSITION', 'HOME'] -> [5, 1] and --no-bench-order preset round-trip
        no_bench_order_params = create_params(bench_order_input=False)
        test_params(no_bench_order_params)
        pos_layer_court = next(
            lp for lp in no_bench_order_params.architecture_params.layer_params_list
            if getattr(lp, 'input_tensor_name', None) == 'positional_encoding_features'
        )
        self.assertEqual(pos_layer_court.embeddings_dims, [5, 1])

        # 4. Bench-slot permutation equivariance (swapping bench slots 5 and 12 produces identical swapped outputs)
        no_bench_order_params.features_params.feature_type_to_input_shape = {
            'tabular': (30, len(no_bench_order_params.features_params.feature_name_to_machine_learning_dataset_params['tabular'].feature_names)),
            'lagged': (30, len(no_bench_order_params.features_params.feature_name_to_machine_learning_dataset_params['lagged'].feature_names)),
            'embedding_features': (30, len(no_bench_order_params.features_params.feature_name_to_machine_learning_dataset_params['embedding_features'].feature_names)),
            'positional_encoding_features': (30, 2),
            'real_player_mask': (30, 1),
            'keep_mask': (30, 1),
            METADATA: (30, 6),
        }
        for lp in no_bench_order_params.architecture_params.layer_params_list:
            if getattr(lp, 'input_tensor_name', None) == 'embedding_features':
                lp.update_after_loading_data({'num_embeddings': [20] * len(lp.embeddings_dims)})
            elif getattr(lp, 'input_tensor_name', None) == 'positional_encoding_features':
                lp.update_after_loading_data({'num_embeddings': [6, 3]})

        model = ModelArchitecture(
            params=no_bench_order_params.architecture_params,
            feature_type_to_input_shape=no_bench_order_params.features_params.feature_type_to_input_shape,
        ).eval()

        torch.manual_seed(123)
        batch = {
            'tabular': torch.randn(2, 30, 26),
            'lagged': torch.randn(2, 30, 36),
            'embedding_features': torch.randint(0, 10, (2, 30, 6)),
            'positional_encoding_features': torch.randint(0, 2, (2, 30, 2)),
            'real_player_mask': torch.ones(2, 30, 1),
            'keep_mask': torch.ones(2, 30, 1),
            METADATA: torch.ones(2, 30, 6),
        }
        perm = list(range(30))
        perm[5], perm[12] = perm[12], perm[5]
        batch_perm = {k: v[:, perm, :].clone() for k, v in batch.items()}

        with torch.no_grad():
            out1 = model(batch)['outputs_pred']
            out2 = model(batch_perm)['outputs_pred']
        torch.testing.assert_close(out1[:, perm, :], out2, atol=1e-5, rtol=1e-5)

    def test_two_stage_params_and_forward_backward(self):
        """Verify two-stage config layouts, metric tracking, param parity, and forward/backward pass."""
        baseline = create_params(two_stage=False)
        for cb in baseline.training_params.callbacks_params:
            if isinstance(cb, (ModelCheckpointParams, EarlyStoppingParams, ReduceLearningRateOnPlateauParams)):
                self.assertEqual(cb.value_to_track, 'eval_loss')

        self.assertEqual(
            sum(lp.num_layers for lp in baseline.architecture_params.layer_params_list if isinstance(lp, SplitAttentionCombineWrapperParams)),
            1,
        )
        self.assertEqual(
            sum(lp.num_layers for lp in baseline.architecture_params.layer_params_list if isinstance(lp, TransformerEncoderPytorchParams)),
            2,
        )

        with self.assertRaises(ValueError):
            create_params(two_stage=True, stage1_layout='invalid_layout')

        for layout, kl_weight in [('split1_global1_global1', 10.0), ('split2_global1', 5.0)]:
            params = create_params(two_stage=True, stage1_layout=layout, stage1_loss_weight=kl_weight)
            test_params(params)

            for cb in params.training_params.callbacks_params:
                if isinstance(cb, (ModelCheckpointParams, EarlyStoppingParams, ReduceLearningRateOnPlateauParams)):
                    self.assertEqual(cb.value_to_track, 'eval_mse')

            expected_split = 1 if layout == 'split1_global1_global1' else 2
            expected_global = 2 if layout == 'split1_global1_global1' else 1
            self.assertEqual(
                sum(lp.num_layers for lp in params.architecture_params.layer_params_list if isinstance(lp, SplitAttentionCombineWrapperParams)),
                expected_split,
            )
            self.assertEqual(
                sum(lp.num_layers for lp in params.architecture_params.layer_params_list if isinstance(lp, TransformerEncoderPytorchParams)),
                expected_global,
            )

            for lp in params.architecture_params.layer_params_list:
                if hasattr(lp, 'embeddings_dims') and hasattr(lp, 'num_embeddings'):
                    lp.update_after_loading_data({'num_embeddings': [50] * len(lp.embeddings_dims)})

            model = ModelArchitecture(
                params=params.architecture_params,
                feature_type_to_input_shape=params.features_params.feature_type_to_input_shape,
            ).train()
            losses = LossesCombined(params.training_params.loss_params, device=torch.device('cpu'), name='loss')

            torch.manual_seed(42)
            batch = {}
            for name, shape in params.features_params.feature_type_to_input_shape.items():
                if name in ('embedding_features', 'positional_encoding_features'):
                    batch[name] = torch.randint(0, 5, (2, *shape))
                elif name in ('real_player_mask', 'keep_mask', 'stage1_keep_mask') or 'weight' in name:
                    batch[name] = torch.ones((2, *shape), dtype=torch.float32)
                elif name == 'stage1_label':
                    batch[name] = torch.full((2, *shape), 1.0 / 11.0, dtype=torch.float32)
                else:
                    batch[name] = torch.randn((2, *shape), dtype=torch.float32)

            outputs = model(batch)
            self.assertIn('outputs_pred', outputs)
            self.assertIn('stage1_log_share', outputs)
            self.assertIn('stage1_alloc', outputs)
            self.assertEqual(outputs['stage1_log_share'].shape, (2, 22, 2))
            self.assertEqual(outputs['stage1_alloc'].shape, (2, 22, 2))

            all_tensors = {**batch, **outputs}
            total_loss = losses.forward(all_tensors)
            self.assertTrue(torch.isfinite(total_loss).all())
            total_loss.backward()

            grad_norms = [p.grad.abs().sum().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
            self.assertGreater(len(grad_norms), 0)
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())

    def test_multiplicative_stage2_head_validation_params_and_ckpt_parity(self):
        """Verify --stage2-head multiplicative requires --two-stage, passes round-trips, and matches across reloads."""
        feat_params = create_features_params(two_stage=False)
        with self.assertRaisesRegex(ValueError, "requires two_stage=True"):
            create_architecture_params(feat_params, num_days_per_batch=4, two_stage=False, stage2_head='multiplicative')
        with self.assertRaisesRegex(ValueError, "requires two_stage=True"):
            create_params(two_stage=False, stage2_head='multiplicative')
        with self.assertRaisesRegex(ValueError, "requires two_stage=True"):
            create_params(two_stage=False, detach_z_min=True)

        detached_params = create_params(two_stage=True, detach_z_min=True)
        ta_det = next(lp for lp in detached_params.architecture_params.layer_params_list if isinstance(lp, TeamAllocationParams))
        self.assertTrue(ta_det.detach_z_min)
        test_params(detached_params)

        params = create_params(two_stage=True, stage2_head='multiplicative')
        ta_default = next(lp for lp in params.architecture_params.layer_params_list if isinstance(lp, TeamAllocationParams))
        self.assertFalse(ta_default.detach_z_min)
        rsh_list = [lp for lp in params.architecture_params.layer_params_list if isinstance(lp, RateScaleHeadParams)]
        self.assertEqual(len(rsh_list), 1)
        self.assertEqual(rsh_list[0].input_tensor_name, {'raw': 'stage2_raw_rates', 'alloc': 'stage1_alloc'})
        self.assertEqual(rsh_list[0].output_tensor_name, 'outputs_pred')

        for lp in params.architecture_params.layer_params_list:
            if hasattr(lp, 'embeddings_dims') and hasattr(lp, 'num_embeddings'):
                lp.update_after_loading_data({'num_embeddings': [50] * len(lp.embeddings_dims)})

        stat_names = params.features_params.feature_name_to_machine_learning_dataset_params['true_label'].feature_names
        class DummyScaler:
            feature_names_in_ = np.array(stat_names)
            mean_ = np.arange(1.0, len(stat_names) + 1.0, dtype=np.float64)
            scale_ = np.linspace(0.5, 5.5, len(stat_names), dtype=np.float64)

            def inverse_transform(self, x):
                return np.asarray(x, dtype=np.float64) * self.scale_ + self.mean_

        combined = type("Combined", (), {
            "machine_learning_datasets_stacked": type("Stacked", (), {
                "datasets": {"true_label": type("DS", (), {"scaler": DummyScaler()})()}
            })()
        })()
        update_calculate_fantasy_points_params(params=params, dataset_combined=combined)
        self.assertIsNotNone(rsh_list[0].scaler_mean)
        cfp = next(lp for lp in params.architecture_params.layer_params_list if isinstance(lp, CalculateFantasyPointsParams))
        self.assertEqual(rsh_list[0].scaler_mean, cfp.scaler_mean)
        self.assertEqual(rsh_list[0].scaler_scale, cfp.scaler_scale)

        test_params(params)

        module = FantasyBasketballLightningModule(params=params)
        module.eval()

        torch.manual_seed(7)
        batch = {}
        for name, shape in params.features_params.feature_type_to_input_shape.items():
            if name in ('embedding_features', 'positional_encoding_features'):
                batch[name] = torch.randint(0, 5, (2, *shape))
            elif name in ('real_player_mask', 'keep_mask', 'stage1_keep_mask') or 'weight' in name:
                batch[name] = torch.ones((2, *shape), dtype=torch.float32)
            elif name == 'stage1_label':
                batch[name] = torch.full((2, *shape), 1.0 / 11.0, dtype=torch.float32)
            else:
                batch[name] = torch.randn((2, *shape), dtype=torch.float32)

        with torch.no_grad():
            orig_out = module.model(batch)

        # Verify Lightning .ckpt save & reload parity (including non-persistent scaler buffers restored from params)
        ckpt_path = os.path.join(self.temp_dir, 'mult_head.ckpt')
        trainer = pl.Trainer(default_root_dir=self.temp_dir, accelerator='cpu', logger=False, enable_checkpointing=False)
        trainer.strategy.connect(module)
        trainer.save_checkpoint(ckpt_path)

        reloaded_module = FantasyBasketballLightningModule.load_from_checkpoint(ckpt_path)
        reloaded_module.eval()
        with torch.no_grad():
            reloaded_ckpt_out = reloaded_module.model(batch)
        torch.testing.assert_close(orig_out['outputs_pred'], reloaded_ckpt_out['outputs_pred'], atol=0.0, rtol=0.0)
        torch.testing.assert_close(orig_out['fantasy_points'], reloaded_ckpt_out['fantasy_points'], atol=0.0, rtol=0.0)

        # Verify params.json + ModelArchitecture.load_state_dict parity
        params_path = os.path.join(self.temp_dir, 'params.json')
        params.save_to_json(path=params_path)
        loaded_params = MachineLearningParamsFantasyBasketballPredictions.load_params(params_path)
        rebuilt_model = ModelArchitecture(
            params=loaded_params.architecture_params,
            feature_type_to_input_shape=loaded_params.features_params.feature_type_to_input_shape,
        )
        rebuilt_model.load_state_dict(module.model.state_dict(), strict=True)
        rebuilt_model.eval()
        with torch.no_grad():
            rebuilt_out = rebuilt_model(batch)
        torch.testing.assert_close(orig_out['outputs_pred'], rebuilt_out['outputs_pred'], atol=0.0, rtol=0.0)

    def test_cl4_no_histogram_weights_and_fp_loss_weight(self):
        """Verify --no-histogram-weights and --fp-loss-weight wiring, round-trips, and forward/backward passes."""
        with self.assertRaisesRegex(ValueError, "fp_loss_weight must be >= 0.0"):
            create_params(fp_loss_weight=-0.01)

        # 1. Single-stage with --no-histogram-weights and --fp-loss-weight 0.05 (bench=10 -> 30 players/game)
        p_single = create_params(
            bench_order_input=False,
            two_stage=False,
            use_histogram_weights=False,
            fp_loss_weight=0.05,
        )
        test_params(p_single)
        ds_single = p_single.features_params.feature_name_to_machine_learning_dataset_params
        self.assertNotIn('sample_weights', ds_single)
        self.assertIn('yahoo_points_label', ds_single)
        self.assertEqual(ds_single['yahoo_points_label'].feature_names, ['YAHOO_POINTS'])
        self.assertIn('fantasy_points_3d', p_single.architecture_params.output_tensor_names)

        reshape_layers = [
            lp for lp in p_single.architecture_params.layer_params_list
            if isinstance(lp, ReshapeLayerParams) and lp.output_tensor_name == 'fantasy_points_3d'
        ]
        self.assertEqual(len(reshape_layers), 1)
        self.assertEqual(reshape_layers[0].shape, (30, 1))

        loss_names_single = [lp.name_for_logs for lp in p_single.training_params.loss_params]
        self.assertEqual(loss_names_single, ['mse', 'fp_mse'])
        self.assertIsNone(p_single.training_params.loss_params[0].tensor_names_sample_weights)
        self.assertEqual(p_single.training_params.loss_params[1].weight, 0.05)

        for cb in p_single.training_params.callbacks_params:
            if isinstance(cb, (ModelCheckpointParams, EarlyStoppingParams, ReduceLearningRateOnPlateauParams)):
                self.assertEqual(cb.value_to_track, 'eval_loss')

        # 2. Two-stage multiplicative combined with --no-histogram-weights and --fp-loss-weight 0.005
        p_two = create_params(
            two_stage=True,
            stage2_head='multiplicative',
            use_histogram_weights=False,
            fp_loss_weight=0.005,
        )
        test_params(p_two)
        for cb in p_two.training_params.callbacks_params:
            if isinstance(cb, (ModelCheckpointParams, EarlyStoppingParams, ReduceLearningRateOnPlateauParams)):
                self.assertEqual(cb.value_to_track, 'eval_mse')

        for lp in p_two.architecture_params.layer_params_list:
            if hasattr(lp, 'embeddings_dims') and hasattr(lp, 'num_embeddings'):
                lp.update_after_loading_data({'num_embeddings': [50] * len(lp.embeddings_dims)})

        stat_names = p_two.features_params.feature_name_to_machine_learning_dataset_params['true_label'].feature_names
        class DummyScaler:
            feature_names_in_ = np.array(stat_names)
            mean_ = np.arange(1.0, len(stat_names) + 1.0, dtype=np.float64)
            scale_ = np.linspace(0.5, 5.5, len(stat_names), dtype=np.float64)

            def inverse_transform(self, x):
                return np.asarray(x, dtype=np.float64) * self.scale_ + self.mean_

        combined = type("Combined", (), {
            "machine_learning_datasets_stacked": type("Stacked", (), {
                "datasets": {"true_label": type("DS", (), {"scaler": DummyScaler()})()}
            })()
        })()
        update_calculate_fantasy_points_params(params=p_two, dataset_combined=combined)

        model = ModelArchitecture(
            params=p_two.architecture_params,
            feature_type_to_input_shape=p_two.features_params.feature_type_to_input_shape,
        ).train()
        losses = LossesCombined(p_two.training_params.loss_params, device=torch.device('cpu'), name='loss')

        torch.manual_seed(19)
        batch = {}
        for name, shape in p_two.features_params.feature_type_to_input_shape.items():
            if name in ('embedding_features', 'positional_encoding_features'):
                batch[name] = torch.randint(0, 5, (2, *shape))
            elif name in ('real_player_mask', 'keep_mask', 'stage1_keep_mask'):
                batch[name] = torch.ones((2, *shape), dtype=torch.float32)
            elif name == 'stage1_label':
                batch[name] = torch.full((2, *shape), 1.0 / 11.0, dtype=torch.float32)
            else:
                batch[name] = torch.randn((2, *shape), dtype=torch.float32)

        outputs = model(batch)
        self.assertEqual(outputs['fantasy_points_3d'].shape, (2, 22, 1))
        total_loss = losses.forward({**batch, **outputs})
        self.assertTrue(torch.isfinite(total_loss).all())
        total_loss.backward()
        losses.summarize()
        logs = losses.get_logs()
        self.assertIn('loss', logs)
        self.assertIn('mse', logs)
        self.assertIn('stage1_kl', logs)
        self.assertIn('fp_mse', logs)
        self.assertAlmostEqual(
            float(logs['loss']),
            float(logs['mse']) + 10.0 * float(logs['stage1_kl']) + 0.005 * float(logs['fp_mse']),
            places=5,
        )

    def test_fp_quantiles_wiring_gradient_isolation_and_lightning_export(self):
        """Verify --fp-quantiles parameter wiring, gradient isolation, single/two-stage support, and Lightning export."""
        p_single_q = create_params(fp_quantiles=True, two_stage=False)
        test_params(p_single_q)
        self.assertIn('yahoo_points_label', p_single_q.features_params.feature_name_to_machine_learning_dataset_params)
        self.assertIn('completed_real_mask', p_single_q.features_params.feature_name_to_machine_learning_dataset_params)
        self.assertIn('MIN', p_single_q.features_params.all_feature_names_list)
        self.assertEqual(p_single_q.training_params.alpha_cash, 0.35)
        self.assertEqual(p_single_q.training_params.alpha_gpp, 0.85)
        for cb in p_single_q.training_params.callbacks_params:
            if isinstance(cb, (ModelCheckpointParams, EarlyStoppingParams, ReduceLearningRateOnPlateauParams)):
                self.assertEqual(cb.value_to_track, 'eval_mse')

        q_layers_single = [lp for lp in p_single_q.architecture_params.layer_params_list if isinstance(lp, NonCrossingQuantilesParams)]
        self.assertEqual(len(q_layers_single), 1)
        self.assertEqual(
            q_layers_single[0].input_tensor_name,
            {'features': 'transformer_output', 'stats': 'outputs_pred', 'fp': 'fantasy_points_3d'},
        )

        p_two_q = create_params(fp_quantiles=True, two_stage=True, stage2_head='multiplicative')
        test_params(p_two_q)
        q_layers = [lp for lp in p_two_q.architecture_params.layer_params_list if isinstance(lp, NonCrossingQuantilesParams)]
        self.assertEqual(len(q_layers), 1)
        self.assertEqual(
            q_layers[0].input_tensor_name,
            {'features': 'post_transformer_combined', 'stats': 'outputs_pred', 'fp': 'fantasy_points_3d'},
        )
        self.assertEqual(list(q_layers[0].output_tensor_name), ['fp_q20_pred', 'fp_q80_pred'])

        q_losses = [lp for lp in p_two_q.training_params.loss_params if isinstance(lp, QuantileRegressionLossParams)]
        self.assertEqual([(ql.name_for_logs, ql.quantile, ql.tensor_names_keep_mask, ql.weight) for ql in q_losses], [
            ('fp_q20', 0.20, 'completed_real_mask', 0.1),
            ('fp_q80', 0.80, 'completed_real_mask', 0.1),
        ])

        for lp in p_two_q.architecture_params.layer_params_list:
            if hasattr(lp, 'embeddings_dims') and hasattr(lp, 'num_embeddings'):
                lp.update_after_loading_data({'num_embeddings': [50] * len(lp.embeddings_dims)})

        stat_names = p_two_q.features_params.feature_name_to_machine_learning_dataset_params['true_label'].feature_names
        class DummyScaler:
            feature_names_in_ = np.array(stat_names)
            mean_ = np.arange(1.0, len(stat_names) + 1.0, dtype=np.float64)
            scale_ = np.linspace(0.5, 5.5, len(stat_names), dtype=np.float64)

            def inverse_transform(self, x):
                return np.asarray(x, dtype=np.float64) * self.scale_ + self.mean_

        combined = type("Combined", (), {
            "machine_learning_datasets_stacked": type("Stacked", (), {
                "datasets": {"true_label": type("DS", (), {"scaler": DummyScaler()})()}
            })()
        })()
        update_calculate_fantasy_points_params(params=p_two_q, dataset_combined=combined)

        model = ModelArchitecture(
            params=p_two_q.architecture_params,
            feature_type_to_input_shape=p_two_q.features_params.feature_type_to_input_shape,
        ).eval()

        torch.manual_seed(23)
        batch = {}
        for name, shape in p_two_q.features_params.feature_type_to_input_shape.items():
            if name in ('embedding_features', 'positional_encoding_features'):
                batch[name] = torch.randint(0, 5, (2, *shape))
            elif name in ('real_player_mask', 'keep_mask', 'stage1_keep_mask', 'completed_real_mask'):
                batch[name] = torch.ones((2, *shape), dtype=torch.float32)
            elif name == 'stage1_label':
                batch[name] = torch.full((2, *shape), 1.0 / 11.0, dtype=torch.float32)
            else:
                batch[name] = torch.randn((2, *shape), dtype=torch.float32)

        outputs = model(batch)
        self.assertEqual(outputs['fp_q20_pred'].shape, (2, 22, 1))
        self.assertEqual(outputs['fp_q80_pred'].shape, (2, 22, 1))
        self.assertTrue(torch.all(outputs['fp_q20_pred'] < outputs['fantasy_points_3d']))
        self.assertTrue(torch.all(outputs['fantasy_points_3d'] < outputs['fp_q80_pred']))

        # Pinball-only backward must leave trunk / Stage-1 / Stage-2 gradients untouched (None).
        pinball_only = LossesCombined(q_losses, device=torch.device('cpu'), name='pinball_only')
        model.zero_grad()
        loss_q = pinball_only.forward({**batch, **outputs})
        loss_q.backward()
        q_layer_idx = p_two_q.architecture_params.layer_params_list.index(q_layers[0])
        for idx, sublayer in enumerate(model.layers):
            if idx == q_layer_idx:
                self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0) for p in sublayer.parameters()))
            else:
                for p in sublayer.parameters():
                    self.assertIsNone(p.grad)

        # Lightning export propagates unscaled YAHOO_POINTS_q20_pred & YAHOO_POINTS_q80_pred
        lm = FantasyBasketballLightningModule(
            params=p_two_q,
            model=model,
        ).eval()
        lm._target_scaler = DummyScaler()
        pred_batch = {**batch}
        pred_batch['metadata'] = torch.zeros(
            (2, *p_two_q.features_params.feature_type_to_input_shape['metadata']),
            dtype=torch.float32,
        )
        pred_batch['metadata'][:, :, 0] = 20000.0
        pred_batch['metadata'][:, :, 1] = torch.arange(1, 23).unsqueeze(0).expand(2, 22).float()
        pred_out = lm.predict_step(pred_batch, 0)
        self.assertIn('fp_q20_pred', pred_out)
        self.assertIn('fp_q80_pred', pred_out)
        pred_df = lm.format_predictions([pred_out])
        self.assertIn('YAHOO_POINTS_q20_pred', pred_df.columns)
        self.assertIn('YAHOO_POINTS_q80_pred', pred_df.columns)
        self.assertTrue((pred_df['YAHOO_POINTS_q20_pred'] < pred_df['YAHOO_POINTS_pred']).all())
        self.assertTrue((pred_df['YAHOO_POINTS_pred'] < pred_df['YAHOO_POINTS_q80_pred']).all())


if __name__ == "__main__":
    unittest.main()
