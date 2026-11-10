import datetime
import os
import pickle as pkl
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from calc_features import CalcFeatures, COUNTING_STATS
from constants import DATA_DIR, NY_TZ, POSITION_TO_NUM
from machine_learning.callbacks import CollectTruePredictedOutputs, CollectTruePredictedOutputsParams
from machine_learning.enums import DataSplitType
from machine_learning.layers.layers import EmbeddingsLayer
from machine_learning.machine_learning_datasets import (
    MachineLearningDatasetCombined,
)
from machine_learning.params.layers_params import EmbeddingsLayerParams
from machine_learning.params.machine_learning_dataset_params import (
    MachineLearningDatasetTabularCategoricalEncodingParams,
    MachineLearningDatasetTabularScalingParams,
)
from machine_learning.params.params import MachineLearningParams, FeaturesParams
from machine_learning.run_model import RunModel
from machine_learning.utils import CustomOrdinalEncoder
from machine_learning_fantasy_basketball_predictions.lightning_model import FantasyBasketballLightningModule
from machine_learning_fantasy_basketball_predictions.params.params import (
    MachineLearningParamsFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.run_model import (
    RunModelFantasyBasketballPredictions,
)
from run_full_pipeline import RunFullPipeline
from utils import get_current_date


class DummyLegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 4)
        self.emb.padding_idx = None
        with torch.no_grad():
            self.emb.weight.fill_(1.0)  # Row 4 is 1.0 initially


class TestPR1ReliabilityAndEmbeddings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_todays_predictions_merges_clean_yahoo_salary_and_position(self):
        """Verify _get_todays_predictions resolves authoritative salaries/positions without _x/_y collisions."""
        today = get_current_date()
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )
        pipeline._run_date = today

        # 1. Mock predictions with metadata prefix and placeholder salaries
        mock_preds = pd.DataFrame({
            "DATE": [today, today],
            "PLAYER_ID": [101, 102],
            "CollectTruePredictedOutputsParamsFantasyBasketballPredictions_YAHOO_POINTS": [32.5, 48.0],
            "CollectTruePredictedOutputsParamsFantasyBasketballPredictions_YAHOO_POSITION": [1.0, 2.0],
            "CollectTruePredictedOutputsParamsFantasyBasketballPredictions_YAHOO_SALARY": [-999.0, -999.0],
            "split_type": ["TRAIN", "TRAIN"],
        })

        pipeline.train_nn = MagicMock()
        pipeline.train_nn._change_metrics_params_tuple_to_per_sample = MagicMock(return_value=())
        pipeline.train_nn.params.training_params.metrics_params = ()
        pipeline.train_nn.params.training_params.loss_params = ()
        pipeline.train_nn.predict = MagicMock(return_value=mock_preds)

        # 2. Mock stats_loaders_players.df containing overlapping columns
        pipeline.stats_loaders_players = MagicMock()
        pipeline.stats_loaders_players.df = pd.DataFrame({
            "PLAYER_ID": [101, 102],
            "DATE": [today, today],
            "GAME_ID": [9001, 9002],
            "TEAM_ID": [1610612738, 1610612744],
            "YAHOO_POSITION": [1, 2],
            "YAHOO_SALARY": [-999.0, -999.0],
        })
        pipeline.stats_loaders_players.get_player_name_to_id.return_value = {
            "Jayson Tatum": 101,
            "Stephen Curry": 102,
        }

        # 3. Authoritative Yahoo template
        teams_template = pd.DataFrame({
            "Name": ["Jayson Tatum", "Stephen Curry"],
            "Salary": [45.0, 42.0],
            "Position": ["SF/PF", "PG"],
            "Game": ["BOS@GSW", "BOS@GSW"],
            "Time": ["7:30PM", "7:30PM"],
        })

        # Mock game times addition to avoid external schedule dependencies
        game_time = datetime.datetime.now(NY_TZ) + datetime.timedelta(hours=2)
        pipeline._add_game_times_to_df = MagicMock(
            side_effect=lambda df: df.assign(GAME_DATE_TIME=game_time)
        )

        with patch("run_full_pipeline.get_next_game_time", return_value=datetime.datetime.now(NY_TZ)):
            result = pipeline._get_todays_predictions(teams_template=teams_template)

        # Verify no column collisions (_x or _y)
        self.assertFalse(any(col.endswith("_x") or col.endswith("_y") for col in result.columns))
        self.assertIn("YAHOO_SALARY", result.columns)
        self.assertIn("YAHOO_POSITION", result.columns)

        # Verify salaries mapped from template
        tatum_row = result.loc[result["PLAYER_ID"] == 101].iloc[0]
        curry_row = result.loc[result["PLAYER_ID"] == 102].iloc[0]
        self.assertEqual(tatum_row["YAHOO_SALARY"], 45.0)
        self.assertEqual(curry_row["YAHOO_SALARY"], 42.0)

        # Verify positions parsed from template
        self.assertEqual(tatum_row["YAHOO_POSITION"], POSITION_TO_NUM["SF"])
        self.assertEqual(curry_row["YAHOO_POSITION"], POSITION_TO_NUM["PG"])

    def test_collect_outputs_uses_active_model_dir_scalers_and_fails_when_missing(self):
        """Verify CollectTruePredictedOutputs uses model_dir scalers and fails fast when missing."""
        # 1. Missing model_dir when scaler_dir is specified -> raises ValueError
        params = CollectTruePredictedOutputsParams(
            scaler_dir="scalers",
            tensor_names_true=["true_label"],
            tensor_names_pred=["true_label_pred"],
        )
        cb_missing_dir = CollectTruePredictedOutputs(params=params, model_dir=None)
        with self.assertRaises(ValueError):
            cb_missing_dir._resolve_scaler_dir()

        # 2. scaler_dir is specified with model_dir, but directory does not exist -> raises FileNotFoundError
        model_dir = os.path.join(self.temp_dir, "test_model")
        os.makedirs(model_dir, exist_ok=True)
        cb_no_scalers = CollectTruePredictedOutputs(params=params, model_dir=model_dir)
        scaler_dir = cb_no_scalers._resolve_scaler_dir()

        dummy_df = pd.DataFrame({
            "YAHOO_POINTS_true": [10.0, 20.0],
            "YAHOO_POINTS_pred": [12.0, 18.0],
        })
        with self.assertRaises(FileNotFoundError):
            cb_no_scalers._unscale_results(dummy_df, scaler_dir)

        # 3. Scaler directory exists and contains true_label.pkl -> unscaling succeeds
        scalers_dir = os.path.join(model_dir, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)

        scaler = StandardScaler()
        scaler.fit(pd.DataFrame({"YAHOO_POINTS": [0.0, 100.0]}))
        with open(os.path.join(scalers_dir, "true_label.pkl"), "wb") as f:
            pkl.dump(scaler, f)

        cb_valid = CollectTruePredictedOutputs(params=params, model_dir=model_dir)
        scaler_dir_valid = cb_valid._resolve_scaler_dir()
        self.assertEqual(scaler_dir_valid, scalers_dir)

        test_df = dummy_df.copy()
        cb_valid._unscale_results(test_df, scaler_dir_valid)
        expected_unscaled_0 = 10.0 * scaler.scale_[0] + scaler.mean_[0]
        self.assertAlmostEqual(test_df["YAHOO_POINTS_true"].iloc[0], expected_unscaled_0, places=4)

    def test_assert_transformers_complete_catches_missing_scalers_and_allows_no_categorical_config(self):
        """Verify assert_transformers_complete strictly validates scalers and encoders."""
        fp = FeaturesParams(
            feature_name_to_machine_learning_dataset_params={
                "tabular": MachineLearningDatasetTabularScalingParams(feature_names=["feat_a"]),
                "embedding_features": MachineLearningDatasetTabularCategoricalEncodingParams(feature_names=["cat_b"]),
            }
        )

        # Missing tabular scaler
        with self.assertRaises(FileNotFoundError) as cm:
            MachineLearningDatasetCombined.assert_transformers_complete(
                features_params=fp,
                scalers={},
                encoders={"embedding_features": MagicMock()},
            )
        self.assertIn("tabular", str(cm.exception))

        # Missing categorical encoder
        with self.assertRaises(FileNotFoundError) as cm:
            MachineLearningDatasetCombined.assert_transformers_complete(
                features_params=fp,
                scalers={"tabular": MagicMock()},
                encoders={},
            )
        self.assertIn("embedding_features", str(cm.exception))

        # Parameters with no categorical columns configured -> should succeed without encoders
        fp_no_cat = FeaturesParams(
            feature_name_to_machine_learning_dataset_params={
                "tabular": MachineLearningDatasetTabularScalingParams(feature_names=["feat_a"]),
            }
        )
        try:
            MachineLearningDatasetCombined.assert_transformers_complete(
                features_params=fp_no_cat,
                scalers={"tabular": MagicMock()},
                encoders={},
            )
        except Exception as e:
            self.fail(f"assert_transformers_complete raised unexpectedly with no categorical features: {e}")

    @patch("run_full_pipeline.update_todays_schedule_from_scoreboard")
    @patch("run_full_pipeline.scrape_season_schedules")
    @patch("run_full_pipeline.load_and_cache_players_list")
    @patch("run_full_pipeline.update_all_stats")
    def test_run_once_per_day_falls_back_to_latest_valid_daily_model_with_matching_params(
        self,
        mock_update_stats,
        mock_load_players,
        mock_scrape_schedules,
        mock_update_scoreboard,
    ):
        """Verify run_once_per_day safely falls back to latest valid daily model on training failure."""
        models_base = os.path.join(
            self.temp_dir, "machine_learning_fantasy_basketball_predictions", "models"
        )
        os.makedirs(models_base, exist_ok=True)

        # 1. Create base model 0
        model_0_dir = os.path.join(models_base, "0")
        os.makedirs(os.path.join(model_0_dir, "scalers"), exist_ok=True)
        os.makedirs(os.path.join(model_0_dir, "categorical_encoder"), exist_ok=True)
        with open(os.path.join(model_0_dir, "params.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(model_0_dir, "model.pt"), "w") as f:
            f.write("model0")
        with open(os.path.join(model_0_dir, "scalers", "true_label.pkl"), "w") as f:
            f.write("scaler")
        with open(os.path.join(model_0_dir, "categorical_encoder", "embedding_features.pkl"), "w") as f:
            f.write("encoder")

        # 2. Create previous valid daily model: model_2026_11_07
        model_yesterday_dir = os.path.join(models_base, "model_2026_11_07")
        os.makedirs(os.path.join(model_yesterday_dir, "scalers"), exist_ok=True)
        os.makedirs(os.path.join(model_yesterday_dir, "categorical_encoder"), exist_ok=True)
        with open(os.path.join(model_yesterday_dir, "params.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(model_yesterday_dir, "model.pt"), "w") as f:
            f.write("model_yesterday")
        with open(os.path.join(model_yesterday_dir, "scalers", "true_label.pkl"), "w") as f:
            f.write("scaler")
        with open(os.path.join(model_yesterday_dir, "categorical_encoder", "embedding_features.pkl"), "w") as f:
            f.write("encoder")

        # 3. Setup pipeline for 2026-11-08 with scheduled games
        run_date = datetime.date(2026, 11, 8)
        pipeline = RunFullPipeline(
            load_params_from_model_id=0,
            run_date=run_date,
            run_immediately=True,
            initialize_daily_job=False,
            force_retrain=True,
        )
        pipeline.dates_to_game_times = {run_date: [datetime.datetime.now(NY_TZ)]}
        pipeline._get_game_times_by_day_dict = MagicMock(return_value={run_date: [datetime.datetime.now(NY_TZ)]})

        # Mock RunModelFantasyBasketballPredictions: first instantiation fails during train()
        mock_failing_runner = MagicMock()
        failed_temp_dir = os.path.join(models_base, "failed_tmp_model")
        os.makedirs(failed_temp_dir, exist_ok=True)
        mock_failing_runner._model_dir = failed_temp_dir
        mock_failing_runner.train.side_effect = RuntimeError("Simulated GPU OutOfMemoryError")

        mock_fallback_runner = MagicMock()

        def runner_side_effect(params, description=None):
            if params.continue_model_id == "model_2026_11_07":
                return mock_fallback_runner
            return mock_failing_runner

        with patch("run_full_pipeline.DATA_DIR", self.temp_dir), \
             patch("run_full_pipeline.MachineLearningParamsFantasyBasketballPredictions.load_params") as mock_load_params, \
             patch("run_full_pipeline.RunModelFantasyBasketballPredictions", side_effect=runner_side_effect):

            mock_params = MagicMock()
            mock_params.features_params = MagicMock()
            mock_params.training_params = MagicMock()
            mock_load_params.return_value = mock_params

            pipeline.run_once_per_day()

        # Verify failed temporary directory was cleaned up
        self.assertFalse(os.path.exists(failed_temp_dir))
        # Verify base model 0 was preserved
        self.assertTrue(os.path.isdir(model_0_dir))
        # Verify fallback model was set to model_2026_11_07
        self.assertEqual(mock_params.continue_model_id, "model_2026_11_07")
        self.assertIs(pipeline.train_nn, mock_fallback_runner)

    def test_unseen_category_maps_to_zero_frozen_padding_embedding_and_undrafted_rookie_kept(self):
        """Verify unseen category maps to padding index with 0 output & 0 grad, and undrafted rookie is retained."""
        # --- Part A: Unseen Category Padding Embedding ---
        encoder = CustomOrdinalEncoder(handle_unknown="use_encoded_value")
        categories_train = [["Boston"], ["Golden State"], ["Los Angeles"]]
        encoder.fit(categories_train)

        # Unseen category "Seattle"
        unseen_test = [["Seattle"]]
        encoded = encoder.transform(unseen_test)
        # Expected encoded index: 3 (len of categories)
        self.assertEqual(encoded[0, 0], 3)

        num_unique = len(encoder.categories_[0]) + 1  # 3 + 1 = 4
        emb_params = EmbeddingsLayerParams(
            input_tensor_name="categorical_input",
            num_embeddings=[num_unique],
            embeddings_dims=[8],
            output_tensor_name=["cat_emb"],
        )
        layer = EmbeddingsLayer(params=emb_params, input_shape=(1, 1))
        self.assertEqual(layer.embedding_layers[0].padding_idx, 3)

        # Forward pass with unseen index
        x = torch.tensor([[3]], dtype=torch.long)
        out = layer(x)["cat_emb"]
        self.assertTrue(torch.all(out == 0.0))

        # Backward pass: gradient at padding_idx must remain zero
        loss = (out ** 2).sum()
        loss.backward()
        padding_grad = layer.embedding_layers[0].weight.grad[3]
        self.assertTrue(torch.all(padding_grad == 0.0))

        # --- Part B: Undrafted Rookie Retained in _drop_nans ---
        cf = CalcFeatures()
        # Row 1: Undrafted rookie (YEARS_SINCE_DRAFT is NaN, but counting stats and key cols present)
        # Row 2: Missing key stat (MIN_mean5 is NaN) -> must be dropped
        sample_data = {
            "PLAYER_ID": [201, 202],
            "TEAM_ID": [1610612738, 1610612738],
            "DATE": [datetime.date(2026, 11, 10), datetime.date(2026, 11, 10)],
            "DAYS_REST": [2, 2],
            "MIN_mean5": [28.5, np.nan],
            "START_POSITION_CALCULATED": [1, 1],
            "YEARS_SINCE_DRAFT": [np.nan, 2.0],
        }
        for stat in COUNTING_STATS:
            sample_data[stat] = [10.0, 10.0]

        cf.updates_df = pd.DataFrame(sample_data)
        cf._drop_nans()

        # Verify only row 201 (undrafted rookie) survived; row 202 was dropped
        self.assertEqual(len(cf.updates_df), 1)
        self.assertEqual(cf.updates_df.iloc[0]["PLAYER_ID"], 201)
        self.assertTrue(pd.isna(cf.updates_df.iloc[0]["YEARS_SINCE_DRAFT"]))

    def test_collect_outputs_supports_explicit_valid_scaler_dir_when_model_dir_is_none(self):
        """Verify CollectTruePredictedOutputs accepts existing directory in params.scaler_dir when model_dir is None."""
        scalers_dir = os.path.join(self.temp_dir, "custom_scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        params = CollectTruePredictedOutputsParams(
            scaler_dir=scalers_dir,
            tensor_names_true=["true_label"],
            tensor_names_pred=["true_label_pred"],
        )
        cb = CollectTruePredictedOutputs(params=params, model_dir=None)
        resolved = cb._resolve_scaler_dir()
        self.assertEqual(resolved, scalers_dir)

    def test_lightning_format_predictions_fails_fast_when_scaler_missing(self):
        """Verify FantasyBasketballLightningModule.format_predictions raises FileNotFoundError when scaler is missing."""
        mock_params = MagicMock()
        mock_params.features_params.feature_name_to_machine_learning_dataset_params = {
            "true_label": MachineLearningDatasetTabularScalingParams(feature_names=["YAHOO_POINTS"])
        }
        empty_scaler_dir = os.path.join(self.temp_dir, "empty_scalers")
        os.makedirs(empty_scaler_dir, exist_ok=True)

        module = MagicMock(spec=FantasyBasketballLightningModule)
        module.params = mock_params
        module.scaler_dir = empty_scaler_dir
        module._target_scaler = None
        module._load_target_scaler = MagicMock(return_value=None)

        with self.assertRaises(FileNotFoundError) as cm:
            FantasyBasketballLightningModule.format_predictions(module, predict_batches=[])
        self.assertIn("Target scaler true_label.pkl not found", str(cm.exception))

    def test_run_model_create_data_loaders_asserts_transformers_complete(self):
        """Verify RunModel.create_data_loaders calls assert_transformers_complete and fails on missing scaler."""
        checkpoint_dir = os.path.join(self.temp_dir, "ckpt_missing_scalers")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "scalers"), exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, "categorical_encoder"), exist_ok=True)

        class TestRunner(RunModel):
            def __init__(self):
                pass

        runner = TestRunner()
        runner.params = MagicMock()
        runner.params.continue_model_id = "test_ckpt"
        runner._continue_model_dir = checkpoint_dir
        runner.use_lightning = False
        runner.params.features_params = FeaturesParams(
            feature_name_to_machine_learning_dataset_params={
                "tabular": MachineLearningDatasetTabularScalingParams(feature_names=["PTS"]),
            }
        )

        with self.assertRaises(FileNotFoundError) as cm:
            runner.create_data_loaders()
        self.assertIn("Missing scalers for datasets", str(cm.exception))

    def test_loaded_legacy_model_acquires_padding_idx_and_zeroes_row(self):
        """Verify that loading a model with legacy EmbeddingsLayer (padding_idx=None) retrofits padding_idx and zeroes row."""
        model_path = os.path.join(self.temp_dir, "dummy_model.pt")
        legacy_model = DummyLegacyModel()
        torch.save(legacy_model, model_path)

        runner = MagicMock(spec=RunModel)
        runner.params = MagicMock()
        runner.params.continue_model_id = "dummy"
        runner.use_lightning = False
        runner.device = torch.device("cpu")
        runner._continue_model_dir = self.temp_dir

        with patch("machine_learning.callbacks.ModelCheckpoint.load_model", return_value=torch.load(model_path, weights_only=False)):
            RunModel._create_model(runner)

        # Verify padding_idx retrofitted to 4, and row 4 zeroed
        self.assertEqual(runner.model.emb.padding_idx, 4)
        self.assertTrue(torch.all(runner.model.emb.weight[4] == 0.0))
        # Non-padding rows still intact
        self.assertTrue(torch.all(runner.model.emb.weight[0] == 1.0))


if __name__ == "__main__":
    unittest.main()
