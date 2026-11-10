import datetime
import os
import pickle as pkl
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from machine_learning.callbacks import CollectTruePredictedOutputs, CollectTruePredictedOutputsParams
from machine_learning.enums import Reduction
from machine_learning.layers.layers import EmbeddingsLayer
from machine_learning.machine_learning_datasets import (
    MachineLearningDatasetCombined,
    MachineLearningDatasetMetadata,
)
from machine_learning.metrics.abstract_metrics import AbstractMetric
from machine_learning.params.abstract_params import AbstractMetricParams
from machine_learning.params.layers_params import EmbeddingsLayerParams
from machine_learning.params.machine_learning_dataset_params import (
    MachineLearningDatasetTabularCategoricalEncodingParams,
    MachineLearningDatasetTabularScalingParams,
)
from machine_learning.params.params import FeaturesParams
from machine_learning.run_model import RunModel
from machine_learning.utils import CustomOrdinalEncoder, get_next_model_id
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import (
    MachineLearningDatasetCombinedFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.params.params import (
    FeaturesParamsFantasyBasketballPredictions,
)
from utils import convert_date_col_to_date


class DummyLegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5, 4)
        self.emb.padding_idx = None
        with torch.no_grad():
            self.emb.weight.fill_(1.0)


class TestGetNextModelId(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("machine_learning.utils.models_dir")
    def test_directory_does_not_exist(self, mock_models_dir):
        """Tests that get_next_model_id creates the directory if missing and returns 0."""
        target_dir = os.path.join(self.temp_dir, "non_existent_models_dir")
        mock_models_dir.return_value = target_dir

        self.assertFalse(os.path.exists(target_dir))
        next_id = get_next_model_id("fantasy_basketball_predictions")

        self.assertEqual(next_id, 0)
        self.assertTrue(os.path.isdir(target_dir))

    @patch("machine_learning.utils.models_dir")
    def test_directory_exists_and_empty(self, mock_models_dir):
        """Tests that get_next_model_id returns 0 when the models directory is empty."""
        target_dir = os.path.join(self.temp_dir, "empty_models_dir")
        os.makedirs(target_dir, exist_ok=True)
        mock_models_dir.return_value = target_dir

        next_id = get_next_model_id("fantasy_basketball_predictions")
        self.assertEqual(next_id, 0)

    @patch("machine_learning.utils.models_dir")
    def test_increment_with_existing_numeric_ids(self, mock_models_dir):
        """Tests that get_next_model_id returns max(digits) + 1."""
        target_dir = os.path.join(self.temp_dir, "existing_models_dir")
        os.makedirs(target_dir, exist_ok=True)
        for model_id in [0, 1, 4, 12]:
            os.makedirs(os.path.join(target_dir, str(model_id)))

        mock_models_dir.return_value = target_dir

        next_id = get_next_model_id("fantasy_basketball_predictions")
        self.assertEqual(next_id, 13)

    @patch("machine_learning.utils.models_dir")
    def test_ignores_non_digit_directories_and_files(self, mock_models_dir):
        """Tests that non-digit files/folders (e.g., .gitkeep, archive, temp_01) are ignored."""
        target_dir = os.path.join(self.temp_dir, "mixed_models_dir")
        os.makedirs(target_dir, exist_ok=True)

        # Create numeric folders
        os.makedirs(os.path.join(target_dir, "0"))
        os.makedirs(os.path.join(target_dir, "5"))

        # Create non-numeric folders and files
        os.makedirs(os.path.join(target_dir, "archive"))
        os.makedirs(os.path.join(target_dir, "temp_model_10"))
        with open(os.path.join(target_dir, ".gitkeep"), "w") as f:
            f.write("")
        with open(os.path.join(target_dir, "notes.txt"), "w") as f:
            f.write("")

        mock_models_dir.return_value = target_dir

        next_id = get_next_model_id("fantasy_basketball_predictions")
        self.assertEqual(next_id, 6)


class TestConvertDatetimeToEpochTime(unittest.TestCase):
    def test_date_objects_convert_to_epoch_seconds(self):
        """DATE as datetime.date (as produced by CalcFeatures.load_df) must map to true epoch seconds."""
        dates = [datetime.date(2014, 10, 28), datetime.date(2021, 10, 19), datetime.date(2026, 10, 20)]
        df = pd.DataFrame({"DATE": dates})

        result = MachineLearningDatasetCombined._convert_datetime_to_epoch_time(df)

        expected = [(pd.Timestamp(d) - pd.Timestamp("1970-01-01")).total_seconds() for d in dates]
        self.assertEqual(result["EPOCH_DATE"].tolist(), expected)
        self.assertEqual(result["EPOCH_DATE"].iloc[0], 1414454400.0)

    def test_datetime64_column_is_left_untouched(self):
        """An already-datetime64 DATE column skips conversion (existing behavior)."""
        df = pd.DataFrame({"DATE": pd.to_datetime(["2021-10-19"])})
        result = MachineLearningDatasetCombined._convert_datetime_to_epoch_time(df)
        self.assertNotIn("EPOCH_DATE", result.columns)


class DummyScalarMetricParams(AbstractMetricParams):
    name_for_logs: str = "dummy_scalar"
    accumulate: bool = True
    reduction: Reduction = Reduction.MEAN
    reduction_dim: int = -1


class DummyScalarMetric(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class TestAbstractMetricAccumulation(unittest.TestCase):
    def test_accumulate_with_0d_scalar_tensors_does_not_raise(self):
        """Tests that 0-D scalar metrics can accumulate across batches without concatenation errors."""
        params = DummyScalarMetricParams(
            name_for_logs="dummy",
            accumulate=True,
            reduction=Reduction.MEAN,
            reduction_dim=-1,
        )
        metric = DummyScalarMetric(params=params)

        # Batch 1 with 0-D scalar reduction
        metric._add_batch_results_to_epoch_results(torch.tensor(2.0), torch.tensor(1.0))
        # Batch 2 with 0-D scalar reduction
        metric._add_batch_results_to_epoch_results(torch.tensor(4.0), torch.tensor(1.0))

        metric.summarize()
        self.assertIsInstance(metric.epoch_results, pd.DataFrame)
        self.assertEqual(len(metric.epoch_results), 2)
        self.assertListEqual(metric.epoch_results["dummy"].tolist(), [2.0, 4.0])


class TestLaggedFeaturesFallback(unittest.TestCase):
    def test_missing_lag_falls_back_to_previous_lag(self):
        """Player with 2 career games requesting 3 lags: t-2 and t-3 should carry forward t-1."""
        df = pd.DataFrame({
            "PLAYER_ID": [10, 10],
            "DATE": [datetime.date(2023, 11, 1), datetime.date(2023, 11, 3)],
            "GAME_ID": [101, 102],
            "TEAM_ID": [1, 1],
            "TEAM": ["BOS", "BOS"],
            "SEASON_YEAR": [2023, 2023],
            "STARTER": [True, True],
            "START_POSITION_CALCULATED": ["PG", "PG"],
            "EJECTED": [False, False],
            "LEFT_MIDGAME": [False, False],
            "PTS": [20.0, 30.0],
        })
        params = FeaturesParamsFantasyBasketballPredictions()
        params.all_feature_names_list = ["PTS_t-1", "PTS_t-2", "PTS_t-3"]
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_additional_columns(df.copy(), params)

        # Game 2 (index 1) has game 1 prior (PTS=20.0). Lags 2 and 3 should fall back to 20.0
        self.assertEqual(res.loc[1, "PTS_t-1"], 20.0)
        self.assertEqual(res.loc[1, "PTS_t-2"], 20.0)
        self.assertEqual(res.loc[1, "PTS_t-3"], 20.0)

    def test_debut_player_uses_role_baseline(self):
        """Debut game with no prior history fills with role mean instead of 0.0."""
        df = pd.DataFrame({
            "PLAYER_ID": [10, 10],
            "DATE": [datetime.date(2023, 11, 1), datetime.date(2023, 11, 3)],
            "GAME_ID": [101, 102],
            "TEAM_ID": [1, 1],
            "TEAM": ["BOS", "BOS"],
            "SEASON_YEAR": [2023, 2023],
            "STARTER": [True, True],
            "START_POSITION_CALCULATED": ["PG", "PG"],
            "EJECTED": [False, False],
            "LEFT_MIDGAME": [False, False],
            "PTS": [20.0, 30.0],
        })
        params = FeaturesParamsFantasyBasketballPredictions()
        params.all_feature_names_list = ["PTS_t-1", "PTS_t-2", "PTS_t-3"]
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_additional_columns(df.copy(), params)

        # Debut game (index 0) has no prior history -> role mean for Starting PG is (20+30)/2 = 25.0
        self.assertEqual(res.loc[0, "PTS_t-1"], 25.0)
        self.assertEqual(res.loc[0, "PTS_t-2"], 25.0)
        self.assertEqual(res.loc[0, "PTS_t-3"], 25.0)


class TestDateConversion(unittest.TestCase):
    """Tests date conversion from days, seconds, strings, and tensors."""

    def test_convert_date_col_from_days_float32(self):
        s = pd.Series([19676.0, 19677.0], dtype="float32")
        converted = convert_date_col_to_date(s)
        self.assertEqual(converted.iloc[0], datetime.date(2023, 11, 15))
        self.assertEqual(converted.iloc[1], datetime.date(2023, 11, 16))

    def test_convert_date_col_from_seconds_int64(self):
        s = pd.Series([1700006400, 1700092800], dtype="int64")
        converted = convert_date_col_to_date(s)
        self.assertEqual(converted.iloc[0], datetime.date(2023, 11, 15))
        self.assertEqual(converted.iloc[1], datetime.date(2023, 11, 16))

    def test_convert_date_col_from_strings(self):
        s = pd.Series(["2023-11-15", "2023-11-16"])
        converted = convert_date_col_to_date(s)
        self.assertEqual(converted.iloc[0], datetime.date(2023, 11, 15))
        self.assertEqual(converted.iloc[1], datetime.date(2023, 11, 16))

    def test_metadata_tensor_to_df_converts_date_days(self):
        tensor = torch.tensor([[19676.0, 1.0]], dtype=torch.float32)
        df = MachineLearningDatasetMetadata.tensor_to_df(tensor, column_names=["DATE", "split_type"])
        self.assertEqual(df["DATE"].iloc[0], datetime.date(2023, 11, 15))


class TestCollectTruePredictedOutputs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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


class TestDatasetTransformersValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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


class TestEmbeddingsLayerPaddingAndRetrofit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unseen_category_maps_to_zero_frozen_padding_embedding(self):
        """Verify unseen category maps to padding index with 0 output & 0 grad."""
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
