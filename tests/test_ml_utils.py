import datetime
import os
import pickle as pkl
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from constants import IGNORE_VAL, Sites, START_POSITIONS, UNKNOWN_PLAYER_ID
from machine_learning.callbacks import CollectTruePredictedOutputs, CollectTruePredictedOutputsParams
from machine_learning.constants import IGNORE, METADATA
from machine_learning.enums import Reduction, ModelMode, DataSplitType
from machine_learning.layers.layers import (
    EmbeddingsLayer,
    _build_safe_key_padding_mask,
    TransformerEncoderPytorch,
    SplitAttentionCombineWrapper,
    SplitAttentionCombine,
    TransformerEncoderCustom,
)
from machine_learning.machine_learning_datasets import (
    MachineLearningDatasetCombined,
    MachineLearningDatasetMetadata,
    AbstractScalingMachineLearningDataset,
    dates_to_epoch_days,
)
from machine_learning.metrics.abstract_metrics import AbstractMetric
from machine_learning.metrics.combined_metrics import LossesCombined
from machine_learning.metrics.metrics import KLDivergenceLoss, MSELoss
from machine_learning.params.abstract_params import AbstractMetricParams
from machine_learning.params.metrics_params import KLDivergenceLossParams, MSELossParams
from machine_learning.params.layers_params import (
    EmbeddingsLayerParams,
    TransformerEncoderPytorchParams,
    SplitAttentionCombineParams,
    SplitAttentionCombineWrapperParams,
    TransformerEncoderCustomParams,
)
from machine_learning.params.machine_learning_dataset_params import (
    MachineLearningDatasetTabularCategoricalEncodingParams,
    MachineLearningDatasetTabularScalingParams,
)
from machine_learning.lightning_model import MachineLearningLightningModule
from machine_learning.params.params import FeaturesParams, TrainingParams
from machine_learning.run_model import RunModel
from machine_learning.utils import CustomOrdinalEncoder, get_next_model_id
from machine_learning_fantasy_basketball_predictions.layers import (
    CalculateFantasyPoints,
    ReshapeToSamplePerDay,
    OptimizeTeam,
    OptimizeTeamFn,
    TeamAllocation,
)
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import (
    MachineLearningDatasetCombinedFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.params.layers_params import (
    CalculateFantasyPointsParams,
    ReshapeToSamplePerDayParams,
    OptimizeTeamParams,
    TeamAllocationParams,
    update_calculate_fantasy_points_params,
)
from machine_learning_fantasy_basketball_predictions.params.metrics_params import OptimizeTeamsLossParams
from machine_learning_fantasy_basketball_predictions.params.params import (
    FeaturesParamsFantasyBasketballPredictions,
)
from run_all import create_params
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
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df.copy(), params)

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
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df.copy(), params)

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
        """Verify that loading a model with legacy EmbeddingsLayer (padding_idx=None) appends row K, sets padding_idx=K, and preserves rows 0..K-1."""
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

        # Verify num_embeddings expanded from 5 to 6, padding_idx retrofitted to 5, row 5 zeroed, and all 5 original rows preserved
        self.assertEqual(runner.model.emb.num_embeddings, 6)
        self.assertEqual(runner.model.emb.padding_idx, 5)
        self.assertTrue(torch.all(runner.model.emb(torch.tensor([5])) == 0.0))
        self.assertTrue(torch.all(runner.model.emb.weight[:5] == 1.0))

    def test_load_lightning_checkpoint_weights_raises_on_shape_mismatch(self):
        """Verify _load_lightning_checkpoint_weights re-raises RuntimeError instead of falling back to random weights."""
        ckpt_path = os.path.join(self.temp_dir, "epoch=0.ckpt")
        torch.save({"state_dict": {"emb.weight": torch.ones((10, 4))}}, ckpt_path)

        runner = MagicMock(spec=RunModel)
        runner._continue_model_dir = self.temp_dir
        runner._find_checkpoint_path.return_value = ckpt_path
        runner.device = torch.device("cpu")
        runner.model = DummyLegacyModel()  # emb has shape (5, 4) -> mismatch with (10, 4)

        with self.assertRaises(RuntimeError):
            RunModel._load_lightning_checkpoint_weights(runner)


class TestCalculateFantasyPointsUnscaling(unittest.TestCase):
    def test_unscaling_with_scaler_buffers(self):
        """Verify CalculateFantasyPoints unscales inputs back to physical units before computing points."""
        stat_names = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        params = CalculateFantasyPointsParams(
            input_tensor_name='outputs_pred',
            stat_names=stat_names,
            site=Sites.YAHOO,
            scaler_mean=[10.0, 5.0, 3.0, 1.0, 0.5, 2.0],
            scaler_scale=[2.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        )
        layer = CalculateFantasyPoints(params=params, input_shape=(1, 6))
        # Normalized input: PTS=1.0 -> 12.0 raw; REB=2.0 -> 7.0 raw; AST=-1.0 -> 2.5 raw; STL=0->1; BLK=0->0.5; TOV=0->2
        # Yahoo scoring: 12*1 + 7*1.2 + 2.5*1.5 + 1*3 + 0.5*3 + 2*(-1) = 26.65
        x = torch.tensor([[1.0, 2.0, -1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        fp = layer(x)
        self.assertTrue(torch.isclose(fp, torch.tensor([26.65]), atol=1e-4))

    def test_unscaling_fallback_when_buffers_none(self):
        """Verify CalculateFantasyPoints preserves backward compatibility when scalers are None."""
        stat_names = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        params = CalculateFantasyPointsParams(
            input_tensor_name='outputs_pred',
            stat_names=stat_names,
            site=Sites.YAHOO,
        )
        layer = CalculateFantasyPoints(params=params, input_shape=(1, 6))
        # Without scaler, raw calculation: 1*1 + 2*1.2 + 3*1.5 + 1*3 + 1*3 + 1*(-1) = 12.9
        x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        fp = layer(x)
        self.assertTrue(torch.isclose(fp, torch.tensor([12.9]), atol=1e-4))

    def test_legacy_unpickled_layer_without_scaler_attributes(self):
        """Verify legacy unpickled CalculateFantasyPoints without scaler_mean/scaler_scale attributes does not raise AttributeError."""
        stat_names = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
        params = CalculateFantasyPointsParams(
            input_tensor_name='outputs_pred',
            stat_names=stat_names,
            site=Sites.YAHOO,
        )
        layer = CalculateFantasyPoints(params=params, input_shape=(1, 6))
        delattr(layer, 'scaler_mean')
        delattr(layer, 'scaler_scale')
        x = torch.tensor([[1.0, 2.0, 3.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
        fp = layer(x)
        self.assertTrue(torch.isclose(fp, torch.tensor([12.9]), atol=1e-4))

    def test_collect_true_predicted_outputs_resolve_scaler_dir_respects_none(self):
        """Verify CollectTruePredictedOutputs._resolve_scaler_dir returns None when params.scaler_dir is None even if model_dir/scalers exists."""
        temp_dir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(temp_dir, "scalers"), exist_ok=True)
            cb_params = CollectTruePredictedOutputsParams(
                tensor_names_true=['true_label'],
                tensor_names_pred=['outputs_pred'],
                scaler_dir=None,
            )
            cb = CollectTruePredictedOutputs(params=cb_params, model_dir=temp_dir)
            self.assertIsNone(cb._resolve_scaler_dir())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestReshapeToSamplePerDayDynamic(unittest.TestCase):
    def test_dynamic_sizing_arbitrary_dates_and_games(self):
        """Verify ReshapeToSamplePerDay dynamically accommodates slates larger than max_num_games_per_day."""
        params = ReshapeToSamplePerDayParams(
            input_tensor_name=['game_pred', METADATA],
            sample_per_game_tensor_name='game_pred',
            metadata_name=METADATA,
            max_num_games_per_day=2,  # Configured small
            num_players_per_game=2,
            num_days_per_batch=1,
            date_ind_in_metadata=0,
        )
        layer = ReshapeToSamplePerDay(params=params, input_shape=(5, 2))

        # 5 games: 3 on date 100.0, 2 on date 200.0 (exceeds max_num_games_per_day=2 and num_days_per_batch=1)
        pred_tensor = torch.arange(10, dtype=torch.float32).view(5, 2)
        meta_tensor = torch.tensor([
            [[100.0], [100.0]],
            [[100.0], [100.0]],
            [[100.0], [100.0]],
            [[200.0], [200.0]],
            [[200.0], [200.0]],
        ], dtype=torch.float32)

        batch = {'game_pred': pred_tensor, METADATA: meta_tensor}
        reshaped = layer(batch)

        # Output shape should be (num_days >= 2, max_games >= 3, 2)
        self.assertGreaterEqual(reshaped.shape[0], 2)
        self.assertGreaterEqual(reshaped.shape[1], 3)
        self.assertEqual(reshaped.shape[2], 2)
        # Verify content intact
        self.assertTrue(torch.all(reshaped[0, 0] == pred_tensor[0]))
        self.assertTrue(torch.all(reshaped[0, 2] == pred_tensor[2]))
        self.assertTrue(torch.all(reshaped[1, 1] == pred_tensor[4]))


class TestOptimizeTeamGating(unittest.TestCase):
    def test_optimize_team_skipped_in_training_when_run_in_train_false(self):
        """Verify OptimizeTeam returns zeros during training when run_in_train=False without calling solver."""
        params = OptimizeTeamParams(
            input_tensor_name=['pred', 'true', 'salary', 'pos', 'date'],
            run_in_train=False,
        )
        layer = OptimizeTeam(params=params, input_shape=(1, 8))
        layer.train()  # In training mode

        dummy_tensor = torch.ones((1, 8), dtype=torch.float32)
        batch = {
            'fantasy_points_pred_sample_per_day': dummy_tensor,
            'fantasy_points_true_sample_per_day': dummy_tensor,
            'salary_sample_per_day': dummy_tensor,
            'position_sample_per_day': dummy_tensor,
            'date_sample_per_day': dummy_tensor,
        }

        with patch.object(layer.optimizer, 'run_inner') as mock_solve:
            out = layer(batch)
            mock_solve.assert_not_called()
            self.assertTrue(torch.all(out == 0.0))

    def test_optimize_team_fn_no_grad_skips_true_lp(self):
        """Verify OptimizeTeamFn does not solve true team when y_pred does not require grad."""
        points_pred = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32, requires_grad=False)
        points_true = torch.tensor([[5.0, 15.0, 25.0]], dtype=torch.float32)
        salaries = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)
        positions = torch.tensor([[1, 2, 3]], dtype=torch.float32)
        dates = torch.tensor([[20000.0, 20000.0, 20000.0]], dtype=torch.float32)

        mock_optimizer = MagicMock()
        mock_optimizer.get_results_boolean_array.return_value = np.array([[True, False, True]])

        num_to_pos = lambda arr: np.array(['PG', 'SG', 'SF'])

        out = OptimizeTeamFn.apply(points_pred, points_true, salaries, positions, mock_optimizer, num_to_pos, dates)
        self.assertEqual(out.shape, (1, 3))
        # run_inner should only be called once (for pred), NOT twice (never for true)
        self.assertEqual(mock_optimizer.run_inner.call_count, 1)


class TestDatesToEpochDays(unittest.TestCase):
    def test_dates_to_epoch_days_conversions(self):
        """Verify dates_to_epoch_days converts dates, datetimes, and strings to exact floating epoch days."""
        d = datetime.date(2026, 11, 10)
        dt = datetime.datetime(2026, 11, 10, 0, 0, 0)
        s = "2026-11-10"

        res_d = dates_to_epoch_days(d)
        res_dt = dates_to_epoch_days(dt)
        res_s = dates_to_epoch_days(s)
        res_list = dates_to_epoch_days([d, s])

        self.assertEqual(res_d[0], res_dt[0])
        self.assertEqual(res_d[0], res_s[0])
        self.assertEqual(len(res_list), 2)
        self.assertEqual(res_list[0], res_list[1])
        # Invert back to date
        recovered = pd.to_datetime(res_d, unit='D').date[0]
        self.assertEqual(recovered, d)


class TestRunModelPredictDataLoader(unittest.TestCase):
    def test_predict_passes_custom_data_loader_to_run_epoch(self):
        """Verify RunModel.predict passes custom data_loader through to _run_epoch."""
        runner = MagicMock(spec=RunModel)
        runner.use_lightning = False
        runner.model = MagicMock()
        runner._define_callbacks.return_value = MagicMock()
        runner._loss_function = MagicMock()
        runner._metrics = MagicMock()
        runner._define_metrics.return_value = MagicMock()
        runner._run_epoch.return_value = {}

        custom_loader = [MagicMock()]
        RunModel.predict(runner, data_split_type=DataSplitType.TRAIN, data_loader=custom_loader)
        runner._run_epoch.assert_called_once()
        self.assertEqual(runner._run_epoch.call_args.kwargs['data_loader'], custom_loader)


class TestUpdateCalculateFantasyPointsParams(unittest.TestCase):
    def test_update_populates_scalers(self):
        """Verify update_calculate_fantasy_points_params copies scaler values to layer params."""
        stat_names = ['PTS', 'REB', 'AST']
        cfp = CalculateFantasyPointsParams(input_tensor_name='pred', stat_names=stat_names)
        arch = type("Arch", (), {"layer_params_list": [cfp]})()
        params = type("Params", (), {"architecture_params": arch})()

        class DummyScaler:
            feature_names_in_ = np.array(['PTS', 'REB', 'AST'])
            mean_ = np.array([15.0, 7.0, 5.0])
            scale_ = np.array([3.0, 2.0, 1.0])

        combined = type("Combined", (), {
            "machine_learning_datasets_stacked": type("Stacked", (), {
                "datasets": {"true_label": type("DS", (), {"scaler": DummyScaler()})()}
            })()
        })()

        update_calculate_fantasy_points_params(params=params, dataset_combined=combined)
        self.assertEqual(cfp.scaler_mean, [15.0, 7.0, 5.0])
        self.assertEqual(cfp.scaler_scale, [3.0, 2.0, 1.0])


class TestDateFilteredDataLoader(unittest.TestCase):
    def test_create_date_filtered_data_loader(self):
        """Verify create_date_filtered_data_loader extracts only samples for the specified target date."""
        class MockStacked:
            def __init__(self):
                self.datasets = {'true_label': torch.arange(50)}
            def __len__(self):
                return 50

        class MockDS(MachineLearningDatasetCombinedFantasyBasketballPredictions):
            def __init__(self):
                self._original_index_to_date = np.array([20000.0] * 20 + [20001.0] * 30)
                self.machine_learning_datasets_stacked = MockStacked()
                self.training_params = type("TP", (), {"batch_size": 10, "num_days_per_batch": 1})()

            def _remove_unused_indices_by_keep_mask(self, indices):
                return indices

            def _get_data_loader_batch_kwargs(self, batch_size, shuffle, machine_learning_datasets_stacked_subset):
                return {"batch_size": batch_size, "shuffle": False}

        ds = MockDS()
        target_d = pd.to_datetime(20001.0, unit='D').date()
        loader = ds.create_date_filtered_data_loader([target_d], batch_size=10)
        self.assertEqual(len(loader.dataset), 30)

        missing_d = pd.to_datetime(29999.0, unit='D').date()
        with self.assertRaises(ValueError):
            ds.create_date_filtered_data_loader([missing_d], batch_size=10)


class TestKeyPaddingMaskAndAttention(unittest.TestCase):
    def test_build_safe_key_padding_mask(self):
        """Verify _build_safe_key_padding_mask produces correct booleans and avoids all-masked NaNs."""
        # Standard mask: 1 is real (False), 0 is dummy (True)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        pad_mask = _build_safe_key_padding_mask(mask)
        self.assertTrue(torch.equal(pad_mask, torch.tensor([[False, False, True, True]])))

        # 3D mask with trailing 1 dimension: (B, S, 1) -> (B, S)
        mask_3d = torch.tensor([[[1.0], [0.0], [1.0]]])
        pad_mask_3d = _build_safe_key_padding_mask(mask_3d)
        self.assertEqual(pad_mask_3d.shape, (1, 3))
        self.assertTrue(torch.equal(pad_mask_3d, torch.tensor([[False, True, False]])))

        # All-masked row protection: should NOT be all-True (prevents softmax -inf NaN)
        all_zeros = torch.tensor([[0.0, 0.0, 0.0]])
        safe_mask = _build_safe_key_padding_mask(all_zeros)
        self.assertFalse(safe_mask.all().item(), "All-masked sequence must be unmasked to prevent softmax NaN")

    def test_transformer_encoder_custom_with_mask(self):
        """Verify TransformerEncoderCustom accepts mask and outputs finite values."""
        params = TransformerEncoderCustomParams(
            input_tensor_name='x',
            d_model=32,
            n_heads=4,
            ffdim=64,
        )
        layer = TransformerEncoderCustom(params=params, input_shape=(10, 32))
        x = torch.randn(2, 10, 32)
        mask = torch.tensor([[1.0] * 8 + [0.0] * 2, [1.0] * 5 + [0.0] * 5])

        out = layer(x, mask=mask)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out).any())

        # Test dictionary input
        out_dict = layer({'x': x, 'mask': mask})
        self.assertEqual(out_dict.shape, (2, 10, 32))
        self.assertFalse(torch.isnan(out_dict).any())

    def test_split_attention_combine_with_dict_and_mask(self):
        """Verify SplitAttentionCombine splits mask into m1/m2, is invariant to dummy perturbation, and handles all-masked teams."""
        params = SplitAttentionCombineParams(
            input_tensor_name={'x': 'inputs', 'mask': 'real_player_mask'},
            d_model=32,
            n_heads=4,
            ffdim=64,
            n_bench_players=6,  # 5 + 6 = 11 players per team, 22 total
        )
        layer = SplitAttentionCombine(params=params, input_shape={'x': (22, 32), 'mask': (22, 1)})
        layer.eval()
        self.assertEqual(layer.output_shape(), (22, 32))

        x = torch.randn(2, 22, 32)
        # Team 1 has 9 real players, Team 2 has 10 real players
        mask = torch.tensor([
            [1.0] * 9 + [0.0] * 2 + [1.0] * 10 + [0.0] * 1,
            [1.0] * 11 + [1.0] * 8 + [0.0] * 3,
        ])
        out = layer({'x': x, 'mask': mask})
        self.assertEqual(out.shape, (2, 22, 32))
        self.assertFalse(torch.isnan(out).any())

        # Perturb masked dummy slots by +1000.0; real player outputs must be strictly invariant
        x_perturbed = x.clone()
        x_perturbed[mask == 0] += 1000.0
        out_perturbed = layer({'x': x_perturbed, 'mask': mask})
        real_positions = mask == 1
        self.assertTrue(torch.allclose(out[real_positions], out_perturbed[real_positions], atol=1e-5))

        # All-masked team (Team 1 all 0.0): must produce finite outputs without softmax NaNs
        mask_all_zero_team = torch.tensor([
            [0.0] * 11 + [1.0] * 10 + [0.0] * 1,
            [0.0] * 22,
        ])
        out_all_zero = layer({'x': x, 'mask': mask_all_zero_team})
        self.assertFalse(torch.isnan(out_all_zero).any())

    def test_split_attention_combine_wrapper_with_dict(self):
        """Verify SplitAttentionCombineWrapper passes mask through all inner layers."""
        params = SplitAttentionCombineWrapperParams(
            input_tensor_name={'x': 'inputs', 'mask': 'real_player_mask'},
            d_model=32,
            n_heads=4,
            ffdim=64,
            n_bench_players=6,
            num_layers=2,
        )
        wrapper = SplitAttentionCombineWrapper(params=params, input_shape={'x': (22, 32), 'mask': (22, 1)})
        self.assertEqual(wrapper.output_shape(), (22, 32))

        x = torch.randn(2, 22, 32)
        mask = torch.ones(2, 22)
        mask[:, -3:] = 0.0  # 3 dummy players
        out = wrapper({'x': x, 'mask': mask})
        self.assertEqual(out.shape, (2, 22, 32))
        self.assertFalse(torch.isnan(out).any())

    def test_transformer_encoder_pytorch_with_dict(self):
        """Verify TransformerEncoderPytorch extracts mask from dict, is invariant to dummy perturbation, and runs attention."""
        params = TransformerEncoderPytorchParams(
            input_tensor_name={'x': 'inputs', 'mask': 'real_player_mask'},
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            num_layers=1,
            norm_first=True,
        )
        layer = TransformerEncoderPytorch(params=params, input_shape={'x': (22, 32), 'mask': (22, 1)})
        layer.eval()
        self.assertEqual(layer.output_shape(), (22, 32))

        x = torch.randn(2, 22, 32)
        mask = torch.ones(2, 22)
        mask[:, -2:] = 0.0
        out = layer({'x': x, 'mask': mask})
        self.assertEqual(out.shape, (2, 22, 32))
        self.assertFalse(torch.isnan(out).any())

        x_perturbed = x.clone()
        x_perturbed[mask == 0] += 1000.0
        out_perturbed = layer({'x': x_perturbed, 'mask': mask})
        real_positions = mask == 1
        self.assertTrue(torch.allclose(out[real_positions], out_perturbed[real_positions], atol=1e-5))


class TestMaskedFeatureScaling(unittest.TestCase):
    def test_scale_features_with_masked_fill_value(self):
        """Verify _scale_features replaces masked entries with masked_fill_value instead of unscaled numbers."""
        scaler = StandardScaler()
        df = pd.DataFrame({
            'stat1': [10.0, 20.0, 30.0, -999.0],
            'stat2': [1.0, 2.0, 3.0, -999.0],
        })
        keep_mask = np.array([1, 1, 1, 0])
        # Fit scaler on non-masked rows
        scaler.fit(df.iloc[:3])

        # Without masked_fill_value: row 3 keeps raw -999.0
        res_raw = AbstractScalingMachineLearningDataset._scale_features(
            scaler=scaler, features_to_scale=df, keep_mask=keep_mask, masked_fill_value=None
        )
        self.assertEqual(res_raw.loc[3, 'stat1'], -999.0)

        # With masked_fill_value=0.0: row 3 gets 0.0
        res_filled = AbstractScalingMachineLearningDataset._scale_features(
            scaler=scaler, features_to_scale=df, keep_mask=keep_mask, masked_fill_value=0.0
        )
        self.assertEqual(res_filled.loc[3, 'stat1'], 0.0)
        self.assertEqual(res_filled.loc[3, 'stat2'], 0.0)
        # Scaled values on real rows are preserved
        self.assertAlmostEqual(res_filled.loc[0, 'stat1'], (10.0 - 20.0) / scaler.scale_[0])


class TestRegularizationAndWeightDecay(unittest.TestCase):
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(10, 8)
            self.linear = nn.Linear(8, 16)
            self.norm = nn.LayerNorm(16)
            self.head = nn.Linear(16, 2)

    def test_training_params_weight_decay(self):
        """Verify TrainingParams default weight_decay=1e-2 and validation."""
        tp = TrainingParams(learning_rate=0.001)
        self.assertEqual(tp.weight_decay, 1e-2)

        tp_custom = TrainingParams(learning_rate=0.001, weight_decay=1e-3)
        self.assertEqual(tp_custom.weight_decay, 1e-3)

        with self.assertRaises(ValueError):
            TrainingParams(learning_rate=0.001, weight_decay=-0.01)

    def test_uniform_weight_decay_in_lightning_and_legacy_optimizers(self):
        """Verify both Lightning and legacy optimizers apply weight_decay=1e-2 uniformly across all parameters."""
        model = self.SimpleModule()
        mock_params = MagicMock()
        mock_params.training_params.learning_rate = 0.001
        mock_params.training_params.weight_decay = 1e-2
        mock_params.training_params.callbacks_params = ()

        module = MachineLearningLightningModule(params=mock_params, model=model)
        optimizer = module.configure_optimizers()
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-2)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), len(list(model.parameters())))

        runner = MagicMock(spec=RunModel)
        runner.params = mock_params
        runner.model = model
        legacy_opt = RunModel._define_optimizer(runner)
        self.assertIsInstance(legacy_opt, torch.optim.AdamW)
        self.assertEqual(len(legacy_opt.param_groups), 1)
        self.assertEqual(legacy_opt.param_groups[0]["weight_decay"], 1e-2)

    def test_zero_weight_decay_is_explicitly_preserved(self):
        """Verify weight_decay=0.0 sets param_group weight_decay=0.0 instead of falling back to PyTorch's 0.01 default."""
        model = self.SimpleModule()
        mock_params = MagicMock()
        mock_params.training_params.learning_rate = 0.001
        mock_params.training_params.weight_decay = 0.0
        mock_params.training_params.callbacks_params = ()

        module = MachineLearningLightningModule(params=mock_params, model=model)
        optimizer = module.configure_optimizers()
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)

        runner = MagicMock(spec=RunModel)
        runner.params = mock_params
        runner.model = model
        legacy_opt = RunModel._define_optimizer(runner)
        self.assertEqual(len(legacy_opt.param_groups), 1)
        self.assertEqual(legacy_opt.param_groups[0]["weight_decay"], 0.0)


class TestZeroSampleAndKLLoss(unittest.TestCase):
    def test_zero_sample_guard_accumulate_false(self):
        mse_params = MSELossParams(
            tensor_names_pred="outputs_pred",
            tensor_names_true="true_label",
            tensor_names_keep_mask="keep_mask",
            name_for_logs="mse",
            reduction=Reduction.MEAN,
            accumulate=False,
        )
        loss_fn = MSELoss(params=mse_params)
        y_pred = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
        y_true = torch.tensor([[[0.0, 0.0]]])
        keep_mask = torch.zeros_like(y_true)

        loss = loss_fn(y_pred=y_pred, y_true=y_true, keep_mask=keep_mask)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(y_pred.grad)
        self.assertTrue(torch.isfinite(y_pred.grad).all())
        self.assertTrue(torch.all(y_pred.grad == 0.0))

    def test_zero_sample_guard_accumulate_true_and_reduction_none(self):
        mse_params = MSELossParams(
            tensor_names_pred="outputs_pred",
            tensor_names_true="true_label",
            tensor_names_keep_mask="keep_mask",
            name_for_logs="mse",
            reduction=Reduction.MEAN,
            reduction_dim=(1,),
            accumulate=True,
        )
        loss_fn = MSELoss(params=mse_params)

        # Batch 1: completely masked out -> forward yields NaN so accumulate ignores empty rows
        y_pred_empty = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
        y_true_empty = torch.zeros_like(y_pred_empty)
        keep_mask_empty = torch.zeros_like(y_pred_empty)
        out_empty = loss_fn(y_pred=y_pred_empty, y_true=y_true_empty, keep_mask=keep_mask_empty)
        self.assertTrue(torch.isnan(out_empty).all())

        # Batch 2: one valid row -> (2^2 + 4^2) / 2 = 10.0, second row empty -> NaN
        y_pred_valid = torch.tensor([[[2.0, 4.0], [5.0, 5.0]]])
        y_true_valid = torch.zeros_like(y_pred_valid)
        keep_mask_valid = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
        out_valid = loss_fn(y_pred=y_pred_valid, y_true=y_true_valid, keep_mask=keep_mask_valid)
        self.assertAlmostEqual(out_valid[0].item(), 10.0, places=5)
        self.assertTrue(torch.isnan(out_valid[1]))

        loss_fn.summarize()
        self.assertAlmostEqual(float(np.nanmean(loss_fn.epoch_results["mse"].values)), 10.0, places=5)

        mse_none = MSELoss(
            params=MSELossParams(
                tensor_names_pred="outputs_pred",
                tensor_names_true="true_label",
                tensor_names_keep_mask="keep_mask",
                name_for_logs="mse_none",
                reduction=Reduction.NONE,
                accumulate=False,
            )
        )
        out_none = mse_none(y_pred=y_pred_empty, y_true=y_true_empty, keep_mask=keep_mask_empty)
        self.assertEqual(out_none.shape, y_pred_empty.shape)
        self.assertTrue(torch.isnan(out_none).all())

    def test_fractional_sample_weights(self):
        mse_fn = MSELoss(
            params=MSELossParams(
                tensor_names_pred="outputs_pred",
                tensor_names_true="true_label",
                tensor_names_keep_mask="keep_mask",
                tensor_names_sample_weights="sample_weights",
                name_for_logs="mse",
                reduction=Reduction.MEAN,
                accumulate=False,
            )
        )
        y_pred = torch.tensor([[[3.0]]])
        y_true = torch.tensor([[[1.0]]])
        keep_mask = torch.ones_like(y_true)
        sample_weights = torch.tensor([[[0.25]]])
        loss = mse_fn(y_pred=y_pred, y_true=y_true, keep_mask=keep_mask, sample_weights=sample_weights)
        # (3 - 1)^2 * 0.25 / 0.25 == 4.0 (verifies > 0 guard does not clamp fractional weight to 1.0)
        self.assertAlmostEqual(loss.item(), 4.0, places=5)

    def test_kl_divergence_and_losses_combined_logs(self):
        p = torch.tensor([[[0.5, 0.0], [0.5, 1.0]]])
        q = torch.tensor([[[0.25, 0.2], [0.75, 0.8]]])
        log_q = torch.log(q)
        kl_mask = torch.ones_like(p)

        kl_fn = KLDivergenceLoss(
            params=KLDivergenceLossParams(
                tensor_names_pred="stage1_log_share",
                tensor_names_true="stage1_label",
                tensor_names_keep_mask="stage1_keep_mask",
                name_for_logs="stage1_kl",
                weight=10.0,
                column_names=["MIN_SHARE", "PLAYS_SHARE"],
                accumulate=False,
            )
        )
        kl_val = kl_fn(y_pred=log_q, y_true=p, keep_mask=kl_mask)
        expected_elements = torch.tensor([
            0.5 * np.log(0.5 / 0.25),
            0.0,
            0.5 * np.log(0.5 / 0.75),
            1.0 * np.log(1.0 / 0.8),
        ])
        self.assertAlmostEqual(kl_val.item(), float(expected_elements.mean()), places=5)

        for accumulate in (False, True):
            red_dim = (1,) if accumulate else None
            mse_params = MSELossParams(
                tensor_names_pred="outputs_pred",
                tensor_names_true="true_label",
                tensor_names_keep_mask="keep_mask",
                name_for_logs="mse",
                weight=1.0,
                reduction_dim=red_dim,
                accumulate=accumulate,
            )
            kl_params = KLDivergenceLossParams(
                tensor_names_pred="stage1_log_share",
                tensor_names_true="stage1_label",
                tensor_names_keep_mask="stage1_keep_mask",
                name_for_logs="stage1_kl",
                weight=10.0,
                reduction_dim=red_dim,
                accumulate=accumulate,
            )
            combined = LossesCombined((mse_params, kl_params), device=torch.device("cpu"), name="loss")
            tensors = {
                "outputs_pred": torch.tensor([[[2.0, 2.0], [1.0, 1.0]]]),
                "true_label": torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]),
                "keep_mask": torch.ones((1, 2, 2)),
                "stage1_log_share": log_q,
                "stage1_label": p,
                "stage1_keep_mask": kl_mask,
            }
            total = combined.forward(tensors)
            self.assertTrue(torch.isfinite(total).all())
            combined.summarize()
            logs = combined.get_logs()
            self.assertIn("loss", logs)
            self.assertIn("mse", logs)
            self.assertIn("stage1_kl", logs)
            if not accumulate:
                self.assertTrue(np.isfinite(logs["loss"]))
                self.assertAlmostEqual(float(logs["loss"]), float(logs["mse"]) + 10.0 * float(logs["stage1_kl"]), places=5)
            else:
                self.assertTrue(np.isfinite(logs["loss"].values).all())
                self.assertTrue(np.isfinite(logs["mse"].values).all())
                self.assertTrue(np.isfinite(logs["stage1_kl"].values).all())


class TestTeamAllocation(unittest.TestCase):
    @staticmethod
    def _make_layer(n_bench_players: int = 6) -> TeamAllocation:
        total_players = 2 * (5 + n_bench_players)
        params = TeamAllocationParams(
            input_tensor_name={"x": "stage1_raw", "mask": "real_player_mask"},
            output_tensor_name=["stage1_log_share", "stage1_alloc"],
            n_bench_players=n_bench_players,
        )
        return TeamAllocation(params=params, input_shape={"x": (total_players, 2), "mask": (total_players, 1)})

    def _assert_team_invariants(
        self,
        away_logits_min: list[float],
        n_valid_away: int,
        n_bench_players: int = 6,
        dtype: torch.dtype = torch.float32,
        atol: float = 1e-5,
    ):
        p = 5 + n_bench_players
        layer = self._make_layer(n_bench_players=n_bench_players)
        raw = torch.zeros(1, 2 * p, 2, dtype=dtype)
        mask = torch.zeros(1, 2 * p, 1, dtype=dtype)
        raw[0, : len(away_logits_min), 0] = torch.tensor(away_logits_min, dtype=dtype)
        raw[0, :n_valid_away, 1] = torch.linspace(2.0, -1.0, n_valid_away, dtype=dtype)
        mask[0, :n_valid_away, 0] = 1.0
        # Give Home team valid players too
        mask[0, p : p + 6, 0] = 1.0
        raw[0, p : p + 6, 0] = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0], dtype=dtype)

        out = layer({"x": raw, "mask": mask})
        alloc = out["stage1_alloc"]
        self.assertFalse(alloc.requires_grad)

        away_alloc = alloc[0, :p, :].float()
        away_valid = away_alloc[:n_valid_away, :]
        away_dummy = away_alloc[n_valid_away:, :]

        # Masked dummy players must be exact zero
        self.assertTrue(torch.all(away_dummy == 0.0))
        # Per-team sums equal 5.0 (for both MIN_ALLOC and PLAYS_ALLOC)
        self.assertAlmostEqual(away_valid[:, 0].sum().item(), 5.0, delta=atol)
        self.assertAlmostEqual(away_valid[:, 1].sum().item(), 5.0, delta=atol)

        expected_cap_alloc = 5.0 * max(1.0 / n_valid_away, 0.20)
        if n_valid_away >= 5:
            self.assertTrue(torch.all(away_valid[:, 0] <= 1.0 + max(1e-6, atol)))
        else:
            self.assertTrue(torch.all(away_valid[:, 0] <= expected_cap_alloc + max(1e-6, atol)))

    def test_kl_gradient_nonzero_when_saturated_and_zero_cross_team(self):
        layer = self._make_layer(n_bench_players=6)
        raw = torch.zeros(1, 22, 2, requires_grad=True)
        with torch.no_grad():
            # Player 0 has unconstrained MIN softmax share > 0.20
            raw_init = torch.zeros(1, 22, 2)
            raw_init[0, 0, 0] = 5.0
            raw_init[0, 11:16, :] = 1.0
        raw = raw_init.clone().detach().requires_grad_(True)
        mask = torch.ones(1, 22, 1)

        out = layer({"x": raw, "mask": mask})
        log_share = out["stage1_log_share"]
        alloc = out["stage1_alloc"]
        self.assertFalse(alloc.requires_grad)

        target = torch.full((1, 22, 2), 1.0 / 11.0)
        kl = torch.mean(torch.xlogy(target, target) - target * log_share)
        kl.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(abs(raw.grad[0, 0, 0].item()), 1e-4)

        # Cross-team invariance
        raw_perturbed = raw_init.clone()
        raw_perturbed[0, :11, :] += 7.5
        out_base = layer({"x": raw_init, "mask": mask})
        out_pert = layer({"x": raw_perturbed, "mask": mask})
        self.assertTrue(torch.equal(out_base["stage1_log_share"][:, 11:, :], out_pert["stage1_log_share"][:, 11:, :]))
        self.assertTrue(torch.equal(out_base["stage1_alloc"][:, 11:, :], out_pert["stage1_alloc"][:, 11:, :]))

    def test_capped_simplex_projection_edge_cases(self):
        # Star-heavy logits
        self._assert_team_invariants([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0], n_valid_away=8)
        # 6-man cascade: 4 saturate 0.20 and 5th exceeds 0.20 on first redistribution
        self._assert_team_invariants([10.0, 10.0, 10.0, 10.0, 3.0, 0.0], n_valid_away=6)
        # Extreme z_min: [+40*4, -40]
        self._assert_team_invariants([40.0, 40.0, 40.0, 40.0, -40.0], n_valid_away=5)
        # 13 valid players: [30*4, 7*9]
        self._assert_team_invariants([30.0] * 4 + [7.0] * 9, n_valid_away=13, n_bench_players=10)
        # Fewer than 5 valid players: 3 valid players -> cap == 5/3, sum == 5.0
        self._assert_team_invariants([5.0, 0.0, -5.0], n_valid_away=3)
        # float16 logits producing unconstrained [~0.25*4, 3e-7, ~0.0]
        self._assert_team_invariants(
            [0.0, 0.0, 0.0, 0.0, -13.6, -20.0],
            n_valid_away=6,
            dtype=torch.float16,
            atol=5e-3,
        )
        self._assert_team_invariants(
            [0.0, 0.0, 0.0, 0.0, -13.6, -100.0],
            n_valid_away=6,
            dtype=torch.float16,
            atol=5e-3,
        )


class TestStage1PostGridTargets(unittest.TestCase):
    def test_add_additional_columns_computes_team_shares_and_exit_masks(self):
        params = create_params(two_stage=True).features_params
        positions = START_POSITIONS + [f"BN{i}" for i in range(1, 7)]

        rows = []
        # Away team (HOME=0): 6 real players, player 1 left midgame
        away_mins = [30.0, 10.0, 40.0, 40.0, 40.0, 80.0]
        away_fga = [10.0, 4.0, 12.0, 8.0, 6.0, 20.0]
        away_fta = [0.0, 0.0, 5.0, 0.0, 5.0, 0.0]
        away_tov = [1.0, 1.0, 2.0, 1.0, 1.0, 4.0]
        for idx in range(len(positions)):
            is_real = idx < len(away_mins)
            rows.append({
                "DATE": datetime.date(2024, 1, 10),
                "SEASON_YEAR": 2023,
                "GAME_ID": 1001,
                "HOME": 0.0,
                "PLAYER_ID": 100 + idx if is_real else UNKNOWN_PLAYER_ID,
                "MIN": away_mins[idx] if is_real else 0.0,
                "FGA": away_fga[idx] if is_real else 0.0,
                "FTA": away_fta[idx] if is_real else 0.0,
                "TOV": away_tov[idx] if is_real else 0.0,
                "EJECTED": False,
                "LEFT_MIDGAME": is_real and idx == 1,
            })

        # Home team (HOME=1): 5 real players, no exits/ejections
        home_mins = [48.0, 48.0, 48.0, 48.0, 48.0]
        home_fga = [15.0, 10.0, 10.0, 5.0, 10.0]
        home_fta = [5.0, 0.0, 0.0, 0.0, 0.0]
        home_tov = [2.8, 1.0, 1.0, 1.0, 2.0]
        for idx in range(len(positions)):
            is_real = idx < len(home_mins)
            rows.append({
                "DATE": datetime.date(2024, 1, 10),
                "SEASON_YEAR": 2023,
                "GAME_ID": 1001,
                "HOME": 1.0,
                "PLAYER_ID": 200 + idx if is_real else UNKNOWN_PLAYER_ID,
                "MIN": home_mins[idx] if is_real else 0.0,
                "FGA": home_fga[idx] if is_real else 0.0,
                "FTA": home_fta[idx] if is_real else 0.0,
                "TOV": home_tov[idx] if is_real else 0.0,
                "EJECTED": False,
                "LEFT_MIDGAME": False,
            })

        df = pd.DataFrame(rows)
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_additional_columns(df, params)

        away = res[res["HOME"] == 0.0]
        home = res[res["HOME"] == 1.0]
        away_real = away[away["PLAYER_ID"] != UNKNOWN_PLAYER_ID]
        away_dummy = away[away["PLAYER_ID"] == UNKNOWN_PLAYER_ID]
        home_real = home[home["PLAYER_ID"] != UNKNOWN_PLAYER_ID]
        home_dummy = home[home["PLAYER_ID"] == UNKNOWN_PLAYER_ID]

        # Shares sum to 1.0 per team over real players and 0.0 on dummy rows
        self.assertAlmostEqual(float(away_real["MIN_SHARE"].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(away_real["PLAYS_SHARE"].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(home_real["MIN_SHARE"].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(home_real["PLAYS_SHARE"].sum()), 1.0, places=5)
        self.assertTrue((away_dummy["MIN_SHARE"] == 0.0).all())
        self.assertTrue((away_dummy["PLAYS_SHARE"] == 0.0).all())
        self.assertTrue((home_dummy["MIN_SHARE"] == 0.0).all())
        self.assertTrue((home_dummy["PLAYS_SHARE"] == 0.0).all())

        # Any exit/ejection masks the entire team's STAGE1_NOT_IGNORE, without affecting opposing clean team
        self.assertTrue((away["STAGE1_NOT_IGNORE"] == IGNORE_VAL).all())
        self.assertTrue((home_real["STAGE1_NOT_IGNORE"] == 1).all())
        self.assertTrue((home_dummy["STAGE1_NOT_IGNORE"] == IGNORE_VAL).all())


if __name__ == "__main__":
    unittest.main()
