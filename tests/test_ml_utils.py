import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from machine_learning.enums import Reduction
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning.metrics.abstract_metrics import AbstractMetric
from machine_learning.params.abstract_params import AbstractMetricParams
from machine_learning.utils import get_next_model_id
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import (
    MachineLearningDatasetCombinedFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.params.params import (
    FeaturesParamsFantasyBasketballPredictions,
)


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
        from utils import convert_date_col_to_date
        s = pd.Series([19676.0, 19677.0], dtype="float32")
        converted = convert_date_col_to_date(s)
        self.assertEqual(converted.iloc[0], datetime.date(2023, 11, 15))
        self.assertEqual(converted.iloc[1], datetime.date(2023, 11, 16))

    def test_convert_date_col_from_seconds_int64(self):
        from utils import convert_date_col_to_date
        s = pd.Series([1700006400, 1700092800], dtype="int64")
        converted = convert_date_col_to_date(s)
        self.assertEqual(converted.iloc[0], datetime.date(2023, 11, 15))
        self.assertEqual(converted.iloc[1], datetime.date(2023, 11, 16))

    def test_convert_date_col_from_strings(self):
        from utils import convert_date_col_to_date
        s = pd.Series(["2023-11-15", "2023-11-16"])
        converted = convert_date_col_to_date(s)
        self.assertEqual(converted.iloc[0], datetime.date(2023, 11, 15))
        self.assertEqual(converted.iloc[1], datetime.date(2023, 11, 16))

    def test_metadata_tensor_to_df_converts_date_days(self):
        from machine_learning.machine_learning_datasets import MachineLearningDatasetMetadata
        tensor = torch.tensor([[19676.0, 1.0]], dtype=torch.float32)
        df = MachineLearningDatasetMetadata.tensor_to_df(tensor, column_names=["DATE", "split_type"])
        self.assertEqual(df["DATE"].iloc[0], datetime.date(2023, 11, 15))


if __name__ == "__main__":
    unittest.main()
