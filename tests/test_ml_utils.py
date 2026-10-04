import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning.utils import get_next_model_id


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


if __name__ == "__main__":
    unittest.main()
