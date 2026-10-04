import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pytz

from run_full_pipeline import RunFullPipeline
from constants import DATA_DIR, NY_TZ
from utils import get_current_date, get_current_season


class TestRunFullPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_defaults_and_safe_params_dict(self):
        """Verify RunFullPipeline initializes with default parameters without raising AttributeError."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )

        self.assertGreaterEqual(pipeline.season, 2026)
        self.assertIsInstance(pipeline._model_params_update_dict, dict)
        self.assertIn("training", pipeline._model_params_update_dict)
        self.assertFalse(pipeline._model_params_update_dict["training"]["run_extra_callbacks"])
        self.assertIsNotNone(pipeline.stats_loaders_players)
        self.assertIsNotNone(pipeline.stats_loaders_teams)
        self.assertIsNotNone(pipeline.calc_features)
        self.assertEqual(pipeline.dates_to_game_times, {})

    def test_init_custom_params_update_dict(self):
        """Verify custom params_update_dict is preserved while enforcing run_extra_callbacks=False."""
        custom_dict = {
            "training": {"num_epochs": 20, "learning_rate": 0.001},
            "features": {"scale": True}
        }
        pipeline = RunFullPipeline(
            load_params_from_model_id=42,
            params_update_dict=custom_dict,
            run_immediately=True,
            initialize_daily_job=False
        )

        self.assertEqual(pipeline._model_params_update_dict["training"]["num_epochs"], 20)
        self.assertEqual(pipeline._model_params_update_dict["training"]["learning_rate"], 0.001)
        self.assertFalse(pipeline._model_params_update_dict["training"]["run_extra_callbacks"])
        self.assertEqual(pipeline._model_params_update_dict["features"]["scale"], True)

    def test_reset_run_date_resolution(self):
        """Verify _reset correctly sets _run_date from date object or 'today' string."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )

        with patch.object(pipeline, "run_once_per_day"), patch.object(pipeline, "_wait_until_certain_time"):
            # Explicit date
            test_date = datetime.date(2026, 11, 15)
            pipeline._reset(test_date)
            self.assertEqual(pipeline._run_date, test_date)

            # String 'today'
            pipeline._reset("today")
            self.assertEqual(pipeline._run_date, get_current_date())

    def test_has_games_today(self):
        """Verify _has_games_today returns True when games are scheduled for _run_date."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )
        today = get_current_date()
        pipeline._run_date = today

        # No games scheduled
        pipeline.dates_to_game_times = {}
        self.assertFalse(pipeline._has_games_today())

        # Games scheduled
        pipeline.dates_to_game_times = {today: [datetime.datetime.now(NY_TZ)]}
        self.assertTrue(pipeline._has_games_today())

    def test_todays_model_already_trained(self):
        """Verify _todays_model_already_trained detects model.pt and .ckpt files in model directory."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )
        test_date = datetime.date(2026, 10, 25)
        pipeline._run_date = test_date

        target_model_name = f"model_{str(test_date).replace('-', '_')}"
        models_base_dir = os.path.join(self.temp_dir, "machine_learning_fantasy_basketball_predictions", "models")
        target_model_dir = os.path.join(models_base_dir, target_model_name)

        with patch("run_full_pipeline.DATA_DIR", self.temp_dir):
            # 1. Directory does not exist -> False
            self.assertFalse(pipeline._todays_model_already_trained())

            # 2. Directory exists but empty -> False
            os.makedirs(target_model_dir, exist_ok=True)
            self.assertFalse(pipeline._todays_model_already_trained())

            # 3. model.pt exists -> True
            pt_file = os.path.join(target_model_dir, "model.pt")
            with open(pt_file, "w") as f:
                f.write("model")
            self.assertTrue(pipeline._todays_model_already_trained())

            # 4. Checkpoint exists -> True
            os.remove(pt_file)
            ckpt_file = os.path.join(target_model_dir, "epoch_1.ckpt")
            with open(ckpt_file, "w") as f:
                f.write("ckpt")
            self.assertTrue(pipeline._todays_model_already_trained())

    def test_wait_until_certain_time_past_time_does_not_sleep(self):
        """Verify _wait_until_certain_time does not sleep if the target time is already in the past."""
        pipeline = RunFullPipeline(load_params_from_model_id=1, initialize_daily_job=False)
        past_time = datetime.datetime.now(NY_TZ) - datetime.timedelta(hours=2)

        with patch("run_full_pipeline.time.sleep") as mock_sleep:
            pipeline._wait_until_certain_time(past_time)
            mock_sleep.assert_not_called()

    def test_browser_profile_path_returns_str_or_none(self):
        """Verify _browser_profile_path returns either a valid path string or None without raising an exception."""
        result = RunFullPipeline._browser_profile_path()
        self.assertTrue(result is None or isinstance(result, str))

    @patch("run_full_pipeline.scrape_season_schedules")
    @patch("run_full_pipeline.load_and_cache_players_list")
    @patch("run_full_pipeline.RunModelFantasyBasketballPredictions")
    @patch("run_full_pipeline.MachineLearningParamsFantasyBasketballPredictions.load_params")
    def test_run_once_per_day_execution_flow(
        self,
        mock_load_params,
        mock_run_model_cls,
        mock_load_players,
        mock_scrape_schedules
    ):
        """Verify run_once_per_day executes data collection, feature calculation, and model training in order."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )
        today = get_current_date()
        pipeline._run_date = today

        # Mock schedule and game times
        mock_game_time = datetime.datetime.now(NY_TZ) + datetime.timedelta(hours=3)
        pipeline._get_game_times_by_day_dict = MagicMock(return_value={today: [mock_game_time]})

        # Mock loaders & features
        pipeline.stats_loaders_players.call = MagicMock()
        pipeline.stats_loaders_teams.call = MagicMock()
        pipeline.calc_features.call = MagicMock()

        # Mock players list
        mock_players = pd.DataFrame({"NAME": ["Player A"], "PLAYER_ID": [100]})
        mock_load_players.return_value = mock_players

        # Mock ML params and runner
        mock_params = MagicMock()
        mock_params.features_params = MagicMock()
        mock_load_params.return_value = mock_params

        mock_runner = MagicMock()
        mock_runner._model_dir = os.path.join(self.temp_dir, "temp_model_dir")
        os.makedirs(mock_runner._model_dir, exist_ok=True)
        mock_run_model_cls.return_value = mock_runner

        with patch.object(pipeline, "_todays_model_already_trained", return_value=False):
            pipeline.run_once_per_day()

        # Assert correct flow
        mock_scrape_schedules.assert_called_once()
        mock_load_players.assert_called_once_with(season=pipeline.season)
        pipeline.stats_loaders_players.call.assert_called_once_with(
            players_list=mock_players, add_upcoming_games=False
        )
        pipeline.stats_loaders_teams.call.assert_called_once()
        pipeline.calc_features.call.assert_called_once_with(
            players_list=mock_players, add_upcoming_games=False
        )
        mock_runner.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
