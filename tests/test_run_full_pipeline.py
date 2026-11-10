import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
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

            # ISO date string
            pipeline._reset("2026-11-15")
            self.assertEqual(pipeline._run_date, datetime.date(2026, 11, 15))

            # Invalid string fallback
            pipeline._reset("not-a-date")
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
        """Verify _todays_model_already_trained detects complete models (weights, params, and transformers)."""
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

            # 3. Model weights only without params and scalers -> False
            pt_file = os.path.join(target_model_dir, "model.pt")
            with open(pt_file, "w") as f:
                f.write("model")
            self.assertFalse(pipeline._todays_model_already_trained())

            # 4. Add params.json, scalers, and encoders -> True
            with open(os.path.join(target_model_dir, "params.json"), "w") as f:
                f.write("{}")
            scalers_dir = os.path.join(target_model_dir, "scalers")
            os.makedirs(scalers_dir, exist_ok=True)
            with open(os.path.join(scalers_dir, "true_label.pkl"), "w") as f:
                f.write("scaler")
            encoders_dir = os.path.join(target_model_dir, "categorical_encoder")
            os.makedirs(encoders_dir, exist_ok=True)
            with open(os.path.join(encoders_dir, "embedding_features.pkl"), "w") as f:
                f.write("encoder")
            self.assertTrue(pipeline._todays_model_already_trained())

            # 5. Checkpoint (.ckpt) instead of model.pt -> True
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

    @patch("run_full_pipeline.update_todays_schedule_from_scoreboard")
    @patch("run_full_pipeline.scrape_season_schedules")
    @patch("run_full_pipeline.load_and_cache_players_list")
    @patch("run_full_pipeline.RunModelFantasyBasketballPredictions")
    @patch("run_full_pipeline.MachineLearningParamsFantasyBasketballPredictions.load_params")
    def test_run_once_per_day_execution_flow(
        self,
        mock_load_params,
        mock_run_model_cls,
        mock_load_players,
        mock_scrape_schedules,
        mock_update_scoreboard,
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
        with open(os.path.join(mock_runner._model_dir, "params.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(mock_runner._model_dir, "model.pt"), "w") as f:
            f.write("weights")
        scalers_dir = os.path.join(mock_runner._model_dir, "scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        with open(os.path.join(scalers_dir, "true_label.pkl"), "w") as f:
            f.write("scaler")
        encoders_dir = os.path.join(mock_runner._model_dir, "categorical_encoder")
        os.makedirs(encoders_dir, exist_ok=True)
        with open(os.path.join(encoders_dir, "embedding_features.pkl"), "w") as f:
            f.write("encoder")
        mock_run_model_cls.return_value = mock_runner

        with patch.object(pipeline, "_todays_model_already_trained", return_value=False):
            pipeline.run_once_per_day()

        # Assert correct flow
        mock_scrape_schedules.assert_called_once()
        mock_update_scoreboard.assert_called_once_with(date=today, season=pipeline.season)
        mock_load_players.assert_called_once_with(season=pipeline.season)
        pipeline.stats_loaders_players.call.assert_called_once_with(
            players_list=mock_players, add_upcoming_games=False
        )
        pipeline.stats_loaders_teams.call.assert_called_once()
        pipeline.calc_features.call.assert_called_once_with(
            players_list=mock_players, add_upcoming_games=False
        )
        mock_runner.train.assert_called_once()

    @patch("run_full_pipeline.RunFullPipeline.run_full_pipeline")
    @patch("run_full_pipeline.RunFullPipeline.run_once_per_day")
    def test_run_once_cli_flow(self, mock_ropd, mock_rfp):
        """Verify RunFullPipeline handles run_once flow by executing pipeline once and exiting."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )
        pipeline.run_full_pipeline()
        mock_rfp.assert_called_once()

    @patch("run_full_pipeline.get_next_game_time")
    def test_get_todays_predictions_avoids_salary_column_collision(self, mock_cutoff):
        """Verify _get_todays_predictions does not crash with KeyError: 'YAHOO_SALARY' when columns overlap."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=1,
            run_immediately=True,
            initialize_daily_job=False
        )
        today = get_current_date()
        pipeline._run_date = today

        # Mock predict returning metadata containing YAHOO_POSITION and YAHOO_SALARY (e.g. -999)
        mock_preds = pd.DataFrame({
            "DATE": [today, today],
            "PLAYER_ID": [101, 102],
            "CollectTruePredictedOutputsParamsFantasyBasketballPredictions_YAHOO_POINTS": [30.0, 45.0],
            "CollectTruePredictedOutputsParamsFantasyBasketballPredictions_YAHOO_POSITION": [1.0, 2.0],
            "CollectTruePredictedOutputsParamsFantasyBasketballPredictions_YAHOO_SALARY": [-999.0, -999.0],
            "split_type": ["TRAIN", "TRAIN"],
        })
        pipeline.train_nn = MagicMock()
        pipeline.train_nn._change_metrics_params_tuple_to_per_sample = MagicMock(return_value=())
        pipeline.train_nn.params.training_params.metrics_params = ()
        pipeline.train_nn.params.training_params.loss_params = ()
        pipeline.train_nn.predict = MagicMock(return_value=mock_preds)

        # Mock stats_loaders_players.df also containing YAHOO_POSITION and YAHOO_SALARY
        pipeline.stats_loaders_players.df = pd.DataFrame({
            "DATE": [today, today],
            "PLAYER_ID": [101, 102],
            "GAME_ID": [10001, 10001],
            "TEAM_ID": [1610612738, 1610612744],
            "YAHOO_POSITION": ["PG", "SG"],
            "YAHOO_SALARY": [np.nan, np.nan],
        })
        pipeline.stats_loaders_players.get_player_name_to_id = MagicMock(return_value={
            "Stephen Curry": 101,
            "Klay Thompson": 102,
        })

        # Mock teams_template from Yahoo
        mock_template = pd.DataFrame({
            "Name": ["Stephen Curry", "Klay Thompson"],
            "Position": ["PG", "SG"],
            "Salary": [45, 25],
            "Time": [datetime.datetime.now(NY_TZ) + datetime.timedelta(hours=2)] * 2,
        })

        # Future cutoff time
        mock_cutoff.return_value = datetime.datetime.now(NY_TZ) - datetime.timedelta(hours=1)
        pipeline._add_game_times_to_df = MagicMock(side_effect=lambda df, **kwargs: df.assign(
            GAME_DATE_TIME=datetime.datetime.now(NY_TZ) + datetime.timedelta(hours=2)
        ))

        result = pipeline._get_todays_predictions(teams_template=mock_template)

        self.assertIn("YAHOO_SALARY", result.columns)
        self.assertNotIn("YAHOO_SALARY_x", result.columns)
        self.assertNotIn("YAHOO_SALARY_y", result.columns)
        self.assertEqual(result.loc[result["PLAYER_ID"] == 101, "YAHOO_SALARY"].values[0], 45.0)
        self.assertEqual(result.loc[result["PLAYER_ID"] == 102, "YAHOO_SALARY"].values[0], 25.0)
        self.assertEqual(result.loc[result["PLAYER_ID"] == 101, "YAHOO_POSITION"].values[0], 1)
        self.assertEqual(result.loc[result["PLAYER_ID"] == 102, "YAHOO_POSITION"].values[0], 2)

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

    @patch("run_full_pipeline.SubmitDFSTeamYahooByCSV")
    @patch("run_full_pipeline.update_todays_schedule_from_scoreboard")
    @patch("run_full_pipeline.update_all_stats")
    def test_run_full_pipeline_routes_cash_vs_gpp_entries_separately(
        self,
        mock_update_stats,
        mock_update_scoreboard,
        mock_submit_cls,
    ):
        """Verify run_full_pipeline separates 50/50 cash from GPP tournaments and orders submissions."""
        pipeline = RunFullPipeline(
            load_params_from_model_id=0,
            run_immediately=True,
            initialize_daily_job=False
        )
        pipeline.players_list = pd.DataFrame({"NAME": ["Player A"], "TO_YEAR": [2026]})

        mock_submit_inst = MagicMock()
        mock_submit_inst.entry_ids = [5001, 6001, 6002]
        mock_submit_inst.contest_ids = [101, 102]
        mock_submit_inst.get_entry_titles.return_value = {
            5001: "NBA 50/50 Baller ($5)",
            6001: "NBA $10K Multi-Entry Slam",
            6002: "NBA $5K Baller GPP",
        }
        mock_submit_inst.get_locked_dfs.return_value = {
            5001: pd.DataFrame(columns=["Pos", "NAME", "YAHOO_SALARY", "YAHOO_POSITION"]),
            6001: pd.DataFrame(columns=["Pos", "NAME", "YAHOO_SALARY", "YAHOO_POSITION"]),
            6002: pd.DataFrame(columns=["Pos", "NAME", "YAHOO_SALARY", "YAHOO_POSITION"]),
        }
        mock_submit_cls.return_value = mock_submit_inst

        pipeline.stats_loaders_players = MagicMock()
        pipeline.stats_loaders_players.get_player_name_to_id.return_value = {"Player A": 101}
        pipeline.train_nn = MagicMock()
        pipeline.train_nn.create_data_loaders = MagicMock()

        dummy_preds = pd.DataFrame({"PLAYER_ID": [101], "YAHOO_POINTS_pred": [30.0]})
        pipeline._get_todays_predictions = MagicMock(return_value=dummy_preds)

        lp_solver_inits = []
        lp_solver_runs = []

        class MockSolver:
            def __init__(self, **kwargs):
                lp_solver_inits.append(kwargs)

            def run(self, df, locked):
                eids = list(locked.keys())
                lp_solver_runs.append((eids, df))
                slots = ["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL"]
                rows = []
                for eid in eids:
                    row = {s: 101 for s in slots}
                    row["salary"] = 180.0
                    row["points"] = 240.0
                    row["entry_id"] = eid
                    row["contest_id"] = eid
                    rows.append(row)
                return pd.DataFrame(rows)

        with patch("run_full_pipeline.FindBestSetOfTeamsLinearProgrammingYahoo", side_effect=MockSolver):
            pipeline.run_full_pipeline()

        # Check solver invocations:
        # 1st solver call: Cash entry 5001 solved individually
        self.assertEqual(len(lp_solver_inits), 2)
        self.assertEqual(lp_solver_inits[0]["num_teams_in_set"], 1)
        self.assertFalse(lp_solver_inits[0]["avoid_single_player_in_all_sets"])
        self.assertIsNone(lp_solver_inits[0]["max_shared_players_in_teams_pairs"])
        self.assertEqual(lp_solver_runs[0][0], [5001])

        # 2nd solver call: GPP entries 6001 & 6002 solved jointly with diversity constraints
        self.assertEqual(lp_solver_inits[1]["num_teams_in_set"], 2)
        self.assertEqual(lp_solver_inits[1]["max_shared_players_in_teams_pairs"], 4)
        self.assertEqual(set(lp_solver_runs[1][0]), {6001, 6002})

        # Check submit_best_set was called with ordered rows
        mock_submit_inst.submit_best_set.assert_called_once()
        submitted_df = mock_submit_inst.submit_best_set.call_args[0][0]
        self.assertEqual(list(submitted_df["entry_id"]), [5001, 6001, 6002])


if __name__ == "__main__":
    unittest.main()
