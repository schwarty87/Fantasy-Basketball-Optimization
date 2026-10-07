import datetime
import unittest
from unittest.mock import patch
import pandas as pd
from data_collectors.stats_loaders import (
    StatsLoader,
    StatsLoaderPlayers,
    StatsLoaderTeams,
    _coalesce_merge_columns,
)
from utils import assert_valid_df_all, assert_valid_df_players, downcast_df


class TestStatsLoaders(unittest.TestCase):

    def test_coalesce_merge_columns_basic(self):
        """Verify _coalesce_merge_columns fills NaNs in base columns and drops suffixed columns."""
        df = pd.DataFrame({
            "NAME": ["LeBron James", None, "Stephen Curry"],
            "NAME_x": ["LeBron J.", "Luka Doncic", "Steph C."],
            "PTS": [25.0, 30.0, None],
            "PTS_x": [20.0, 28.0, 35.0],
            "TEAM": ["LAL", "DAL", "GSW"],
        })
        result = _coalesce_merge_columns(df, suffix="_x")
        self.assertNotIn("NAME_x", result.columns)
        self.assertNotIn("PTS_x", result.columns)
        self.assertEqual(result["NAME"].tolist(), ["LeBron James", "Luka Doncic", "Stephen Curry"])
        self.assertEqual(result["PTS"].tolist(), [25.0, 30.0, 35.0])

    def test_coalesce_merge_columns_preserves_names_ending_in_x(self):
        """Verify _coalesce_merge_columns does not truncate column names ending with 'x' (unlike .strip('_x'))."""
        df = pd.DataFrame({
            "box": [1, None],
            "box_x": [10, 20],
            "complex": [None, 3],
            "complex_x": [5, 6],
            "max": [None, None],
            "max_x": [100, 200],
        })
        result = _coalesce_merge_columns(df, suffix="_x")
        self.assertIn("box", result.columns)
        self.assertIn("complex", result.columns)
        self.assertIn("max", result.columns)
        self.assertNotIn("bo", result.columns)
        self.assertNotIn("comple", result.columns)
        self.assertNotIn("ma", result.columns)
        self.assertEqual(result["box"].tolist(), [1, 20])
        self.assertEqual(result["complex"].tolist(), [5, 3])
        self.assertEqual(result["max"].tolist(), [100, 200])

    def test_coalesce_merge_columns_new_base_column(self):
        """Verify suffixed column becomes a base column if the base column did not originally exist."""
        df = pd.DataFrame({
            "A": [1, 2],
            "NEW_COL_x": [10, 20],
        })
        result = _coalesce_merge_columns(df, suffix="_x")
        self.assertIn("NEW_COL", result.columns)
        self.assertNotIn("NEW_COL_x", result.columns)
        self.assertEqual(result["NEW_COL"].tolist(), [10, 20])

    def test_coalesce_merge_columns_no_suffix(self):
        """Verify DataFrame is unchanged when no suffixed columns exist."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = _coalesce_merge_columns(df, suffix="_x")
        self.assertEqual(list(result.columns), ["A", "B"])

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_load_saved_data_empty_safe(self, mock_schedule):
        """Verify load_saved_data returns empty DataFrame cleanly if file does not exist."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)
        with patch.object(StatsLoader, "load_saved_data", return_value=None):
            df = loader.load_saved_data()
            self.assertTrue(df.empty)

    @patch("data_collectors.stats_loaders.os.replace")
    @patch("pandas.DataFrame.to_feather")
    @patch("data_collectors.stats_loaders.load_schedule")
    def test_save_df_available_flag_boolean(self, mock_schedule, mock_to_feather, mock_replace):
        """Verify _save_df properly converts AVAILABLE_FLAG to boolean without .value_counts() corruption."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)
        loader.updates_df = pd.DataFrame({"GAME_ID": [1, 2], "TEAM_ID": [10, 10]})
        loader.df = pd.DataFrame({
            "DATE": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
            "GAME_ID": [1, 2],
            "TEAM_ID": [10, 10],
            "PLAYER_ID": [100, 101],
            "AVAILABLE_FLAG": [1, 0],
        })
        with patch.object(loader, "assert_valid_df"):
            loader._save_df()
            self.assertEqual(loader.df["AVAILABLE_FLAG"].dtype, bool)
            self.assertEqual(loader.df["AVAILABLE_FLAG"].tolist(), [True, False])

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_double_double_and_triple_double_computation(self, mock_schedule):
        """Verify double-double and triple-double calculation in player game logs."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)

        mock_data = pd.DataFrame({
            "SEASON_ID": ["22026", "22026", "22026", "22026"],
            "PTS": [25, 10, 12, 8],
            "REB": [11, 10, 10, 4],
            "AST": [4, 10, 10, 2],
            "STL": [1, 1, 10, 0],
            "BLK": [0, 0, 1, 1],
        })

        with patch("data_collectors.stats_loaders.load_nbacom_data", return_value=mock_data):
            df = loader.scrape_single_boxscore_type_from_nbacom(boxscore_type=loader.BOXSCORE_TYPES[0])
            # Row 0: PTS 25, REB 11 -> 2 double-digit categories -> Double-Double
            self.assertTrue(df.loc[0, "DOUBLE_DOUBLE"])
            self.assertFalse(df.loc[0, "TRIPLE_DOUBLE"])

            # Row 1: PTS 10, REB 10, AST 10 -> 3 double-digit categories -> Triple-Double
            self.assertFalse(df.loc[1, "DOUBLE_DOUBLE"])
            self.assertTrue(df.loc[1, "TRIPLE_DOUBLE"])

            # Row 2: PTS 12, REB 10, AST 10, STL 10 -> 4 double-digit categories (Quadruple-Double) -> Triple-Double
            self.assertFalse(df.loc[2, "DOUBLE_DOUBLE"])
            self.assertTrue(df.loc[2, "TRIPLE_DOUBLE"])

            # Row 3: Single category -> neither
            self.assertFalse(df.loc[3, "DOUBLE_DOUBLE"])
            self.assertFalse(df.loc[3, "TRIPLE_DOUBLE"])

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_missing_player_ids_safe_update(self, mock_schedule):
        """Verify missing player IDs in load_saved_data update dictionaries without overwriting or index errors."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)
        loader._player_id_to_name = {101: "Existing Player"}
        loader._player_name_to_id = {"Existing Player": 101}

        loader.df = pd.DataFrame({
            "DATE": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
            "GAME_ID": [1, 2],
            "TEAM_ID": [10, 10],
            "PLAYER_ID": [101, 102],
            "NAME": ["Existing Player", "New Player"],
            "MIN": [20.0, 15.0],
            "COMMENT": ["", ""],
            "LEFT_MIDGAME": [False, False],
            "EJECTED": [False, False],
            "IS_OUT": [False, False],
        })

        with patch.object(StatsLoader, "load_saved_data", return_value=loader.df), \
             patch.object(loader, "assert_valid_df"):
            loader.load_saved_data()

            self.assertEqual(loader._player_id_to_name[102], "New Player")
            self.assertEqual(loader._player_name_to_id["New Player"], 102)
            self.assertEqual(loader._player_name_to_id["Existing Player"], 101)

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_scrape_boxscores_parsing_and_filtering(self, mock_schedule):
        """Verify scrape_boxscores parses HOME, OPPONENT, W_L, SEASON_YEAR and drops unneeded columns."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)

        mock_boxscore = pd.DataFrame({
            "GAME_DATE": ["2026-11-10", "2026-11-12"],
            "MATCHUP": ["BOS vs. NYK", "BOS @ MIA"],
            "WL": ["W", "L"],
            "SEASON_YEAR": ["2026-27", "2026-27"],
            "PLAYER_NAME": ["Jayson Tatum", "Jaylen Brown"],
            "TEAM_ABBREVIATION": ["BOS", "BOS"],
            "TEAM_ID": [1610612738, 1610612738],
            "GAME_ID": [22600001, 22600002],
            "PLAYER_ID": [1628369, 1627759],
            "PTS_RANK": [1, 2],
            "E_OFF_RATING": [115.0, 108.0],
            "sp_work_pace": [100.0, 99.0],
        })

        with patch.object(loader, "scrape_single_boxscore_type_from_nbacom", return_value=mock_boxscore), \
             patch.object(loader, "assert_valid_df"), \
             patch("data_collectors.stats_loaders.get_current_time", return_value=datetime.datetime(2026, 11, 20)):
            loader.scrape_boxscores_from_nbacom()

            self.assertIsInstance(loader.updates_df, pd.DataFrame)
            self.assertEqual(loader.updates_df["HOME"].tolist(), [1, 0])
            self.assertEqual(loader.updates_df["OPPONENT"].tolist(), ["NYK", "MIA"])
            self.assertEqual(loader.updates_df["W_L"].tolist(), [1, 0])
            self.assertEqual(loader.updates_df["SEASON_YEAR"].tolist(), [2026, 2026])

            # Verify unneeded columns ending in _RANK or starting with E_/sp_work_ were dropped
            self.assertNotIn("PTS_RANK", loader.updates_df.columns)
            self.assertNotIn("E_OFF_RATING", loader.updates_df.columns)
            self.assertNotIn("sp_work_pace", loader.updates_df.columns)

    def test_combine_old_new_dfs(self):
        """Verify _combine_old_new_dfs overwrites matching dates and retains non-overlapping past dates."""
        loader = StatsLoader(season=2026)
        loader.SORT_COLS = ["DATE", "GAME_ID", "TEAM_ID"]

        d1 = datetime.date(2026, 1, 1)
        d2 = datetime.date(2026, 1, 2)
        d3 = datetime.date(2026, 1, 3)

        loader.df = pd.DataFrame({
            "DATE": [d1, d2],
            "GAME_ID": [1, 2],
            "TEAM_ID": [10, 10],
            "VAL": ["old_1", "old_2"],
        })
        loader.updates_df = pd.DataFrame({
            "DATE": [d2, d3],
            "GAME_ID": [2, 3],
            "TEAM_ID": [10, 10],
            "VAL": ["new_2", "new_3"],
        })

        loader._combine_old_new_dfs()

        self.assertEqual(len(loader.df), 3)
        self.assertEqual(loader.df["DATE"].tolist(), [d1, d2, d3])
        # d1 preserved from old df
        self.assertEqual(loader.df.loc[loader.df["DATE"] == d1, "VAL"].values[0], "old_1")
        # d2 overwritten by updates df
        self.assertEqual(loader.df.loc[loader.df["DATE"] == d2, "VAL"].values[0], "new_2")
        # d3 added from updates df
        self.assertEqual(loader.df.loc[loader.df["DATE"] == d3, "VAL"].values[0], "new_3")

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_postprocessing_fantasy_points_and_confirmation(self, mock_schedule):
        """Verify _postprocessing computes fantasy points and marks past games confirmed."""
        mock_schedule.return_value = pd.DataFrame({
            "DATE": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
            "GAME_TIME": ["19:00", "20:00"],
            "GAME_ID": [101, 102],
            "TEAM_ID": [1610612738, 1610612738],
            "TEAM_ID_OPPONENT": [1610612752, 1610612752],
            "HOME": [1, 0],
        })
        loader = StatsLoaderPlayers(season=2026)

        past_date = datetime.date(2026, 1, 1)
        today = datetime.date(2026, 1, 2)

        loader.updates_df = pd.DataFrame({
            "DATE": [past_date, today],
            "GAME_ID": [101, 102],
            "TEAM_ID": [1610612738, 1610612738],
            "PLAYER_ID": [1001, 1002],
            "PTS": [20.0, 10.0],
            "3PM": [2.0, 1.0],
            "REB": [10.0, 5.0],
            "AST": [10.0, 5.0],
            "STL": [2.0, 1.0],
            "BLK": [1.0, 0.0],
            "TOV": [3.0, 2.0],
            "DOUBLE_DOUBLE": [False, False],
            "TRIPLE_DOUBLE": [True, False],
            "YAHOO_SALARY": [35.0, 20.0],
            "DRAFTKINGS_SALARY": [8000.0, 5000.0],
            "FANDUEL_SALARY": [7500.0, 4500.0],
        })

        with patch("data_collectors.stats_loaders.get_current_date", return_value=today):
            loader._postprocessing()

            # Past date marked CONFIRMED = True, today is not confirmed
            self.assertTrue(loader.updates_df.loc[loader.updates_df["DATE"] == past_date, "CONFIRMED"].values[0])
            self.assertFalse(loader.updates_df.loc[loader.updates_df["DATE"] == today, "CONFIRMED"].values[0])

            # Check Yahoo fantasy points for row 0:
            # PTS (20*1) + REB (10*1.2) + AST (10*1.5) + STL (2*3) + BLK (1*3) + TOV (3*-1) = 53.0
            yahoo_pts = loader.updates_df.loc[loader.updates_df["DATE"] == past_date, "YAHOO_POINTS"].values[0]
            self.assertAlmostEqual(yahoo_pts, 53.0, places=1)

            # Check DraftKings fantasy points for row 0:
            # PTS (20*1) + 3PM (2*0.5) + REB (10*1.25) + AST (10*1.5) + STL (2*2) + BLK (1*2) + TOV (3*-0.5) + TRIPLE_DOUBLE (1*3)
            # = 20 + 1 + 12.5 + 15 + 4 + 2 - 1.5 + 3 = 56.0
            dk_pts = loader.updates_df.loc[loader.updates_df["DATE"] == past_date, "DRAFTKINGS_POINTS"].values[0]
            self.assertAlmostEqual(dk_pts, 56.0, places=1)

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_read_single_game_boxscores_comments_drops_start_position(self, mock_schedule):
        """Verify _read_single_game_boxscores_comments drops START_POSITION to avoid overwriting 5-position starters."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)
        loader.latest_date_with_data = datetime.date(2026, 1, 1)

        loader.updates_df = pd.DataFrame({
            "DATE": [datetime.date(2026, 1, 2)],
            "GAME_ID": [22600001],
            "TEAM_ID": [1610612738],
            "PLAYER_ID": [1628369],
            "NAME": ["Jayson Tatum"],
            "MIN": [35.0],
            "SEASON_YEAR": [2026],
            "TEAM": ["BOS"],
            "OPPONENT": ["NYK"],
            "W_L": [1],
            "HOME": [1],
        })

        comments_df = pd.DataFrame({
            "GAME_ID": [22600001],
            "TEAM_ID": [1610612738],
            "PLAYER_ID": [1628369],
            "START_POSITION": ["F"],
            "COMMENT": [""],
        })

        with patch("data_collectors.stats_loaders.os.path.isfile", return_value=True), \
             patch("data_collectors.stats_loaders.pd.read_feather", return_value=comments_df), \
             patch.object(loader, "assert_valid_df"):
            loader._read_single_game_boxscores_comments()

            # START_POSITION from comments must be dropped
            self.assertNotIn("START_POSITION", loader.updates_df.columns)
            self.assertIn("COMMENT", loader.updates_df.columns)

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_load_saved_data_boolean_nan_handling(self, mock_schedule):
        """Verify load_saved_data converts NaN in boolean columns to False, not True."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)

        mock_saved = pd.DataFrame({
            "DATE": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2), datetime.date(2026, 1, 3)],
            "GAME_ID": [1, 2, 3],
            "TEAM_ID": [10, 10, 10],
            "PLAYER_ID": [101, 102, 103],
            "NAME": ["Player A", "Player B", "Player C"],
            "MIN": [20.0, 15.0, 10.0],
            "COMMENT": ["", "", ""],
            "LEFT_MIDGAME": [None, False, True],
            "EJECTED": [None, False, True],
            "IS_OUT": [None, False, True],
        })

        with patch.object(StatsLoader, "load_saved_data", return_value=mock_saved), \
             patch.object(loader, "assert_valid_df"):
            loader.load_saved_data()

            self.assertEqual(loader.df["LEFT_MIDGAME"].tolist(), [False, False, True])
            self.assertEqual(loader.df["EJECTED"].tolist(), [False, False, True])
            self.assertEqual(loader.df["IS_OUT"].tolist(), [False, False, True])

    @patch("data_collectors.stats_loaders.os.replace")
    @patch("data_collectors.stats_loaders.os.makedirs")
    @patch("pandas.DataFrame.to_feather")
    @patch("data_collectors.stats_loaders.load_schedule")
    def test_save_df_downcasts_and_creates_dir(self, mock_schedule, mock_to_feather, mock_makedirs, mock_replace):
        """Verify _save_df creates parent directory, drops VIDEO_AVAILABLE before validation, and downcasts."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers(season=2026)
        loader.updates_df = pd.DataFrame({"GAME_ID": [1], "TEAM_ID": [10]})
        loader.df = pd.DataFrame({
            "DATE": [datetime.date(2026, 1, 1)],
            "GAME_ID": [1],
            "TEAM_ID": [10],
            "PLAYER_ID": [101],
            "MIN": [30.5],
            "VIDEO_AVAILABLE": [1],
        })

        mock_makedirs.reset_mock()
        with patch.object(loader, "assert_valid_df"):
            loader._save_df()

            # Ensure directory creation was called
            mock_makedirs.assert_called_once()
            # Ensure VIDEO_AVAILABLE was dropped before saving
            self.assertNotIn("VIDEO_AVAILABLE", loader.df.columns)
            # Ensure float was downcasted to float32
            self.assertEqual(loader.df["MIN"].dtype, "float32")

    def test_assert_valid_df_incremental_flag(self):
        """Verify assert_valid_df with is_incremental=True allows partial seasons, while False enforces 1230 games."""
        loader = StatsLoader(season=2021)
        loader.SORT_COLS = ["DATE", "GAME_ID", "TEAM_ID"]
        partial_df = pd.DataFrame({
            "GAME_ID": [22100001, 22100002],
            "DATE": [datetime.date(2021, 11, 1), datetime.date(2021, 11, 2)],
            "TEAM_ID": [1610612738, 1610612747],
            "TEAM": ["BOS", "LAL"],
            "SEASON_YEAR": [2021, 2021],
        })

        # Incremental validation (delta updates) should pass despite having only 2 games
        loader.assert_valid_df(partial_df, is_incremental=True)

        # Full validation should raise AssertionError for missing games in completed season
        with self.assertRaises(AssertionError):
            loader.assert_valid_df(partial_df, is_incremental=False)

    def test_assert_valid_df_all_handles_string_dates_and_game_ids(self):
        """Verify assert_valid_df_all parses string dates and string GAME_IDs without TypeErrors."""
        df = pd.DataFrame({
            "GAME_ID": ["0022300001", "0022300002"],
            "DATE": ["2023-11-01", "2023-11-02"],
            "TEAM_ID": [1, 2],
            "TEAM": ["BOS", "LAL"],
        })
        # Should execute cleanly without TypeError: '<' or '>'
        assert_valid_df_all(df, check_total_games=False)

    def test_assert_valid_df_players_duplicates_and_multiple_names(self):
        """Verify assert_valid_df_players detects duplicate rows and inconsistent names per player ID."""
        dup_df = pd.DataFrame({
            "PLAYER_ID": [101, 101],
            "DATE": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 1)],
            "NAME": ["LeBron James", "LeBron James"],
        })
        with self.assertRaises(ValueError):
            assert_valid_df_players(dup_df)

        diff_names_df = pd.DataFrame({
            "PLAYER_ID": [101, 101],
            "DATE": [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2)],
            "NAME": ["LeBron James", "King James"],
        })
        with self.assertRaises(ValueError):
            assert_valid_df_players(diff_names_df)

    def test_assert_valid_df_players_starters_validation(self):
        """Verify assert_valid_df_players correctly validates starter count and 5-position set."""
        date = datetime.date(2026, 1, 1)
        valid_starters = pd.DataFrame({
            "DATE": [date] * 5,
            "TEAM_ID": [1] * 5,
            "PLAYER_ID": [101, 102, 103, 104, 105],
            "STARTER": [True] * 5,
            "START_POSITION": ["PG", "SG", "SF", "PF", "C"],
        })
        assert_valid_df_players(valid_starters)

        # Missing one position
        invalid_pos = valid_starters.copy()
        invalid_pos.loc[4, "START_POSITION"] = "PG"
        with self.assertRaises(AssertionError):
            assert_valid_df_players(invalid_pos)

    def test_downcast_df_preserves_numeric_and_downcasts(self):
        """Verify downcast_df converts float64 to float32, downcasts ints, and preserves counting stats."""
        df = pd.DataFrame({
            "MIN": [32.5, 20.0],
            "PTS": [25.0, 10.0],
            "BLK": [1, 0],
            "PLUS_MINUS": [-1, 2],
            "PLAYER_ID": [201939, 201940],
            "CONFIRMED": [1, 0],
        })
        downcast_df(df, print_usage=False)

        # Float columns should be downcast to float32
        self.assertEqual(df["MIN"].dtype, "float32")
        self.assertEqual(df["PTS"].dtype, "float32")

        # BLK and PLUS_MINUS must NOT be converted to bool; negative values preserved
        self.assertNotEqual(df["BLK"].dtype, bool)
        self.assertNotEqual(df["PLUS_MINUS"].dtype, bool)
        self.assertEqual(df["PLUS_MINUS"].tolist(), [-1, 2])

        # _ID columns must retain 64-bit integer type to avoid merge schema mismatches
        self.assertEqual(df["PLAYER_ID"].dtype, "int64")

        # True boolean flags should be converted to bool
        self.assertEqual(df["CONFIRMED"].dtype, bool)
        self.assertEqual(df["CONFIRMED"].tolist(), [True, False])

    def test_downcast_df_empty(self):
        """Verify downcast_df safely handles empty DataFrames."""
        df = pd.DataFrame(columns=["MIN", "PTS", "BLK"])
        result = downcast_df(df, print_usage=False)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()

