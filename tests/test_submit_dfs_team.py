import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from submit_dfs_team import SubmitDFSTeamYahooByCSV
from constants import Browsers, Sites, DFS_POSITIONS


class TestSubmitDFSTeamYahooByCSV(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_submitted_teams_and_download_dirs_exist(self):
        """Verify directories are created properly."""
        teams_dir = SubmitDFSTeamYahooByCSV._submitted_teams_dir()
        download_dir = SubmitDFSTeamYahooByCSV._download_dir()
        self.assertTrue(os.path.isdir(teams_dir))
        self.assertTrue(os.path.isdir(download_dir))

    def test_create_browser_options_chrome(self):
        """Verify Chrome options configure download dir and anti-automation flags."""
        with patch("submit_dfs_team.BROWSER", Browsers.CHROME):
            mock_inst = MagicMock(spec=SubmitDFSTeamYahooByCSV)
            mock_inst.headless = True
            mock_inst.profile_path = "/fake/profile"
            mock_inst._download_dir = MagicMock(return_value=self.temp_dir)
            options = SubmitDFSTeamYahooByCSV._create_browser_options(mock_inst)

            # Check that options were generated
            self.assertIsNotNone(options)
            args = options.arguments
            self.assertIn("--user-data-dir=/fake/profile", args)
            self.assertIn("--headless=new", args)
            self.assertIn("--disable-blink-features=AutomationControlled", args)

    def test_id_to_value(self):
        """Verify _id_to_value looks up keys by exact match or ID prefix."""
        lookup = {
            "12345 - Stephen Curry": 45,
            "67890 - LeBron James": 50,
        }
        # Exact match
        self.assertEqual(SubmitDFSTeamYahooByCSV._id_to_value(lookup, "12345 - Stephen Curry"), 45)
        # Prefix match
        self.assertEqual(SubmitDFSTeamYahooByCSV._id_to_value(lookup, "67890"), 50)
        # Non-matching
        with self.assertRaises(KeyError):
            SubmitDFSTeamYahooByCSV._id_to_value(lookup, "99999")

    def test_is_valid_contest_id(self):
        """Verify _is_valid_contest_id accepts positive integers and floats, rejecting NaNs and non-positives."""
        self.assertTrue(SubmitDFSTeamYahooByCSV._is_valid_contest_id(15935109))
        self.assertTrue(SubmitDFSTeamYahooByCSV._is_valid_contest_id(np.int64(15935109)))
        self.assertTrue(SubmitDFSTeamYahooByCSV._is_valid_contest_id(15935109.0))
        self.assertFalse(SubmitDFSTeamYahooByCSV._is_valid_contest_id(np.nan))
        self.assertFalse(SubmitDFSTeamYahooByCSV._is_valid_contest_id(0))
        self.assertFalse(SubmitDFSTeamYahooByCSV._is_valid_contest_id(-1))
        self.assertFalse(SubmitDFSTeamYahooByCSV._is_valid_contest_id("invalid"))

    def test_filter_dfs_league_contests(self):
        """Verify private DFS league contests (e.g. 'DFS league 2026-27') are filtered out of template."""
        left_cols = ['Entry ID', 'Contest ID', 'Contest Title', 'Contest Entry Fee', 'PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL', ' ']
        right_cols = ['ID', 'First Name', 'Last Name', 'Position', 'Salary', 'Time']
        cols = left_cols + right_cols

        mock_template = pd.DataFrame(index=range(4), columns=cols)
        mock_template.loc[0, 'Contest ID'] = 15935109
        mock_template.loc[0, 'Contest Title'] = 'NBA $5K Baller'
        mock_template.loc[1, 'Contest ID'] = 15935799
        mock_template.loc[1, 'Contest Title'] = 'DFS league 2026-27 Round 1'

        mock_template.loc[0:3, 'ID'] = ['101', '102', '103', '104']
        mock_template.loc[0:3, 'First Name'] = ['Player', 'Player', 'Player', 'Player']
        mock_template.loc[0:3, 'Last Name'] = ['One', 'Two', 'Three', 'Four']
        mock_template.loc[0:3, 'Position'] = ['PG', 'SG', 'SF', 'PF']
        mock_template.loc[0:3, 'Salary'] = [20, 25, 30, 35]

        filtered = SubmitDFSTeamYahooByCSV._filter_excluded_contests(mock_template)

        # Contest 0 ('NBA $5K Baller') should remain, contest 1 ('DFS league 2026-27') should be removed
        self.assertEqual(filtered.loc[0, 'Contest ID'], 15935109)
        self.assertTrue(pd.isna(filtered.loc[1, 'Contest ID']))

        # Verify player list on right side is completely unharmed
        self.assertEqual(list(filtered['ID'].dropna()), ['101', '102', '103', '104'])

    def test_multi_entry_same_contest_keys_by_entry_id(self):
        """Verify multiple entries in the same contest are keyed by distinct entry_id and preserved."""
        cols = ['Entry ID', 'Contest ID', 'Contest Title', 'Contest Entry Fee',
                'PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL', ' ',
                'ID', 'ID + Name', 'Name', 'Position', 'Salary', 'Time']
        template_df = pd.DataFrame(index=range(2), columns=cols)
        # Entry 1 and Entry 2 share the same contest ID 15935109
        template_df.loc[0, 'Entry ID'] = 88001
        template_df.loc[0, 'Contest ID'] = 15935109
        template_df.loc[0, 'Contest Title'] = 'NBA $5K Baller'
        for s in ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']:
            template_df.loc[0, s] = ''

        template_df.loc[1, 'Entry ID'] = 88002
        template_df.loc[1, 'Contest ID'] = 15935109
        template_df.loc[1, 'Contest Title'] = 'NBA $5K Baller'
        for s in ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']:
            template_df.loc[1, s] = ''

        template_df['ID'] = ['101', '102']
        template_df['ID + Name'] = ['101 - Player One', '102 - Player Two']
        template_df['Name'] = ['Player One', 'Player Two']
        template_df['Position'] = ['PG', 'SG']
        template_df['Salary'] = [25, 30]
        template_df['Time'] = ['7:00PM', '7:00PM']

        inst = MagicMock(spec=SubmitDFSTeamYahooByCSV)
        inst._site = Sites.YAHOO
        inst._positions = DFS_POSITIONS[Sites.YAHOO]
        inst._is_valid_contest_id = SubmitDFSTeamYahooByCSV._is_valid_contest_id
        inst.template = template_df
        inst.id_to_game_time = {'101 - Player One': '7:00PM', '102 - Player Two': '7:00PM'}
        inst.id_to_salary = {'101 - Player One': 25, '102 - Player Two': 30}
        inst.id_to_name = {'101 - Player One': 'Player One', '102 - Player Two': 'Player Two'}

        # Compute entry_ids & titles
        inst.entry_ids = [
            int(eid) for eid, cid in zip(template_df['Entry ID'], template_df['Contest ID'])
            if inst._is_valid_contest_id(cid) and pd.notna(eid)
        ]
        self.assertEqual(inst.entry_ids, [88001, 88002])

        titles = SubmitDFSTeamYahooByCSV.get_entry_titles(inst)
        self.assertEqual(titles[88001], 'NBA $5K Baller')
        self.assertEqual(titles[88002], 'NBA $5K Baller')

        # get_locked_dfs keys by entry_id
        with patch("submit_dfs_team.get_next_game_time", return_value="6:00PM"):
            locked_dfs = SubmitDFSTeamYahooByCSV.get_locked_dfs(inst)
        self.assertEqual(set(locked_dfs.keys()), {88001, 88002})

        # _fill_template_with_predicted_teams matches on Entry ID
        mock_best_set = pd.DataFrame([
            {'entry_id': 88001, 'PG': '101 - Player One', 'SG': '102 - Player Two',
             'G': '', 'SF': '', 'PF': '', 'F': '', 'C': '', 'UTIL': ''},
            {'entry_id': 88002, 'PG': '102 - Player Two', 'SG': '101 - Player One',
             'G': '', 'SF': '', 'PF': '', 'F': '', 'C': '', 'UTIL': ''},
        ])
        SubmitDFSTeamYahooByCSV._fill_template_with_predicted_teams(inst, mock_best_set)
        # Verify row 0 got 88001's lineup and row 1 got 88002's lineup
        self.assertEqual(inst.template.loc[0, 'PG'], '101 - Player One')
        self.assertEqual(inst.template.loc[1, 'PG'], '102 - Player Two')

    def test_get_template_df_raises_on_failure_without_stale_fallback(self):
        """Verify _get_template_df raises ValueError on download failure instead of loading stale CSV."""
        inst = MagicMock(spec=SubmitDFSTeamYahooByCSV)
        inst._template_path = MagicMock(return_value=os.path.join(self.temp_dir, "stale.csv"))
        # Create a stale file on disk
        with open(inst._template_path(), "w") as f:
            f.write("Stale,CSV,Data\n")

        inst._start_driver = MagicMock(side_effect=RuntimeError("Selenium initialization error"))

        with self.assertRaises(ValueError) as cm:
            SubmitDFSTeamYahooByCSV._get_template_df(inst)
        self.assertIn("we weren't able to load the template", str(cm.exception))

    def test_get_locked_dfs_skips_nan_empty_slots_and_fill_template_rejects_missing_entry(self):
        """Verify get_locked_dfs handles NaN empty slots and _fill_template_with_predicted_teams rejects unknown Entry IDs."""
        cols = ['Entry ID', 'Contest ID'] + DFS_POSITIONS[Sites.YAHOO] + ['ID', 'ID + Name', 'Name', 'Position', 'Salary', 'Time']
        template_df = pd.DataFrame(index=[0], columns=cols)
        template_df.loc[0, 'Entry ID'] = 88001
        template_df.loc[0, 'Contest ID'] = 15935109
        for s in DFS_POSITIONS[Sites.YAHOO]:
            template_df.loc[0, s] = np.nan

        inst = MagicMock(spec=SubmitDFSTeamYahooByCSV)
        inst._positions = DFS_POSITIONS[Sites.YAHOO]
        inst._is_valid_contest_id = SubmitDFSTeamYahooByCSV._is_valid_contest_id
        inst.template = template_df

        with patch("submit_dfs_team.get_next_game_time", return_value="7:00PM"):
            locked_dfs = SubmitDFSTeamYahooByCSV.get_locked_dfs(inst)

        self.assertIn(88001, locked_dfs)
        self.assertTrue(locked_dfs[88001].empty)

        unknown_best_set = pd.DataFrame([
            {'entry_id': 99999, **{s: '' for s in DFS_POSITIONS[Sites.YAHOO]}}
        ])
        with self.assertRaises(ValueError) as cm:
            SubmitDFSTeamYahooByCSV._fill_template_with_predicted_teams(inst, unknown_best_set)
        self.assertIn("Expected exactly 1 template row for Entry ID 99999", str(cm.exception))

    def test_submit_best_set_waits_for_entry_ids_count_on_shared_contest(self):
        """Verify submit_best_set checks 'We have detected 2 lineups' when 2 entries share 1 Contest ID."""
        positions = DFS_POSITIONS[Sites.YAHOO]
        cols = ['Entry ID', 'Contest ID'] + positions + ['ID', 'ID + Name', 'Name', 'Position', 'Salary', 'Time']
        template_df = pd.DataFrame(index=range(8), columns=cols)
        template_df.loc[0, 'Entry ID'] = 88001
        template_df.loc[0, 'Contest ID'] = 15935109
        template_df.loc[1, 'Entry ID'] = 88002
        template_df.loc[1, 'Contest ID'] = 15935109

        player_names = [f"Player {i}" for i in range(1, 9)]
        player_id_names = [f"10{i} - Player {i}" for i in range(1, 9)]
        template_df['ID'] = [f"10{i}" for i in range(1, 9)]
        template_df['ID + Name'] = player_id_names
        template_df['Name'] = player_names
        template_df['Position'] = ['PG', 'SG', 'PG', 'SF', 'PF', 'SF', 'C', 'C']
        template_df['Salary'] = [25] * 8
        template_df['Time'] = ['7:00PM'] * 8

        inst = MagicMock(spec=SubmitDFSTeamYahooByCSV)
        inst._positions = positions
        inst._is_valid_contest_id = SubmitDFSTeamYahooByCSV._is_valid_contest_id
        inst.template = template_df
        inst.contest_ids = [15935109]
        inst.entry_ids = [88001, 88002]
        inst.name_to_id = dict(zip(player_names, player_id_names))
        inst._sort_players_by_time = lambda teams_df: teams_df
        inst._fill_template_with_predicted_teams = lambda df: SubmitDFSTeamYahooByCSV._fill_template_with_predicted_teams(inst, df)
        inst._template_path = MagicMock(return_value=os.path.join(self.temp_dir, "upload.csv"))
        inst._assert_template_csv_finished_saving = MagicMock()
        inst._start_driver = MagicMock()
        inst._navigate_to_submit_csv_page = MagicMock(return_value=True)

        mock_driver = MagicMock()
        mock_file_input = MagicMock()
        mock_driver.find_elements.return_value = [mock_file_input]
        inst.driver = mock_driver

        upload_btn = MagicMock()
        marker_calls = []

        def fake_find_by_two_markers(by1, value1, attribute_name, attribute_value):
            marker_calls.append((value1, attribute_name, attribute_value))
            if value1 == 'button' and attribute_value == 'Upload and edit entries':
                return upload_btn
            return MagicMock()

        inst._find_element_by_two_markers = MagicMock(side_effect=fake_find_by_two_markers)

        best_set = pd.DataFrame([
            {'entry_id': 88001, **dict(zip(positions, player_names))},
            {'entry_id': 88002, **dict(zip(positions, player_names))},
        ])

        SubmitDFSTeamYahooByCSV.submit_best_set(inst, best_set)

        self.assertIn(('span', 'text', 'We have detected 2 lineups'), marker_calls)
        upload_btn.click.assert_called_once()


if __name__ == '__main__':
    unittest.main()
