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


if __name__ == '__main__':
    unittest.main()
