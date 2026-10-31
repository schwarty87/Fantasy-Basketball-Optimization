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


if __name__ == '__main__':
    unittest.main()
