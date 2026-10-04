import os
import datetime
import json
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from constants import DATA_DIR

from bs4 import BeautifulSoup
from data_collectors.injuries import Injuries, InjuriesSources
from data_collectors.ejections import Ejections
from data_collectors.starters import (LoadStartersBasketballMonster, LoadStartersRotowire,
                                     LoadStartersNBACOM, Starters)
from data_collectors.stats_loaders import StatsLoaderPlayers, StatsLoaderTeams
from data_collectors.fantasy_data import LoadFantasyData
from utils import (read_and_cache, fetch_soup, parse_html_table, fetch_html_table,
                   get_current_season, season_int_to_str, date_to_season_str, assert_valid_df_all,
                   add_game_times_to_df, load_and_cache_players_list, date_to_game_ids,
                   update_dir_name)
from run_full_pipeline import RunFullPipeline


class TestDataCollectors(unittest.TestCase):

    # -------------------------------------------------------------------------
    # 1. Injuries - Rotowire Status Parsing Tests
    # -------------------------------------------------------------------------
    @patch("data_collectors.injuries.requests.get")
    def test_rotowire_injuries_status_mapping(self, mock_get):
        """Verify Rotowire maps 'Questionable', 'Doubtful', 'Out', 'GTD' correctly."""
        mock_payload = [
            {"player": "Santi Aldama", "team": "DAL", "status": "Questionable"},
            {"player": "Bradley Beal", "team": "LAC", "status": "Out"},
            {"player": "Paul George", "team": "PHI", "status": "Doubtful"},
            {"player": "Kawhi Leonard", "team": "LAC", "status": "Out For Season"},
            {"player": "Stephen Curry", "team": "GSW", "status": "Game Time Decision"},
            {"player": "Luka Doncic", "team": "DAL", "status": "Probable"},
        ]
        mock_resp = MagicMock()
        mock_resp.content = json.dumps(mock_payload).encode("utf-8")
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        injuries = Injuries(injuries_sources=InjuriesSources.ROTOWIRE)
        df = injuries._load_injuries_rotowire()

        self.assertEqual(len(df), 6)

        # Check IS_OUT
        out_players = df.loc[df["IS_OUT"], "NAME"].tolist()
        self.assertIn("Bradley Beal", out_players)
        self.assertIn("Kawhi Leonard", out_players)
        self.assertNotIn("Santi Aldama", out_players)
        self.assertNotIn("Luka Doncic", out_players)

        # Check IS_QUESTIONABLE
        q_players = df.loc[df["IS_QUESTIONABLE"], "NAME"].tolist()
        self.assertIn("Santi Aldama", q_players)
        self.assertIn("Paul George", q_players)
        self.assertIn("Stephen Curry", q_players)
        self.assertNotIn("Bradley Beal", q_players)
        self.assertNotIn("Luka Doncic", q_players)

    # -------------------------------------------------------------------------
    # 2. Injuries - CBS Sports DOM & Dynamic Header Parsing Tests
    # -------------------------------------------------------------------------
    @patch("utils.fetch_soup")
    def test_cbs_injuries_dom_parsing_and_clean_names(self, mock_fetch):
        """Verify CBS parser extracts exact clean player names without character duplication."""
        mock_html = """
        <html>
            <body>
                <div class="TeamName">Atlanta</div>
                <table>
                    <thead>
                        <tr>
                            <th>Player</th>
                            <th>Position</th>
                            <th>Updated</th>
                            <th>Injury</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>
                                <span class="CellPlayerName--short">K. Gilbert</span>
                                <span class="CellPlayerName--long">Keshon Gilbert</span>
                            </td>
                            <td>PG</td>
                            <td>Thu, Oct 1</td>
                            <td>Hip</td>
                            <td>Game Time Decision</td>
                        </tr>
                        <tr>
                            <td>
                                <span class="CellPlayerName--short">H. Veesaar</span>
                                <span class="CellPlayerName--long">Henri Veesaar</span>
                            </td>
                            <td>C</td>
                            <td>Thu, Oct 1</td>
                            <td>Knee</td>
                            <td>Out for the season</td>
                        </tr>
                    </tbody>
                </table>
            </body>
        </html>
        """
        mock_fetch.return_value = BeautifulSoup(mock_html, "html.parser")

        injuries = Injuries(injuries_sources=InjuriesSources.CBS)
        df = injuries._load_injuries_cbs()

        self.assertEqual(len(df), 2)
        # Verify clean name extraction (no "K. GilbertKeshon Gilbert")
        self.assertEqual(df.loc[0, "NAME"], "Keshon Gilbert")
        self.assertEqual(df.loc[0, "TEAM"], "ATL")
        self.assertTrue(df.loc[0, "IS_QUESTIONABLE"])
        self.assertFalse(df.loc[0, "IS_OUT"])

        self.assertEqual(df.loc[1, "NAME"], "Henri Veesaar")
        self.assertTrue(df.loc[1, "IS_OUT"])
        self.assertFalse(df.loc[1, "IS_QUESTIONABLE"])

    @patch("utils.fetch_soup")
    def test_cbs_injuries_dynamic_header_reordering(self, mock_fetch):
        """Verify CBS parser adapts dynamically if Status column is placed before Injury."""
        reordered_html = """
        <html>
            <body>
                <div class="TeamName">Boston</div>
                <table>
                    <thead>
                        <tr>
                            <th>Player</th>
                            <th>Status</th>
                            <th>Injury</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>
                                <span class="CellPlayerName--long">Jaylen Brown</span>
                            </td>
                            <td>Questionable</td>
                            <td>Hamstring</td>
                        </tr>
                    </tbody>
                </table>
            </body>
        </html>
        """
        mock_fetch.return_value = BeautifulSoup(reordered_html, "html.parser")

        injuries = Injuries(injuries_sources=InjuriesSources.CBS)
        df = injuries._load_injuries_cbs()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "NAME"], "Jaylen Brown")
        self.assertEqual(df.loc[0, "TEAM"], "BOS")
        self.assertTrue(df.loc[0, "IS_QUESTIONABLE"])
        self.assertFalse(df.loc[0, "IS_OUT"])

    # -------------------------------------------------------------------------
    # 2b. Utils - fetch_soup Robustness Tests
    # -------------------------------------------------------------------------
    @patch("utils.time.sleep")
    @patch("utils.requests.get")
    def test_fetch_soup_success_and_error_handling(self, mock_get, mock_sleep):
        """Verify fetch_soup returns BeautifulSoup on 200 and None on HTTP error or exception."""
        # Success case
        mock_resp = MagicMock()
        mock_resp.content = b"<html><body><h1>NBA</h1></body></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        soup = fetch_soup("https://example.com")
        self.assertIsNotNone(soup)
        self.assertEqual(soup.find("h1").text, "NBA")

        # Failure case (e.g. 404 or connection failure) - retries 3 times then returns None
        mock_get.side_effect = Exception("Connection refused")
        soup_failed = fetch_soup("https://broken.example.com")
        self.assertIsNone(soup_failed)
        self.assertEqual(mock_sleep.call_count, 3)

    # -------------------------------------------------------------------------
    # 3. Ejections - Error Handling & Modification Time
    # -------------------------------------------------------------------------
    @patch("utils.time.sleep")
    @patch("utils.requests.get")
    def test_ejections_spotrac_non_200_returns_empty_df(self, mock_get, mock_sleep):
        """Verify that HTTP 403 or non-200 responses return EMPTY_DF without crashing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
        mock_get.return_value = mock_resp

        ejections = Ejections(season=2025)
        df = ejections._load_ejections_spotrac()
        self.assertTrue(df.empty)
        mock_get.assert_called()

    @patch("data_collectors.ejections.os.path.isfile", return_value=True)
    @patch("data_collectors.ejections.os.path.getmtime")
    @patch("data_collectors.ejections.pd.read_feather")
    def test_ejections_uses_mtime_not_ctime(self, mock_read_feather, mock_getmtime, mock_isfile):
        """Verify collect_data checks mtime against yesterday's date."""
        now_ts = datetime.datetime.now().timestamp()
        mock_getmtime.return_value = now_ts
        mock_read_feather.return_value = pd.DataFrame({"NAME": ["Player 1"], "TEAM": ["BOS"], "DATE": [datetime.date.today()]})

        ejections = Ejections(season=2025)
        df = ejections.collect_data()

        mock_getmtime.assert_called_once()
        mock_read_feather.assert_called_once()
        self.assertEqual(len(df), 1)

    # -------------------------------------------------------------------------
    # 4. StatsLoaderPlayers - Date Truncation Filter Removal
    # -------------------------------------------------------------------------
    @patch.object(StatsLoaderPlayers, "assert_valid_df")
    @patch("data_collectors.stats_loaders.load_schedule")
    @patch("data_collectors.stats_loaders.os.path.exists", return_value=True)
    @patch("data_collectors.stats_loaders.pd.read_feather")
    def test_stats_loader_retains_dates_after_august_2025(self, mock_read_feather, mock_exists, mock_schedule, mock_assert):
        """Verify that games played after 2025-08-01 are not truncated when loading saved data."""
        mock_schedule.return_value = pd.DataFrame()
        mock_df = pd.DataFrame({
            "DATE": [
                datetime.date(2025, 4, 1),
                datetime.date(2025, 11, 15),
                datetime.date(2026, 3, 10),
            ],
            "GAME_ID": [22400001, 22500001, 22500500],
            "TEAM_ID": [1610612738, 1610612738, 1610612738],
            "PLAYER_ID": [203935, 203935, 203935],
            "NAME": ["Marcus Smart", "Marcus Smart", "Marcus Smart"],
            "MIN": [30.0, 25.0, 28.0],
            "LEFT_MIDGAME": [False, False, False],
            "EJECTED": [False, False, False],
            "IS_OUT": [False, False, False],
            "COMMENT": ["", "", ""],
        })
        mock_read_feather.return_value = mock_df

        loader = StatsLoaderPlayers(season=2025)
        loader.load_saved_data()

        # All 3 dates must be retained, including the 2025-11-15 and 2026-03-10 records
        self.assertEqual(len(loader.df), 3)
        self.assertIn(datetime.date(2026, 3, 10), loader.df["DATE"].values)

    # -------------------------------------------------------------------------
    # 5. Utils - read_and_cache Zero-Copy Feather Read
    # -------------------------------------------------------------------------
    @patch("utils.os.path.isfile", return_value=True)
    @patch("utils.pd.read_feather")
    def test_read_and_cache_returns_read_feather_directly(self, mock_read_feather, mock_isfile):
        """Verify read_and_cache returns pd.read_feather without calling .to_dict()."""
        expected_df = pd.DataFrame({"a": [1, 2, 3]})
        mock_read_feather.return_value = expected_df

        dummy_func = MagicMock()
        result = read_and_cache(dummy_func, {}, "test_cache.feather")

        mock_read_feather.assert_called_once()
        dummy_func.assert_not_called()
        self.assertIs(result, expected_df)

    # -------------------------------------------------------------------------
    # 6. Injuries - combine_dfs Multi-Source Integration Tests
    # -------------------------------------------------------------------------
    def test_injuries_combine_dfs_merges_and_mutual_exclusion(self):
        """Verify combine_dfs correctly merges sources and enforces that OUT overrides QUESTIONABLE."""
        cbs_df = pd.DataFrame([
            {"NAME": "LeBron James", "TEAM": "LAL", "IS_OUT": False, "IS_QUESTIONABLE": True},
            {"NAME": "Austin Reaves", "TEAM": "LAL", "IS_OUT": True, "IS_QUESTIONABLE": False},
        ])
        rw_df = pd.DataFrame([
            {"NAME": "LeBron James", "TEAM": "LAL", "IS_OUT": True, "IS_QUESTIONABLE": False},
            {"NAME": "Anthony Davis", "TEAM": "LAL", "IS_OUT": False, "IS_QUESTIONABLE": True},
        ])

        injuries = Injuries()
        injuries.combine_dfs({
            InjuriesSources.CBS: cbs_df,
            InjuriesSources.ROTOWIRE: rw_df,
        })

        res = injuries.injuries_df.set_index("NAME")

        # LeBron was Questionable on CBS and Out on Rotowire -> must be OUT=True, QUESTIONABLE=False
        self.assertTrue(res.loc["LeBron James", "IS_OUT"])
        self.assertFalse(res.loc["LeBron James", "IS_QUESTIONABLE"])

        # Austin Reaves was Out on CBS -> OUT=True, QUESTIONABLE=False
        self.assertTrue(res.loc["Austin Reaves", "IS_OUT"])
        self.assertFalse(res.loc["Austin Reaves", "IS_QUESTIONABLE"])

        # Anthony Davis was Questionable on Rotowire -> OUT=False, QUESTIONABLE=True
        self.assertFalse(res.loc["Anthony Davis", "IS_OUT"])
        self.assertTrue(res.loc["Anthony Davis", "IS_QUESTIONABLE"])

        self.assertCountEqual(injuries.out_players(), ["LeBron James", "Austin Reaves"])
        self.assertCountEqual(injuries.questionable_players(), ["Anthony Davis"])

    # -------------------------------------------------------------------------
    # 7. Starters - Basketball Monster BeautifulSoup Parsing Tests
    # -------------------------------------------------------------------------
    @patch("data_collectors.starters.fetch_soup")
    def test_basketball_monster_starters_parsing(self, mock_fetch):
        """Verify Basketball Monster extracts starters and cleans suffix tags."""
        html = """
        <html>
            <body>
                <table class="datatable">
                    <tr><th>BOS @ DET - 7:00 PM</th></tr>
                    <tr><td></td><td>BOS</td><td>@ DET</td></tr>
                    <tr><td>PG</td><td>Derrick White</td><td>Cade Cunningham Q</td></tr>
                    <tr><td>SG</td><td>Jaylen Brown</td><td>Duncan Robinson</td></tr>
                    <tr><td>SF</td><td>Jayson Tatum</td><td>Ausar Thompson</td></tr>
                    <tr><td>PF</td><td>Al Horford</td><td>Tobias Harris</td></tr>
                    <tr><td>C</td><td>Kristaps Porzingis</td><td>Jalen Duren</td></tr>
                </table>
            </body>
        </html>
        """
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")

        bm = LoadStartersBasketballMonster()
        bm._load_starters()

        self.assertEqual(len(bm.df), 10)
        # Suffix "Q" must be stripped from Cade Cunningham Q
        cade = bm.df.loc[bm.df["NAME"] == "Cade Cunningham"].iloc[0]
        self.assertEqual(cade["START_POSITION"], "PG")
        self.assertEqual(cade["TEAM"], "DET")
        self.assertFalse(cade["CONFIRMED"])

        # Derrick White for BOS
        dwhite = bm.df.loc[bm.df["NAME"] == "Derrick White"].iloc[0]
        self.assertEqual(dwhite["START_POSITION"], "PG")
        self.assertEqual(dwhite["TEAM"], "BOS")

    # -------------------------------------------------------------------------
    # 8. Ejections - Spotrac BeautifulSoup Table Parsing Tests
    # -------------------------------------------------------------------------
    @patch("utils.fetch_soup")
    def test_ejections_spotrac_beautifulsoup_table_parsing(self, mock_fetch):
        """Verify Spotrac table is parsed with BeautifulSoup and filters by ejection infraction."""
        html = """
        <html>
            <body>
                <table>
                    <thead>
                        <tr>
                            <th>Player</th>
                            <th>Team</th>
                            <th>Date</th>
                            <th>Infraction</th>
                            <th>Amount</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Draymond Green</td>
                            <td>Golden State Warriors</td>
                            <td>01/15/26</td>
                            <td>Technical Foul - Ejection</td>
                            <td>$2,000</td>
                        </tr>
                        <tr>
                            <td>Dillon Brooks</td>
                            <td>Houston Rockets</td>
                            <td>01/15/26</td>
                            <td>Flopping fine</td>
                            <td>$2,000</td>
                        </tr>
                    </tbody>
                </table>
            </body>
        </html>
        """
        mock_fetch.return_value = BeautifulSoup(html, "html.parser")

        ejections = Ejections(season=2025)
        df = ejections._load_ejections_spotrac()

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["NAME"], "Draymond Green")
        self.assertEqual(df.iloc[0]["TEAM"], "GSW")
        self.assertEqual(df.iloc[0]["DATE"], datetime.date(2026, 1, 15))

    # -------------------------------------------------------------------------
    # 9. Utils - Dynamic & Configurable HTML Table Parsing Tests
    # -------------------------------------------------------------------------
    def test_parse_html_table_standard_and_transforms(self):
        """Verify parse_html_table headers, uppercase, header_transform, and column_rename."""
        html = """
        <table>
            <thead>
                <tr><th>Player Name</th><th>Team Code</th><th>Points</th></tr>
            </thead>
            <tbody>
                <tr><td>Jayson Tatum</td><td>BOS</td><td>30</td></tr>
                <tr><td>Luka Doncic</td><td>DAL</td><td>33</td></tr>
            </tbody>
        </table>
        """
        table = BeautifulSoup(html, "html.parser").find("table")

        # Uppercase headers & column rename
        df = parse_html_table(
            table,
            uppercase_headers=True,
            column_rename={"PLAYER NAME": "NAME", "TEAM CODE": "TEAM"}
        )
        self.assertEqual(list(df.columns), ["NAME", "TEAM", "POINTS"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["NAME"], "Jayson Tatum")

        # Custom header transform
        df_lower = parse_html_table(
            table,
            header_transform=lambda h: h.strip().lower().replace(" ", "_")
        )
        self.assertEqual(list(df_lower.columns), ["player_name", "team_code", "points"])

    def test_parse_html_table_cell_parser_and_row_parser(self):
        """Verify custom cell_parser and row_parser handling in parse_html_table."""
        # Custom cell parser extracting link href
        html_links = """
        <table>
            <thead><tr><th>Player</th><th>Profile</th></tr></thead>
            <tbody>
                <tr><td>Nikola Jokic</td><td><a href="/players/jokic">Profile</a></td></tr>
            </tbody>
        </table>
        """
        table_links = BeautifulSoup(html_links, "html.parser").find("table")
        df_links = parse_html_table(
            table_links,
            cell_parser=lambda td: td.find("a")["href"] if td.find("a") else td.get_text(strip=True)
        )
        self.assertEqual(df_links.iloc[0]["Profile"], "/players/jokic")

        # Custom row parser returning a list of dicts (multi-entity rows)
        html_multi = """
        <table>
            <tbody>
                <tr><td>PG</td><td>Player A</td><td>Player B</td></tr>
                <tr><td>C</td><td>Player C</td><td>Player D</td></tr>
            </tbody>
        </table>
        """
        table_multi = BeautifulSoup(html_multi, "html.parser").find("table")

        def multi_row_parser(tr):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(tds) >= 3:
                pos, p1, p2 = tds[0], tds[1], tds[2]
                return [
                    {"POS": pos, "PLAYER": p1, "SIDE": "HOME"},
                    {"POS": pos, "PLAYER": p2, "SIDE": "AWAY"},
                ]
            return None

        df_multi = parse_html_table(table_multi, row_parser=multi_row_parser)
        self.assertEqual(len(df_multi), 4)
        self.assertEqual(df_multi.iloc[0]["PLAYER"], "Player A")
        self.assertEqual(df_multi.iloc[1]["PLAYER"], "Player B")

    def test_parse_html_table_headerless_and_context(self):
        """Verify headerless tables, custom headers, and context injection."""
        html = """
        <table>
            <tbody>
                <tr><td>1</td><td>Stephen Curry</td></tr>
                <tr><td>2</td><td>Klay Thompson</td></tr>
                <tr><td>  </td><td>  </td></tr>
            </tbody>
        </table>
        """
        table = BeautifulSoup(html, "html.parser").find("table")
        df = parse_html_table(
            table,
            header_row=None,
            custom_headers=["RANK", "NAME"],
            context={"TEAM": "GSW", "SEASON": 2026},
            drop_empty_rows=True
        )
        # Empty row must be dropped
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["RANK", "NAME", "TEAM", "SEASON"])
        self.assertEqual(df.iloc[0]["TEAM"], "GSW")
        self.assertEqual(df.iloc[0]["SEASON"], 2026)

    def test_fetch_html_table_all_tables_with_context_and_concat(self):
        """Verify fetch_html_table across multiple tables with context extraction and concatenation."""
        multi_table_html = """
        <html>
            <body>
                <div class="division">Atlantic</div>
                <table class="standings">
                    <thead><tr><th>Team</th><th>Wins</th></tr></thead>
                    <tbody><tr><td>Boston Celtics</td><td>60</td></tr></tbody>
                </table>
                <div class="division">Pacific</div>
                <table class="standings">
                    <thead><tr><th>Team</th><th>Wins</th></tr></thead>
                    <tbody><tr><td>Golden State Warriors</td><td>50</td></tr></tbody>
                </table>
            </body>
        </html>
        """
        soup = BeautifulSoup(multi_table_html, "html.parser")

        def extract_division(t):
            div = t.find_previous(class_="division")
            return {"DIVISION": div.get_text(strip=True) if div else ""}

        # Test concat_tables=True
        merged_df = fetch_html_table(
            soup,
            table_selector="table.standings",
            all_tables=True,
            concat_tables=True,
            table_context_parser=extract_division
        )
        self.assertEqual(len(merged_df), 2)
        self.assertEqual(merged_df.iloc[0]["DIVISION"], "Atlantic")
        self.assertEqual(merged_df.iloc[0]["Team"], "Boston Celtics")
        self.assertEqual(merged_df.iloc[1]["DIVISION"], "Pacific")
        self.assertEqual(merged_df.iloc[1]["Team"], "Golden State Warriors")

        # Test table_index selection
        second_table_df = fetch_html_table(
            soup,
            table_selector="table.standings",
            table_index=1,
            table_context_parser=extract_division
        )
        self.assertEqual(len(second_table_df), 1)
        self.assertEqual(second_table_df.iloc[0]["Team"], "Golden State Warriors")

    @patch("utils.requests.get")
    def test_fetch_soup_with_params_and_custom_options(self, mock_get):
        """Verify fetch_soup forwards query params, custom headers, and timeout."""
        mock_resp = MagicMock()
        mock_resp.content = b"<html><body><p>Data</p></body></html>"
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        params = {"season": 2026, "type": "traditional"}
        headers = {"X-Custom-Header": "TestValue"}
        soup = fetch_soup("https://example.com/api", params=params, headers=headers, timeout=25)

        self.assertIsNotNone(soup)
        mock_get.assert_called_once_with(
            "https://example.com/api",
            params=params,
            headers=headers,
            timeout=25
        )

    # -------------------------------------------------------------------------
    # 12. Stage 2 - Dynamic Season Resolution, Defaults & Invariant Tests
    # -------------------------------------------------------------------------
    def test_get_current_season_boundary_dates(self):
        """Verify NBA season resolution based on month/year boundaries."""
        # October start of 2026-27 season
        self.assertEqual(get_current_season(datetime.date(2026, 10, 1)), 2026)
        self.assertEqual(get_current_season(datetime.date(2026, 12, 25)), 2026)
        # Mid-season next calendar year
        self.assertEqual(get_current_season(datetime.date(2027, 1, 15)), 2026)
        self.assertEqual(get_current_season(datetime.date(2027, 4, 11)), 2026)
        self.assertEqual(get_current_season(datetime.date(2027, 6, 20)), 2026)
        # September before season begins belongs to previous season
        self.assertEqual(get_current_season(datetime.date(2026, 9, 30)), 2025)
        # String parsing support
        self.assertEqual(get_current_season("2026-10-04"), 2026)
        # Default with no arguments resolves to an integer >= 2026
        self.assertGreaterEqual(get_current_season(), 2026)

    def test_season_int_to_str_formatting(self):
        """Verify season integer to formatted string conversion and defaults."""
        self.assertEqual(season_int_to_str(2026), "2026-27")
        self.assertEqual(season_int_to_str(2023), "2023-24")
        self.assertEqual(season_int_to_str("2026-27"), "2026-27")
        # None default should return string of current season
        self.assertEqual(season_int_to_str(None), season_int_to_str(get_current_season()))
        self.assertEqual(season_int_to_str(), season_int_to_str(get_current_season()))

    def test_date_to_season_str_resolution(self):
        """Verify date to season string conversion."""
        self.assertEqual(date_to_season_str(datetime.date(2026, 10, 15)), "2026-27")
        self.assertEqual(date_to_season_str(datetime.date(2027, 3, 1)), "2026-27")
        self.assertEqual(date_to_season_str(datetime.date(2024, 2, 10)), "2023-24")
        self.assertEqual(date_to_season_str(), season_int_to_str(get_current_season()))

    @patch("data_collectors.stats_loaders.load_schedule")
    def test_stats_loader_defaults_to_dynamic_season(self, mock_schedule):
        """Verify StatsLoaderPlayers defaults to get_current_season() when season is None."""
        mock_schedule.return_value = pd.DataFrame()
        loader = StatsLoaderPlayers()
        expected_season = get_current_season()
        self.assertEqual(loader.season, expected_season)
        self.assertEqual(loader.season_str, season_int_to_str(expected_season))

    def test_ejections_defaults_to_dynamic_season(self):
        """Verify Ejections defaults to get_current_season() when season is None."""
        ejections = Ejections()
        self.assertEqual(ejections.season, get_current_season())

    def test_assert_valid_df_all_validates_completed_seasons(self):
        """Verify assert_valid_df_all asserts 1230 games for all completed seasons."""
        current_season = get_current_season()
        completed_cutoff = current_season - 1

        rows = []
        for s in range(2018, current_season + 1):
            if s == 2019:
                n_games = 1059
            elif s == 2020:
                n_games = 1080
            elif s == current_season:
                n_games = 10  # Ongoing season has fewer games
            else:
                n_games = 1230

            for g in range(n_games):
                rows.append({
                    "GAME_ID": s * 100000 + g,
                    "DATE": datetime.date(s, 11, 1),
                    "TEAM_ID": 1,
                    "TEAM": "BOS",
                    "SEASON_YEAR": s,
                })
        valid_df = pd.DataFrame(rows)

        # Should pass without assertion errors
        assert_valid_df_all(valid_df)

        # Now simulate a completed season missing games (e.g. completed_cutoff having only 1200)
        invalid_rows = [r for r in rows if not (r["SEASON_YEAR"] == completed_cutoff and r["GAME_ID"] >= completed_cutoff * 100000 + 1200)]
        invalid_df = pd.DataFrame(invalid_rows)
        with self.assertRaises(AssertionError):
            assert_valid_df_all(invalid_df)

    # -------------------------------------------------------------------------
    # 13. Pipeline & Game Times Optimization Tests
    # -------------------------------------------------------------------------
    def test_add_game_times_to_df_clean_merge(self):
        """Verify add_game_times_to_df merges GAME_DATE_TIME cleanly on GAME_ID."""
        mock_schedule = pd.DataFrame({
            "GAME_ID": [1001, 1002],
            "GAME_DATE_TIME": [
                pd.Timestamp("2026-10-25 19:30:00-0400"),
                pd.Timestamp("2026-10-25 22:00:00-0400")
            ],
            "TEAM_ID": [1, 2]
        })
        input_df = pd.DataFrame({
            "PLAYER_ID": [101, 102],
            "GAME_ID": [1001, 1002],
            "TEAM_ID": [1, 2],
            "NAME": ["Player A", "Player B"]
        })

        result_df = add_game_times_to_df(input_df, schedule=mock_schedule)
        self.assertIn("GAME_DATE_TIME", result_df.columns)
        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df.iloc[0]["GAME_DATE_TIME"], pd.Timestamp("2026-10-25 19:30:00-0400"))
        # Verify static wrapper on RunFullPipeline produces exact same result
        wrapper_df = RunFullPipeline._add_game_times_to_df(input_df, schedule=mock_schedule)
        self.assertTrue(result_df.equals(wrapper_df))

    def test_add_game_times_to_df_missing_game_raises_assertion(self):
        """Verify add_game_times_to_df asserts when a GAME_ID is not found in schedule."""
        mock_schedule = pd.DataFrame({
            "GAME_ID": [1001],
            "GAME_DATE_TIME": [pd.Timestamp("2026-10-25 19:30:00-0400")],
        })
        input_df = pd.DataFrame({
            "GAME_ID": [1001, 9999],  # 9999 not in schedule
        })
        with self.assertRaises(AssertionError):
            add_game_times_to_df(input_df, schedule=mock_schedule)

    @patch("run_full_pipeline.time.sleep")
    def test_wait_until_certain_time_handles_naive_and_aware(self, mock_sleep):
        """Verify _wait_until_certain_time handles both tz-naive and tz-aware datetimes without crashing."""
        from constants import NY_TZ
        future_aware = datetime.datetime.now(tz=NY_TZ) + datetime.timedelta(seconds=5)
        future_naive = datetime.datetime.now() + datetime.timedelta(seconds=5)

        # Neither should raise TypeError: can't subtract offset-naive and offset-aware datetimes
        RunFullPipeline._wait_until_certain_time(future_aware)
        self.assertTrue(mock_sleep.called)

        mock_sleep.reset_mock()
        RunFullPipeline._wait_until_certain_time(future_naive)
        self.assertTrue(mock_sleep.called)

    @patch("run_full_pipeline.os.path.isdir")
    @patch("run_full_pipeline.os.path.isfile")
    def test_todays_model_already_trained_respects_custom_run_date(self, mock_isfile, mock_isdir):
        """Verify _todays_model_already_trained respects self._run_date and custom date argument."""
        pipeline = MagicMock(spec=RunFullPipeline)
        pipeline._run_date = datetime.date(2026, 11, 15)
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        # Call the actual unbound method with our mock instance
        is_trained = RunFullPipeline._todays_model_already_trained(pipeline)
        self.assertTrue(is_trained)
        mock_isdir.assert_called_with(
            os.path.join(
                DATA_DIR, 'machine_learning_fantasy_basketball_predictions', 'models',
                'model_2026_11_15'
            )
        )

    def test_stats_loader_teams_defaults_to_dynamic_season(self):
        """Verify StatsLoaderTeams defaults to get_current_season() when season is None."""
        loader = StatsLoaderTeams()
        expected_season = get_current_season()
        self.assertEqual(loader.season, expected_season)
        self.assertEqual(loader.season_str, season_int_to_str(expected_season))

    @patch("utils.load_nbacom_data")
    @patch("utils.os.path.isfile", return_value=False)
    @patch("pandas.DataFrame.to_feather")
    def test_load_and_cache_players_list_defaults_to_current_season(self, mock_to_feather, mock_isfile, mock_load):
        """Verify load_and_cache_players_list defaults to current season."""
        mock_load.return_value = pd.DataFrame({
            "PLAYER_FIRST_NAME": ["Cooper"],
            "PLAYER_LAST_NAME": ["Flagg"],
            "PERSON_ID": [1642250],
            "DRAFT_NUMBER": [1]
        })
        res = load_and_cache_players_list(print_usage=False)
        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["NAME"], "Cooper Flagg")

    @patch("run_full_pipeline.define_logger")
    @patch("run_full_pipeline.StatsLoaderPlayers")
    @patch("run_full_pipeline.StatsLoaderTeams")
    @patch("run_full_pipeline.CalcFeatures")
    @patch.object(RunFullPipeline, "_reset")
    def test_run_full_pipeline_season_propagation(self, mock_reset, mock_calc, mock_teams, mock_players, mock_logger):
        """Verify RunFullPipeline propagates dynamic season to child loaders."""
        pipeline = RunFullPipeline(load_params_from_model_id=1, season=None, params_update_dict={})
        expected_season = get_current_season()
        self.assertEqual(pipeline.season, expected_season)
        mock_players.assert_called_once_with(season=expected_season)
        mock_teams.assert_called_once_with(season=expected_season)

    def test_load_fantasy_data_teams_match(self):
        """Verify _teams_match_todays_teams works with array/list inputs."""
        teams_a = np.array(["BOS", "LAL"])
        teams_b = ["LAL", "BOS"]
        self.assertTrue(LoadFantasyData._teams_match_todays_teams(teams_a, teams_b))
        self.assertFalse(LoadFantasyData._teams_match_todays_teams(["BOS"], ["LAL"]))

    def test_load_fantasy_data_convert_salary_cols(self):
        """Verify salary columns ending with _SALARY are converted to floats."""
        df = pd.DataFrame({
            "NAME": ["Player A"],
            "YAHOO_SALARY": ["25"],
            "DRAFTKINGS_SALARY": [30],
            "POINTS": [10]
        })
        LoadFantasyData._convert_salary_cols_to_num(df)
        self.assertEqual(df["YAHOO_SALARY"].dtype, float)
        self.assertEqual(df["YAHOO_SALARY"].iloc[0], 25.0)
        self.assertEqual(df["DRAFTKINGS_SALARY"].dtype, float)
        self.assertEqual(df["DRAFTKINGS_SALARY"].iloc[0], 30.0)

    @patch("data_collectors.fantasy_data.load_schedule")
    @patch("data_collectors.fantasy_data.load_fantasy_salaries_positions_lineups_hq")
    @patch("data_collectors.fantasy_data.get_first_slate_id", return_value=None)
    @patch("data_collectors.fantasy_data.load_and_cache_players_list")
    def test_load_fantasy_data_rotogrinders_fallback_calls_dynamic_players_list(
        self, mock_load_players, mock_first_slate, mock_rg_salaries, mock_schedule
    ):
        """Verify rotogrinders fallback calls load_and_cache_players_list dynamically."""
        mock_schedule.return_value = pd.DataFrame({"TEAM_ID": [1], "DATE": [datetime.date(2026, 11, 1)]})
        mock_rg_salaries.return_value = pd.DataFrame({"NAME": ["Player A"], "YAHOO_SALARY": [15]})
        mock_load_players.return_value = pd.DataFrame({"NAME": ["Player A"], "TEAM_ABBREVIATION": ["BOS"]})

        lfd = LoadFantasyData(nbacom_players_list=[])
        res = lfd._load_data_rotogrinders(date=datetime.date(2026, 11, 1))

        mock_load_players.assert_called_once_with()
        self.assertIn("TEAM", res.columns)
        self.assertEqual(res.iloc[0]["TEAM"], "BOS")

    @patch("run_full_pipeline.load_schedule")
    def test_get_game_times_by_day_dict(self, mock_schedule):
        """Verify _get_game_times_by_day_dict correctly groups game times by date."""
        mock_schedule.return_value = pd.DataFrame({
            "DATE": [datetime.date(2026, 10, 25), datetime.date(2026, 10, 25), datetime.date(2026, 10, 26)],
            "GAME_DATE_TIME": [
                pd.Timestamp("2026-10-25 19:30:00"),
                pd.Timestamp("2026-10-25 22:00:00"),
                pd.Timestamp("2026-10-26 20:00:00"),
            ]
        })
        result = RunFullPipeline._get_game_times_by_day_dict()
        self.assertIn(datetime.date(2026, 10, 25), result)
        self.assertEqual(len(result[datetime.date(2026, 10, 25)]), 2)
        self.assertEqual(len(result[datetime.date(2026, 10, 26)]), 1)

    @patch("data_collectors.starters.fetch_soup")
    def test_rotowire_starters_parser(self, mock_soup):
        """Verify LoadStartersRotowire correctly parses lineup boxes, positions, names, and confirmation."""
        html = """
        <div class="lineup__box is-confirmed">
            <div class="lineup__teams">
                <a href="#">UTA</a>
                <a href="#">DEN</a>
            </div>
            <ul class="lineup__list is-visit">
                <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Keyonte George">K. George</a></li>
                <li class="lineup__player"><div class="lineup__pos">SG</div><a title="Darryn Peterson">D. Peterson</a></li>
                <li class="lineup__player"><div class="lineup__pos">SF</div><a title="Lauri Markkanen">L. Markkanen</a></li>
                <li class="lineup__player"><div class="lineup__pos">PF</div><a title="Jaren Jackson">J. Jackson</a></li>
                <li class="lineup__player"><div class="lineup__pos">C</div><a title="Jusuf Nurkic">J. Nurkic</a></li>
            </ul>
            <ul class="lineup__list is-home">
                <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Jamal Murray">J. Murray</a></li>
                <li class="lineup__player"><div class="lineup__pos">SG</div><a title="Christian Braun">C. Braun</a></li>
                <li class="lineup__player"><div class="lineup__pos">SF</div><a title="Cameron Johnson">C. Johnson</a></li>
                <li class="lineup__player"><div class="lineup__pos">PF</div><a title="Aaron Gordon">A. Gordon</a></li>
                <li class="lineup__player"><div class="lineup__pos">C</div><a title="Nikola Jokic">N. Jokic</a></li>
            </ul>
        </div>
        """
        mock_soup.return_value = BeautifulSoup(html, "html.parser")
        loader = LoadStartersRotowire()
        loader.run()

        self.assertEqual(len(loader.df), 10)
        self.assertTrue(loader.df["CONFIRMED"].all())
        uta_starters = loader.df.query('TEAM == "UTA"')
        self.assertEqual(len(uta_starters), 5)
        self.assertEqual(set(uta_starters["START_POSITION"]), {"PG", "SG", "SF", "PF", "C"})
        self.assertIn("Keyonte George", uta_starters["NAME"].values)

        den_starters = loader.df.query('TEAM == "DEN"')
        self.assertEqual(len(den_starters), 5)
        self.assertEqual(set(den_starters["START_POSITION"]), {"PG", "SG", "SF", "PF", "C"})
        self.assertIn("Nikola Jokic", den_starters["NAME"].values)

    @patch("data_collectors.starters.fetch_soup")
    def test_rotowire_starters_empty_or_past_date(self, mock_soup):
        """Verify LoadStartersRotowire returns EMPTY_DF for past dates or empty soup."""
        mock_soup.return_value = None
        loader = LoadStartersRotowire()
        loader.run()
        self.assertTrue(loader.df.empty)

        # Past date
        past_date = datetime.date(2026, 1, 1)
        loader._load_starters(date=past_date)
        self.assertTrue(loader.df.empty)

    @patch("data_collectors.starters.requests.get")
    def test_nbacom_starters_graceful_error_handling(self, mock_get):
        """Verify LoadStartersNBACOM handles 404/empty responses gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""
        mock_get.return_value = mock_resp

        loader = LoadStartersNBACOM()
        schedule = pd.DataFrame({"DATE": [datetime.date.today()], "GAME_ID": [100]})
        loader._load_starters(schedule=schedule, date=datetime.date.today())
        self.assertTrue(loader.df.empty)

    @patch("data_collectors.starters.requests.get")
    def test_nbacom_starters_malformed_json(self, mock_get):
        """Verify LoadStartersNBACOM handles invalid JSON gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"Not a JSON object"
        mock_get.return_value = mock_resp

        loader = LoadStartersNBACOM()
        schedule = pd.DataFrame({"DATE": [datetime.date.today()], "GAME_ID": [100]})
        loader._load_starters(schedule=schedule, date=datetime.date.today())
        self.assertTrue(loader.df.empty)

    @patch("data_collectors.starters.requests.get")
    def test_nbacom_starters_happy_path(self, mock_get):
        """Verify LoadStartersNBACOM parses valid JSON payload into starters DataFrame."""
        payload = {
            "games": [{
                "gameId": "0022600100",
                "homeTeam": {
                    "teamAbbreviation": "BOS",
                    "players": [
                        {"personId": 1628369, "playerName": "Jayson Tatum", "position": "SF",
                         "rosterStatus": "Active", "lineupStatus": "Confirmed"},
                        {"personId": 1627759, "playerName": "Jaylen Brown", "position": "SG",
                         "rosterStatus": "Active", "lineupStatus": "Confirmed"}
                    ]
                },
                "awayTeam": {
                    "teamAbbreviation": "MIA",
                    "players": [
                        {"personId": 1628389, "playerName": "Bam Adebayo", "position": "C",
                         "rosterStatus": "Active", "lineupStatus": "Confirmed"}
                    ]
                }
            }]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(payload).encode("utf-8")
        mock_get.return_value = mock_resp

        loader = LoadStartersNBACOM()
        schedule = pd.DataFrame({"DATE": [datetime.date.today()], "GAME_ID": [22600100]})
        loader._load_starters(schedule=schedule, date=datetime.date.today())

        self.assertEqual(len(loader.df), 3)
        self.assertIn("Jayson Tatum", loader.df["NAME"].values)
        self.assertIn("Bam Adebayo", loader.df["NAME"].values)
        self.assertTrue(loader.df["CONFIRMED"].all())

    @patch("data_collectors.starters.fetch_soup")
    def test_rotowire_starters_per_team_confirmation_and_filtering(self, mock_soup):
        """Verify per-team confirmation status, out_players filtering, and non-starter rejection."""
        html = """
        <div class="lineup__box">
            <!-- Box with missing team links should be skipped -->
        </div>
        <div class="lineup__box">
            <div class="lineup__teams">
                <a href="#">UTA</a>
                <a href="#">DEN</a>
            </div>
            <ul class="lineup__list is-visit">
                <li class="lineup__status is-confirmed">Confirmed Lineup</li>
                <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Keyonte George">K. George</a></li>
                <li class="lineup__player"><div class="lineup__pos">UTIL</div><a title="Bench Player">B. Player</a></li>
            </ul>
            <ul class="lineup__list is-home">
                <li class="lineup__status is-expected">Expected Lineup</li>
                <li class="lineup__player"><div class="lineup__pos">PG</div><a title="Jamal Murray">J. Murray</a></li>
            </ul>
        </div>
        """
        mock_soup.return_value = BeautifulSoup(html, "html.parser")
        loader = LoadStartersRotowire()
        # Keyonte George is confirmed, so shouldn't be dropped even if in out_players.
        # Jamal Murray is unconfirmed, so should be dropped when in out_players.
        loader.run(out_players=["Keyonte George", "Jamal Murray"])

        self.assertEqual(len(loader.df), 1)
        self.assertEqual(loader.df.iloc[0]["NAME"], "Keyonte George")
        self.assertTrue(loader.df.iloc[0]["CONFIRMED"])

    @patch("data_collectors.injuries.requests.get")
    def test_injuries_nbacom_graceful_error_handling(self, mock_get):
        """Verify Injuries._load_injuries_nbacom handles HTTP errors gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.content = b""
        mock_get.return_value = mock_resp

        injuries = Injuries()
        res = injuries._load_injuries_nbacom(date=datetime.date.today())
        self.assertTrue(res.empty)

    @patch("data_collectors.injuries.requests.get")
    def test_injuries_nbacom_malformed_json(self, mock_get):
        """Verify Injuries._load_injuries_nbacom handles malformed JSON gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"{invalid"
        mock_get.return_value = mock_resp

        injuries = Injuries()
        res = injuries._load_injuries_nbacom(date=datetime.date.today())
        self.assertTrue(res.empty)

    @patch("data_collectors.injuries.date_to_game_ids", return_value=[22600100])
    @patch("data_collectors.injuries.requests.get")
    def test_injuries_nbacom_happy_path(self, mock_get, mock_gids):
        """Verify Injuries._load_injuries_nbacom correctly maps inactive players to IS_OUT."""
        payload = {
            "games": [{
                "gameId": "0022600100",
                "homeTeam": {
                    "teamAbbreviation": "BOS",
                    "players": [
                        {"personId": 1, "playerName": "Active Player", "position": "SF", "rosterStatus": "Active"},
                        {"personId": 2, "playerName": "Inactive Player", "position": "PG", "rosterStatus": "Inactive"}
                    ]
                },
                "awayTeam": {
                    "teamAbbreviation": "MIA",
                    "players": []
                }
            }]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(payload).encode("utf-8")
        mock_get.return_value = mock_resp

        injuries = Injuries()
        res = injuries._load_injuries_nbacom(date=datetime.date.today())

        self.assertEqual(len(res), 1)
        self.assertEqual(res.iloc[0]["NAME"], "Inactive Player")
        self.assertTrue(res.iloc[0]["IS_OUT"])
        self.assertFalse(res.iloc[0]["IS_QUESTIONABLE"])

    # -------------------------------------------------------------------------
    # 20. Stage 4 - Directory Structure & Cache Unification Tests
    # -------------------------------------------------------------------------
    def test_update_dir_name_canonical_cache_mapping(self):
        """Verify update_dir_name routes cache categories canonically under cached_data."""
        expected_cached = [
            'starters',
            'injuries',
            'fantasy_data',
            'ejections',
            'nbacom_single_game_boxscores',
            'nbacom_single_game_pbp',
            'rotogrinders_lineups',
            'left_midgame',
        ]
        for cat in expected_cached:
            resolved = update_dir_name(cat)
            expected = os.path.join(DATA_DIR, 'cached_data', cat)
            self.assertEqual(resolved, expected, f"Failed canonical routing for {cat}")

            # Explicit cached_data/cat should resolve identically
            resolved_explicit = update_dir_name(f'cached_data/{cat}')
            self.assertEqual(resolved_explicit, expected)

        # Root-level data directories should resolve directly under DATA_DIR
        for root_cat in ['stats', 'season_schedules', 'features', 'comments_nbacom', 'dfs_contests', 'logs']:
            resolved_root = update_dir_name(root_cat)
            expected_root = os.path.join(DATA_DIR, root_cat)
            self.assertEqual(resolved_root, expected_root)

        # Base cached_data resolution
        self.assertEqual(update_dir_name('cached_data'), os.path.join(DATA_DIR, 'cached_data'))

    def test_collectors_save_dir_consistency(self):
        """Verify all collectors initialize save_dir using canonical update_dir_name paths."""
        starters = Starters()
        self.assertEqual(starters.save_dir, update_dir_name('starters'))
        self.assertEqual(starters.save_dir, os.path.join(DATA_DIR, 'cached_data', 'starters'))

        injuries = Injuries()
        self.assertEqual(injuries.save_dir, update_dir_name('injuries'))
        self.assertEqual(injuries.save_dir, os.path.join(DATA_DIR, 'cached_data', 'injuries'))

        fantasy = LoadFantasyData()
        self.assertEqual(fantasy.save_dir, update_dir_name('fantasy_data'))
        self.assertEqual(fantasy.save_dir, os.path.join(DATA_DIR, 'cached_data', 'fantasy_data'))

        ejections = Ejections()
        self.assertEqual(ejections.save_dir, update_dir_name('ejections'))
        self.assertEqual(ejections.save_dir, os.path.join(DATA_DIR, 'cached_data', 'ejections'))

    def test_cached_data_unified_coverage(self):
        """Verify that cached_data contains unified historical data across seasons with zero loss."""
        # 1. Boxscores: must contain both 2024-25 (2240xxxx) and 2025-26 (2250xxxx) games
        boxscore_dir = update_dir_name('nbacom_single_game_boxscores')
        if os.path.exists(boxscore_dir):
            boxscore_files = os.listdir(boxscore_dir)
            self.assertGreaterEqual(len(boxscore_files), 2460, "Must contain all 2,460 boxscores for 24-25 and 25-26")
            has_2024 = any("2240" in f for f in boxscore_files)
            has_2025 = any("2250" in f for f in boxscore_files)
            self.assertTrue(has_2024, "Missing 2024-25 season boxscores in unified cache")
            self.assertTrue(has_2025, "Missing 2025-26 season boxscores in unified cache")

        # 2. Starters & Injuries: must contain March-April 2025 files and 2025-26 files
        for cat in ['starters', 'injuries', 'fantasy_data']:
            cat_dir = update_dir_name(cat)
            if os.path.exists(cat_dir):
                files = os.listdir(cat_dir)
                self.assertGreaterEqual(len(files), 189, f"Must contain at least 189 unified daily files for {cat}")
                has_2025_spring = any("2025_03_" in f or "2025_04_" in f for f in files)
                has_2025_autumn = any("2025_10_" in f or "2025_11_" in f for f in files)
                self.assertTrue(has_2025_spring, f"Missing 2025 spring historical data in {cat}")
                self.assertTrue(has_2025_autumn, f"Missing 2025 autumn data in {cat}")


if __name__ == "__main__":
    unittest.main()
