import datetime
import json
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from bs4 import BeautifulSoup
from data_collectors.injuries import Injuries, InjuriesSources
from data_collectors.ejections import Ejections
from data_collectors.starters import LoadStartersBasketballMonster
from data_collectors.stats_loaders import StatsLoaderPlayers
from utils import (read_and_cache, fetch_soup, parse_html_table, fetch_html_table,
                   get_current_season, season_int_to_str, date_to_season_str, assert_valid_df_all)


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
    @patch("utils.requests.get")
    def test_ejections_spotrac_non_200_returns_empty_df(self, mock_get):
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


if __name__ == "__main__":
    unittest.main()
