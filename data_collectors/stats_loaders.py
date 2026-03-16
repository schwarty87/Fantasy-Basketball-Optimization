import os
import datetime
import logging
import json
import time
import requests
import numpy as np
import pandas as pd
from typing import Optional, Dict
from tqdm import tqdm
from nba_api.stats.endpoints import boxscoretraditionalv2, playergamelogs, teamgamelogs
from constants import URLS, SCORING, TEAM_NAME_TO_ID, TEAM_ID_TO_NAME
from enums import StatsTypes, BoxscoreTypes, Sites
from utils import downcast_df, load_and_cache_players_list, season_int_to_str, load_nbacom_data, read_and_cache, \
    get_latest_date_with_data, assert_valid_df_all, assert_valid_df_players, load_schedule, get_current_time, \
    get_current_date, FixPlayerNames, get_header, update_dir_name
from data_collectors.fantasy_data import LoadFantasyData
from data_collectors.injuries import Injuries
from data_collectors.starters import Starters
from data_collectors.ejections import Ejections

FILL_ZEROS_COLS = ['MIN', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL',
                   'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'FG_PCT', '3P_PCT', 'FT_PCT']


class StatsLoader:
    """
    parent class for loading stats. both the classes for loading/scraping players stats and team stats inherit form
    this class
    """

    LOADER_TYPE = None
    BOXSCORE_TYPES = [BoxscoreTypes.Base]
    SEASON_TYPES = ['Regular Season']  #, 'IST']
    MERGE_ON = ['DATE', 'TEAM_ID']
    DROP_COLS = []
    REPLACE_COLUMN_NAMES = {'PLAYER_NAME': 'NAME', 'TEAM_ABBREVIATION': 'TEAM', 'GAME_DATE': 'DATE',
                            'MATCHUP': 'OPPONENT', 'WL': 'W_L', 'FG3M': '3PM', 'FG3A': '3PA', 'OFF_RATING': 'OFFRTG',
                            'FG3_PCT': '3P_PCT', 'PCT_FG3M': 'PCT_3PM', 'DEF_RATING': 'DEFRTG', 'AST_TOV': 'AST_TO',
                            'PCT_FG3A': 'PCT_3PA', 'NET_RATING': 'NETRTG', 'DD2': 'DOUBLE_DOUBLE',
                            'TD3': 'TRIPLE_DOUBLE'}
    SORT_COLS = []

    def __init__(self, season: int = 2023, **kwargs):
        """
        :param season:the season to collect stats for (2022 will collect stats for the 2022-23 season)
        """
        self.season = season
        self.season_str = season_int_to_str(season)

        self._stats_dir = update_dir_name('stats')
        self.filepath = os.path.join(self._stats_dir, f'{self.LOADER_TYPE.name}_boxscores.feather')

        self.df = pd.DataFrame(columns=self.SORT_COLS)
        self.latest_date_with_data = datetime.date(2000, 1, 1)
        self._reset_updates_df()

    def  call(self, add_upcoming_games: bool = True):
        self._reset_updates_df()
        self.load_saved_data()
        self.scrape_boxscores_from_nbacom()
        self._add_extra_data_to_updates_df(add_upcoming_games=add_upcoming_games)
        self._postprocessing()
        self._combine_old_new_dfs()
        self._save_df()

    def _reset_updates_df(self):
        self.updates_df = pd.DataFrame(columns=self.SORT_COLS)

    def load_saved_data(self) -> pd.DataFrame | None:
        if os.path.exists(self.filepath):
            self.df = pd.read_feather(self.filepath)
            self.df['DATE'] = pd.to_datetime(self.df['DATE']).dt.date

            if 'OPPONENT_ID' in self.df:
                self.df.drop(columns=['OPPONENT_ID'], inplace=True)

            self.df.sort_values(self.SORT_COLS, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            downcast_df(self.df)
            self.df[['GAME_ID', 'TEAM_ID']] = self.df[['GAME_ID', 'TEAM_ID']].astype('int64')
            self.df.drop_duplicates(inplace=True)
            self.latest_date_with_data = get_latest_date_with_data(self.df)
            self.assert_valid_df(self.df)

            return self.df
        return None

    def scrape_single_boxscore_type_from_nbacom(self, boxscore_type: BoxscoreTypes,
                                                season_type: str = 'Regular Season') -> pd.DataFrame:
        raise NotImplementedError

    def scrape_boxscores_from_nbacom(self):
        yesterdays_date = (get_current_time() - datetime.timedelta(days=1)).date()
        if self.latest_date_with_data >= yesterdays_date:
            return
        self.updates_df = list()
        for season_type in self.SEASON_TYPES:
            season_type_dfs = list()
            for i, boxscore_type in tqdm(enumerate(self.BOXSCORE_TYPES), total=len(self.BOXSCORE_TYPES),
                                         desc=f'scraping boxscores from nba.com for {self.LOADER_TYPE.name}'):
                curr_df = self.scrape_single_boxscore_type_from_nbacom(boxscore_type=boxscore_type,
                                                                       season_type=season_type)

                curr_df.rename(columns=self.REPLACE_COLUMN_NAMES, inplace=True)
                curr_df['DATE'] = pd.to_datetime(curr_df['DATE']).dt.date

                # only save new data that we don't have already
                curr_df = curr_df.loc[curr_df['DATE'] > self.latest_date_with_data]
                if curr_df.empty:
                    break

                # drop unneeded or duplicate columns
                drop_cols = (set(curr_df.columns) & set(self.DROP_COLS)).union(
                    set([c for c in curr_df if c.endswith('_RANK') or c.startswith('E_') or c.startswith('sp_work_')]))
                duplicate_cols = set(curr_df.columns) & set(sum([list(df.columns) for df in season_type_dfs], []))
                drop_cols = drop_cols.union(duplicate_cols)
                curr_df.drop(columns=list(drop_cols), inplace=True)

                curr_df.set_index(self.MERGE_ON, inplace=True)
                season_type_dfs.append(curr_df)
            if len(season_type_dfs) > 0:
                assert len(np.unique([df.shape[0] for df in season_type_dfs])) == 1, \
                    "the season_type_dfs don't have the same number of rows!"
                self.updates_df.append(pd.concat(season_type_dfs, axis=1).reset_index(drop=False))
        if len(self.updates_df) == 0:
            self.updates_df = pd.DataFrame(columns=self.SORT_COLS)
            return
        assert len(np.unique([df.shape[1] for df in self.updates_df])) == 1, \
            "the updates_dfs don't have the same number of columns!"
        self.updates_df = pd.concat(self.updates_df, axis=0).reset_index(drop=True)
        self.updates_df.drop_duplicates(self.MERGE_ON, inplace=True)

        split_opp = list(map(lambda x: x.split(), self.updates_df['OPPONENT']))
        vals = [(int(i[1] == 'vs.'), i[2]) for i in split_opp]
        self.updates_df['HOME'] = [i[0] for i in vals]
        self.updates_df['OPPONENT'] = [i[1] for i in vals]
        self.updates_df['W_L'] = (self.updates_df['W_L'] == 'W').astype(int)
        self.updates_df['DATE'] = pd.to_datetime(self.updates_df['DATE']).dt.date
        self.updates_df['SEASON_YEAR'] = self.updates_df['SEASON_YEAR'].str.split('-').str[0].astype(int)
        self.updates_df.sort_values(self.SORT_COLS, inplace=True)
        self.updates_df = self.updates_df.loc[self.updates_df['TEAM_ID'].isin(TEAM_ID_TO_NAME.keys())]
        downcast_df(self.updates_df)
        self.assert_valid_df(self.updates_df)

    def _add_extra_data_to_updates_df(self, add_upcoming_games: bool = True):
        pass

    def _postprocessing(self):
        pass

    def _combine_old_new_dfs(self):
        if self.df is None or self.df.empty:
            self.df = self.updates_df
        elif not self.updates_df.empty:
            dates_not_in_updates = ~self.df['DATE'].isin(self.updates_df['DATE'].unique())
            self.df = pd.concat([self.df.loc[dates_not_in_updates], self.updates_df], axis=0)
            self.df.sort_values(self.SORT_COLS, inplace=True)

    def assert_valid_df(self, df: pd.DataFrame):
        assert_valid_df_all(df)

    def _save_df(self):
        if not self.updates_df.empty:
            if 'AVAILABLE_FLAG' in self.df.columns:
                self.df['AVAILABLE_FLAG'] = self.df['AVAILABLE_FLAG'].fillna(0).astype(bool).value_counts()
            self.df.sort_values(self.SORT_COLS, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.assert_valid_df(self.df)
            logging.info(f'saving {self.LOADER_TYPE.name} boxscores to {self.filepath}')

            if 'VIDEO_AVAILABLE' in self.df.columns:
                self.df.drop(columns=['VIDEO_AVAILABLE'], inplace=True)

            self.df.to_feather(self.filepath)


class StatsLoaderPlayers(StatsLoader):
    LOADER_TYPE = StatsTypes.player
    BOXSCORE_TYPES = [BoxscoreTypes.Base]  # , BoxscoreTypes.Advanced, BoxscoreTypes.Usage]
    DROP_COLS = ['NICKNAME', 'TEAM_NAME', 'NBA_FANTASY_PTS', 'WNBA_FANTASY_PTS', 'VIDEO_AVAILABLE_FLAG']
    SORT_COLS = ['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID']
    MERGE_ON = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']

    def __init__(self, season: Optional[int] = 2025, out_players: Optional[Dict] = None, **kwargs):
        """
        :param season: the season we're loading data for (for example if season==2022, then we're working on 2022-23)
        :param out_players: a dict of players that are out today and their replacements in the starting lineup.
                            used when scraping data from rotogrinders
        """
        super(StatsLoaderPlayers, self).__init__(season=season, **kwargs)

        self.out_players = out_players

        self._players_list = None
        self._player_id_to_name = None
        self._player_name_to_id = None

        self._schedule = load_schedule(drop_ppd=True)

    def call(self, players_list: Optional[pd.DataFrame] = None, add_upcoming_games: bool = True):
        if players_list is not None:
            self._players_list = players_list
        else:
            self._players_list = load_and_cache_players_list(season=self.season)
        self._player_id_to_name = dict(zip(self._players_list['PLAYER_ID'], self._players_list['NAME']))
        self._player_name_to_id = dict(zip(self._players_list['NAME'], self._players_list['PLAYER_ID']))

        super(StatsLoaderPlayers, self).call(add_upcoming_games=add_upcoming_games)

    def load_saved_data(self) -> pd.DataFrame:
        super().load_saved_data()
        self.df['MIN'] = self.df['MIN'].fillna(0)
        self.df = self.df.loc[self.df['NAME'].notnull()]
        for c in ('LEFT_MIDGAME', 'EJECTED', 'IS_OUT'):
            self.df[c] = self.df[c].astype(bool)
            self.df[c] = self.df[c].fillna(False)
        self.df['PLAYER_ID'] = self.df['PLAYER_ID'].astype('int64')

        self.df['COMMENT'] = self.df['COMMENT'].fillna('')
        self.df.loc[(self.df['MIN'] > 0) & (self.df['COMMENT'].str.contains('DNP')), 'COMMENT'] = ''

        if not self.df.empty:
            self.df['NAME'] = FixPlayerNames().fix_names(self.df['NAME'])

        # adding missing ids from df
        if self._player_name_to_id is not None:
            already_in = set(self._player_id_to_name.keys()) if self._player_name_to_id is not None else set()
            ids_to_add = list(set(self.df['PLAYER_ID'].unique()) - already_in)
            add_dict = self.df.drop_duplicates('PLAYER_ID').set_index('PLAYER_ID').loc[ids_to_add, 'NAME'].to_dict()
            self._player_id_to_name.update(add_dict)
            self._player_name_to_id.update({v: k for k, v in add_dict.items()})

        if 'VIDEO_AVAILABLE' in self.df.columns:
            self.df.drop(columns=['VIDEO_AVAILABLE'], inplace=True)

        return self.df

    def scrape_boxscores_from_nbacom(self):
        super(StatsLoaderPlayers, self).scrape_boxscores_from_nbacom()
        if not self.updates_df.empty:
            self.updates_df['NAME'] = FixPlayerNames().fix_names(self.updates_df['NAME'])

    def scrape_single_boxscore_type_from_nbacom(self, boxscore_type: BoxscoreTypes,
                                                season_type: str = 'Regular Season') -> pd.DataFrame:
        try:
            curr_url = URLS().boxscores_nbacom2(player_or_team=self.LOADER_TYPE.name.upper()[0],
                                                season=self.season_str,
                                                season_type=season_type)
            curr_df = load_nbacom_data(curr_url)
            curr_df['SEASON_YEAR'] = curr_df['SEASON_ID'].astype(str).str[1:]
            curr_df.drop(columns=['SEASON_ID'], inplace=True)
            for c in ('BLKA', 'PFD'):
                if c not in curr_df.columns:
                    curr_df[c] = -1
            double_digits = (curr_df[['PTS', 'REB', 'AST', 'STL', 'BLK']] >= 10).sum(axis=1)
            curr_df['DOUBLE_DOUBLE'] = double_digits == 2
            curr_df['TRIPLE_DOUBLE'] = double_digits == 3
            # print('aaa >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        except requests.exceptions.ConnectionError:
            try:
                curr_url = URLS().boxscores_nbacom(player_or_team=self.LOADER_TYPE.name,
                                                   measure_type=boxscore_type.name,
                                                   season=self.season_str,
                                                   season_type=season_type)
                curr_df = load_nbacom_data(curr_url)
                # print('bbb >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            except requests.exceptions.ConnectionError:
                player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=self.season_str,
                                                                 measure_type_player_game_logs_nullable=boxscore_type.name,
                                                                 season_type_nullable=season_type,
                                                                 headers=get_header())
                curr_df = player_game_logs.get_data_frames()[0]
                # print('ccc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        return curr_df

    def get_player_name_to_id(self) -> Dict[str, int]:
        return self._player_name_to_id

    def _read_single_game_boxscores_comments(self):
        if self.updates_df is None or self.updates_df.empty:
            return

        game_ids = self.updates_df.loc[self.updates_df['DATE'] > self.latest_date_with_data, 'GAME_ID'].unique()
        if len(game_ids) == 0:
            return

        # if we're running on a past season then we should have a file with all the information for the entire season,
        # otherwise we have a file for each game
        full_season_comments_path = f'comments_nbacom/{self.season}.feather'
        if os.path.isfile(full_season_comments_path):
            all_dfs = pd.read_feather(full_season_comments_path)
            all_dfs = all_dfs.loc[all_dfs['GAME_ID'].isin(game_ids)]
        else:
            keep_cols = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'START_POSITION', 'COMMENT']
            all_dfs = list()
            for count, gid in tqdm(enumerate(sorted(game_ids)), desc='getting boxscore comments from nba.com',
                                   total=len(game_ids)):
                cache_name = f'nbacom_single_game_boxscores/nbacom_boxscoretraditionalv2_gid_{gid}.feather'
                gid_str = str(gid)
                if len(gid_str) < 10:
                    gid_str = '0' * (10 - len(gid_str)) + gid_str

                try:
                    boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2
                    tmp_df = read_and_cache(
                        func=lambda x: boxscore(game_id=x, headers=get_header(), timeout=10).get_data_frames()[0][
                            keep_cols],
                        func_params={'x': gid_str},
                        cache_name=cache_name
                    )
                except (requests.exceptions.ReadTimeout, IndexError, json.decoder.JSONDecodeError):
                    curr_url = URLS().single_boxscores_nbacom(gid=gid_str)
                    tmp_df = read_and_cache(lambda x: load_nbacom_data(url=x)[keep_cols], func_params={'x': curr_url},
                                            cache_name=cache_name)
                all_dfs.append(tmp_df)
                time.sleep(0.6)
            all_dfs = pd.concat(all_dfs, axis=0).reset_index(drop=True)

        # # changing the starters positions from G/F/C to PG/SG/SF/PF/C
        # first_ind_in_team = np.where((all_dfs['TEAM_ID'] != all_dfs['TEAM_ID'].shift(1)) |
        #                              (all_dfs['GAME_ID'] != all_dfs['GAME_ID'].shift(1)))[0]
        # for i, prefix in enumerate(['S', 'P']):
        #     add_inds = (first_ind_in_team + i).tolist() + (first_ind_in_team + i + 3).tolist()
        #     assert all_dfs.loc[add_inds, 'START_POSITION'].isin(['F', 'G']).all(), \
        #         f'indexing is off when adding prefix {prefix} to START_POSITION!!!'
            # all_dfs.loc[add_inds, 'START_POSITION'] = prefix + all_dfs.loc[add_inds, 'START_POSITION']

        # all_dfs['START_POSITION'].replace({'': np.nan}, inplace=True)
        for c in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']:
            all_dfs[c] = all_dfs[c].astype('int64')
        if 'START_POSITION' in all_dfs:
            all_dfs.drop(columns=['START_POSITION'], inplace=True)
        self.updates_df = self.updates_df.merge(all_dfs, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='outer')

        # filling nans for NAME
        self.updates_df['NAME'] = \
            self.updates_df['NAME'].fillna(self.updates_df['PLAYER_ID'].replace(self._player_id_to_name))
        self.updates_df.loc[self.updates_df['PLAYER_ID'] == self.updates_df['NAME'], 'NAME'] = np.nan

        # filling nans for team stats
        fill_cols = ['DATE', 'SEASON_YEAR', 'TEAM', 'OPPONENT', 'W_L', 'HOME']
        merge_on = ['TEAM_ID', 'GAME_ID']
        merge_df = self.updates_df.dropna(subset=['DATE', 'TEAM'], how='any').drop_duplicates(['DATE', 'TEAM'])
        merge_df = merge_df[merge_on + fill_cols]
        for c in merge_on:
            merge_df[c] = merge_df[c].astype('int64')
        self.updates_df = self.updates_df.merge(merge_df, on=merge_on, how='left', suffixes=(None, '_x'))
        for c in fill_cols:
            self.updates_df[c] = self.updates_df[c].fillna(self.updates_df[c + '_x'])
        self.updates_df.drop(columns=[c + '_x' for c in fill_cols], inplace=True)

        self.updates_df[FILL_ZEROS_COLS] = self.updates_df[FILL_ZEROS_COLS].fillna(0)
        self.updates_df[['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE']] = \
            self.updates_df[['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE']].fillna(False)

        drop_inds = self.updates_df['NAME'].isnull() & (self.updates_df['MIN'] == 0)
        self.updates_df = self.updates_df.loc[~drop_inds]

        self.updates_df.sort_values(self.SORT_COLS, inplace=True)
        self.updates_df['NAME'] = FixPlayerNames().fix_names(self.updates_df['NAME'])
        downcast_df(self.updates_df)
        self.assert_valid_df(self.updates_df)

    # def _read_salaries_data_rotogrinders(self, add_upcoming_games: bool = True):
    #     full_season_data_path = glob.glob(f'*_data/{self.season}.feather')
    #     if len(full_season_data_path) > 0:
    #         full_season_data_path = full_season_data_path[0]
    #         rotogrinders_df = pd.read_feather(full_season_data_path)
    #         self.updates_df = self.updates_df.merge(rotogrinders_df, on=['NAME', 'DATE'], how='left',
    #                                                 suffixes=(None, '_ROTOGRINDERS'))
    #         self.updates_df.sort_values(self.SORT_COLS, inplace=True)
    #         downcast_df(self.updates_df)
    #         self.assert_valid_df(self.updates_df)
    #         return
    #     elif self.season < 2022:
    #         copy_cols = ['GAME_ID', 'PLAYER_ID', 'YAHOO_POSITION', 'YAHOO_SALARY', 'DRAFTKINGS_POSITION',
    #                      'DRAFTKINGS_SALARY', 'FANDUEL_POSITION', 'FANDUEL_SALARY']
    #         old_boxscores = pd.read_csv(f'/home/danieli/DFS/stats/player_boxscores_{self.season}_{self.season+1}.csv')
    #         old_boxscores.columns = [c.upper() for c in old_boxscores.columns]
    #         for c in [cc + '_POSITION' for cc in ALL_SITES]:
    #             old_boxscores.loc[old_boxscores[c] <= 0, c] = np.nan
    #         old_boxscores['YAHOO_POSITION'] = old_boxscores['YAHOO_POSITION'].replace(NUM_TO_POSITION)
    #         not_nan = old_boxscores['FANDUEL_POSITION'].notnull()
    #         old_boxscores.loc[not_nan, 'FANDUEL_POSITION'] = \
    #             old_boxscores.loc[not_nan, 'FANDUEL_POSITION'].astype(int).apply(
    #                 lambda x: '/'.join([NUM_TO_POSITION[int(xx)] for xx in str(x)]))
    #         not_nan = old_boxscores['DRAFTKINGS_POSITION'].notnull()
    #         old_boxscores.loc[not_nan, 'DRAFTKINGS_POSITION'] = \
    #             old_boxscores.loc[not_nan, 'DRAFTKINGS_POSITION'].astype(int).apply(
    #                 lambda x: '/'.join([NUM_TO_POSITION[int(xx)] for xx in str(x)]))
    #         self.updates_df = self.updates_df.merge(old_boxscores[copy_cols], on=['GAME_ID', 'PLAYER_ID'], how='left')
    #         self.updates_df.sort_values(self.SORT_COLS, inplace=True)
    #         downcast_df(self.updates_df)
    #         self.assert_valid_df(self.updates_df)
    #         return
    #
    #     dates_to_run = list()
    #     if add_upcoming_games:
    #         dates_to_run.append(get_current_date())
    #     if not self.updates_df.empty:
    #         dates_to_run = sorted(set(self.updates_df['DATE'].unique()).union(dates_to_run))
    #
    #     rotogrinders_lineup = RotoGrindersLineup(add_date_column=True, add_id_columns=False)
    #     salary_df = rotogrinders_lineup.run(dates_to_run, out_players=self.out_players)
    #     if not salary_df.empty:
    #         compare_names = self._players_list.query(f'TO_YEAR>={self.season}')['NAME']
    #         salary_df['NAME'] = FixPlayerNames(compare_names=compare_names).fix_names(salary_df['NAME'])
    #
    #     salary_df = salary_df.loc[salary_df['NAME'] != '']
    #     if salary_df.empty:
    #         return
    #
    #     salary_df['PLAYER_ID'] = salary_df['NAME'].replace(self._player_name_to_id)
    #
    #     # filling nan team names
    #     salary_df = salary_df.merge(self._players_list[['PLAYER_ID', 'TEAM_ABBREVIATION']], on='PLAYER_ID', how='left')
    #     salary_df['TEAM'] = salary_df['TEAM'].fillna(salary_df['TEAM_ABBREVIATION'])
    #     salary_df.drop(columns=['TEAM_ABBREVIATION', 'PLAYER_ID'], inplace=True)
    #
    #     if 'GAME_ID' in salary_df.columns:
    #         salary_df.drop(columns='GAME_ID', inplace=True)
    #     salary_df['TEAM'] = salary_df['TEAM'].replace(FIX_TEAM_NAMES)
    #     if 'OPPONENT' in salary_df.columns:
    #         salary_df['OPPONENT'] = salary_df['OPPONENT'].replace(FIX_TEAM_NAMES)
    #     salary_df['TEAM_ID'] = salary_df['TEAM'].replace(TEAM_IDS)
    #
    #     # removing postponed games
    #     schedule = load_schedule(drop_ppd=False)
    #     postponed_games = schedule.query('gameStatusText == "PPD"')
    #     salary_df = salary_df.merge(postponed_games[['DATE', 'TEAM_ID']], on=['DATE', 'TEAM_ID'], how='left',
    #                                 indicator='xxx')
    #     salary_df = salary_df.loc[salary_df['xxx'] == "left_only"].reset_index(drop=True).drop(columns='xxx')
    #
    #     salary_df.loc[salary_df['STARTER'].isnull(), 'STARTER'] = np.nan
    #     salary_df.rename(columns={'STARTER': 'START_POSITION'}, inplace=True)
    #     salary_df['SEASON_YEAR'] = self.season
    #
    #     # removing players who don't have an id and aren't starters
    #     remove_inds = (salary_df['PLAYER_ID'] == salary_df['NAME']) & salary_df['START_POSITION'].isnull()
    #     if remove_inds.any():
    #         remove_names = salary_df.loc[remove_inds, 'NAME'].unique()
    #         logging.info(f"#####################################################################\n"
    #                      f"removing the following players since they don't have a PLAYER_ID:\n{remove_names}\n"
    #                      f"#####################################################################")
    #         salary_df = salary_df.loc[~remove_inds].reset_index(drop=True)
    #
    #     if self.updates_df.empty:
    #         self.updates_df = salary_df
    #         for c in FILL_ZEROS_COLS + ['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE']:
    #             if c not in self.updates_df.columns:
    #                 self.updates_df[c] = 0 if c in FILL_ZEROS_COLS else False
    #     else:
    #         todays_date = get_current_date()
    #         self.updates_df = self.updates_df.merge(salary_df, on=['PLAYER_ID', 'DATE'], how='left',
    #                                                 suffixes=(None, '_ROTOGRINDERS'))
    #         salary_df_today = salary_df.loc[salary_df['DATE'] == todays_date]
    #         if not salary_df_today.empty:
    #             self.updates_df = pd.concat([self.updates_df, salary_df_today], axis=0)
    #             self.updates_df.loc[self.updates_df['DATE'] == todays_date, FILL_ZEROS_COLS] = 0
    #
    #     # filling nans for today's games
    #     if not self.updates_df.empty:
    #         rotogrinders_cols = [c for c in self.updates_df if c.endswith('_ROTOGRINDERS')]
    #         today_inds = self.updates_df['DATE'] == get_current_date()
    #         for c in rotogrinders_cols:
    #             c_stripped = c.replace('_ROTOGRINDERS', '')
    #             if c_stripped in self.updates_df.columns:
    #                 self.updates_df.loc[today_inds, c_stripped] = \
    #                     self.updates_df.loc[today_inds, c_stripped].fillna(self.updates_df.loc[today_inds, c])
    #         self.updates_df.drop(columns=rotogrinders_cols, inplace=True)
    #
    #     # adding GAME_ID/HOME
    #     self.updates_df = self.updates_df.merge(self._schedule[['TEAM_ID', 'DATE', 'GAME_ID', 'HOME']],
    #                                             on=['TEAM_ID', 'DATE'], how='left', suffixes=(None, '_x'))
    #     if 'GAME_ID_x' in self.updates_df.columns:
    #         self.updates_df['GAME_ID'] = self.updates_df['GAME_ID'].fillna(self.updates_df['GAME_ID_x'])
    #         self.updates_df.drop(columns=['GAME_ID_x'], inplace=True)
    #     if 'HOME_x' in self.updates_df.columns:
    #         self.updates_df['HOME'] = self.updates_df['HOME'].fillna(self.updates_df['HOME_x'])
    #         self.updates_df.drop(columns=['HOME_x'], inplace=True)
    #
    #     downcast_df(self.updates_df)
    #     self.assert_valid_df(self.updates_df)

    # def _remove_injured_players(self, add_upcoming_games: bool = True):
    #     if add_upcoming_games and get_current_date() in self.updates_df['DATE'].values:
    #         injured_players = Injuries(nbacom_players_list=self._players_list['NAME'])
    #         injured_players.collect_data()
    #         out_player_names = injured_players.out_players()
    #
    #         # dealing with starters who are listed as injured
    #         drop_starters = (self.updates_df['DATE'].values == get_current_date()) & \
    #                         self.updates_df['NAME'].isin(out_player_names).values & \
    #                         self.updates_df['START_POSITION'].notnull().values
    #         if drop_starters.any():
    #             logging.warning('There is an injured starter!!!!')
    #             for i, injured_starter_row in self.updates_df.loc[drop_starters].iterrows():
    #                 possible_replacements = \
    #                     self.updates_df.loc[(self.updates_df['DATE'] == injured_starter_row['DATE']) &
    #                                         (self.updates_df['TEAM_ID'] == injured_starter_row['TEAM_ID']) &
    #                                         self.updates_df['START_POSITION'].isnull() &
    #                                         (self.updates_df['YAHOO_POSITION'] == injured_starter_row['START_POSITION'])]
    #                 if len(possible_replacements) > 0:
    #                     replacement_ind = possible_replacements['YAHOO_SALARY'].idxmax()
    #                     self.updates_df.loc[replacement_ind, 'START_POSITION'] = injured_starter_row['START_POSITION']
    #                     self.updates_df.drop(index=i, inplace=True)
    #                     logging.info(f'replacing injured starter {injured_starter_row["NAME"]} with '
    #                                  f'{self.updates_df.loc[replacement_ind, "NAME"]}')
    #                 else:
    #                     logging.info(f'replacing injured starter {injured_starter_row["NAME"]} with fake name/id...')
    #                     self.updates_df.loc[i, 'NAME'] = 'Injured Starter'
    #                     self.updates_df.loc[i, 'PLAYER_NAME'] = self._players_list['PLAYER_ID'].max() + 1
    #
    #         # removing bench players
    #         drop_bench = (self.updates_df['DATE'].values == get_current_date()) & \
    #                      self.updates_df['NAME'].isin(out_player_names).values & \
    #                      (~self.updates_df['START_POSITION'].notnull()).values
    #         if drop_bench.any():
    #             logging.info(f'dropping injured bench players: {self.updates_df.loc[drop_bench, "NAME"].to_list()}')
    #             self.updates_df = self.updates_df.loc[~drop_bench]
    #
    #         self.updates_df.reset_index(drop=True, inplace=True)

    def _add_extra_data_to_updates_df(self, add_upcoming_games: bool = True):
        if (not add_upcoming_games) and (self.updates_df is None or self.updates_df.empty):
            return

        self._read_single_game_boxscores_comments()

        players_list_names = self._players_list['NAME'].to_list()
        ejections = Ejections(nbacom_players_list=players_list_names)
        ejections_df = ejections.collect_data()

        update_dates = self.updates_df['DATE'].unique().tolist()
        if add_upcoming_games:
            update_dates.append(get_current_date())
        extra_data = list()
        injuries = Injuries(nbacom_players_list=players_list_names, player_id_to_name=self._player_id_to_name)
        load_fantasy_data = LoadFantasyData(nbacom_players_list=players_list_names)
        load_starters = Starters(nbacom_players_list=players_list_names, player_name_to_id=self._player_name_to_id)
        for date in tqdm(update_dates, desc='loading extra data per date'):
            # loading injuries data
            injuries.collect_data(date=date)
            injured_players = injuries.out_players()

            # removing players who played that day
            # TODO this should be added to Injuries!
            if not self.updates_df.empty:
                players_played_today = self.updates_df.loc[(self.updates_df['DATE'] == date) &
                                                           (self.updates_df['MIN'] > 0), 'NAME']
                left_midgame = set(players_played_today) & set(injured_players)
                injured_players = [p for p in injured_players if p not in left_midgame]
            else:
                left_midgame = list()

            # loading fantasy positions/salaries data
            if self.updates_df.empty:
                name_to_team = None
            else:
                name_to_team = \
                    self.updates_df.loc[self.updates_df['DATE'] == date][['NAME', 'TEAM']].set_index('NAME').iloc[:, 0].to_dict()
            fantasy_data_df = load_fantasy_data.collect_data(date=date, name_to_team=name_to_team)
            fantasy_data_df['IS_OUT'] = fantasy_data_df['NAME'].isin(injured_players).values

            # loading starters
            starters_df = load_starters.collect_data(date=date, out_players=injured_players,
                                                     salaries_df=fantasy_data_df)

            # combine starters and salaries/positions
            combined = starters_df.merge(fantasy_data_df, on=['NAME', 'TEAM'], how='outer', suffixes=(None, '_x'))
            x_cols = [c for c in combined if c.endswith('_x')]
            non_x_cols = [c.strip('_x') for c in x_cols]
            combined[non_x_cols] = \
                combined[non_x_cols].fillna(combined[x_cols].rename(columns=lambda x: x.rstrip('_x')))
            combined.drop(columns=x_cols, inplace=True)
            combined['DATE'] = date
            combined['LEFT_MIDGAME'] = combined['NAME'].isin(left_midgame)

            extra_data.append(combined)
        extra_data = pd.concat(extra_data, axis=0)

        # adding ejections data
        extra_data = extra_data.merge(ejections_df, on=['NAME', 'TEAM', 'DATE'], how='left', indicator='EJECTED')
        extra_data['EJECTED'] = extra_data['EJECTED'] == 'both'
        extra_data.loc[extra_data['START_POSITION'].notnull(), 'IS_OUT'] = False

        # adding extra data to updates df
        if 'PLAYER_ID' not in extra_data.columns:
            extra_data['PLAYER_ID'] = np.nan
        if 'TEAM_ID' not in extra_data.columns:
            extra_data['TEAM_ID'] = np.nan
        if self.updates_df.empty:
            self.updates_df = extra_data
        else:
            # we'll only keep players who played or have a comment about why they didn't play
            self.updates_df = self.updates_df.merge(extra_data.loc[extra_data['DATE'] != get_current_date()],
                                                    on=['NAME', 'TEAM', 'DATE'], how='left', suffixes=(None, '_x'))
            x_cols = [c for c in self.updates_df if c.endswith('_x')]
            non_x_cols = [c.strip('_x') for c in x_cols]
            self.updates_df[non_x_cols] = \
                self.updates_df[non_x_cols].fillna(self.updates_df[x_cols].rename(columns=lambda x: x.rstrip('_x')))
            self.updates_df.drop(columns=x_cols, inplace=True)
            self.updates_df = pd.concat([self.updates_df, extra_data.loc[extra_data['DATE'] == get_current_date()]],
                                        axis=0)
        is_null = self.updates_df['PLAYER_ID'].isnull()
        self.updates_df.loc[is_null, 'PLAYER_ID'] = self.updates_df.loc[is_null, 'NAME'].map(self._player_name_to_id)
        is_null = self.updates_df['TEAM_ID'].isnull()
        self.updates_df.loc[is_null, 'TEAM_ID'] = self.updates_df.loc[is_null, 'TEAM'].map(TEAM_NAME_TO_ID)

        no_player_id = (self.updates_df['PLAYER_ID'] == self.updates_df['NAME']) | self.updates_df['PLAYER_ID'].isnull()
        if no_player_id.any():
            logging.info(f'removing players with player ID:\n{self.updates_df.loc[no_player_id, "NAME"].to_list()}')

        self.updates_df = self.updates_df.loc[(~no_player_id) | self.updates_df['START_POSITION'].notnull()]

    def assert_valid_df(self, df: pd.DataFrame):
        super(StatsLoaderPlayers, self).assert_valid_df(df=df)
        assert_valid_df_players(df)

    def _postprocessing(self):
        """process the data in updates_df"""

        super(StatsLoaderPlayers, self)._postprocessing()

        # filling 0 in the counting stats for players who didn't play
        if not self.updates_df.empty:
            add_cols = [c for c in FILL_ZEROS_COLS + ['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE'] if c not in self.updates_df]
            self.updates_df[add_cols] = np.nan

            self.updates_df[FILL_ZEROS_COLS] = self.updates_df[FILL_ZEROS_COLS].fillna(0)

            fill_cols = ['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE']
            self.updates_df[fill_cols] = self.updates_df[fill_cols].astype('boolean').fillna(False)

            # converting confirmed to true for all past games
            not_today_inds = self.updates_df['DATE'] != get_current_date()
            self.updates_df.loc[not_today_inds, 'CONFIRMED'] = True
            if 'IS_QUESTIONABLE' in self.updates_df.columns:
                self.updates_df.loc[not_today_inds, 'IS_QUESTIONABLE'] = False
            else:
                self.updates_df['IS_QUESTIONABLE'] = False

            # fixing salaries/positions from older versions
            site_salary_cols = [s.name + '_SALARY' for s in Sites]
            self.updates_df[site_salary_cols] = self.updates_df[site_salary_cols].astype(float)
            for c in site_salary_cols:
                self.updates_df.loc[self.updates_df[c] <= 0, c] = np.nan

            # adding fantasy points
            for site, scoring in SCORING.items():
                self.updates_df[f'{site.name.upper()}_POINTS'] = np.sum([self.updates_df[c] * m for c, m in scoring.items()],
                                                                   axis=0)

            # adding game times and game ids
            schedule = self._schedule.assign(OPPONENT=self._schedule['TEAM_ID_OPPONENT'].replace(TEAM_ID_TO_NAME))
            self.updates_df = self.updates_df.merge(schedule[['DATE', 'GAME_TIME', 'GAME_ID', 'TEAM_ID', 'OPPONENT',
                                                              'HOME']],
                                                    how='left', on=['DATE', 'TEAM_ID'], suffixes=(None, '_x'))
            x_cols = [c for c in self.updates_df.columns if c.endswith('_x')]
            non_x_cols = [c.replace('_x', '') for c in x_cols]
            self.updates_df[non_x_cols] = \
                self.updates_df[non_x_cols].fillna(self.updates_df[x_cols].rename(columns=lambda x: x.rstrip('_x')))
            self.updates_df.drop(columns=x_cols, inplace=True)

            if 'SEASON_YEAR' in self.updates_df:
                self.updates_df['SEASON_YEAR'] = self.updates_df['SEASON_YEAR'].fillna(self.season)
            else:
                self.updates_df['SEASON_YEAR'] = self.season


class StatsLoaderTeams(StatsLoader):
    LOADER_TYPE = StatsTypes.team
    DROP_COLS = ['TEAM_NAME']
    BOXSCORE_TYPES = [BoxscoreTypes.Base, BoxscoreTypes.Advanced, BoxscoreTypes.FourFactors]
    SORT_COLS = ['DATE', 'GAME_ID', 'TEAM_ID']
    MERGE_ON = ['GAME_ID', 'TEAM_ID']

    def scrape_single_boxscore_type_from_nbacom(self, boxscore_type: BoxscoreTypes,
                                                season_type: str = 'Regular Season') -> pd.DataFrame:
        boxscore_type_name = boxscore_type.name
        if boxscore_type_name == 'FourFactors':
            boxscore_type_name = 'Four Factors'

        try:
            print('aaa >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            return load_nbacom_data(url=URLS().boxscores_nbacom_team(measure_type=boxscore_type_name,
                                                                     season=self.season_str,
                                                                     season_type=season_type),
                                    player_or_team=False)
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            try:
                print('bbb >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                return load_nbacom_data(url=URLS().BOXSCORES_NBACOM.format(player_or_team='team',
                                                                           measure_type=boxscore_type_name,
                                                                           season=self.season_str,
                                                                           season_type=season_type),
                                        player_or_team=False)
            except requests.exceptions.ConnectionError:
                print('ccc >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
                team_game_logs = teamgamelogs.TeamGameLogs(season_nullable=self.season_str,
                                                           measure_type_player_game_logs_nullable=boxscore_type_name,
                                                           season_type_nullable=season_type)
                return team_game_logs.get_data_frames()[0]


if __name__ == '__main__':
    # from utils import get_header
    # header = get_header()
    # df = pd.read_html('https://basketballmonster.com/nbalineups.aspx')

    # slp = StatsLoaderPlayers(season=2024)
    # slp.call()
    # df = slp.load_saved_data()
    # left_midgame2 = (slp.df['MIN'] == 0) & (slp.df['PF'] > 0)
    # slp.df['LEFT_MIDGAME'] |= left_midgame2
    # left_mid_game = pd.read_csv('C:/Users/schwa/Downloads/possibly_left_midgame - possibly_left_midgame.csv')
    # left_mid_game['NAME'] = left_mid_game['NAME'].replace({'Cam Payne': 'Cameron Payne',
    #                                                        "Mo Wagner": "Moritz Wagner",
    #                                                        "Trey Murhoy III": 'Trey Murphy III'})
    # left_mid_game['DATE'] = pd.to_datetime(left_mid_game['DATE']).dt.date
    # left_mid_game['NAME'] = FixPlayerNames(compare_names=df['NAME'].unique()).fix_names(left_mid_game['NAME'])
    # drop = [{'NAME': "Anfernee Simons", 'DATE': datetime.date(2024, 11, 13)},
    #         {'NAME': "Coby White", 'DATE': datetime.date(2024, 12, 2)},
    #         {'NAME': "Keyonte George", 'DATE': datetime.date(2025, 3, 24)},
    #         ]
    # left_mid_game = left_mid_game.merge(pd.DataFrame(drop), on=['NAME', 'DATE'], indicator='xxx', how='left')
    # left_mid_game = left_mid_game.loc[left_mid_game['xxx'] == 'left_only'].drop(columns=['xxx'])
    # slp.df = slp.df.merge(left_mid_game, on=['NAME', 'DATE'], how='outer', indicator='xxx', suffixes=(None, '_x'))
    # if not slp.df.query('xxx=="right_only"').empty:
    #     raise ValueError('aaaaaaaaaaaaaaaa')
    # print(slp.df['LEFT_MIDGAME'].sum())
    # slp.df.loc[slp.df['xxx'] == 'both', 'LEFT_MIDGAME'] = True
    # print(slp.df['LEFT_MIDGAME'].sum())
    # slp.df.drop(columns=['xxx', 'LEFT_MIDGAME_x'], inplace=True)
    # #
    # slp._save_df()

    # for curr_season in range(2014, 2024):
    for curr_season in (2025,):
        print(curr_season, '**************************************************')
        # slp = StatsLoaderPlayers(season=curr_season)
        slp = StatsLoaderTeams(season=curr_season)
        slp.call()
        del slp
