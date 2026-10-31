import os
import datetime
import logging
import json
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from nba_api.stats.endpoints import playergamelogs, teamgamelogs, boxscoretraditionalv3
import data_collectors
from constants import URLS, SCORING, TEAM_NAME_TO_ID, TEAM_ID_TO_NAME
from enums import StatsTypes, BoxscoreTypes, Sites
from utils import (
    downcast_df,
    load_and_cache_players_list,
    season_int_to_str,
    load_nbacom_data,
    read_and_cache,
    get_latest_date_with_data,
    assert_valid_df_all,
    assert_valid_df_players,
    assert_reconciled_player_team_stats,
    load_schedule,
    get_current_time,
    get_current_date,
    FixPlayerNames,
    get_header,
    update_dir_name,
    get_current_season,
    define_logger,
    scrape_season_schedules,
)

FILL_ZEROS_COLS = [
    'MIN', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL',
    'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS', 'FG_PCT', '3P_PCT', 'FT_PCT'
]


def _coalesce_merge_columns(df: pd.DataFrame, suffix: str = '_x') -> pd.DataFrame:
    """
    Fills NaN values in base columns using suffixed columns (e.g. from an outer or left merge),
    then drops all suffixed columns. Safely uses str.removesuffix instead of character-set strip.
    """
    suffix_cols = [c for c in df.columns if c.endswith(suffix)]
    if not suffix_cols:
        return df
    base_cols = [c.removesuffix(suffix) for c in suffix_cols]
    for base_col, s_col in zip(base_cols, suffix_cols):
        if base_col in df.columns:
            df[base_col] = df[base_col].fillna(df[s_col])
        else:
            df[base_col] = df[s_col]
    df.drop(columns=suffix_cols, inplace=True)
    return df


class StatsLoader:
    """
    Parent class for loading stats. Both the classes for loading/scraping players stats and team stats inherit from
    this class.
    """

    LOADER_TYPE = None
    BOXSCORE_TYPES = [BoxscoreTypes.Base]
    SEASON_TYPES = ['Regular Season']
    MERGE_ON = ['DATE', 'TEAM_ID']
    DROP_COLS = []
    REPLACE_COLUMN_NAMES = {
        'PLAYER_NAME': 'NAME', 'TEAM_ABBREVIATION': 'TEAM', 'GAME_DATE': 'DATE',
        'MATCHUP': 'OPPONENT', 'WL': 'W_L', 'FG3M': '3PM', 'FG3A': '3PA', 'OFF_RATING': 'OFFRTG',
        'FG3_PCT': '3P_PCT', 'PCT_FG3M': 'PCT_3PM', 'DEF_RATING': 'DEFRTG', 'AST_TOV': 'AST_TO',
        'PCT_FG3A': 'PCT_3PA', 'NET_RATING': 'NETRTG', 'DD2': 'DOUBLE_DOUBLE',
        'TD3': 'TRIPLE_DOUBLE'
    }
    SORT_COLS = []

    def __init__(self, season: int | None = None, **kwargs):
        """
        :param season: the season to collect stats for (e.g. 2026 for 2026-27, defaults to current season)
        """
        self.season = season or get_current_season()
        self.season_str = season_int_to_str(self.season)

        self._stats_dir = update_dir_name('stats')
        loader_name = self.LOADER_TYPE.name if self.LOADER_TYPE is not None else 'base'
        self.filepath = os.path.join(self._stats_dir, f'{loader_name}_boxscores.feather')

        self.df = pd.DataFrame(columns=self.SORT_COLS)
        self.latest_date_with_data = datetime.date(2000, 1, 1)
        self._reset_updates_df()

    def call(self, add_upcoming_games: bool = True):
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

        updates_list: list[pd.DataFrame] = []
        for season_type in self.SEASON_TYPES:
            season_type_dfs: list[pd.DataFrame] = []
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
                existing_cols = {col for df in season_type_dfs for col in df.columns}
                drop_cols = (set(curr_df.columns) & set(self.DROP_COLS)) | {
                    c for c in curr_df.columns if c.endswith('_RANK') or c.startswith(('E_', 'sp_work_'))
                } | (set(curr_df.columns) & existing_cols)
                curr_df.drop(columns=list(drop_cols), inplace=True)

                curr_df.set_index(self.MERGE_ON, inplace=True)
                season_type_dfs.append(curr_df)

            if season_type_dfs:
                assert len(np.unique([df.shape[0] for df in season_type_dfs])) == 1, \
                    "the season_type_dfs don't have the same number of rows!"
                updates_list.append(pd.concat(season_type_dfs, axis=1).reset_index(drop=False))

        if not updates_list:
            self.updates_df = pd.DataFrame(columns=self.SORT_COLS)
            return

        assert len(np.unique([df.shape[1] for df in updates_list])) == 1, \
            "the updates_dfs don't have the same number of columns!"
        self.updates_df = pd.concat(updates_list, axis=0).reset_index(drop=True)
        self.updates_df.drop_duplicates(self.MERGE_ON, inplace=True)

        split_opp = [x.split() for x in self.updates_df['OPPONENT']]
        self.updates_df['HOME'] = [int(i[1] == 'vs.') for i in split_opp]
        self.updates_df['OPPONENT'] = [i[2] for i in split_opp]
        self.updates_df['W_L'] = (self.updates_df['W_L'] == 'W').astype(int)
        self.updates_df['DATE'] = pd.to_datetime(self.updates_df['DATE']).dt.date
        if 'SEASON_YEAR' in self.updates_df.columns:
            self.updates_df['SEASON_YEAR'] = self.updates_df['SEASON_YEAR'].astype(str).str.split('-').str[0].astype(int)
        if 'GAME_ID' in self.updates_df.columns:
            self.updates_df['GAME_ID'] = self.updates_df['GAME_ID'].astype('int64')
        if 'TEAM_ID' in self.updates_df.columns:
            self.updates_df['TEAM_ID'] = self.updates_df['TEAM_ID'].astype('int64')
        self.updates_df.sort_values(self.SORT_COLS, inplace=True)
        self.updates_df = self.updates_df.loc[self.updates_df['TEAM_ID'].isin(TEAM_ID_TO_NAME.keys())]
        downcast_df(self.updates_df)
        self.assert_valid_df(self.updates_df, is_incremental=True)

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

    def assert_valid_df(self, df: pd.DataFrame, is_incremental: bool = False):
        assert_valid_df_all(df, check_total_games=not is_incremental)

    def _save_df(self):
        if not self.updates_df.empty:
            if 'AVAILABLE_FLAG' in self.df.columns:
                self.df['AVAILABLE_FLAG'] = self.df['AVAILABLE_FLAG'].fillna(0).astype(bool)
            if 'VIDEO_AVAILABLE' in self.df.columns:
                self.df.drop(columns=['VIDEO_AVAILABLE'], inplace=True)

            for col in ('GAME_ID', 'TEAM_ID', 'PLAYER_ID'):
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype('int64')

            if 'W_L' in self.df.columns:
                if self.LOADER_TYPE == StatsTypes.player:
                    self.df['W_L'] = self.df['W_L'].fillna(False).astype(bool)
                elif self.LOADER_TYPE == StatsTypes.team:
                    self.df['W_L'] = self.df['W_L'].fillna(0).astype('int8')

            if 'HOME' in self.df.columns:
                if self.LOADER_TYPE == StatsTypes.player:
                    self.df['HOME'] = self.df['HOME'].fillna(False).astype(bool)
                elif self.LOADER_TYPE == StatsTypes.team:
                    self.df['HOME'] = self.df['HOME'].fillna(0).astype('int8')

            self.df.drop_duplicates(inplace=True)
            self.df.sort_values(self.SORT_COLS, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            downcast_df(self.df)
            self.assert_valid_df(self.df)

            loader_name = self.LOADER_TYPE.name if self.LOADER_TYPE is not None else 'base'
            logging.info(f'saving {loader_name} boxscores to {self.filepath}')

            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            tmp_path = self.filepath + '.tmp'
            self.df.to_feather(tmp_path)
            os.replace(tmp_path, self.filepath)


class StatsLoaderPlayers(StatsLoader):
    LOADER_TYPE = StatsTypes.player
    BOXSCORE_TYPES = [BoxscoreTypes.Base]
    DROP_COLS = ['NICKNAME', 'TEAM_NAME', 'NBA_FANTASY_PTS', 'WNBA_FANTASY_PTS', 'VIDEO_AVAILABLE_FLAG']
    SORT_COLS = ['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID']
    MERGE_ON = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']

    def __init__(self, season: int | None = None, out_players: dict | None = None, **kwargs):
        """
        :param season: the season we're loading data for (defaults to current season)
        :param out_players: a dict of players that are out today and their replacements in the starting lineup.
                            used when scraping data from rotogrinders
        """
        super().__init__(season=season, **kwargs)

        self.out_players = out_players

        self._players_list: pd.DataFrame | None = None
        self._player_id_to_name: dict[int, str] | None = None
        self._player_name_to_id: dict[str, int] | None = None
        self._name_fixer = FixPlayerNames()

        self._schedule = load_schedule(drop_ppd=True)

    def call(self, players_list: pd.DataFrame | None = None, add_upcoming_games: bool = True):
        if players_list is not None:
            self._players_list = players_list
        else:
            self._players_list = load_and_cache_players_list(season=self.season)
        self._player_id_to_name = dict(zip(self._players_list['PLAYER_ID'], self._players_list['NAME']))
        self._player_name_to_id = dict(zip(self._players_list['NAME'], self._players_list['PLAYER_ID']))

        super().call(add_upcoming_games=add_upcoming_games)

    def load_saved_data(self) -> pd.DataFrame:
        saved_df = super().load_saved_data()
        if saved_df is not None:
            self.df = saved_df
        if self.df is None or self.df.empty:
            return self.df

        self.df['MIN'] = self.df['MIN'].fillna(0)
        self.df = self.df.loc[self.df['NAME'].notnull()]
        for c in ('LEFT_MIDGAME', 'EJECTED', 'IS_OUT', 'W_L', 'HOME'):
            if c in self.df.columns:
                self.df[c] = self.df[c].fillna(False).astype(bool)
        self.df['PLAYER_ID'] = self.df['PLAYER_ID'].astype('int64')

        self.df['COMMENT'] = self.df['COMMENT'].fillna('')
        self.df.loc[(self.df['MIN'] > 0) & (self.df['COMMENT'].str.contains('DNP')), 'COMMENT'] = ''

        if not self.df.empty:
            self.df['NAME'] = self._name_fixer.fix_names(self.df['NAME'])

        # adding missing ids from df
        if self._player_name_to_id is not None and self._player_id_to_name is not None:
            missing_ids = set(self.df['PLAYER_ID'].unique()) - set(self._player_id_to_name.keys())
            if missing_ids:
                new_records = (
                    self.df.loc[self.df['PLAYER_ID'].isin(missing_ids), ['PLAYER_ID', 'NAME']]
                    .drop_duplicates('PLAYER_ID')
                )
                add_dict = dict(zip(new_records['PLAYER_ID'], new_records['NAME']))
                self._player_id_to_name.update(add_dict)
                self._player_name_to_id.update(
                    {name: pid for pid, name in add_dict.items() if name not in self._player_name_to_id}
                )

        if 'VIDEO_AVAILABLE' in self.df.columns:
            self.df.drop(columns=['VIDEO_AVAILABLE'], inplace=True)

        self.latest_date_with_data = get_latest_date_with_data(self.df)

        return self.df

    def scrape_boxscores_from_nbacom(self):
        super().scrape_boxscores_from_nbacom()
        if not self.updates_df.empty:
            self.updates_df['NAME'] = self._name_fixer.fix_names(self.updates_df['NAME'])

    def scrape_single_boxscore_type_from_nbacom(self, boxscore_type: BoxscoreTypes,
                                                season_type: str = 'Regular Season') -> pd.DataFrame:
        try:
            curr_url = URLS().boxscores_nbacom2(player_or_team=self.LOADER_TYPE.name.upper()[0],
                                                season=self.season_str,
                                                season_type=season_type)
            curr_df = load_nbacom_data(curr_url)
            if curr_df is None or curr_df.empty:
                raise requests.exceptions.RequestException('Empty data from boxscores_nbacom2')
            curr_df['SEASON_YEAR'] = curr_df['SEASON_ID'].astype(str).str[1:]
            curr_df.drop(columns=['SEASON_ID'], inplace=True)
            for c in ('BLKA', 'PFD'):
                if c not in curr_df.columns:
                    curr_df[c] = -1
            double_digits = (curr_df[['PTS', 'REB', 'AST', 'STL', 'BLK']] >= 10).sum(axis=1)
            curr_df['DOUBLE_DOUBLE'] = double_digits == 2
            curr_df['TRIPLE_DOUBLE'] = double_digits >= 3
        except (requests.exceptions.RequestException, KeyError):
            try:
                curr_url = URLS().boxscores_nbacom(player_or_team=self.LOADER_TYPE.name,
                                                   measure_type=boxscore_type.name,
                                                   season=self.season_str,
                                                   season_type=season_type)
                curr_df = load_nbacom_data(curr_url)
                if curr_df is None or curr_df.empty:
                    raise requests.exceptions.RequestException('Empty data from boxscores_nbacom')
            except (requests.exceptions.RequestException, KeyError):
                try:
                    player_game_logs = playergamelogs.PlayerGameLogs(season_nullable=self.season_str,
                                                                     measure_type_player_game_logs_nullable=boxscore_type.name,
                                                                     season_type_nullable=season_type,
                                                                     headers=get_header(),
                                                                     timeout=60)
                    dfs = player_game_logs.get_data_frames()
                    curr_df = dfs[0] if dfs else pd.DataFrame()
                except Exception as e:
                    logging.warning(f"PlayerGameLogs fallback failed: {e}")
                    curr_df = pd.DataFrame()
        return curr_df if curr_df is not None else pd.DataFrame()

    def get_player_name_to_id(self) -> dict[str, int]:
        return self._player_name_to_id

    def _read_single_game_boxscores_comments(self):
        if self.updates_df is None or self.updates_df.empty:
            return

        game_ids = self.updates_df.loc[self.updates_df['DATE'] > self.latest_date_with_data, 'GAME_ID'].unique()
        if len(game_ids) == 0:
            return

        # if we're running on a past season then we should have a file with all the information for the entire season,
        # otherwise we have a file for each game
        full_season_comments_path = os.path.join(update_dir_name('comments_nbacom'), f'{self.season}.feather')
        if os.path.isfile(full_season_comments_path):
            all_dfs = pd.read_feather(full_season_comments_path)
            all_dfs = all_dfs.loc[all_dfs['GAME_ID'].isin(game_ids)]
        else:
            keep_cols = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'START_POSITION', 'COMMENT']
            rename_cols = {
                'gameId': 'GAME_ID', 'teamId': 'TEAM_ID', 'personId': 'PLAYER_ID',
                'position': 'START_POSITION', 'comment': 'COMMENT'
            }
            all_dfs = list()
            for count, gid in tqdm(enumerate(sorted(game_ids)), desc='getting boxscore comments from nba.com',
                                   total=len(game_ids)):
                cache_name = f'nbacom_single_game_boxscores/nbacom_boxscoretraditionalv2_gid_{gid}.feather'
                gid_str = str(gid).zfill(10)
                tmp_df = None
                try:
                    boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3
                    tmp_df = read_and_cache(
                        func=lambda x: boxscore(game_id=x, headers=get_header(), timeout=5).player_stats.get_data_frame().rename(columns=rename_cols)[keep_cols],
                        func_params={'x': gid_str},
                        cache_name=cache_name,
                        sleep_time_after_reading_from_web=0.5
                    )
                except Exception as e:
                    logging.debug(f"Could not load boxscore comments for game {gid}: {e}. Proceeding with default empty comments.")
                    tmp_df = pd.DataFrame(columns=keep_cols)

                if tmp_df is not None and not tmp_df.empty:
                    all_dfs.append(tmp_df)

            if all_dfs:
                all_dfs = pd.concat(all_dfs, axis=0).reset_index(drop=True)
            else:
                all_dfs = pd.DataFrame(columns=keep_cols)

        for c in ['GAME_ID', 'TEAM_ID', 'PLAYER_ID']:
            all_dfs[c] = all_dfs[c].astype('int64')
        if 'START_POSITION' in all_dfs:
            all_dfs.drop(columns=['START_POSITION'], inplace=True)
        self.updates_df = self.updates_df.merge(all_dfs, on=['GAME_ID', 'TEAM_ID', 'PLAYER_ID'], how='outer')

        # filling nans for NAME
        if self._player_id_to_name:
            self.updates_df['NAME'] = self.updates_df['NAME'].fillna(
                self.updates_df['PLAYER_ID'].map(self._player_id_to_name)
            )

        # filling nans for team stats
        fill_cols = ['DATE', 'SEASON_YEAR', 'TEAM', 'OPPONENT', 'W_L', 'HOME']
        merge_on = ['TEAM_ID', 'GAME_ID']
        merge_df = self.updates_df.dropna(subset=['DATE', 'TEAM'], how='any').drop_duplicates(['DATE', 'TEAM'])
        merge_df = merge_df[merge_on + fill_cols]
        for c in merge_on:
            merge_df[c] = merge_df[c].astype('int64')
        self.updates_df = self.updates_df.merge(merge_df, on=merge_on, how='left', suffixes=(None, '_x'))
        _coalesce_merge_columns(self.updates_df)

        existing_zero_cols = [c for c in FILL_ZEROS_COLS if c in self.updates_df.columns]
        if existing_zero_cols:
            self.updates_df[existing_zero_cols] = self.updates_df[existing_zero_cols].fillna(0)
        for c in ('DOUBLE_DOUBLE', 'TRIPLE_DOUBLE'):
            if c in self.updates_df.columns:
                self.updates_df[c] = self.updates_df[c].fillna(False)

        drop_inds = self.updates_df['NAME'].isnull() & (self.updates_df['MIN'] == 0)
        self.updates_df = self.updates_df.loc[~drop_inds]

        self.updates_df.sort_values(self.SORT_COLS, inplace=True)
        self.updates_df['NAME'] = self._name_fixer.fix_names(self.updates_df['NAME'])
        downcast_df(self.updates_df)
        self.assert_valid_df(self.updates_df, is_incremental=True)

    def _add_extra_data_to_updates_df(self, add_upcoming_games: bool = True):
        if (not add_upcoming_games) and (self.updates_df is None or self.updates_df.empty):
            return

        self._read_single_game_boxscores_comments()

        players_list_names = self._players_list['NAME'].to_list() if self._players_list is not None else []
        ejections = data_collectors.ejections.Ejections(nbacom_players_list=players_list_names)
        ejections_df = ejections.collect_data()

        update_dates = list(dict.fromkeys(self.updates_df['DATE'].unique().tolist()))
        if add_upcoming_games:
            today = get_current_date()
            if today not in update_dates:
                update_dates.append(today)

        extra_data = list()
        injuries = data_collectors.injuries.Injuries(
            nbacom_players_list=players_list_names,
            player_id_to_name=self._player_id_to_name
        )
        load_fantasy_data = data_collectors.fantasy_data.LoadFantasyData(nbacom_players_list=players_list_names)
        load_starters = data_collectors.starters.Starters(
            nbacom_players_list=players_list_names,
            player_name_to_id=self._player_name_to_id
        )
        left_midgame_collector = data_collectors.left_midgame.LeftMidgame(
            nbacom_players_list=players_list_names,
            player_id_to_name=self._player_id_to_name,
            player_name_to_id=self._player_name_to_id
        )
        for date in tqdm(update_dates, desc='loading extra data per date'):
            # loading injuries data
            injuries.collect_data(date=date)
            injured_players = injuries.out_players()

            # loading left midgame data using play-by-play and recap parsing
            pbp_left_midgame = left_midgame_collector.collect_data(date=date)

            # removing players who played that day
            if not self.updates_df.empty:
                date_mask = self.updates_df['DATE'] == date
                played_mask = date_mask & (self.updates_df['MIN'] > 0)
                players_played_today = set(self.updates_df.loc[played_mask, 'NAME'])

                # Heuristic: players on injury report who played very short minutes (< 15 mins)
                short_min_mask = played_mask & (self.updates_df['MIN'] < 15)
                short_min_players = set(self.updates_df.loc[short_min_mask, 'NAME'])
                heuristic_left_midgame = short_min_players & set(injured_players)

                left_midgame = (pbp_left_midgame | heuristic_left_midgame) & players_played_today

                # Full-minute players (>= 30 mins) did not have a game-shortening exit
                full_min_players = set(self.updates_df.loc[date_mask & (self.updates_df['MIN'] >= 30), 'NAME'])
                left_midgame = left_midgame - full_min_players

                self.updates_df.loc[date_mask, 'LEFT_MIDGAME'] = (
                    self.updates_df.loc[date_mask, 'NAME'].isin(left_midgame)
                )

                injured_players = [p for p in injured_players if p not in left_midgame]
            else:
                left_midgame = list()

            # loading fantasy positions/salaries data
            if self.updates_df.empty:
                name_to_team = None
            else:
                date_teams = self.updates_df.loc[self.updates_df['DATE'] == date, ['NAME', 'TEAM']].dropna()
                name_to_team = dict(zip(date_teams['NAME'], date_teams['TEAM']))
            fantasy_data_df = load_fantasy_data.collect_data(date=date, name_to_team=name_to_team)
            fantasy_data_df['IS_OUT'] = fantasy_data_df['NAME'].isin(injured_players).values

            # loading starters
            starters_df = load_starters.collect_data(
                date=date, out_players=injured_players, salaries_df=fantasy_data_df
            )

            # combine starters and salaries/positions
            combined = starters_df.merge(fantasy_data_df, on=['NAME', 'TEAM'], how='outer', suffixes=(None, '_x'))
            _coalesce_merge_columns(combined)
            combined['DATE'] = date
            combined['LEFT_MIDGAME'] = combined['NAME'].isin(left_midgame)

            extra_data.append(combined)

        extra_data = pd.concat(extra_data, axis=0) if extra_data else pd.DataFrame()

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
            self.updates_df = self.updates_df.merge(
                extra_data.loc[extra_data['DATE'] != get_current_date()],
                on=['NAME', 'TEAM', 'DATE'],
                how='left',
                suffixes=(None, '_x')
            )
            _coalesce_merge_columns(self.updates_df)
            self.updates_df = pd.concat(
                [self.updates_df, extra_data.loc[extra_data['DATE'] == get_current_date()]],
                axis=0
            )

        if self._player_name_to_id is not None:
            is_null = self.updates_df['PLAYER_ID'].isnull()
            self.updates_df.loc[is_null, 'PLAYER_ID'] = self.updates_df.loc[is_null, 'NAME'].map(self._player_name_to_id)
        is_null = self.updates_df['TEAM_ID'].isnull()
        self.updates_df.loc[is_null, 'TEAM_ID'] = self.updates_df.loc[is_null, 'TEAM'].map(TEAM_NAME_TO_ID)

        no_player_id = (self.updates_df['PLAYER_ID'] == self.updates_df['NAME']) | self.updates_df['PLAYER_ID'].isnull()
        if no_player_id.any():
            logging.info(f'removing players without valid player ID:\n{self.updates_df.loc[no_player_id, "NAME"].to_list()}')

        self.updates_df = self.updates_df.loc[(~no_player_id) | self.updates_df['START_POSITION'].notnull()]

        for c in ('LEFT_MIDGAME', 'EJECTED', 'IS_OUT'):
            if c in self.updates_df.columns:
                self.updates_df[c] = self.updates_df[c].fillna(False).astype(bool)
            else:
                self.updates_df[c] = False

    def _get_team_boxscores_df(self) -> pd.DataFrame | None:
        team_boxscores_path = os.path.join(os.path.dirname(self.filepath), 'team_boxscores.feather')
        if not os.path.isfile(team_boxscores_path):
            return None
        if not hasattr(self, '_cached_team_df') or self._cached_team_df is None:
            self._cached_team_df = pd.read_feather(team_boxscores_path)
        return self._cached_team_df

    def assert_valid_df(self, df: pd.DataFrame, is_incremental: bool = False):
        super().assert_valid_df(df=df, is_incremental=is_incremental)
        assert_valid_df_players(df)

        team_df = self._get_team_boxscores_df()
        if team_df is not None and not df.empty and 'GAME_ID' in df.columns:
            if is_incremental:
                gids = df['GAME_ID'].dropna().unique()
            elif hasattr(self, 'updates_df') and self.updates_df is not None and not self.updates_df.empty:
                gids = self.updates_df['GAME_ID'].dropna().unique()
            else:
                gids = None
            assert_reconciled_player_team_stats(df, team_df, game_ids=gids)

    def _postprocessing(self):
        """process the data in updates_df"""
        super()._postprocessing()

        # filling 0 in the counting stats for players who didn't play
        if not self.updates_df.empty:
            add_cols = [c for c in FILL_ZEROS_COLS + ['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE'] if c not in self.updates_df]
            self.updates_df[add_cols] = np.nan

            self.updates_df[FILL_ZEROS_COLS] = self.updates_df[FILL_ZEROS_COLS].fillna(0)

            fill_cols = ['DOUBLE_DOUBLE', 'TRIPLE_DOUBLE']
            self.updates_df[fill_cols] = self.updates_df[fill_cols].astype('boolean').fillna(False)

            # converting confirmed to true for all past games
            not_today_inds = self.updates_df['DATE'] != get_current_date()
            if 'CONFIRMED' not in self.updates_df.columns:
                self.updates_df['CONFIRMED'] = False
            self.updates_df.loc[not_today_inds, 'CONFIRMED'] = True
            self.updates_df['CONFIRMED'] = self.updates_df['CONFIRMED'].fillna(False).astype(bool)
            if 'IS_QUESTIONABLE' in self.updates_df.columns:
                self.updates_df.loc[not_today_inds, 'IS_QUESTIONABLE'] = False
            else:
                self.updates_df['IS_QUESTIONABLE'] = False

            # ensuring boolean status columns are strictly boolean with zero nulls
            for c in ('LEFT_MIDGAME', 'EJECTED', 'IS_OUT', 'W_L', 'HOME'):
                if c in self.updates_df.columns:
                    self.updates_df[c] = self.updates_df[c].fillna(False).astype(bool)
                else:
                    self.updates_df[c] = False

            # fixing salaries/positions from older versions
            site_salary_cols = [s.name + '_SALARY' for s in Sites]
            self.updates_df[site_salary_cols] = self.updates_df[site_salary_cols].astype(float)
            for c in site_salary_cols:
                self.updates_df.loc[self.updates_df[c] <= 0, c] = np.nan

            # adding fantasy points
            for site, scoring in SCORING.items():
                self.updates_df[f'{site.name.upper()}_POINTS'] = np.sum(
                    [self.updates_df[c] * m for c, m in scoring.items()],
                    axis=0
                )

            # adding game times and game ids
            schedule = self._schedule.assign(OPPONENT=self._schedule['TEAM_ID_OPPONENT'].replace(TEAM_ID_TO_NAME))
            self.updates_df = self.updates_df.merge(
                schedule[['DATE', 'GAME_TIME', 'GAME_ID', 'TEAM_ID', 'OPPONENT', 'HOME']],
                how='left',
                on=['DATE', 'TEAM_ID'],
                suffixes=(None, '_x')
            )
            _coalesce_merge_columns(self.updates_df)

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
            res = load_nbacom_data(
                url=URLS().boxscores_nbacom_team(
                    measure_type=boxscore_type_name,
                    season=self.season_str,
                    season_type=season_type
                ),
                player_or_team=False
            )
            if res is None or res.empty:
                raise requests.exceptions.RequestException('Empty data from boxscores_nbacom_team')
            return res
        except (requests.exceptions.RequestException, KeyError):
            try:
                res = load_nbacom_data(
                    url=URLS().BOXSCORES_NBACOM.format(
                        player_or_team='team',
                        measure_type=boxscore_type_name,
                        season=self.season_str,
                        season_type=season_type
                    ),
                    player_or_team=False
                )
                if res is None or res.empty:
                    raise requests.exceptions.RequestException('Empty data from BOXSCORES_NBACOM')
                return res
            except (requests.exceptions.RequestException, KeyError):
                try:
                    team_game_logs = teamgamelogs.TeamGameLogs(
                        season_nullable=self.season_str,
                        measure_type_player_game_logs_nullable=boxscore_type_name,
                        season_type_nullable=season_type,
                        headers=get_header(),
                        timeout=60
                    )
                    dfs = team_game_logs.get_data_frames()
                    return dfs[0] if dfs else pd.DataFrame()
                except Exception as e:
                    logging.warning(f"TeamGameLogs fallback failed: {e}")
                    return pd.DataFrame()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Update NBA stats data (teams, players, schedules)")
    parser.add_argument('--season', type=int, default=None,
                        help="Season to update (defaults to current season)")
    parser.add_argument('--team-only', action='store_true',
                        help="Only update team stats")
    parser.add_argument('--player-only', action='store_true',
                        help="Only update player stats")
    parser.add_argument('--add-upcoming', action='store_true',
                        help="Include today's upcoming games in player stats update")
    args = parser.parse_args()

    define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO, print_to_console=True)
    target_season = args.season or get_current_season()
    logging.info(f'Starting stats update for season {target_season}...')

    if not args.player_only:
        logging.info(f'Updating team boxscores for season {target_season}...')
        slt = StatsLoaderTeams(season=target_season)
        slt.call()
        logging.info('Successfully updated team boxscores!')

    if not args.team_only:
        logging.info(f'Updating player boxscores for season {target_season}...')
        players_list = load_and_cache_players_list(season=target_season)
        slp = StatsLoaderPlayers(season=target_season)
        slp.call(players_list=players_list, add_upcoming_games=args.add_upcoming)
        logging.info('Successfully updated player boxscores!')

    logging.info('Stats update completed successfully!')
