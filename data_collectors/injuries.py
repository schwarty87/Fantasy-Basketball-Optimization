import os
import datetime
import json
import logging
import requests
import numpy as np
import pandas as pd
from enums import Enum, auto
from requests.exceptions import TooManyRedirects, ReadTimeout
from constants import URLS, FIX_TEAM_NAMES, TEAM_ID_TO_NAME
from utils import FixPlayerNames, retry, get_header, UnableToLoadWebsiteException, get_current_date, \
    date_to_game_ids, update_dir_name, fetch_html_table


INJURIES_DF_COLUMNS = ['NAME', 'TEAM', 'IS_OUT', 'IS_QUESTIONABLE']
EMPTY_DF = pd.DataFrame(columns=INJURIES_DF_COLUMNS)


class InjuriesSources(Enum):
    PREVIOUSLY_SAVED = auto()
    NBACOM_LINEUPS = auto()
    ROTOGRINDERS = auto()
    ROTOWIRE = auto()
    NBACOM_COMMENTS = auto()
    CBS = auto()

    @classmethod
    def all_injuries_sources(cls) -> tuple["InjuriesSources", ...]:
        return tuple([c for c in cls])


class Injuries:
    def __init__(self, injuries_sources: InjuriesSources | tuple[InjuriesSources, ...] = InjuriesSources.all_injuries_sources(),
                 nbacom_players_list: list[str] | pd.Series | np.ndarray | pd.DataFrame | None = None,
                 player_id_to_name: dict[int, str] | None = None):
        self.injuries_sources = injuries_sources
        if isinstance(self.injuries_sources, InjuriesSources):
            self.injuries_sources = (self.injuries_sources,)
        if isinstance(nbacom_players_list, pd.DataFrame):
            nbacom_players_list = nbacom_players_list['NAME'].values

        self.player_id_to_name = player_id_to_name
        self.fix_player_names = FixPlayerNames(compare_names=nbacom_players_list)

        self.injuries_df = EMPTY_DF

        self.save_dir = os.path.join(update_dir_name('cached_data'), 'injuries')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    def _fix_names(self, df: pd.DataFrame):
        if 'NAME' not in df and 'PLAYER_ID' in df:
            df['NAME'] = df['PLAYER_ID'].replace(self.player_id_to_name)
            df.loc[df['NAME'] == df['PLAYER_ID'], 'NAME'] = ''
        if 'TEAM' not in df and 'TEAM_ID' in df:
            df['TEAM'] = df['TEAM_ID'].replace(TEAM_ID_TO_NAME)
        df['NAME'] = self.fix_player_names.fix_names(df['NAME'])
        df['TEAM'] = df['TEAM'].replace(FIX_TEAM_NAMES)

    @retry(max_retries=3, delay_seconds=1, return_on_fail=EMPTY_DF)
    def _load_injuries_rotowire(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        if date != get_current_date():
            return EMPTY_DF

        # headers = get_header()
        headers = None
        html = requests.get(URLS.ROTOWIRE_INJURIES, headers=headers, timeout=30).content
        df = pd.DataFrame(json.loads(html))[['player', 'team', 'status']]
        df.rename(columns={'player': 'NAME', 'team': 'TEAM', 'status': 'STATUS'}, inplace=True)

        self._fix_names(df=df)

        status_lower = df['STATUS'].astype(str).str.lower()
        df['IS_OUT'] = status_lower.str.contains('out')
        df['IS_QUESTIONABLE'] = (
            status_lower.str.contains('questionable') |
            status_lower.str.contains('doubtful') |
            status_lower.str.contains('game time decision') |
            status_lower.str.contains('gtd')
        ) & (~df['IS_OUT'])

        logging.info('finished loading rotowire injuries data')

        return df

    @retry(max_retries=5, delay_seconds=2, return_on_fail=EMPTY_DF)
    def _load_injuries_rotogrinders(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        if date != get_current_date():
            return EMPTY_DF

        df = fetch_html_table(URLS.INJURIES_ROTOGRINDERS, uppercase_headers=True)
        if df.empty or not any(['PLAYER' in c for c in df.columns]):
            return EMPTY_DF

        df.dropna(subset=['PLAYER'], inplace=True)

        # fixing column names
        rename_dict = {c: 'STATUS' for c in df.columns if c.startswith('STATUS') or c in ['KY']}
        for c in df.columns:
            if 'PLAYER' in c:
                rename_dict[c] = 'NAME'
        df.rename(columns=rename_dict, inplace=True)

        self._fix_names(df=df)

        df['IS_OUT'] = df['STATUS'].isin(['O', 'D'])
        df['IS_QUESTIONABLE'] = df['STATUS'] == 'Q'

        logging.info('finished loading rotogrinders injuries data')

        return df

    @retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects), max_retries=3, delay_seconds=1,
           return_on_fail=EMPTY_DF)
    def _load_injuries_nbacom(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        if date != get_current_date():
            return EMPTY_DF

        url = URLS().games_lineups_nbacom(str(get_current_date()))
        headers = get_header()
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200 or len(resp.content) == 0:
                return EMPTY_DF
            data_dict = json.loads(resp.content).get('games', [])
        except Exception as e:
            logging.warning(f"Error fetching NBA lineups from {url}: {e}")
            return EMPTY_DF

        if not data_dict:
            return EMPTY_DF
        dfs = list()
        for curr_dict in data_dict:
            home_dict = curr_dict['homeTeam']
            home_df = pd.DataFrame(home_dict['players'])
            home_df = home_df.assign(TEAM_NAME=home_dict['teamAbbreviation'],
                                     HOME=True)
            away_dict = curr_dict['awayTeam']
            away_df = pd.DataFrame(away_dict['players'])
            away_df = away_df.assign(TEAM_NAME=away_dict['teamAbbreviation'],
                                     HOME=False)
            game_df = pd.concat([home_df, away_df], axis=0).assign(GAME_ID=curr_dict['gameId'])
            dfs.append(game_df)
        df = pd.concat(dfs, axis=0)

        todays_game_ids = date_to_game_ids(get_current_date())
        df = df.loc[df['GAME_ID'].astype(int).isin(todays_game_ids)]

        df.rename(columns={'personId': 'PLAYER_ID', 'teamId': 'TEAM_ID', 'playerName': 'NAME',
                           'position': 'START_POSITION', 'TEAM_NAME': 'TEAM', 'rosterStatus': 'STATUS'},
                  inplace=True)
        df.drop(columns=['firstName', 'lastName', 'timestamp'], inplace=True)

        self._fix_names(df=df)

        df['IS_OUT'] = df['STATUS'] != 'Active'
        df['IS_QUESTIONABLE'] = False
        df = df.loc[df[['IS_OUT', 'IS_QUESTIONABLE']].any(axis=1)]

        logging.info('finished loading nba.com injuries data')

        return df

    @staticmethod
    def _was_out_before_game(comment: str) -> bool:
        comment_lower = comment.lower()
        if 'coach' in comment_lower:
            return False
        for dnp_str in ['dnd', 'nwt', 'dnp']:
            if dnp_str in comment_lower:
                return True
        return False

    def _load_injuries_nbacom_comments(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        if date == get_current_date():
            return EMPTY_DF

        todays_game_ids = date_to_game_ids(date=date)

        all_game_comments = list()
        for gid in todays_game_ids:
            game_path = os.path.join(update_dir_name('cached_data'), 'nbacom_single_game_boxscores',
                                     f'nbacom_boxscoretraditionalv2_gid_{gid}.feather')
            if os.path.isfile(game_path):
                game_comments = pd.read_feather(game_path)
                game_comments = game_comments.loc[game_comments['COMMENT'].notnull() &
                                                  (game_comments['COMMENT'].str.len() > 0)]
                game_comments['IS_OUT'] = game_comments['COMMENT'].apply(self._was_out_before_game)

                all_game_comments.append(game_comments)
        if len(all_game_comments) == 0:
            return EMPTY_DF
        all_game_comments = pd.concat(all_game_comments, axis=0)
        self._fix_names(all_game_comments)
        all_game_comments['IS_QUESTIONABLE'] = False

        return all_game_comments

    def _load_injuries_cbs(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        if date != get_current_date():
            return EMPTY_DF

        def parse_cbs_cell(cell):
            long_name = cell.find(class_='CellPlayerName--long')
            if long_name:
                return long_name.get_text(strip=True)
            a_tag = cell.find('a')
            if a_tag:
                return a_tag.get_text(strip=True)
            return cell.get_text(strip=True)

        def get_cbs_team(table):
            team_el = table.find_previous(class_='TeamName')
            return {'TEAM': team_el.get_text(strip=True) if team_el else ''}

        df = fetch_html_table(
            'https://www.cbssports.com/nba/injuries/',
            all_tables=True,
            concat_tables=True,
            uppercase_headers=True,
            cell_parser=parse_cbs_cell,
            table_context_parser=get_cbs_team,
            column_rename={'PLAYER': 'NAME'}
        )
        if df.empty or 'NAME' not in df.columns or 'STATUS' not in df.columns:
            return EMPTY_DF

        status_lower = df['STATUS'].str.lower()
        df['IS_OUT'] = status_lower.str.contains('out')
        df['IS_QUESTIONABLE'] = (
            status_lower.str.contains('game time decision') |
            status_lower.str.contains('questionable') |
            status_lower.str.contains('doubtful') |
            status_lower.str.contains('gtd')
        ) & (~df['IS_OUT'])

        self._fix_names(df)

        logging.info('finished loading cbs injuries data')

        return df

    def combine_dfs(self, injuries_source_to_df: dict[InjuriesSources, pd.DataFrame]):
        is_out_cols = list()
        is_questionable_cols = list()
        all_dfs = list()
        for source, df in injuries_source_to_df.items():
            if df.empty:
                continue
            df.drop_duplicates(['NAME', 'TEAM'], keep='first', inplace=True)
            df = df.rename(columns={c: f'{c}_{source.name}' for c in df.columns if c not in ['NAME', 'TEAM']})
            df.set_index(['NAME', 'TEAM'], inplace=True)
            all_dfs.append(df)

            is_out_cols.append(f'IS_OUT_{source.name}')
            is_questionable_cols.append(f'IS_QUESTIONABLE_{source.name}')

        if len(all_dfs) == 0:
            self.injuries_df =pd.DataFrame(columns=['NAME', 'TEAM', 'IS_OUT', 'IS_QUESTIONABLE'])
            return
        self.injuries_df = pd.concat(all_dfs, axis=1).reset_index()
        self.injuries_df[is_out_cols + is_questionable_cols] = \
            self.injuries_df[is_out_cols + is_questionable_cols].astype('boolean').fillna(False)
        self.injuries_df['IS_OUT'] = self.injuries_df[is_out_cols].any(axis=1)
        self.injuries_df['IS_QUESTIONABLE'] = self.injuries_df[is_questionable_cols].any(axis=1)
        self.injuries_df.loc[self.injuries_df['IS_OUT'], 'IS_QUESTIONABLE'] = False
        self.injuries_df = self.injuries_df.loc[self.injuries_df[['IS_OUT', 'IS_QUESTIONABLE']].any(axis=1)]
        self.injuries_df.reset_index(drop=True, inplace=True)

    def out_players(self) -> list[str]:
        return self.injuries_df.query('IS_OUT')['NAME'].unique().tolist()

    def questionable_players(self) -> list[str]:
        return self.injuries_df.query('IS_QUESTIONABLE')['NAME'].unique().tolist()

    def out_and_questionable_players(self) -> list[str]:
        return self.out_players() + self.questionable_players()

    def collect_data(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        save_path = os.path.join(self.save_dir, f"{str(date).replace('-', '_')}.feather")

        injuries_source_to_df = dict()

        if InjuriesSources.PREVIOUSLY_SAVED in self.injuries_sources:
            injuries_source_to_df[InjuriesSources.PREVIOUSLY_SAVED] = \
                pd.read_feather(save_path) if (os.path.isfile(save_path) and date != get_current_date()) else EMPTY_DF

        if InjuriesSources.ROTOWIRE in self.injuries_sources:
            injuries_source_to_df[InjuriesSources.ROTOWIRE] = self._load_injuries_rotowire(date=date)

        if InjuriesSources.ROTOGRINDERS in self.injuries_sources:
            injuries_source_to_df[InjuriesSources.ROTOGRINDERS] = self._load_injuries_rotogrinders(date=date)

        if InjuriesSources.NBACOM_LINEUPS in self.injuries_sources:
            injuries_source_to_df[InjuriesSources.NBACOM_LINEUPS] = self._load_injuries_nbacom(date=date)

        if InjuriesSources.NBACOM_COMMENTS in self.injuries_sources:
            injuries_source_to_df[InjuriesSources.NBACOM_COMMENTS] = self._load_injuries_nbacom_comments(date=date)

        if InjuriesSources.CBS in self.injuries_sources:
            injuries_source_to_df[InjuriesSources.CBS] = self._load_injuries_cbs(date=date)

        self.combine_dfs(injuries_source_to_df=injuries_source_to_df)

        self.injuries_df = self.injuries_df[INJURIES_DF_COLUMNS]
        self.injuries_df.to_feather(save_path)

        return self.injuries_df


if __name__ == '__main__':
    from utils import load_and_cache_players_list
    import traceback
    players_list = load_and_cache_players_list(season=2023)

    try:
        # inj = Injuries(injuries_sources=InjuriesSources.ROTOGRINDERS, nbacom_players_list=players_list)
        inj = Injuries(injuries_sources=InjuriesSources.all_injuries_sources(), nbacom_players_list=players_list)
        inj._load_injuries_cbs(date=datetime.date(2024, 3, 26))
        inj.collect_data()
    except:
        traceback.print_exc()
