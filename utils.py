import os

import logging
import datetime
import json
import pickle
import random
import requests
import pyarrow
import time
import unicodedata
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Union, Tuple, Type, Callable, List
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype
from constants import FIX_PLAYER_NAMES, URLS, START_POSITIONS, SCORING, NY_TZ, DATA_DIR
from requests.exceptions import ConnectionError, ReadTimeout, RequestException
from enums import Sites


class UnableToLoadWebsiteException(Exception):
    def __init__(self, site, *args):
        super().__init__(args)
        self.site = site

    def __str__(self):
        return f'unable to load site {self.site}'


def retry(exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception, max_retries: int = 3,
          delay_seconds: float = 60, return_on_fail: Optional = None):
    """
    Decorator function that retries a given function if specific exceptions occur.
    :param exceptions: Exception classes that trigger retries.
    :param max_retries: Retry this many times before failing
    :param delay_seconds: Time delay in seconds between retries.
    :param return_on_fail: If not None, return this value if we failed the max amount of times. Could be a callable or
                           any value
    :return: Decorated function
    """
    def decorator_retry(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for count in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logging.warning(f"{func.__name__} Attempt {count + 1}/{max_retries} failed: "
                                    f"{type(e).__name__}, {e}")
                    time.sleep(delay_seconds)
                except Exception as e:
                    raise e
            fail_str = f"Function {func.__name__} failed after {max_retries} attempts"
            if return_on_fail is None:
                logging.error(fail_str)
                raise last_exception
            else:
                logging.warning(fail_str + ", returning default value...")
                return return_on_fail
        return wrapper
    return decorator_retry


def season_int_to_str(season: Optional[int] = 2022) -> str:
    return f'{season}-{str(season+1)[-2:]}' if season is not None else ''


def downcast_df(df: pd.DataFrame, df_name: Optional[str] = None, print_usage: bool = True, rtol: float = 1e-6,
                atol: float = 1e-5) -> pd.DataFrame:
    memory_usage_before = df.memory_usage().sum() * 1e-6
    int_cols = set(list(df.select_dtypes(('integer', 'bool')).columns) + [c for c in df.columns if c.endswith('_ID')])
    float_cols = set(df.select_dtypes('float').columns)
    for c in df.columns:
        if is_datetime64_any_dtype(df[c]) or is_string_dtype(df[c]) or c == 'DATE':
            continue
        if c not in int_cols and df[c].notnull().all() and (df[c].astype(int) == df[c]).all():
            int_cols.add(c)
            continue
        if c not in int_cols and c not in float_cols and pd.to_numeric(df[c], errors='coerce').notnull().all():
            float_cols.add(c)
    bool_cols = set([c for c in int_cols if set(df[c].unique()) <= {0, 1, -1, True, False}])
    int_cols = int_cols - bool_cols
    float_cols = float_cols - int_cols - bool_cols
    downcasted = df[sorted(float_cols)].apply(pd.to_numeric, downcast='float')
    for c in float_cols:
        if np.allclose(df[c], downcasted[c], rtol=rtol, atol=atol, equal_nan=True):
            df[c] = downcasted[c].values
    downcasted = df[sorted(int_cols)].apply(pd.to_numeric, downcast='integer')
    for c in int_cols:
        if df[c].notnull().all():
            both_null = df[c].isnull() & downcasted[c].isnull()
            if ((df[c].astype(int) == downcasted[c]) | both_null).all():
                df[c] = downcasted[c]
    df[sorted(bool_cols)] = (df[sorted(bool_cols)] == 1) | (df[sorted(bool_cols)] == True)
    if print_usage:
        logging.info(f'{df_name + " " if df_name is not None else ""}df memory usage: {memory_usage_before:.3f}mb -> '
                     f'{df.memory_usage().sum() * 1e-6:.3f}mb')
    return df


def read_and_cache(func, func_params: dict, cache_name: str, suffix: Optional[str] = None, 
                   sleep_time_after_reading_from_web: int = 0):
    cached_data_dir = update_dir_name('cached_data')
    cache_path = os.path.join(cached_data_dir, cache_name)
    if suffix is not None:
        cache_path += f'.{suffix}'
    if os.path.isfile(cache_path):
        return pd.DataFrame(pd.read_feather(cache_path).to_dict())
    df = func(**func_params)
    if df.empty:
        raise ValueError(f'no data returned from {func.__name__}')
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    df.to_feather(cache_path)
    time.sleep(sleep_time_after_reading_from_web)
    return df


def load_and_cache_players_list(print_usage: bool = True, season: int = 2025):
    cached_data_dir = update_dir_name('cached_data')
    cache_path = os.path.join(cached_data_dir, 'players_list.feather')
    if (os.path.isfile(cache_path) and
            (pd.to_datetime(os.path.getmtime(cache_path), unit='s').date() == get_current_date())):
        cache_path_exists = True
        players_list = pd.read_feather(cache_path)
    else:
        cache_path_exists = False
        try:
            players_list = load_nbacom_data(URLS().players_list_nbacom(season=season))
            players_list['NAME'] = players_list['PLAYER_FIRST_NAME'] + " " + players_list['PLAYER_LAST_NAME']
            players_list.rename(columns={'PERSON_ID': 'PLAYER_ID'}, inplace=True)
        except (ConnectionError, ReadTimeout, RequestException):
            try:
                players_list = load_nbacom_data(URLS().players_list_nbacom3(season=season))
                players_list['NAME'] = players_list['PLAYER_FIRST_NAME'] + " " + players_list['PLAYER_LAST_NAME']
                players_list.rename(columns={'PERSON_ID': 'PLAYER_ID'}, inplace=True)
            except (ConnectionError, ReadTimeout, RequestException):
                players_list = load_nbacom_data(URLS().players_list_nbacom2(season=season))
                players_list.rename(columns={'PLAYER_NAME': 'NAME', 'PLAYER_HEIGHT_INCHES': 'HEIGHT',
                                             'PLAYER_WEIGHT': 'WEIGHT'}, inplace=True)

    players_list['NAME'] = FixPlayerNames().fix_names(players_list['NAME'])
    if 'DRAFT_NUMBER' in players_list.columns:
        players_list['DRAFT_NUMBER'] = players_list['DRAFT_NUMBER'].replace({'Undrafted': 61}).fillna(61).astype(int)
    else:
        players_list['DRAFT_NUMBER'] = 61
    for column_name, dtype in players_list.dtypes.items():
        if dtype == object and players_list[column_name].str.isdigit().all():
            players_list[column_name] = players_list[column_name].astype(float)
            as_int = players_list[column_name].astype(int)
            if (players_list[column_name] == as_int).all():
                players_list[column_name] = as_int

    players_list = downcast_df(players_list, df_name='players_list', print_usage=print_usage)

    if not cache_path_exists:
        players_list.to_feather(cache_path)

    players_list['PLAYER_ID'] = players_list['PLAYER_ID'].astype('int64')

    return players_list


def get_header(player_or_team: bool = True, url: Optional[str] = None):
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0"
    ]
    referrers = ['https://www.nba.com/']
    if player_or_team:
        referrers.append('https://www.nba.com/stats/players/traditional')
    else:
        referrers.append('https://www.nba.com/stats/teams/traditional')

    headers = {
        'User-Agent': random.choice(user_agents),
        'Referer': random.choice(referrers),
        'Origin': 'https://www.nba.com',
        'Host': 'stats.nba.com',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Sec-Ch-Ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
    }

    if url and 'cdn.nba.com' in url:
        headers = {k: v for k, v in headers.items() if k not in (
            'Host', 'Origin', 'x-nba-stats-origin', 'x-nba-stats-token',
            'Sec-Fetch-Site', 'Sec-Fetch-Mode', 'Sec-Fetch-Dest'
        )}

    return headers


@retry(exceptions=(requests.exceptions.ReadTimeout, UnableToLoadWebsiteException, requests.exceptions.ConnectionError, KeyError, requests.exceptions.HTTPError), max_retries=3, delay_seconds=2)
def load_nbacom_data(url: str, use_key: str = 'resultSets', player_or_team: bool = True,
                     use_header: bool = True) -> Optional[Union[pd.DataFrame, dict]]:
    headers = None
    if use_header:
        headers = get_header(player_or_team=player_or_team, url=url)
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raises HTTPError if status is 4xx or 5xx
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            logging.error("Access Denied! The server blocked the request.")
            raise requests.exceptions.HTTPError("Access Denied! The server blocked the request.")
        else:
            raise requests.exceptions.HTTPError(e)
    html = response.content
    if len(html) == 0:
        raise UnableToLoadWebsiteException(site=url)
    html_dict = json.loads(html)
    if use_key in html_dict:
        data_dict = html_dict[use_key]
    else:
        raise KeyError(f'key {use_key} not in scraped data!')
    if isinstance(data_dict, list):
        data_dict = data_dict[0]
    if 'headers' in data_dict and 'rowSet' in data_dict:
        header, stats = data_dict['headers'], data_dict['rowSet']
        df = pd.DataFrame(stats, columns=header)
        if not df.empty:
            return df
        return
    else:
        return data_dict


@retry(exceptions=(requests.exceptions.ReadTimeout, UnableToLoadWebsiteException), max_retries=3, delay_seconds=1)
def load_lineups_from_nbacom() -> pd.DataFrame:
    url = URLS().games_lineups_nbacom(str(get_current_date()))
    headers = get_header()
    html = requests.get(url, headers=headers, timeout=30).content
    if len(html) == 0:
        raise UnableToLoadWebsiteException(site=url)

    data_dict = json.loads(html)['games']
    all_games = list()
    for curr_dict in data_dict:
        home_dict = curr_dict['homeTeam']
        home_df = pd.DataFrame(home_dict['players'])
        home_df = home_df.assign(TEAM_ID=home_dict['teamId'],
                                 TEAM_NAME=home_dict['teamAbbreviation'],
                                 HOME=True)
        away_dict = curr_dict['awayTeam']
        away_df = pd.DataFrame(away_dict['players'])
        away_df = away_df.assign(TEAM_ID=away_dict['teamId'],
                                 TEAM_NAME=away_dict['teamAbbreviation'],
                                 HOME=False)
        game_df = pd.concat([home_df, away_df], axis=0).assign(GAME_ID=curr_dict['gameId'])
        all_games.append(game_df)
    all_games = pd.concat(all_games, axis=0)

    schedule = load_schedule()
    todays_game_ids = schedule.loc[schedule['DATE'] == get_current_date(), 'GAME_ID'].astype(int).unique()
    all_games = all_games.loc[all_games['GAME_ID'].astype(int).isin(todays_game_ids)]

    all_games.rename(columns={'personId': 'PLAYER_ID', 'teamId': 'TEAM_ID', 'playerName': 'NAME',
                              'position': 'START_POSITION'},
                     inplace=True)
    all_games['STARTER'] = all_games['START_POSITION'].str.len() > 0
    all_games.drop(columns=['firstName', 'lastName', 'timestamp'], inplace=True)
    all_games['NAME'] = all_games['NAME'].replace(FIX_PLAYER_NAMES)
    return all_games


def get_latest_date_with_data(df: pd.DataFrame) -> datetime.date:
    return df.query('MIN > 0')['DATE'].max()


def assert_valid_df_all(df: pd.DataFrame, allow_less_games_2014: bool = False):
    assert df['GAME_ID'].notnull().all(), 'there are some rows with null GAME_ID!!!'
    previous_dates = df['DATE'] < get_current_date()
    if previous_dates.any():
        assert df.loc[previous_dates, 'GAME_ID'].min() > 10, 'there are some invalid GAME_IDs!!!'

    assert (df.groupby('TEAM_ID')['TEAM'].nunique() == 1).all(), \
        'there is a TEAM_ID with more than one team name!!!'
    assert (df.groupby('TEAM')['TEAM_ID'].nunique() == 1).all(), \
        'there is a team name with more than one TEAM_ID!!!'

    # there are 1230 total games in every regular season
    games_per_year = df.groupby('SEASON_YEAR')['GAME_ID'].nunique()
    games_per_year = games_per_year.loc[games_per_year.index <= 2022]
    if 2019 in games_per_year.index:
        assert games_per_year.loc[2019] == 1059, '2020 doesnt have the correct amount of games!!!'
        games_per_year.drop(index=2019, inplace=True)
    if 2020 in games_per_year.index:
        assert games_per_year.loc[2020] == 1080, '2020 doesnt have the correct amount of games!!!'
        games_per_year.drop(index=2020, inplace=True)
    if allow_less_games_2014:
        games_per_year = games_per_year.loc[games_per_year.index != 2014]
    assert (games_per_year == 1230).all(), 'missing games for a certain year!!!'


def assert_valid_df_players(df: pd.DataFrame):
    num_unique_names_per_id = df.groupby('PLAYER_ID')['NAME'].nunique()
    if not (num_unique_names_per_id <= 1).all():
        ids_and_names = dict()
        bad_ids = num_unique_names_per_id.loc[num_unique_names_per_id > 1].index
        for bad_id in bad_ids:
            ids_and_names[bad_id] = df.query(f'PLAYER_ID == {bad_id}')['NAME'].unique()
        logging.error(f'found PLAYER_ID with more than 1 name!\n{ids_and_names}')
        raise ValueError(f'found PLAYER_ID with more than 1 name!\n{ids_and_names}')
    if 'STARTER' in df.columns:
        assert (df.query('STARTER').groupby(['DATE', 'TEAM_ID']).size() == 5).all(), \
            "there is at least one game that doesn't have 5 starters!!!"
    if 'START_POSITION' in df.columns:
        group_cols = ['DATE', 'TEAM_ID']
        assert (df.assign(STARTER=df['START_POSITION'].notnull()).groupby(group_cols)['STARTER'].sum() == 5).all(), \
            'there is at least one team that doesnt have 5 starters!!!'
        assert (df.dropna(subset='START_POSITION').groupby(group_cols)['START_POSITION'].apply(set) ==
                set(START_POSITIONS)).all(), \
            "there is at least one team that doesn't have the correct positions!!!"
    if df.groupby(['PLAYER_ID', 'DATE']).size().max() > 1:
        raise ValueError('we have duplicate rows in the dataframe!')


def load_from_pkl(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def dates_to_epoch(dates: pd.Series) -> pd.Series:
    return (pd.to_datetime(dates) - datetime.datetime(1970, 1, 1)).dt.total_seconds()


def epoch_to_dates(epoch_times: pd.Series) -> pd.Series:
    return pd.to_datetime(epoch_times, unit='s')


def get_column_names_from_feather_file(path: str, return_dtypes: bool = False):
    try:
        reader = pyarrow.ipc.open_file(path)
    except pyarrow.lib.ArrowInvalid:
        reader = pyarrow.ipc.open_stream(path)
    if return_dtypes:
        return reader.schema.names, reader.schema.types
    return reader.schema.names


def sharpe(daily_account_money: pd.Series) -> float:
    """
    Calculates the sharpe ratio of an account from a daily series of the account balance

    Parameters
    ----------
    daily_account_money
        A series describing the account balance each day (must be continuous, i.e resampled by day)

    Returns
    -------
    float
        The Sharpe ratio of the account

    """
    returns = np.diff(daily_account_money) / daily_account_money.values[:-1]
    return returns.mean() / returns.std() * np.sqrt(252.0)


def add_missing_points_columns_to_predictions(predictions_df: pd.DataFrame):
    """ add PTS and YAHOO_POINTS columns to the predictions df if they're missing. The changes are done inplace"""
    for pp in ['pred', 'true']:
        if f'PTS_{pp}' not in predictions_df.columns and \
                all([c in predictions_df.columns for c in (f'3PM_{pp}', f'2PM_{pp}', f'FTM_{pp}')]):
            predictions_df[f'PTS_{pp}'] = \
                (predictions_df[[f'3PM_{pp}', f'2PM_{pp}', f'FTM_{pp}']] * [3., 2., 1.]).sum(axis=1)

        if f'YAHOO_POINTS_{pp}' not in predictions_df.columns:
            predictions_df[f'YAHOO_POINTS_{pp}'] = pd.concat([predictions_df[f'{c}_{pp}'] * p for c, p in
                                                              SCORING[Sites.YAHOO].items()], axis=1).sum(axis=1)


def scrape_season_schedules(season: Union[int, str], save_dir: Optional[str] = None) -> None:
    if save_dir:
        if isinstance(season, str):
            season = int(season.split('-')[0])
        save_path = os.path.join(save_dir, f'{season}.feather')

        if os.path.isfile(save_path):
            file_mtime = os.path.getmtime(save_path)
            current_time = time.time()
            age_seconds = current_time - file_mtime
            if age_seconds < 30 * 60:
                logging.info(f'found valid file for schedule for {season}, returning cached data')
                return

    logging.info(f'scraping season schedule {season}')
    season_schedule = load_nbacom_data(url=URLS().season_schedule(season=season),
                                       use_header=True,
                                       use_key='leagueSchedule')
    season_schedule = season_schedule['gameDates']
    season_schedule = sum([date_schedule['games'] for date_schedule in season_schedule], [])
    season_schedule = pd.DataFrame(season_schedule)
    season_schedule = \
        season_schedule[['gameId', 'gameDateTimeUTC', 'homeTeam', 'awayTeam', 'postponedStatus', 'gameStatusText']]
    season_schedule = season_schedule.assign(TEAM_ID_HOME=season_schedule['homeTeam'].apply(lambda x: x['teamId']).astype(int),
                                             TEAM_ID_AWAY=season_schedule['awayTeam'].apply(lambda x: x['teamId']).astype(int),
                                             )
    season_schedule = season_schedule.loc[(season_schedule['TEAM_ID_HOME'] != 0) &
                                          (season_schedule['TEAM_ID_AWAY'] != 0)].reset_index(drop=True)
    season_schedule = season_schedule.loc[season_schedule['gameStatusText'] != 'TBD'].reset_index(drop=True)
    season_schedule['GAME_DATE_TIME'] = (
        pd.to_datetime(season_schedule['gameDateTimeUTC'], utc=True).dt.tz_convert('US/Eastern')
    )
    season_schedule['GAME_ID'] = season_schedule['gameId'].astype(int)
    season_schedule.drop(columns=['homeTeam', 'awayTeam', 'gameDateTimeUTC', 'gameId'], inplace=True)
    if save_dir is not None:
        season_schedule.to_feather(save_path)
        logging.info(f'saved season schedule to {save_path}')


def get_current_time(tz=NY_TZ, return_date: bool = False) -> Union[datetime.date, datetime.datetime]:
    curr_time = datetime.datetime.now(tz=tz)
    if return_date:
        curr_time = curr_time.date()
    return curr_time


def get_current_date(tz=NY_TZ) -> datetime.date:
    return get_current_time(tz=tz, return_date=True)


def get_next_game_time(tz=NY_TZ, template: Optional[pd.DataFrame] = None) -> datetime.datetime:
    current_time = get_current_time(tz=tz)
    if template is not None:
        return template.loc[template['Time'] > current_time, 'Time'].min()
    schedule = load_schedule()
    return schedule.loc[schedule['GAME_DATE_TIME'] > current_time, 'GAME_DATE_TIME'].min()


def define_logger(filename: Optional[str] = None, level=logging.INFO,
                  format: Optional[str] = '%(asctime)s [%(levelname)s] - %(message)s', print_to_console: bool = True,
                  filemode: Optional[str] = 'a'):
    assert print_to_console or filename is not None, 'the logger has to print to the console and/or to a file!!!'

    vars = {'level': level
            }
    if format is not None:
        vars['format'] = format

    handlers = list()
    if print_to_console:
        handlers.append(logging.StreamHandler())
    if filename is not None:
        file_vars = {'filename': filename}
        if filemode is not None:
            file_vars['mode'] = filemode
        handlers.append(logging.FileHandler(**file_vars))
    vars['handlers'] = handlers

    logging.basicConfig(**vars)


def load_schedule(schedules_dir: str = 'season_schedules', drop_ppd: bool = True) -> pd.DataFrame:
    schedules_dir = update_dir_name(schedules_dir)
    schedule_names = sorted(os.listdir(schedules_dir))
    schedule = pd.concat([pd.read_feather(os.path.join(schedules_dir, s)) for s in schedule_names], axis=0)
    schedule['DATE'] = schedule['GAME_DATE_TIME'].dt.date
    schedule['GAME_TIME'] = schedule['GAME_DATE_TIME'].dt.hour + schedule['GAME_DATE_TIME'].dt.minute / 60.

    schedule_home = \
        schedule.rename(columns={'TEAM_ID_HOME': 'TEAM_ID', 'TEAM_ID_AWAY': 'TEAM_ID_OPPONENT'}).assign(HOME=True)
    schedule_away = \
        schedule.rename(columns={'TEAM_ID_AWAY': 'TEAM_ID', 'TEAM_ID_HOME': 'TEAM_ID_OPPONENT'}).assign(HOME=False)

    schedule = pd.concat([schedule_home, schedule_away])
    schedule.sort_values(['DATE', 'GAME_TIME', 'GAME_ID', 'HOME'], inplace=True)

    if drop_ppd:
        schedule = schedule.loc[schedule['gameStatusText'] != 'PPD']

    return schedule


def add_game_times_to_df(df: pd.DataFrame) -> pd.DataFrame:
    schedules = load_schedule()
    df = df.merge(schedules, how='left', left_on=['GAME_ID', 'TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID_HOME'])
    df = df.merge(schedules, how='left', left_on=['GAME_ID', 'TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID_AWAY'])
    df['GAME_DATE_TIME'] = df['GAME_DATE_TIME_x'].fillna(df['GAME_DATE_TIME_y'])

    drop_cols = [c + suff for c in schedules for suff in ('_x', '_y') if c + suff in df]
    df.drop(columns=drop_cols, inplace=True)

    assert not df['GAME_DATE_TIME'].isnull().any(), 'missing game times!!!!'

    return df


def remove_accents(input_str: Optional[str]) -> Optional[str]:
    if input_str is None:
        return None
    # Normalize string to decompose accented characters
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Filter out combining characters (accents and diacritics)
    return ''.join(char for char in normalized_str if not unicodedata.combining(char))


class FixPlayerNames:
    SUFFIXES = (' Jr.', ' Jr', ' Sr.', ' Sr', ' II', ' III', ' IV', '.', "'")

    def __init__(self, compare_names: Optional[Union[List[str], np.ndarray, pd.Series]] = None):
        """
        Fix player names, so that all names from all sources will match
        :param compare_names: The names that we want to search in to see if we could find similar names
        """
        if compare_names is not None and isinstance(compare_names, pd.Series):
            compare_names = compare_names.values
        self.compare_names = compare_names
        if self.compare_names is not None:
            self.compare_names = [remove_accents(n) for n in self.compare_names if n is not None]
            self.compare_names = [FIX_PLAYER_NAMES.get(n, n) for n in self.compare_names]

    def fix_names(self, names_to_convert: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        if isinstance(names_to_convert, pd.Series):
            names_to_convert = names_to_convert.values
        elif isinstance(names_to_convert, list):
            names_to_convert = np.array(names_to_convert)

        converted_names = np.array([remove_accents(n) if isinstance(n, str) else n for n in names_to_convert])
        converted_names = np.array([n.strip(' ') for n in converted_names])
        converted_names = np.array([FIX_PLAYER_NAMES.get(n, n) for n in converted_names])
        if self.compare_names is not None:
            names_to_fix = \
                sorted(set([n for n in converted_names if n not in self.compare_names and n is not None and len(n) > 3]))
            original_to_fixed_names = dict()    # keep all the names that were fixed here
            names_not_fixed = list()            # the names that weren't fixed
            for n in names_to_fix:
                inds = np.where(converted_names == n)[0]
                check_names = [f'{n[0]}.{n[1]}.{n[2:]}', f'{n[0]}.{n[1]} {n[2:]}']
                check_names += [n.replace(suff, '') for suff in FixPlayerNames.SUFFIXES if suff in n]
                check_names += [n + suff for suff in FixPlayerNames.SUFFIXES]

                check_names_in_compare_names = sorted(set(check_names) & set(self.compare_names))
                if len(check_names_in_compare_names) > 0:
                    converted_names[inds] = check_names_in_compare_names[0]
                    original_to_fixed_names[n] = check_names_in_compare_names[0]
                else:
                    matches_lower = \
                        [compare_name for compare_name in self.compare_names if n.lower() == compare_name.lower()]
                    if len(matches_lower) > 0:
                        converted_names[inds] = matches_lower[0]
                        original_to_fixed_names[n] = matches_lower[0]
                    else:
                        names_not_fixed.append(n)

            if len(original_to_fixed_names) > 0:
                logging.info(f'\n@@@@@@@@@@@@@@@@@@@ fixed names: @@@@@@@@@@@@@@@@@@@\n{original_to_fixed_names}\n'
                             f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            if len(names_not_fixed) > 0:
                logging.info(f"\n@@@@@@@@@@@@@@@@@@@ couldn't fix names: @@@@@@@@@@@@@@@@@@@\n{names_not_fixed}\n"
                             f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        return converted_names


def date_to_season_str(date: datetime.date = get_current_date()) -> str:
    year = date.year
    month = date.month
    season = year if month >= 10 else year - 1
    return season_int_to_str(season)


def date_to_game_ids(date: datetime.date = get_current_date()) -> np.ndarray:
    schedule = load_schedule()
    return schedule.loc[schedule['DATE'] == date, 'GAME_ID'].astype(int).unique()


@dataclass
class Timer:
    """
    Helper class to time stuff.
    Usage:
        with Timer("Cooking bananas"):
            cook_bananas()
    """
    task_name: str
    # if True, will also log when starting the task
    announce_start: bool = True
    # pass a dict and a key to save the running time
    report_dict: dict = None
    # this is the key under which the running
    report_key: str = None

    def __enter__(self):
        self.t1 = time.perf_counter()
        if self.announce_start:
            logging.info(f"--- {self.task_name}... ---")

    def __exit__(self, *args):
        self.duration = time.perf_counter() - self.t1
        if self.report_dict is not None:
            self.report_dict[self.report_key] = self.duration
        logging.info(f"--- {self.task_name} took {self.duration:.2f} secs ---")


def timeit(method):
    def timed(*args, **kwargs):
        logging.info(f'starting {method.__name__}...')
        ts = time.perf_counter()
        result = method(*args, **kwargs)
        te = time.perf_counter()
        logging.info(f'{method.__name__} time: {te - ts:.2f}s')
        return result
    return timed


def update_dir_name(dir_name: str):
    """Resolve a data subdirectory name to an absolute path under DATA_DIR."""
    dir_name = dir_name.strip('/')
    full_path = os.path.join(DATA_DIR, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

if __name__ == '__main__':
    load_lineups_from_nbacom()
    for curr_season in range(2014, 2023):
        curr_season=2023
        scrape_season_schedules(curr_season, save_dir=update_dir_name('season_schedules'))
        break