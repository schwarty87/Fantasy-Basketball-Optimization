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
from typing import Callable, Any
from bs4 import BeautifulSoup, Tag
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


_NO_DEFAULT = object()


def retry(exceptions: type[Exception] | tuple[type[Exception], ...] = Exception, max_retries: int = 3,
          delay_seconds: float = 60, return_on_fail: Any = _NO_DEFAULT):
    """
    Decorator function that retries a given function if specific exceptions occur.
    :param exceptions: Exception classes that trigger retries.
    :param max_retries: Retry this many times before failing
    :param delay_seconds: Time delay in seconds between retries.
    :param return_on_fail: If not _NO_DEFAULT, return this value if we failed the max amount of times. Could be a callable or
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
            if return_on_fail is _NO_DEFAULT:
                logging.error(fail_str)
                raise last_exception
            else:
                logging.warning(fail_str + ", returning default value...")
                return return_on_fail
        return wrapper
    return decorator_retry


def get_current_season(date: datetime.date | datetime.datetime | str | pd.Timestamp | None = None) -> int:
    """Returns the NBA season starting year (e.g. Oct 2026 -> 2026 for 2026-27 season)."""
    if date is None:
        d = get_current_date()
    elif isinstance(date, str):
        d = pd.to_datetime(date).date()
    else:
        d = date
    return d.year if d.month >= 10 else d.year - 1


def season_int_to_str(season: int | str | None = None) -> str:
    if season is None:
        season = get_current_season()
    elif isinstance(season, str):
        return season
    return f'{season}-{str(season+1)[-2:]}'


def downcast_df(df: pd.DataFrame, df_name: str | None = None, print_usage: bool = True, rtol: float = 1e-6,
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


def read_and_cache(func, func_params: dict, cache_name: str, suffix: str | None = None, 
                   sleep_time_after_reading_from_web: int = 0):
    cached_data_dir = update_dir_name('cached_data')
    cache_path = os.path.join(cached_data_dir, cache_name)
    if suffix is not None:
        cache_path += f'.{suffix}'
    if os.path.isfile(cache_path):
        return pd.read_feather(cache_path)
    df = func(**func_params)
    if df.empty:
        raise ValueError(f'no data returned from {func.__name__}')
    cache_dir = os.path.dirname(cache_path)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    df.to_feather(cache_path)
    time.sleep(sleep_time_after_reading_from_web)
    return df


def load_and_cache_players_list(print_usage: bool = True, season: int | None = None):
    if season is None:
        season = get_current_season()
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


def get_header(player_or_team: bool = True, url: str | None = None):
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
                     use_header: bool = True) -> pd.DataFrame | dict | None:
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


@retry(max_retries=3, delay_seconds=1, return_on_fail=None)
def fetch_soup(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 15,
    session: requests.Session | None = None,
    parser: str = 'html.parser',
    raise_for_status: bool = True,
    encoding: str | None = None
) -> BeautifulSoup | None:
    """
    Fetches a webpage using an HTTP GET request with standard desktop headers and parses
    its content using BeautifulSoup. Retries up to 3 times on network errors with a 1-second delay.

    Handles network errors, non-200 HTTP status codes, and empty response bodies gracefully
    by logging warnings and returning None if all retries fail.

    :param url: The target web address to request and parse.
    :param params: Optional dictionary of URL query parameters.
    :param headers: Optional dictionary of HTTP request headers. If None, defaults to a modern
                    desktop Chrome User-Agent header to avoid bot detection/403 blocks.
    :param timeout: Maximum time in seconds to wait for the server response before timing out
                    (defaults to 15 seconds).
    :param session: Optional requests.Session instance to reuse connection pools / cookies.
    :param parser: Parser library to use ('html.parser', 'lxml', 'xml', etc. Defaults to 'html.parser').
    :param raise_for_status: If True, raises HTTPError for 4xx/5xx responses so @retry can catch them.
    :param encoding: Optional explicit character encoding to decode response content.
    :return: A parsed BeautifulSoup object if the request was successful and non-empty,
             or None if all retry attempts failed or content was empty.
    """
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
    }
    requester = session or requests
    response = requester.get(url, params=params, headers=headers or default_headers, timeout=timeout)
    if raise_for_status:
        response.raise_for_status()
    if not response.content:
        return None
    content = response.content.decode(encoding) if encoding else response.content
    return BeautifulSoup(content, parser)


def parse_html_table(
    table: BeautifulSoup | Tag | None,
    header_row: int | None = 0,
    uppercase_headers: bool = False,
    header_transform: Callable[[str], str] | None = None,
    custom_headers: list[str] | None = None,
    strip_whitespace: bool = True,
    cell_parser: Callable[[Tag], Any] | None = None,
    row_parser: Callable[[Tag], dict[str, Any] | list[Any] | None] | None = None,
    skip_rows: int = 0,
    column_rename: dict[str, str] | None = None,
    context: dict[str, Any] | None = None,
    drop_empty_rows: bool = True
) -> pd.DataFrame:
    """
    Parses a BeautifulSoup <table> tag into a pandas DataFrame.
    Highly configurable and pure-Python (no lxml or external C-libraries required).

    :param table: BeautifulSoup Tag or soup element representing the <table> to parse.
    :param header_row: 0-indexed row position for headers, or None for headerless tables.
    :param uppercase_headers: If True, capitalizes all column header strings.
    :param header_transform: Optional callable f(header_str) -> str to transform column headers.
    :param custom_headers: Optional explicit list of column names to override extracted headers.
    :param strip_whitespace: If True, strips leading/trailing whitespace from cell values.
    :param cell_parser: Optional custom function f(cell_tag) -> Any to parse individual <td>/<th> cells.
    :param row_parser: Optional custom function f(tr_tag) -> Optional[Union[Dict, List[Dict], List]]
                       to parse entire rows. If it returns None, the row is omitted.
    :param skip_rows: Number of body rows to skip at the top of the table.
    :param column_rename: Optional mapping {old_col: new_col} to rename columns.
    :param context: Optional dictionary of key-value pairs (e.g. {'TEAM': 'BOS'}) to inject
                    into every row of the parsed DataFrame.
    :param drop_empty_rows: If True, drops rows that contain only empty strings or nulls.
    :return: A pandas DataFrame containing the parsed table data, or an empty DataFrame on failure.
    """
    if not table:
        return pd.DataFrame()

    parse_cell = cell_parser or (lambda el: el.get_text(strip=strip_whitespace))
    all_trs = table.find_all('tr')
    if not all_trs:
        return pd.DataFrame()

    # Separate thead / tfoot / tbody trs
    thead = table.find('thead')
    thead_trs = set(thead.find_all('tr')) if thead else set()
    tfoot = table.find('tfoot')
    tfoot_trs = set(tfoot.find_all('tr')) if tfoot else set()

    # Determine body trs (exclude thead and tfoot)
    body = table.find('tbody')
    if body:
        all_tbodies = table.find_all('tbody')
        body_trs = []
        for tb in all_tbodies:
            body_trs.extend(tb.find_all('tr'))
    else:
        body_trs = [tr for tr in all_trs if tr not in thead_trs and tr not in tfoot_trs]

    # If a custom row_parser is provided, let it parse each body row directly
    if row_parser is not None:
        custom_rows = []
        for tr in body_trs[skip_rows:]:
            parsed_row = row_parser(tr)
            if parsed_row is None:
                continue
            if isinstance(parsed_row, list) and parsed_row and isinstance(parsed_row[0], dict):
                custom_rows.extend(parsed_row)
            else:
                custom_rows.append(parsed_row)

        if not custom_rows:
            return pd.DataFrame()

        df = pd.DataFrame(custom_rows)
        if context:
            for k, v in context.items():
                df[k] = v
        if column_rename:
            df.rename(columns=column_rename, inplace=True)
        return df

    headers = None
    if custom_headers is not None:
        headers = list(custom_headers)
    elif header_row is not None:
        if thead and thead.find_all('th'):
            headers = [th.get_text(strip=strip_whitespace) for th in thead.find_all('th')]
        elif thead and thead.find_all('td'):
            headers = [td.get_text(strip=strip_whitespace) for td in thead.find_all('td')]
        elif len(body_trs) > header_row:
            first_row = body_trs[header_row]
            header_cells = first_row.find_all('th') or first_row.find_all('td')
            headers = [c.get_text(strip=strip_whitespace) for c in header_cells]
            body_trs = body_trs[header_row + 1:]

    if headers:
        if uppercase_headers:
            headers = [str(h).upper() for h in headers]
        if header_transform is not None:
            headers = [header_transform(str(h)) for h in headers]

    rows = []
    for tr in body_trs[skip_rows:]:
        cells = [parse_cell(td) for td in tr.find_all(['td', 'th'])]
        if cells:
            if drop_empty_rows and all(str(c).strip() == '' or c is None for c in cells):
                continue
            rows.append(cells)

    if not rows:
        df = pd.DataFrame(columns=headers if headers else None)
        if context:
            for k, v in context.items():
                df[k] = v
        if column_rename:
            df.rename(columns=column_rename, inplace=True)
        return df

    max_cols = max(len(r) for r in rows)
    if headers:
        if len(headers) < max_cols:
            headers = headers + [f'col_{i}' for i in range(len(headers), max_cols)]
        headers = headers[:max_cols]
    else:
        headers = [f'col_{i}' for i in range(max_cols)]

    padded_rows = [r + [''] * (max_cols - len(r)) for r in rows]
    df = pd.DataFrame(padded_rows, columns=headers)

    if context:
        for k, v in context.items():
            df[k] = v
    if column_rename:
        df.rename(columns=column_rename, inplace=True)

    return df


def fetch_html_table(
    url_or_soup: str | BeautifulSoup | Tag,
    table_selector: str | None = None,
    table_index: int = 0,
    all_tables: bool = False,
    concat_tables: bool = False,
    table_context_parser: Callable[[Tag], dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    **parse_kwargs
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Fetches a webpage and parses its HTML <table> element(s) into pandas DataFrame(s).
    Acts as a lightweight, zero-dependency, crash-free replacement for pd.read_html.

    :param url_or_soup: Webpage URL string to fetch, or an existing BeautifulSoup/Tag instance.
    :param table_selector: Optional CSS selector to target specific tables (e.g. 'table.datatable').
    :param table_index: The 0-based index of the table to return when multiple tables match (default: 0).
    :param all_tables: If True, returns a list of all parsed DataFrames (or one merged DataFrame
                       if concat_tables=True).
    :param concat_tables: If True, concatenates all matching tables into a single DataFrame.
                          (automatically enables processing of all tables).
    :param table_context_parser: Optional function f(table_tag) -> Dict to extract table metadata
                                 (e.g. preceding team name or header) to inject into that table's rows.
    :param headers: Optional HTTP headers dictionary to pass to requests.
    :param params: Optional URL query parameters dictionary.
    :param timeout: Network timeout in seconds (default: 15).
    :param parse_kwargs: Optional keyword arguments forwarded directly to parse_html_table
                         (e.g. uppercase_headers, header_transform, column_rename, cell_parser,
                         row_parser, custom_headers, header_row, skip_rows, drop_empty_rows).
    :return: A pandas DataFrame (or List[pd.DataFrame] if all_tables=True and concat_tables=False).
    """
    if isinstance(url_or_soup, (BeautifulSoup, Tag)):
        soup = url_or_soup
    else:
        soup = fetch_soup(str(url_or_soup), params=params, headers=headers, timeout=timeout)

    if not soup:
        return [] if (all_tables and not concat_tables) else pd.DataFrame()

    matched_tables = soup.select(table_selector) if table_selector else soup.find_all('table')
    if not matched_tables:
        return [] if (all_tables and not concat_tables) else pd.DataFrame()

    if all_tables or concat_tables:
        dfs = []
        for t in matched_tables:
            ctx = table_context_parser(t) if table_context_parser else None
            parsed_df = parse_html_table(t, context=ctx, **parse_kwargs)
            if not parsed_df.empty:
                dfs.append(parsed_df)

        if concat_tables:
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        return dfs

    if table_index >= len(matched_tables):
        return pd.DataFrame()

    target_table = matched_tables[table_index]
    ctx = table_context_parser(target_table) if table_context_parser else None
    return parse_html_table(target_table, context=ctx, **parse_kwargs)


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
    completed_season_cutoff = get_current_season() - 1
    games_per_year = games_per_year.loc[games_per_year.index <= completed_season_cutoff]
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


def scrape_season_schedules(season: int | str, save_dir: str | None = None) -> None:
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


def get_current_time(tz=NY_TZ, return_date: bool = False) -> datetime.date | datetime.datetime:
    curr_time = datetime.datetime.now(tz=tz)
    if return_date:
        curr_time = curr_time.date()
    return curr_time


def get_current_date(tz=NY_TZ) -> datetime.date:
    return get_current_time(tz=tz, return_date=True)


def get_next_game_time(tz=NY_TZ, template: pd.DataFrame | None = None) -> datetime.datetime:
    current_time = get_current_time(tz=tz)
    if template is not None:
        return template.loc[template['Time'] > current_time, 'Time'].min()
    schedule = load_schedule()
    return schedule.loc[schedule['GAME_DATE_TIME'] > current_time, 'GAME_DATE_TIME'].min()


def define_logger(filename: str | None = None, level=logging.INFO,
                  format: str | None = '%(asctime)s [%(levelname)s] - %(message)s', print_to_console: bool = True,
                  filemode: str | None = 'a'):
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


def add_game_times_to_df(df: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Enriches a DataFrame containing GAME_ID with the exact game start time (GAME_DATE_TIME).
    Can optionally accept a preloaded schedule DataFrame to avoid redundant disk I/O.
    """
    if schedule is None:
        schedule = load_schedule(drop_ppd=False)
    game_times = schedule[['GAME_ID', 'GAME_DATE_TIME']].drop_duplicates('GAME_ID')
    if 'GAME_DATE_TIME' in df.columns:
        df = df.drop(columns=['GAME_DATE_TIME'])
    df = df.merge(game_times, on='GAME_ID', how='left')
    assert not df['GAME_DATE_TIME'].isnull().any(), 'missing game times!!!!'
    return df


def remove_accents(input_str: str | None) -> str | None:
    if input_str is None:
        return None
    # Normalize string to decompose accented characters
    normalized_str = unicodedata.normalize('NFD', input_str)
    # Filter out combining characters (accents and diacritics)
    return ''.join(char for char in normalized_str if not unicodedata.combining(char))


class FixPlayerNames:
    SUFFIXES = (' Jr.', ' Jr', ' Sr.', ' Sr', ' II', ' III', ' IV', '.', "'")

    def __init__(self, compare_names: list[str] | np.ndarray | pd.Series | None = None):
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

    def fix_names(self, names_to_convert: list[str] | np.ndarray | pd.Series) -> np.ndarray:
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


def date_to_season_str(date: datetime.date | datetime.datetime | str | pd.Timestamp | None = None) -> str:
    return season_int_to_str(get_current_season(date))


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
    @wraps(method)
    def timed(*args, **kwargs):
        logging.info(f'starting {method.__name__}...')
        ts = time.perf_counter()
        result = method(*args, **kwargs)
        te = time.perf_counter()
        logging.info(f'{method.__name__} time: {te - ts:.2f}s')
        return result
    return timed


def update_dir_name(dir_name: str) -> str:
    """Resolve a data subdirectory name to a canonical absolute path under DATA_DIR.

    Subdirectories in cached_subdirs canonically live under DATA_DIR/cached_data/.
    Root-level data categories (e.g. 'stats', 'season_schedules', 'features')
    live directly under DATA_DIR.
    """
    clean_name = dir_name.replace('\\', '/').strip('/')
    cached_subdirs = {
        'starters',
        'injuries',
        'fantasy_data',
        'ejections',
        'nbacom_single_game_boxscores',
        'nbacom_single_game_pbp',
        'rotogrinders_lineups',
        'left_midgame',
    }
    parts = clean_name.split('/')
    if parts[0] == 'cached_data':
        full_path = os.path.join(DATA_DIR, *parts)
    elif parts[0] in cached_subdirs:
        full_path = os.path.join(DATA_DIR, 'cached_data', *parts)
    else:
        full_path = os.path.join(DATA_DIR, *parts)

    os.makedirs(full_path, exist_ok=True)
    return full_path

if __name__ == '__main__':
    load_lineups_from_nbacom()
    for curr_season in range(2014, 2023):
        curr_season=2023
        scrape_season_schedules(curr_season, save_dir=update_dir_name('season_schedules'))
        break