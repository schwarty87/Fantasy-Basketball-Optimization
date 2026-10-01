import os
import logging
import datetime
import requests
import json
import bs4
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional, Tuple
from tqdm import tqdm
from requests.exceptions import TooManyRedirects, ReadTimeout

from constants import POSITION_TO_NUM, URLS, NUM_TO_POSITION, TEAM_NAME_TO_ID
from enums import Sites
from utils import get_current_date, load_schedule, retry, UnableToLoadWebsiteException, update_dir_name, fetch_soup


@retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects), max_retries=3, delay_seconds=1)
def get_first_slate_id(date: datetime.date) -> Optional[int]:
    """ Getting the first slate ID for a given date """
    url = f"https://s3.amazonaws.com/json.rotogrinders.com/v2.00/{str(date).replace('-', '/')}/slates/nba-master.json"
    html = requests.get(url, headers=None, timeout=30).content
    data_dict = json.loads(html)["yahoo"]
    slates_df = pd.DataFrame(data_dict).T
    if 'date' not in slates_df.columns:
        return None
    slates_df = slates_df.loc[pd.to_datetime(slates_df["date"]).dt.date == date]
    if len(slates_df) == 0:
        return None
    n_games_per_slate = slates_df["games"].apply(len)
    return slates_df.iloc[np.argmax(n_games_per_slate)]["importId"]


@retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects), max_retries=3, delay_seconds=1)
def load_data_from_rotogrinders_lineups_hq(date: datetime.date, slate_id: int) -> pd.DataFrame:
    url = f"https://rotogrinders.com/schedules/nba/game-attributes?date={date}&site=yahoo&slate_id={slate_id}"
    html = requests.get(url, headers=None, timeout=30).content
    data_dict = json.loads(html)["players"]
    lineups_data = pd.DataFrame(data_dict).T
    lineups_data.rename(columns={"team": "TEAM"}, inplace=True)
    return lineups_data


@retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects), max_retries=3, delay_seconds=1)
def load_fantasy_salaries_positions_lineups_hq(date: datetime.date):
    """
    Load the fantasy salaries/positions for each player on a given day. Should return a df with columns
    ['FANDUEL_SALARY', 'FANDUEL_POSITION', 'DRAFTKINGS_SALARY', 'DRAFTKINGS_POSITION', 'YAHOO_SALARY', 'YAHOO_POSITION',
    'NAME']
    (notice there's no TEAM here, that should be added
    """
    url = f'https://s3.amazonaws.com/json.rotogrinders.com/lineuphq/v1.00/{date}/nba/players.json'
    html = requests.get(url, headers=None, timeout=30).content
    data_dict = json.loads(html)
    all_rows = list()
    for player_id, data_values in data_dict.items():
        if 'name' not in data_values.keys() or 'salaries' not in data_values.keys():
            continue
        name = data_values['name']
        drop_inds = ['ownersbox'] if 'ownersbox' in data_values['salaries'].keys() else None
        curr_row = pd.DataFrame(data_values).drop(columns=['site_ids', 'name'], index=drop_inds)
        curr_row.rename(index=lambda x: x.upper(), columns={'salaries': 'SALARY', 'positions': 'POSITION'},
                        inplace=True)
        curr_row = curr_row.stack()
        curr_row.index = curr_row.index.map('{0[0]}_{0[1]}'.format)
        curr_row['NAME'] = name
        curr_row.name = player_id
        all_rows.append(curr_row)
    if len(all_rows) > 0:
        all_rows = pd.concat(all_rows, axis=1).T
        all_rows = fix_column_names(all_rows)
    else:
        all_rows = (
            pd.DataFrame(columns=['NAME'] + sum([[s.name + '_SALARY', s.name + '_POSITION'] for s in Sites], []))
        )
    return all_rows


def parse_player_tag(player_tag: bs4.element.Tag) -> Dict[str, Any]:
    name = player_tag.find(class_='player-nameplate-name').text
    position = player_tag.find(class_='small muted').text
    salary = player_tag.find(class_='small player-nameplate-salary').text.replace('$', '')
    if len(salary) == 0:
        salary = -1
    elif salary.endswith('K'):
        salary = int(float(salary.rstrip('K')) * 1000)
    else:
        salary = int(salary)
    if position is None or ('/' not in position and position not in POSITION_TO_NUM.keys()):
        position = np.nan
    add_dict = {'NAME': name.strip().strip('\n'),
                'POSITION': position,
                'SALARY': salary,
                }
    return add_dict


def get_lineup_data_per_date_site(site: Sites, date: datetime.date) -> Optional[pd.DataFrame]:
    curr_url = URLS().lineups_rotogrinders(date=date, site=site)
    if date == get_current_date():
        curr_url = curr_url.replace(f"date={date}&", "")
    soup = fetch_soup(curr_url)
    if not soup:
        return None

    games = soup.find_all(class_="module game-card")
    if len(games) == 0:
        return None
    all_players_data = list()
    for game_id, game in enumerate(games):
        team_names = [shrt.text.strip() for shrt in game.find_all(class_="team-nameplate")]
        lineup_cards = game.find_all(class_="lineup-card")
        for team_name, lineup_card in zip(team_names, lineup_cards):
            unconfirmed = len(lineup_card.find_all("div", class_="lineup-card-body unconfirmed")) > 0
            player_tags = lineup_card.find_all(class_="player-nameplate-info")
            parsed_players = [parse_player_tag(player_tag) for player_tag in player_tags]
            team_data_df = pd.DataFrame(parsed_players)
            team_data_df["STARTER"] = ""
            team_data_df.loc[0:4, "STARTER"] = ["PG", "SG", "SF", "PF", "C"]
            team_data_df["TEAM"] = team_name
            team_data_df["OPPONENT"] = [t for t in team_names if t != team_name][0]
            team_data_df["CONFIRMED"] = not unconfirmed
            all_players_data.append(team_data_df)
    return pd.concat(all_players_data, axis=0).reset_index(drop=True)


def scrape_rotogrinders_lineups_data_from_all_sites_for_today(sites: Tuple[Sites, ...] = Sites.list_members()
                                                              ) -> Optional[pd.DataFrame]:
    """ Scrape the rotogrinders site for the fantasy data for today's games """
    date = get_current_date()
    lineups_data = list()
    cols_to_set_as_index = ["NAME", "TEAM", "OPPONENT", "STARTER"]
    for site in sites:
        curr_lineups_data = get_lineup_data_per_date_site(site=site, date=date)
        if curr_lineups_data is not None:
            curr_lineups_data.drop_duplicates(inplace=True)
            curr_lineups_data.set_index(cols_to_set_as_index, inplace=True)
            curr_lineups_data.rename(columns={c: f"{c}_{site.name}" for c in curr_lineups_data}, inplace=True)
            lineups_data.append(curr_lineups_data)
    if len(lineups_data) == 0:
        return None
    lineups_data = pd.concat(lineups_data, axis=1).reset_index()
    confirmed_cols = [c for c in lineups_data if "CONFIRMED_" in c]
    lineups_data["CONFIRMED"] = lineups_data[confirmed_cols].all(axis=1)
    lineups_data.drop(columns=[c for c in lineups_data if c.startswith('CONFIRMED_')], inplace=True)
    lineups_data = fix_column_names(lineups_data)
    return lineups_data


def fix_column_names(lineups_data: pd.DataFrame):
    fix_dict = dict()
    for c in lineups_data.columns:
        if c.lower().startswith('salary') or c.lower().startswith('position'):
            c_split = c.split('_')
            c_fixed = f'{c_split[1].title()}_{c_split[0].title()}'
            c_fixed = c_fixed.replace('Fanduel', 'FanDuel').replace('Draftkings', 'DraftKings')
            fix_dict[c] = c_fixed
    lineups_data.rename(columns=fix_dict, inplace=True)
    lineups_data.columns = [c.upper() for c in lineups_data.columns]
    return lineups_data


class RotoGrindersLineup:
    def __init__(self, add_date_column: bool = False, add_id_columns: bool = True, cache_data: bool = True):
        self.add_date_column = add_date_column
        self.add_id_columns = add_id_columns
        self.cache_data = cache_data

    def run(self, dates: Union[datetime.date, List[datetime.date]], out_players: Optional[Dict] = None) -> pd.DataFrame:
        """
        reads lineups and fantasy salaries data from RotoGrinders
        :param dates: what dates to run
        :param out_players: a dict of players that are out today and their replacements in the starting lineup
        :return: a dataframe with all the lineups and fantasy salaries data
        """
        if not isinstance(dates, list):
            dates = [dates]

        all_lineups_data = list()
        for curr_date in tqdm(dates, desc='getting rotogrinders lineup data...'):
            curr_lineups_data = self.get_lineup_data_per_date(date=curr_date)
            if curr_lineups_data is None:
                continue
            if curr_date == get_current_date():
                if out_players is not None:
                    for out_player, replacement_player in out_players.items():
                        if out_player not in curr_lineups_data['NAME'].values:
                            logging.info(f'{out_player} not in curr_lineups df!!')
                            continue
                        out_row = curr_lineups_data.loc[curr_lineups_data['NAME'] == out_player].iloc[0]
                        if isinstance(out_row['STARTER'], str) and replacement_player is not None:
                            start_position = out_row['STARTER']
                            curr_lineups_data.loc[curr_lineups_data['NAME'] == replacement_player, 'STARTER'] = start_position
                        curr_lineups_data = curr_lineups_data.loc[curr_lineups_data['NAME'] != out_player]

                # making sure that all teams have five starters
                five_starters_per_team = \
                        curr_lineups_data.groupby('TEAM')['STARTER'].apply(lambda x: x.notnull().sum()) == 5
                if not five_starters_per_team.all():
                    raise ValueError(f"teams in rotogrinders that don't have five players: "
                                     f"{list(five_starters_per_team.loc[~five_starters_per_team].index)}")
            if self.add_date_column:
                curr_lineups_data = curr_lineups_data.assign(DATE=curr_date)
            all_lineups_data.append(curr_lineups_data)
        if len(all_lineups_data) > 0:
            all_lineups_data = pd.concat(all_lineups_data, axis=0) if len(all_lineups_data) > 1 else all_lineups_data[0]
            if 'IS_QUESTIONABLE' not in all_lineups_data.columns:
                all_lineups_data['IS_QUESTIONABLE'] = False
            all_lineups_data['IS_QUESTIONABLE'] = all_lineups_data['IS_QUESTIONABLE'].fillna(False)
        else:
            all_lineups_data = pd.DataFrame(columns=['DATE', 'NAME', 'TEAM', 'STARTER', 'CONFIRMED'])

        return all_lineups_data

    def get_lineup_data_per_date(self, date: datetime.date) -> pd.DataFrame:
        cache_path = os.path.join(update_dir_name('cached_data'), 'rotogrinders_lineups', f'rotogrinders_lineups_{str(date)}.feather')
        if self.cache_data and os.path.isfile(cache_path):
            cached_data = pd.read_feather(cache_path)

            # removing postponed games
            postponed_games = load_schedule(drop_ppd=False).query(f'(gameStatusText == "PPD")')
            postponed_games = postponed_games.loc[postponed_games['DATE'] == date]
            if not postponed_games.empty:
                postponed_team_ids = postponed_games['TEAM_ID'].unique()
                lineups_team_ids = cached_data['TEAM'].replace(TEAM_NAME_TO_ID)
                cached_data = cached_data.loc[~lineups_team_ids.isin(postponed_team_ids)].reset_index(drop=True)
            if 'CONFIRMED' in cached_data.columns and cached_data['CONFIRMED'].all():
                cached_data = self.fix_column_names(cached_data)
                return cached_data

        ###################################################################################
        if date != get_current_date():
            df = load_fantasy_salaries_positions_lineups_hq(date=date)

            slate_id = get_first_slate_id(date=date)

            # getting starters
            df_stats = load_data_from_rotogrinders_lineups_hq(date=date, slate_id=slate_id)
            is_starter = df_stats['starter'].notnull()
            df_stats.loc[is_starter, 'STARTER'] = \
                df_stats.loc[is_starter, 'starter'].apply(lambda x: x['order']).replace(NUM_TO_POSITION)
            df_stats.loc[is_starter, 'CONFIRMED'] = df_stats.loc[is_starter, 'starter'].apply(lambda x: x['confirmed'])
            df_stats.sort_values(['TEAM', 'STARTER'], inplace=True)
            df_stats['CONFIRMED'] = df_stats['CONFIRMED'].ffill() == 1

            lineups_data = df.merge(df_stats[['STARTER', 'CONFIRMED', 'TEAM']], left_index=True, right_index=True, how='left')
            lineups_data.sort_values(['TEAM', 'STARTER'], inplace=True)
            lineups_data['CONFIRMED'] = True
        ###################################################################################

        else:
            lineups_data = scrape_rotogrinders_lineups_data_from_all_sites_for_today()
            if lineups_data is None:
                return
        lineups_data = self.fix_column_names(lineups_data)

        lineups_data.drop_duplicates(['NAME', 'TEAM'], keep='first', inplace=True)
        lineups_data.reset_index(drop=True, inplace=True)

        if self.cache_data:
            lineups_data.to_feather(cache_path)

        return lineups_data

    def parse_player_tag(self, player_tag: bs4.element.Tag) -> Dict[str, Any]:
        name = player_tag.find(class_='player-nameplate-name').text
        position = player_tag.find(class_='small muted').text
        salary = player_tag.find(class_='small player-nameplate-salary').text.replace('$', '')
        if len(salary) == 0:
            salary = -1
        elif salary.endswith('K'):
            salary = int(float(salary.rstrip('K')) * 1000)
        else:
            salary = int(salary)
        if position is None or ('/' not in position and position not in POSITION_TO_NUM.keys()):
            position = np.nan
        add_dict = {'NAME': name.strip().strip('\n'),
                    'POSITION': position,
                    'SALARY': salary,
                    }
        return add_dict

    @staticmethod
    def fix_column_names(lineups_data: pd.DataFrame):
        confirmed_cols = [c for c in lineups_data.columns if c.startswith('CONFIRMED_')]
        if len(confirmed_cols) > 0:
            lineups_data['CONFIRMED'] = lineups_data[confirmed_cols].all(axis=1)
        game_id_cols = [c for c in lineups_data.columns if c.startswith('GAME_ID')]
        if len(game_id_cols) > 0:
            lineups_data['GAME_ID'] = lineups_data[game_id_cols[0]].values
        home_cols = [c for c in lineups_data.columns if c.startswith('HOME')]
        if len(home_cols) > 0:
            lineups_data['HOME'] = lineups_data[home_cols[0]].values
        drop_cols = [c for c in confirmed_cols + game_id_cols + home_cols if c not in ['CONFIRMED', 'GAME_ID', 'HOME']]
        lineups_data.drop(columns=drop_cols, inplace=True)

        fix_dict = dict()
        for c in lineups_data.columns:
            if c.lower().startswith('salary') or c.lower().startswith('position'):
                c_split = c.split('_')
                c_fixed = f'{c_split[1].title()}_{c_split[0].title()}'
                c_fixed = c_fixed.replace('Fanduel', 'FanDuel').replace('Draftkings', 'DraftKings')
                fix_dict[c] = c_fixed
        lineups_data.rename(columns=fix_dict, inplace=True)

        # lineups_data.rename(columns={c: c.replace('GANE_TIME', 'GAME_TIME') for c in lineups_data.columns}, inplace=True)

        lineups_data.columns = [c.upper() for c in lineups_data.columns]

        return lineups_data


if __name__ == '__main__':
    date = datetime.date(2023, 12, 1)
    load_fantasy_salaries_positions_lineups_hq(date=get_current_date())

    slate_id = get_first_slate_id(date=date)
    starters_df = load_data_from_rotogrinders_lineups_hq(date=date, slate_id=slate_id)

    lineups_data = RotoGrindersLineup(add_id_columns=False).run(dates=[date])
