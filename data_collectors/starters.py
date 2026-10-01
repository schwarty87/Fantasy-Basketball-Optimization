import os
import datetime
import json
import logging
import re
import requests
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from requests.exceptions import TooManyRedirects, ReadTimeout, ConnectTimeout
from typing import List, Optional, Union, Type, Dict
from constants import URLS, FIX_TEAM_NAMES, NUM_TO_POSITION, TEAM_ID_TO_NAME, START_POSITIONS, UNKNOWN_PLAYER_ID
from enums import Sites
from utils import FixPlayerNames, retry, get_header, UnableToLoadWebsiteException, load_schedule, get_current_date, \
    define_logger, update_dir_name, fetch_soup
from data_collectors.scrape_rotogrinders_data import get_first_slate_id, \
    scrape_rotogrinders_lineups_data_from_all_sites_for_today, load_data_from_rotogrinders_lineups_hq, \
    load_fantasy_salaries_positions_lineups_hq

STARTER_DF_COLUMNS = ["NAME", "PLAYER_ID", "TEAM", "START_POSITION", "CONFIRMED"]
EMPTY_DF = pd.DataFrame(columns=STARTER_DF_COLUMNS)
START_POSITIONS_SET = set(START_POSITIONS)


class LoadStartersAbstract(ABC):
    def __init__(self, fix_player_names_instance: Optional[FixPlayerNames] = None):
        self.fix_player_names = fix_player_names_instance or FixPlayerNames()
        self.df = EMPTY_DF
    
    @abstractmethod
    @retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects), max_retries=3, delay_seconds=1,
           return_on_fail=EMPTY_DF)
    def _load_starters(self, date: datetime.date = get_current_date(), **kwargs):
        """
        Each source will fill the relevant code here to load the starters df from the given source, for a given date.
        The results should be a df with columns ["NAME", "TEAM", "START_POSITION", "CONFIRMED"]
        """
        raise NotImplementedError

    def _fix_names(self):
        self.df["TEAM"] = self.df["TEAM"].replace(FIX_TEAM_NAMES)
        if "OPPONENT" in self.df.columns:
            self.df["OPPONENT"] = self.df["OPPONENT"].replace(FIX_TEAM_NAMES)
        self.df["NAME"] = self.fix_player_names.fix_names(self.df["NAME"])

    def _filter_df(self, out_players: Optional[List[str]] = None) -> pd.DataFrame:
        self.df = self.df.loc[self.df["START_POSITION"].notnull() & (self.df["START_POSITION"].str.len() > 0)]
        self.df.drop_duplicates(["NAME", "TEAM"], keep="first", inplace=True)
        self.df.drop_duplicates(["TEAM", "START_POSITION"], keep="first", inplace=True)
        if out_players is not None:
            drop = self.df["NAME"].isin(out_players) & (~self.df['CONFIRMED'])
            if drop.any():
                logging.info(f"\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
                             f"dropping out players from starters in {type(self).__name__}:\n"
                             f"{self.df.loc[drop, 'NAME'].to_list()}\n"
                             f"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                self.df = self.df.loc[~drop].reset_index(drop=True)
        return self.df.reset_index(drop=True)

    def _postprocess(self, out_players: List[str]):
        if 'PLAYER_ID' not in self.df:
            self.df['PLAYER_ID'] = np.nan

        if not set(STARTER_DF_COLUMNS).issubset(set(self.df.columns)):
            raise ValueError(f"the starters df returned from {self.__name__} has to have all of the following columns: "
                             f"{STARTER_DF_COLUMNS}")
        self._fix_names()
        self._filter_df(out_players=out_players)

    def run(self, date: datetime.date = get_current_date(), out_players: Optional[List[str]] = None, **kwargs):
        self._load_starters(date=date, **kwargs)
        self._postprocess(out_players=out_players)


class LoadStartersNBACOM(LoadStartersAbstract):
    @retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects), max_retries=3, delay_seconds=1,
           return_on_fail=EMPTY_DF)
    def _load_starters(self, schedule: pd.DataFrame, date: datetime.date = get_current_date(), **kwargs):
        todays_schedule = schedule.loc[schedule['DATE'] == date]

        url = URLS().games_lineups_nbacom(str(date))
        headers = get_header()
        html = requests.get(url, headers=headers, timeout=30).content
        if len(html) == 0:
            raise UnableToLoadWebsiteException(site=url)

        data_dict = json.loads(html)["games"]
        dfs = list()
        for curr_dict in data_dict:
            home_dict = curr_dict["homeTeam"]
            home_df = pd.DataFrame(home_dict["players"])
            home_df = home_df.assign(TEAM_NAME=home_dict["teamAbbreviation"],
                                     HOME=True)
            away_dict = curr_dict["awayTeam"]
            away_df = pd.DataFrame(away_dict["players"])
            away_df = away_df.assign(TEAM_NAME=away_dict["teamAbbreviation"],
                                     HOME=False)
            game_df = pd.concat([home_df, away_df], axis=0).assign(GAME_ID=curr_dict["gameId"])
            dfs.append(game_df)
        df = pd.concat(dfs, axis=0)
        todays_game_ids = todays_schedule["GAME_ID"].astype(int).unique()
        df = df.loc[df["GAME_ID"].astype(int).isin(todays_game_ids)]

        df.rename(columns={"personId": "PLAYER_ID", "teamId": "TEAM_ID", "playerName": "NAME",
                           "position": "START_POSITION", "TEAM_NAME": "TEAM", "rosterStatus": "STATUS"},
                  inplace=True)
        df.drop(columns=["firstName", "lastName", "timestamp"], inplace=True)
        df = df.loc[df["START_POSITION"].notnull() & (df["START_POSITION"].str.len() > 0)]
        if date == get_current_date():
            df["CONFIRMED"] = df["lineupStatus"] == "Confirmed"
        else:
            df["CONFIRMED"] = True
        self.df = df


class LoadStartersRotogrinders(LoadStartersAbstract):
    @retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects, ConnectTimeout), max_retries=3,
           delay_seconds=1, return_on_fail=EMPTY_DF)
    def _load_starters(self, date: datetime.date = get_current_date(), **kwargs):
        if date != get_current_date():
            # getting the first slate id
            slate_id = get_first_slate_id(date=date)

            # getting starters
            lineups_data = load_data_from_rotogrinders_lineups_hq(date=date, slate_id=slate_id)
            is_starter = lineups_data["starter"].notnull()
            lineups_data.loc[is_starter, "STARTER"] = \
                lineups_data.loc[is_starter, "starter"].apply(lambda x: x["order"]).replace(NUM_TO_POSITION)
            lineups_data.loc[is_starter, "CONFIRMED"] = \
                lineups_data.loc[is_starter, "starter"].apply(lambda x: x["confirmed"])
            lineups_data.sort_values(["TEAM", "STARTER"], inplace=True)
            lineups_data["CONFIRMED"] = lineups_data["CONFIRMED"].ffill() == 1
            lineups_data["NAME"] = lineups_data["stats"].apply(lambda x: x["name"] if "name" in x.keys() else np.nan)

            # filling missing names
            fantasy_salaries_positions = load_fantasy_salaries_positions_lineups_hq(date=date)
            lineups_data = lineups_data.merge(fantasy_salaries_positions[['NAME']], left_index=True, right_index=True,
                                              suffixes=(None, '_x'), how='left')
            lineups_data['NAME'] = lineups_data['NAME'].fillna(lineups_data['NAME_x'])

            lineups_data.dropna(subset=["STARTER", "NAME"], inplace=True)
            if date != get_current_date():
                lineups_data["CONFIRMED"] = True
        else:
            lineups_data = scrape_rotogrinders_lineups_data_from_all_sites_for_today(sites=(Sites.YAHOO,))
            if lineups_data is None:
                return EMPTY_DF

        lineups_data.rename(columns={"STARTER": "START_POSITION"}, inplace=True)

        self.df = lineups_data


class LoadStartersRotowire(LoadStartersAbstract):
    def _load_starters(self, date: datetime.date = get_current_date(), **kwargs):
        # TODO fill this!
        self.df = EMPTY_DF


class LoadStartersLineupsDotCom(LoadStartersAbstract):
    def _load_starters(self, date: datetime.date = get_current_date(), **kwargs):
        # TODO fill this!
        self.df = EMPTY_DF


class LoadStartersBasketballMonster(LoadStartersAbstract):
    TEAMS_STR_PATTERN = r"([A-Z]{3}) @ ([A-Z]{3})"

    @retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects, ConnectTimeout), max_retries=3,
           delay_seconds=1, return_on_fail=EMPTY_DF)
    def _load_starters(self, date: datetime.date = get_current_date(), **kwargs):
        if date != get_current_date():
            return EMPTY_DF

        soup = fetch_soup(URLS.BASKETBALL_MONSTER_LINEUPS)
        if not soup:
            return EMPTY_DF

        tables = soup.find_all('table', class_='datatable')
        if not tables:
            return EMPTY_DF

        rows = []
        for t in tables:
            m = re.search(self.TEAMS_STR_PATTERN, t.get_text())
            if not m:
                continue
            team1, team2 = m.group(1), m.group(2)
            for tr in t.find_all('tr'):
                cells = [c.get_text(strip=True) for c in tr.find_all(['td', 'th'])]
                if len(cells) >= 3 and cells[0] in START_POSITIONS:
                    pos, p1, p2 = cells[0], cells[1], cells[2]
                    rows.append({'NAME': p1, 'TEAM': team1, 'START_POSITION': pos})
                    rows.append({'NAME': p2, 'TEAM': team2, 'START_POSITION': pos})

        if not rows:
            return EMPTY_DF

        all_dfs = pd.DataFrame(rows)

        for suff in ('P', 'Q', 'Off Inj', 'In', 'S', 'LR', 'Ret', 'D'):
            suff_len = len(suff) + 1
            replace = all_dfs['NAME'].str[-suff_len:].str.lower() == f' {suff.lower()}'
            all_dfs.loc[replace, 'NAME'] = all_dfs.loc[replace, 'NAME'].str[:-suff_len].str.strip()

        self.df = all_dfs.assign(CONFIRMED=False)


class Starters:
    """
    A class to load starters. It will load starters from a list of sources (given in all_load_classes()), and stop
    when all the starters positions are filled for the given day's games.
    Each loading class loads the starters data from a given site, and returns a df with the starters data. We then
    combine all the data and return the combined results
    """
    def __init__(self, nbacom_players_list: Optional[Union[List[str], pd.Series, np.ndarray, pd.DataFrame]] = None,
                 player_name_to_id: Optional[Dict[str, int]] = None):
        if isinstance(nbacom_players_list, pd.DataFrame):
            nbacom_players_list = nbacom_players_list["NAME"].values

        self.fix_player_names = FixPlayerNames(compare_names=nbacom_players_list)
        self.player_name_to_id = player_name_to_id

        self.schedule = load_schedule()
        self.schedule['TEAM'] = self.schedule['TEAM_ID'].replace(TEAM_ID_TO_NAME)

        self.df = EMPTY_DF

        self.save_dir = os.path.join(update_dir_name('cached_data'), 'starters')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    @staticmethod
    def all_load_classes() -> List[Type[LoadStartersAbstract]]:
        return [LoadStartersNBACOM, LoadStartersBasketballMonster, LoadStartersRotogrinders, LoadStartersRotowire,
                LoadStartersLineupsDotCom]

    @staticmethod
    def _is_finished(df: pd.DataFrame, todays_teams: List[str]) -> bool:
        has_all_positions = df.groupby('TEAM')['START_POSITION'].apply(set) == START_POSITIONS_SET
        has_five_starters_teams = has_all_positions.loc[has_all_positions].index.to_list()
        is_finished = (set(todays_teams) == set(has_five_starters_teams))
        if 'PLAYER_ID' in df:
            all_starters_have_id = df.query('START_POSITION.notnull()')['PLAYER_ID'].notnull().all()
            is_finished &= all_starters_have_id
        return is_finished

    @staticmethod
    def _combine_dfs(df: pd.DataFrame, curr_df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return curr_df.reset_index(drop=True)

        # find the teams that have more starters in curr_df than in df, remove them from df and copy them from
        # curr_df into df
        copy_teams = list()
        for team, curr_team_df in curr_df.groupby('TEAM'):
            team_df = df.query(f'TEAM=="{team}"')
            if team_df.empty:
                copy_teams.append(team)
                continue
            team_has_all_positions = set(team_df['START_POSITION'].values) == START_POSITIONS_SET
            team_confirmed = team_df['CONFIRMED'].all()
            if (team_has_all_positions and team_confirmed) or (curr_team_df.shape[0] < team_df.shape[0]):
                continue
            if curr_team_df.shape[0] > team_df.shape[0]:
                copy_teams.append(team)
                continue
            curr_team_has_all_positions = set(curr_team_df['START_POSITION']) == START_POSITIONS_SET
            curr_team_confirmed = curr_team_df['CONFIRMED'].all()
            if (curr_team_confirmed and not team_confirmed) or \
                    (curr_team_has_all_positions and not team_has_all_positions):
                copy_teams.append(team)

        if len(copy_teams) > 0:
            df = pd.concat([df.loc[~df['TEAM'].isin(copy_teams)],
                            curr_df.loc[curr_df['TEAM'].isin(copy_teams)]],
                           axis=0)
        return df.reset_index(drop=True)

    def _fill_missing_starters(self, df: pd.DataFrame, salaries_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if salaries_df is None:
            return df
        if 'IS_OUT' in salaries_df.columns:
            salaries_df = salaries_df.loc[~salaries_df['IS_OUT']]
        salaries_df_bench = salaries_df.loc[~salaries_df['NAME'].isin(df['NAME'].values)]
        add_rows = list()
        for team, team_df in df.groupby('TEAM'):
            unfilled_positions = START_POSITIONS_SET - set(team_df['START_POSITION'].values)
            for p in unfilled_positions:
                possible_replacements = \
                    salaries_df_bench.loc[(salaries_df_bench['TEAM'] == team) &
                                          (salaries_df_bench['YAHOO_POSITION'] == p)]
                if len(possible_replacements) > 0:
                    replace_name = possible_replacements.sort_values('YAHOO_SALARY')['NAME'].iloc[-1]
                else:
                    if p == 'C' and 'PF' in salaries_df_bench['YAHOO_POSITION'].values:
                        possible_replacements = \
                            salaries_df_bench.loc[(salaries_df_bench['TEAM'] == team) &
                                                  (salaries_df_bench['YAHOO_POSITION'] == 'PF')]
                        replace_name = possible_replacements.sort_values('YAHOO_SALARY')['NAME'].iloc[-1]
                    else:
                        replace_name = 'Injured Starter'
                add_rows.append({'NAME': replace_name, 'START_POSITION': p, 'TEAM': team, 'CONFIRMED': False,
                                 'PLAYER_ID': self.player_name_to_id.get(replace_name, UNKNOWN_PLAYER_ID)})
                salaries_df_bench = salaries_df_bench.loc[salaries_df_bench['NAME'] != replace_name]
                logging.info(f'filling missing starter for team {team}, position {p} with {replace_name}')
        return pd.concat([df, pd.DataFrame(add_rows)])

    def collect_data(self, date: datetime.date = get_current_date(), out_players: Optional[List[str]] = None,
                     salaries_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load starters data from the various websites for a given date.
        :param date: The date we want to collect the starters data for
        :param out_players: A list of player names that we want to exclude (due to injury, for example)
        :param salaries_df: A dataframe with the players fantasy salaries for the given date. The df will be used to
                            fill missing starters. The df should have columns
                            ['NAME', 'TEAM', 'YAHOO_SALARY', 'YAHOO_POSITION'].
                            If the df has a column called 'INJURED', we'll exclude those players where the column is
                            True
        :return: The starters df. The df will have columns ['NAME', 'TEAM', 'START_POSITION', 'CONFIRMED']
        """
        todays_schedule = self.schedule.loc[self.schedule['DATE'] == date]
        todays_teams = todays_schedule['TEAM'].unique()

        # loading saved file if exists
        save_path = os.path.join(self.save_dir, f"{str(date).replace('-', '_')}.feather")
        if os.path.isfile(save_path) and date != get_current_date():
            df = pd.read_feather(save_path)
            if self._is_finished(df, todays_teams) and df['CONFIRMED'].all():
                return df
        else:
            df = EMPTY_DF.copy()

        filled_all_teams = False
        for load_class in self.all_load_classes():
            load_instance = load_class(fix_player_names_instance=self.fix_player_names)
            load_instance.run(date=date, schedule=self.schedule, out_players=out_players)
            curr_df = load_instance.df

            if curr_df.empty:
                logging.info(f"finished running {load_class.__name__}, returned empty dataframe")
                continue

            df = self._combine_dfs(df=df, curr_df=curr_df)
            logging.info(f"finished running {load_class.__name__}")

            # getting player ids
            if self.player_name_to_id is not None:
                ids_from_names = df['NAME'].map(self.player_name_to_id)
                ids_from_names.loc[ids_from_names == df['NAME']] = np.nan
                df['PLAYER_ID'] = df['PLAYER_ID'].fillna(ids_from_names) if 'PLAYER_ID' in df else ids_from_names

            if self._is_finished(df, todays_teams=todays_teams):
                filled_all_teams = True
                break

        df = df[STARTER_DF_COLUMNS]
        if filled_all_teams:
            logging.info('finished loading all starters successfully!')
            df.to_feather(save_path)
            return df

        # filling starters from bench players
        df = self._fill_missing_starters(df=df, salaries_df=salaries_df)
        if not self._is_finished(df=df, todays_teams=todays_teams):
            raise ValueError("There was a problem when loading starters!")

        df.to_feather(save_path)
        logging.info('finished loading all starters successfully')
        self.df = df
        return df


if __name__ == "__main__":
    from utils import load_and_cache_players_list
    import traceback
    from injuries import Injuries

    define_logger(print_to_console=True)

    run_date = get_current_date() #- datetime.timedelta(days=1)
    players_list = load_and_cache_players_list(season=2023)
    if run_date == get_current_date():
        injuries = Injuries(nbacom_players_list=players_list)
        injuries.collect_data()
        out_players = injuries.out_players()
    else:
        out_players = None

    # from scrape_rotogrinders_data import RotoGrindersLineup
    # salaries_df = RotoGrindersLineup().run(dates=[run_date])
    # salaries_df['NAME'] = FixPlayerNames(compare_names=players_list['NAME'].values).fix_names(salaries_df['NAME'])
    salaries_df = None

    try:
        starters = Starters(nbacom_players_list=players_list)
        starters.collect_data(date=run_date, out_players=out_players, salaries_df=salaries_df)
    except:
        traceback.print_exc()