import os
import datetime
import numpy as np
import pandas as pd
from requests.exceptions import TooManyRedirects, ReadTimeout, ConnectTimeout
from typing import List, Optional, Union
from constants import FIX_TEAM_NAMES
from utils import (FixPlayerNames, retry, UnableToLoadWebsiteException, get_current_date, get_current_season,
                   season_int_to_str, update_dir_name, fetch_html_table)

EJECTIONS_DF_COLUMNS = ["NAME", "TEAM", "DATE"]
EMPTY_DF = pd.DataFrame(columns=EJECTIONS_DF_COLUMNS)


class Ejections:
    def __init__(self, nbacom_players_list: Optional[Union[List[str], pd.Series, np.ndarray, pd.DataFrame]] = None,
                 season: Optional[int] = None):
        if isinstance(nbacom_players_list, pd.DataFrame):
            nbacom_players_list = nbacom_players_list["NAME"].values
        self.fix_player_names = FixPlayerNames(compare_names=nbacom_players_list)
        self.season = season or get_current_season()

        self.df = EMPTY_DF

        cached_data_dir = update_dir_name('cached_data')
        self.save_dir = os.path.join(cached_data_dir, 'ejections')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    @retry(exceptions=(ReadTimeout, UnableToLoadWebsiteException, TooManyRedirects, ConnectTimeout), max_retries=3,
           delay_seconds=1, return_on_fail=EMPTY_DF)
    def _load_ejections_spotrac(self) -> pd.DataFrame:
        url = f'https://www.spotrac.com/nba/fines-suspensions/_/year/{self.season}'
        ejections_df = fetch_html_table(
            url,
            column_rename={'Player': 'NAME', 'Team': 'TEAM', 'Date': 'DATE'}
        )
        if ejections_df.empty or not all(c in ejections_df.columns for c in ['NAME', 'TEAM', 'DATE', 'Infraction']):
            return EMPTY_DF

        ejections_df.dropna(subset=['NAME', 'TEAM', 'DATE'], how='any', inplace=True)
        ejections_df = ejections_df.loc[ejections_df['Infraction'].str.lower().str.contains('eject')]
        ejections_df['NAME'] = self.fix_player_names.fix_names(ejections_df['NAME'])
        ejections_df['TEAM'] = ejections_df['TEAM'].replace(FIX_TEAM_NAMES)
        ejections_df['DATE'] = pd.to_datetime(ejections_df['DATE'], format='%m/%d/%y').dt.date
        return ejections_df

    def collect_data(self) -> pd.DataFrame:
        date = get_current_date()
        curr_season_str = season_int_to_str(self.season)
        save_path = os.path.join(self.save_dir, f'{curr_season_str}.feather')
        if os.path.isfile(save_path):
            modified_time = os.path.getmtime(save_path)
            modified_time = datetime.datetime.fromtimestamp(modified_time)
            modified_date = modified_time.date()
            if modified_date >= date - datetime.timedelta(days=1):
                return pd.read_feather(save_path)

        ejections_spotrac = self._load_ejections_spotrac()
        self.df = ejections_spotrac

        self.df = self.df[EJECTIONS_DF_COLUMNS]
        self.df.to_feather(save_path)

        return self.df


if __name__ == '__main__':
    ejections = Ejections()
    ejections.collect_data()