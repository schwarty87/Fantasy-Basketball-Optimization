import os
import datetime
import numpy as np
import pandas as pd
from constants import TEAM_ID_TO_NAME, FIX_TEAM_NAMES
from utils import get_current_date, FixPlayerNames, load_schedule, load_and_cache_players_list, update_dir_name
from data_collectors.scrape_rotogrinders_data import load_fantasy_salaries_positions_lineups_hq, get_first_slate_id, \
    load_data_from_rotogrinders_lineups_hq, scrape_rotogrinders_lineups_data_from_all_sites_for_today


FANTASY_DATA_COLUMNS = ['NAME', 'PLAYER_ID', 'TEAM', 'YAHOO_SALARY', 'YAHOO_POSITION', 'DRAFTKINGS_SALARY',
                        'DRAFTKINGS_POSITION', 'FANDUEL_SALARY', 'FANDUEL_POSITION']
EMPTY_DF = pd.DataFrame(columns=FANTASY_DATA_COLUMNS)
TEMPLATE_RENAME_COLS = {'Name': 'NAME', 'Salary': 'YAHOO_SALARY', 'Position': 'YAHOO_POSITION', 'Team': 'TEAM'}


class LoadFantasyData:
    def __init__(self, nbacom_players_list: list[str] | pd.Series | np.ndarray | pd.DataFrame | None = None,
                 player_name_to_id: dict[str, int] | None = None):
        if isinstance(nbacom_players_list, pd.DataFrame):
            nbacom_players_list = nbacom_players_list["NAME"].values
        self.fix_player_names = FixPlayerNames(compare_names=nbacom_players_list)
        self.player_name_to_id = player_name_to_id

        self.schedule = load_schedule()
        self.schedule['TEAM'] = self.schedule['TEAM_ID'].replace(TEAM_ID_TO_NAME)

        self.df = EMPTY_DF

        self.save_dir = update_dir_name('fantasy_data')

    @staticmethod
    def merge_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        df1 = df1.merge(df2, on=['NAME'], how='outer', suffixes=(None, '_x'))
        x_cols = [c for c in df1.columns if c.endswith('_x')]
        non_x_cols = [c.rstrip('_x') for c in x_cols]
        df1[non_x_cols] = df1[non_x_cols].fillna(df1[x_cols].rename(columns=lambda x: x.rstrip('_x')))
        df1.drop(columns=x_cols, inplace=True)
        return df1

    @staticmethod
    def _convert_salary_cols_to_num(df: pd.DataFrame):
        salary_cols = [c for c in df.columns if c.endswith('_SALARY')]
        df[salary_cols] = df[salary_cols].astype(float)

    def _fix_names(self, df: pd.DataFrame):
        df['NAME'] = self.fix_player_names.fix_names(df['NAME'])
        df['TEAM'] = df['TEAM'].replace(FIX_TEAM_NAMES)

    def _load_data_rotogrinders(self, date: datetime.date) -> pd.DataFrame:
        if date == get_current_date():
            rotogrinders_df = scrape_rotogrinders_lineups_data_from_all_sites_for_today()
            if rotogrinders_df is None:
                return EMPTY_DF
        else:
            rotogrinders_df = load_fantasy_salaries_positions_lineups_hq(date=date)
            first_slate = get_first_slate_id(date=date)
            if first_slate is not None:
                extra_data_df = load_data_from_rotogrinders_lineups_hq(date=date, slate_id=first_slate)
                rotogrinders_df = \
                    rotogrinders_df.merge(extra_data_df[['TEAM']], how='left', left_index=True, right_index=True)
            else:
                players_df = load_and_cache_players_list()
                rotogrinders_df = rotogrinders_df.merge(players_df[['TEAM_ABBREVIATION', 'NAME']], on='NAME', how='left').rename(columns={'TEAM_ABBREVIATION': 'TEAM'})
        self._fix_names(rotogrinders_df)
        self._convert_salary_cols_to_num(rotogrinders_df)
        return rotogrinders_df

    def _load_yahoo_template_df(self, date: datetime.date) -> pd.DataFrame:
        submitted_teams_path = os.path.join('submitted_teams', f'{str(date).replace("-", "_")}.csv')
        if os.path.isfile(submitted_teams_path):
            template_df = pd.read_csv(submitted_teams_path, usecols=['Name', 'Position', 'Salary', 'Team'])
            template_df.rename(columns=TEMPLATE_RENAME_COLS, inplace=True)
            template_df = template_df.loc[template_df['NAME'].notnull() &
                                          (template_df['NAME'].str.strip().str.len() > 1)]
            self._fix_names(template_df)
            self._convert_salary_cols_to_num(template_df)
            return template_df
        return EMPTY_DF

    @staticmethod
    def _df_has_all_columns(df: pd.DataFrame) -> bool:
        return set(FANTASY_DATA_COLUMNS).issubset(set(df.columns))

    @staticmethod
    def _teams_match_todays_teams(df_teams: np.ndarray | list[set], todays_teams: np.ndarray | list[set]
                                  ) -> bool:
        return set(df_teams) == set(todays_teams)

    def _postprocess(self, df: pd.DataFrame, todays_teams: np.ndarray | list[set],
                     name_to_team: dict[str, str] | None) -> pd.DataFrame:
        # getting player ids
        df_has_player_id_col = 'PLAYER_ID' in df.columns
        if self.player_name_to_id is not None:
            ids_from_names = df['NAME'].replace(self.player_name_to_id)
            ids_from_names.loc[ids_from_names == df['NAME']] = np.nan
            df['PLAYER_ID'] = df['PLAYER_ID'].fillna(ids_from_names) if df_has_player_id_col else ids_from_names
        elif not df_has_player_id_col:
            df['PLAYER_ID'] = np.nan

        if not self._df_has_all_columns(df):
            raise ValueError(f'The fantasy data df is missing columns: {set(FANTASY_DATA_COLUMNS) - set(df.columns)}!')

        if name_to_team is not None:
            fill_team_inds = df['TEAM'].isnull() & df['NAME'].isin(name_to_team.keys())
            df.loc[fill_team_inds, 'TEAM'] = df.loc[fill_team_inds, 'NAME'].replace(name_to_team)

        # removing players who don't have any salary info
        salary_cols = [c for c in df if c.endswith('_SALARY')]
        remove_inds = df[salary_cols].isnull().all(axis=1)
        if remove_inds.any():
            remove_teams = df.query('TEAM.notnull()').loc[remove_inds, 'TEAM'].unique()
            keep_teams = df.loc[~remove_inds, 'TEAM'].unique()
            remove_teams = [t for t in set(remove_teams) - set(keep_teams) if (df['TEAM'] == t).sum() > 6]
            if len(remove_teams) > 0:
                todays_teams = np.array(list(set(todays_teams) - set(remove_teams)))
        df = df.loc[df[salary_cols].notnull().any(axis=1)]

        df = df.loc[df['TEAM'].isin(todays_teams) | df['TEAM'].isnull()]
        if not self._teams_match_todays_teams(df['TEAM'].dropna().unique(), todays_teams=todays_teams):
            missing_teams = set(todays_teams) - set(df['TEAM'].dropna().values)
            raise ValueError(f"The fantasy data df has the wrong teams:\nmissing teams: {missing_teams}")

        df['NAME'] = df['NAME'].replace({'nan': np.nan})
        df['TEAM'] = df['TEAM'].replace({'nan': np.nan})
        df = df.dropna(subset=['NAME', 'TEAM'], how='any', axis=0)
        df.drop_duplicates(['NAME', 'TEAM'], inplace=True)

        return df[FANTASY_DATA_COLUMNS].reset_index(drop=True)

    def _is_finished(self, df: pd.DataFrame, todays_teams: np.ndarray | list[set]) -> bool:
        return self._df_has_all_columns(df) and \
               self._teams_match_todays_teams(df['TEAM'].unique(), todays_teams=todays_teams) and \
               df['TEAM'].value_counts().min() > 5

    def collect_data(self, date: datetime.date, name_to_team: dict[str, str] | None = None) -> pd.DataFrame:
        todays_schedule = self.schedule.loc[self.schedule['DATE'] == date]
        todays_teams = todays_schedule['TEAM'].unique()

        df = EMPTY_DF

        save_path = os.path.join(self.save_dir, f"{str(date).replace('-', '_')}.feather")
        if os.path.isfile(save_path):
            df = pd.read_feather(save_path)
            df = df.loc[df['TEAM'].isin(todays_teams)]
        if not self._is_finished(df=df, todays_teams=todays_teams):
            rotogrinders_df = self._load_data_rotogrinders(date=date)
            if not rotogrinders_df.empty:
                df = self.merge_dfs(rotogrinders_df, df)
                yahoo_template_df = self._load_yahoo_template_df(date=date)
                df = self.merge_dfs(yahoo_template_df, df)
                df = self._postprocess(df=df, todays_teams=todays_teams, name_to_team=name_to_team)
                df.to_feather(save_path)

        self.df = df
        return df


if __name__ == '__main__':
    use_date = get_current_date() #- datetime.timedelta(days=1)
    use_date = datetime.date(2023, 12, 6)

    lfd = LoadFantasyData()
    lfd.collect_data(date=use_date)

    df = load_fantasy_salaries_positions_lineups_hq(date=use_date)
    aaa=1