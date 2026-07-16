import os
import datetime
import logging
import numpy as np
import pandas as pd
from enums import Sites
from utils import (get_column_names_from_feather_file, downcast_df, get_latest_date_with_data,
    load_and_cache_players_list, assert_valid_df_all, assert_valid_df_players, dates_to_epoch, get_current_date,
    define_logger, timeit, update_dir_name)
from typing import Optional, List, Tuple
from constants import YAHOO_CATEGORIES, POSITION_TO_NUM, START_POSITIONS, SUPERSTARS, \
    NUM_TO_POSITION, FIX_PLAYER_NAMES, FIX_TEAM_NAMES, IGNORE_VAL
from data_collectors.stats_loaders import StatsLoaderPlayers, StatsLoaderTeams


COUNTING_STATS = ['MIN', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK',
                  'BLKA', 'PTS', '2PM', 'FG_MISSED', 'FT_MISSED', 'PLUS_MINUS', 'YAHOO_POINTS', 'DRAFTKINGS_POINTS',
                  'FANDUEL_POINTS']
COPY_FROM_PLAYER_STATS = ['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'SEASON_YEAR', 'NAME', 'TEAM', 'OPPONENT',
                          'HOME', 'START_POSITION', 'DRAFTKINGS_POSITION', 'DRAFTKINGS_SALARY', 'YAHOO_POSITION',
                          'YAHOO_SALARY', 'FANDUEL_POSITION', 'FANDUEL_SALARY', 'IS_QUESTIONABLE', 'YAHOO_POINTS',
                          "DRAFTKINGS_POINTS", 'FANDUEL_POINTS', 'EPOCH', 'STARTER', 'EJECTED', 'LEFT_MIDGAME']  # , 'GAME_TIME']
COPY_FROM_PLAYER_STATS += sorted(set(YAHOO_CATEGORIES).union(set(COUNTING_STATS)) - set(COPY_FROM_PLAYER_STATS))
MEAN_WINDOWS = (5, 10)


class CalcFeatures:
    SORT_COLS = ['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID']

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath if filepath is not None else (
            os.path.join(update_dir_name('features'), 'features.feather'))

        self.updates_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.latest_date_with_data = datetime.date(2000, 1, 1)

    @timeit
    def _add_opponent_id(self):
        team_to_id = self.player_stats[['TEAM', 'TEAM_ID']].drop_duplicates()
        team_to_id = dict(zip(team_to_id['TEAM'], team_to_id['TEAM_ID']))
        self.updates_df['OPPONENT_ID'] = self.updates_df['OPPONENT'].map(team_to_id)

    @timeit
    def _calc_time_features(self):
        dates = pd.to_datetime(self.updates_df['DATE'])
        month = dates.dt.month
        day = dates.dt.day
        self.updates_df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 11)
        self.updates_df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 11)
        self.updates_df['day_sin'] = np.sin(2 * np.pi * (day - 1) / 30)
        self.updates_df['day_cos'] = np.cos(2 * np.pi * (day - 1) / 30)

        # adding season progress
        df = pd.concat([self.team_stats[['DATE', 'TEAM_ID', 'SEASON_YEAR']],
                        self.updates_df[['DATE', 'TEAM_ID', 'SEASON_YEAR']]],
                       axis=0).drop_duplicates(['DATE', 'TEAM_ID', 'SEASON_YEAR']).reset_index(drop=True).sort_values(['DATE', 'TEAM_ID'])
        df = df.loc[df['SEASON_YEAR'].isin(self.updates_df['SEASON_YEAR'].unique())]
        is_latest_season = df['SEASON_YEAR'] == df['SEASON_YEAR'].max()
        season_progress = \
            df.loc[~is_latest_season].groupby(['TEAM_ID', 'SEASON_YEAR'])['DATE'].rank(method='first', pct=True)
        latest_season_progress = \
            df.loc[is_latest_season].groupby(['TEAM_ID', 'SEASON_YEAR'])['DATE'].rank(method='first') / 82
        season_progress = pd.concat([season_progress, latest_season_progress]).rename("SEASON_PROGRESS")
        self.updates_df = self.updates_df.merge(pd.concat([season_progress, df[['DATE', 'TEAM_ID']]], axis=1),
                                                on=['TEAM_ID', 'DATE'], how='left')
        assert not self.updates_df['SEASON_PROGRESS'].isnull().any(), 'SEASON_PROGRESS has null values!'

    @timeit
    def _add_data_from_nbacom_players_list(self, players_list: Optional[pd.DataFrame] = None):
        if players_list is None:
            players_list = load_and_cache_players_list(season=self.player_stats['SEASON_YEAR'].max())
        players_list = players_list[['PLAYER_ID', 'HEIGHT', "WEIGHT", 'FROM_YEAR', 'DRAFT_NUMBER']]
        self.updates_df = self.updates_df.merge(players_list, on='PLAYER_ID', how='left')

        # fixing FROM_YEAR where possible
        from_year = self.player_stats.groupby('PLAYER_ID')['SEASON_YEAR'].min().reset_index()
        from_year.rename(columns={'SEASON_YEAR': 'FROM_YEAR'}, inplace=True)
        from_year = from_year.loc[from_year['FROM_YEAR'] > self.player_stats['SEASON_YEAR'].min()]
        self.updates_df = self.updates_df.merge(from_year, on='PLAYER_ID', how='left', suffixes=(None, '_x'))
        replace_inds = self.updates_df['FROM_YEAR_x'].notnull() & \
                       (self.updates_df['FROM_YEAR'] > self.player_stats['SEASON_YEAR'].min() - 2)
        self.updates_df.loc[replace_inds, 'FROM_YEAR'] = self.updates_df.loc[replace_inds, 'FROM_YEAR_x'].values
        self.updates_df.drop(columns='FROM_YEAR_x', inplace=True)
        self.updates_df['YEARS_SINCE_DRAFT'] = self.updates_df['SEASON_YEAR'] - self.updates_df['FROM_YEAR']

    @timeit
    def _add_days_rest(self):
        self.updates_df['PID'] = self.updates_df['PLAYER_ID'].astype('int64')
        self.updates_df['D1'] = pd.to_datetime(self.updates_df['DATE'])
        tmp = self.player_stats.query('MIN > 0').rename(columns={'DATE': 'D2', 'PLAYER_ID': 'PID'})
        tmp = tmp.astype({'D2': self.updates_df['D1'].dtype, 'PID': 'int64'})
        self.updates_df = pd.merge_asof(self.updates_df, tmp[['D2', 'PID']], by='PID', left_on='D1', right_on='D2',
                                        direction='backward', allow_exact_matches=False)
        self.updates_df['DAYS_REST'] = \
            (self.updates_df['D1'] - self.updates_df['D2']).dt.days.fillna(10).astype(int).clip(lower=1, upper=10)
        self.updates_df.drop(columns=['D1', 'D2', 'PID'], inplace=True)

    @staticmethod
    def _calc_rolling_means(stats_array: np.ndarray, ignore_ind: int, counting_stats_inds: np.ndarray,
                            window_lens: np.ndarray) -> np.ndarray:
        """
        Calculate rolling sum efficiently, while taking ignores into account
        :param stats_array: the array of stats we want to calculate the rolling means over
        :param ignore_ind: the column ind with the ignore flag
        :param counting_stats_inds: the inds of the stats we want to calculate the means for in stats_array
        :param window_lens: The lengths of the windows we want to use for calculating the means
        :return: A dataframe with the mean values, each row should correspond to the row in the stats_array
        """
        n_counting_stats = len(counting_stats_inds)
        not_ignore = 1 - stats_array[:, ignore_ind]
        curr_player_stats_without_ignore = \
            stats_array[:, counting_stats_inds] * not_ignore[:, None]
        cumsum_vals = np.vstack([np.zeros((1, n_counting_stats)),
                                 curr_player_stats_without_ignore[:-1].cumsum(axis=0)])
        cum_not_ignore = np.append([0], not_ignore[:-1].cumsum())

        v = cum_not_ignore[:, None] - window_lens
        start_inds = np.searchsorted(cum_not_ignore, v)
        sums = np.expand_dims(cumsum_vals, 1) - cumsum_vals[start_inds]
        w = np.expand_dims(cum_not_ignore, 1) - cum_not_ignore[start_inds]
        w = w[..., None]
        means = np.where(w != 0, np.divide(sums, w, where=w != 0), np.nan)
        return np.reshape(means, (stats_array.shape[0], len(window_lens) * n_counting_stats))

    @staticmethod
    def calc_mean_stats(player_stats: pd.DataFrame, stats_to_calculate: List[str] = COUNTING_STATS,
                        mean_windows: Tuple[int, ...] = MEAN_WINDOWS) -> pd.DataFrame:
        player_stats = player_stats.assign(LEFT_MIDGAME2=player_stats['LEFT_MIDGAME'] | player_stats['EJECTED'])
        cols = stats_to_calculate + ['LEFT_MIDGAME2', 'PLAYER_ID', 'STARTER']
        player_stats_numpy = player_stats[cols].values
        cols_inds = {k: i for i, k in enumerate(cols)}
        player_stats_numpy = player_stats_numpy.astype(float)
        n_counting_stats = len(stats_to_calculate)
        counting_stats_inds = np.arange(n_counting_stats)
        all_means = np.ones((player_stats_numpy.shape[0], n_counting_stats * len(mean_windows))) * np.nan
        all_means_starter = np.ones((player_stats_numpy.shape[0], n_counting_stats * len(mean_windows))) * np.nan
        window_lens = np.array(mean_windows)
        for player_id in np.unique(player_stats_numpy[:, cols_inds['PLAYER_ID']]):
            curr_player_inds = np.where(player_stats_numpy[:, cols_inds['PLAYER_ID']] == player_id)[0]
            curr_player_stats_numpy = player_stats_numpy[curr_player_inds]
            all_means[curr_player_inds] = \
                CalcFeatures._calc_rolling_means(stats_array=curr_player_stats_numpy,
                                                 ignore_ind=cols_inds['LEFT_MIDGAME2'],
                                                 counting_stats_inds=counting_stats_inds, window_lens=window_lens)
            for starter in np.unique(curr_player_stats_numpy[:, cols_inds['STARTER']]):
                curr_player_starter_inds_bool = curr_player_stats_numpy[:, cols_inds['STARTER']] == starter
                if sum(curr_player_starter_inds_bool) <= 1:
                    continue
                curr_player_starter_inds = curr_player_inds[curr_player_starter_inds_bool]
                curr_player_starter_stats_numpy = curr_player_stats_numpy[curr_player_starter_inds_bool]

                all_means_starter[curr_player_starter_inds] = \
                    CalcFeatures._calc_rolling_means(stats_array=curr_player_starter_stats_numpy,
                                                     ignore_ind=cols_inds['LEFT_MIDGAME2'],
                                                     counting_stats_inds=counting_stats_inds, window_lens=window_lens)

        # filling nulls from the starters array and converting to dfs
        ind_to_check = 0
        mean_dfs = list()
        for w, means_starter, means in zip(mean_windows, np.split(all_means_starter, len(mean_windows), axis=1),
                                           np.split(all_means, len(mean_windows), axis=1)):
            null_inds = (np.isnan(means_starter[:, [ind_to_check]])) & (~np.isnan(means[:, [ind_to_check]]))
            starter = player_stats_numpy[:, [cols_inds['STARTER']]].astype(bool)
            means_starter = np.where(null_inds, np.where(starter, means * 1.25, means / 1.25), means_starter)
            curr_df = pd.DataFrame(means_starter, columns=[f'{c}_mean{w}' for c in stats_to_calculate],
                                   index=player_stats.index, dtype=np.float32)
            mean_dfs.append(curr_df)

        return pd.concat(mean_dfs, axis=1)

    @timeit
    def _calc_mean_stats(self):
        use_inds = self.player_stats['PLAYER_ID'].isin(self.updates_df['PLAYER_ID'].unique())
        means_df = self.calc_mean_stats(player_stats=self.player_stats.loc[use_inds], stats_to_calculate=COUNTING_STATS,
                                        mean_windows=MEAN_WINDOWS)
        self.updates_df = pd.concat([self.updates_df.set_index('index_in_stats_df'), means_df], axis=1, join='inner')
        self.updates_df = self.updates_df.reset_index().rename(columns={'index': 'index_in_stats_df'})

    @timeit
    def _calc_positions_from_sites_positions(self) -> pd.DataFrame:
        """
        aggregate the positions that were assigned by each of the sites (Yahoo/DraftKings/FanDuel) and then order the
        players by the mean values. Each team's starters will have to have the values (PG/SG/SF/F/C)
        :return: dataframe with a single column that has the start positions (PG/SG/SF/F/C) for each player, and the
                 indexes correspond to the indexes in player_stats/updates_df
        """
        site_position_cols = [f'{site.name}_POSITION' for site in Sites]
        site_positions = \
            self.updates_df[site_position_cols + ['DATE', 'TEAM_ID', 'START_POSITION', 'index_in_stats_df',
                                                  'PLAYER_ID']].copy()
        for c in site_position_cols:
            site_positions[c] = site_positions[c].replace({np.nan: None}).apply(
                lambda x: [POSITION_TO_NUM[xx] for xx in str(x).split('/')] if x is not None else [])
        site_positions['SITE_POSITION'] = \
            site_positions[site_position_cols].apply(lambda x: np.mean(sum(x, [])), axis=1)

        site_positions['SITE_POSITION'] = \
            site_positions['SITE_POSITION'].fillna(site_positions['START_POSITION'].map(POSITION_TO_NUM))

        # filling nans
        position_per_player = \
            site_positions.dropna(subset=['SITE_POSITION']).groupby('PLAYER_ID')['SITE_POSITION'].mean().to_frame()
        site_positions = site_positions.merge(position_per_player.reset_index().astype({'PLAYER_ID': 'int64'}),
                                              on='PLAYER_ID',
                                              suffixes=(None, '_x'),
                                              how='left'
                                              )
        site_positions['SITE_POSITION'] = site_positions['SITE_POSITION'].fillna(site_positions['SITE_POSITION_x'])

        return site_positions.set_index('index_in_stats_df')['SITE_POSITION']

    @staticmethod
    @timeit
    def _calc_expected_order(player_stats: pd.DataFrame, updates_df: pd.DataFrame,
                             use_ids: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Calculate the expected playing time order based on previous games. We'll find, for each pair of players, who
        played more in each of the previous games where they played together, then rank them based on the totals for
        each player.
        :param player_stats: a dataframe with all player stats
        :param updates_df: a dataframe with the players that we want to use
        :param use_ids: the player ids we want to use
        :return: dataframe with a single column, EXPECTED_ORDER that has the players orders, and the index should
                 correspond to the indexes in player_stats/updates_df
        """
        if use_ids is None:
            use_ids = updates_df['PLAYER_ID'].unique()
        use_inds = player_stats['PLAYER_ID'].isin(use_ids)

        merge_df = player_stats.loc[use_inds, ['DATE', 'TEAM_ID', 'PLAYER_ID', 'MIN', 'STARTER']]
        ignore = player_stats.loc[use_inds, 'EJECTED'] | player_stats.loc[use_inds, 'LEFT_MIDGAME']
        merge_df.loc[ignore, 'MIN'] = np.nan
        merged = merge_df.merge(merge_df, on=['DATE', 'TEAM_ID'], how='inner')
        merged = merged.loc[merged['PLAYER_ID_x'] != merged['PLAYER_ID_y']]
        merged['MIN1_bigger_than_MIN2'] = (merged['MIN_x'] > merged['MIN_y']).astype(float)
        merged.loc[merged['MIN_x'] == merged['MIN_y'], 'MIN1_bigger_than_MIN2'] = 0.5
        merged.loc[merged['MIN_x'].isnull() | merged['MIN_y'].isnull(), 'MIN1_bigger_than_MIN2'] = np.nan
        rolling_results = \
            merged.groupby(['PLAYER_ID_x', 'PLAYER_ID_y']).rolling(6, min_periods=1, closed='left')[
                'MIN1_bigger_than_MIN2'].mean()
        rolling_results = rolling_results.reset_index().set_index('level_2')
        rolling_results = rolling_results.merge(merged[['DATE', 'TEAM_ID', 'STARTER_x']], left_index=True,
                                                right_index=True)
        totals = rolling_results.groupby(['DATE', 'TEAM_ID', 'PLAYER_ID_x'])['MIN1_bigger_than_MIN2'].sum().to_frame()
        totals.reset_index(inplace=True)
        totals = totals.sort_values(['DATE', 'TEAM_ID', 'MIN1_bigger_than_MIN2'], ascending=[True, True, False])
        totals['EXPECTED_ORDER'] = totals.assign(dummy=1).groupby(['DATE', 'TEAM_ID'])['dummy'].cumsum()
        totals = totals.merge(updates_df[['DATE', 'PLAYER_ID', 'index_in_stats_df']],
                              left_on=['DATE', 'PLAYER_ID_x'], right_on=['DATE', 'PLAYER_ID'], how='left')
        return totals[['EXPECTED_ORDER', 'index_in_stats_df']].set_index('index_in_stats_df')

    @timeit
    def _calc_start_positions(self):
        # adding starter positions
        starter_df = self.updates_df.loc[self.updates_df['STARTER']].copy()
        start_position_nums = starter_df['START_POSITION'].map(POSITION_TO_NUM)
        starter_df['SITE_POSITION'] += 0.01 * start_position_nums
        starter_df.sort_values(['DATE', 'TEAM_ID', 'SITE_POSITION'], inplace=True)
        starter_df['START_POSITION_CALCULATED'] = np.tile(START_POSITIONS, starter_df.shape[0] // 5)

        # adding bench positions
        bench_df = self.updates_df.loc[~self.updates_df['STARTER']].copy()
        bench_df.sort_values(['DATE', 'TEAM_ID', 'EXPECTED_ORDER'], inplace=True)
        bench_df['START_POSITION_CALCULATED'] = bench_df.assign(dummy=1).groupby(['DATE', 'TEAM_ID'])['dummy'].cumsum()
        bench_df['START_POSITION_CALCULATED'] = 'BN' + bench_df['START_POSITION_CALCULATED'].astype(str)

        starter_df['START_POSITION_CALCULATED_BY_POSITION'] = (starter_df['START_POSITION_CALCULATED'] + '1').values
        bench_df.sort_values(['DATE', 'TEAM_ID', 'SITE_POSITION_ROUNDED', 'EXPECTED_ORDER_BY_POSITION'], inplace=True)
        bench_df['START_POSITION_CALCULATED_BY_POSITION'] = \
            bench_df.assign(dummy=1).groupby(['DATE', 'TEAM_ID', 'SITE_POSITION_ROUNDED'])['dummy'].cumsum() + 1
        bench_df['START_POSITION_CALCULATED_BY_POSITION'] = \
            bench_df['SITE_POSITION_ROUNDED'].replace(NUM_TO_POSITION) + \
            bench_df['START_POSITION_CALCULATED_BY_POSITION'].astype(str)

        concatted = pd.concat([starter_df, bench_df], axis=0).set_index('index_in_stats_df')

        return concatted[['START_POSITION_CALCULATED', 'START_POSITION_CALCULATED_BY_POSITION']]

    @timeit
    def _calc_positions(self):
        """
        getting start positions. we'll calculate the starters positions from the sites positions, and we'll find
        the bench order based on previous games
        """
        site_positions = self._calc_positions_from_sites_positions()
        self.updates_df = self.updates_df.merge(site_positions, left_on='index_in_stats_df', right_index=True,
                                                how='left')

        # adding expected playing order per team
        expected_order = self._calc_expected_order(player_stats=self.player_stats, updates_df=self.updates_df)
        self.updates_df = self.updates_df.merge(expected_order, left_on='index_in_stats_df', right_index=True,
                                                how='left')

        # calculating expected order by position
        self.updates_df['SITE_POSITION_ROUNDED'] = self.updates_df['SITE_POSITION'].fillna(1).round().astype(int)
        starter_inds = self.updates_df['STARTER']
        self.updates_df['SITE_POSITION_ROUNDED'] = np.where(starter_inds,
                                                            self.updates_df['START_POSITION'].map(POSITION_TO_NUM).values,
                                                            self.updates_df['START_POSITION']).astype(float)
        # self.updates_df.loc[starter_inds, 'SITE_POSITION_ROUNDED'] = \
        #     self.updates_df.loc[starter_inds, 'START_POSITION'].replace(POSITION_TO_NUM).values
        self.updates_df['EXPECTED_ORDER_BY_POSITION'] = np.nan
        all_results = []
        for position_num, group_df in self.updates_df.groupby('SITE_POSITION_ROUNDED'):
            use_ids = group_df['PLAYER_ID'].unique()
            expected_order = self._calc_expected_order(player_stats=self.player_stats, updates_df=self.updates_df,
                                                       use_ids=use_ids)
            all_results.append(expected_order)

        if all_results:
            combined = pd.concat(all_results)
            # Handle duplicate indices from players having multiple positions on different dates
            combined = combined[~combined.index.duplicated(keep='first')]
            combined = combined.rename(columns={'EXPECTED_ORDER': 'EXPECTED_ORDER_BY_POSITION'})
            self.updates_df = self.updates_df.merge(combined, left_on='index_in_stats_df', right_index=True, how='left')

        self.updates_df.loc[starter_inds, 'EXPECTED_ORDER_BY_POSITION'] = \
            self.updates_df.loc[starter_inds, 'EXPECTED_ORDER_BY_POSITION'].fillna(1)

        start_positions = self._calc_start_positions()
        self.updates_df = self.updates_df.merge(start_positions, left_on='index_in_stats_df', right_index=True,
                                                how='left')

    @staticmethod
    @timeit
    def _calc_filler_features(df: pd.DataFrame):
        replace_dict = {IGNORE_VAL: np.nan}
        df['2PM'] = df['FGM'].replace(replace_dict).values - df['3PM'].replace(replace_dict).values
        df['2PM'] = df['2PM'].fillna(IGNORE_VAL)
        df['FG_MISSED'] = df['FGA'].replace(replace_dict).values - df['FGM'].replace(replace_dict).values
        df['FG_MISSED'] = df['FG_MISSED'].fillna(IGNORE_VAL)
        df['FT_MISSED'] = df['FTA'].replace(replace_dict).values - df['FTM'].replace(replace_dict).values
        df['FT_MISSED'] = df['FT_MISSED'].fillna(IGNORE_VAL)

    @timeit
    def _combine_year_team_ids(self):
        all_team_ids = list(self.updates_df['TEAM_ID'].unique())
        if not self.df.empty:
            all_team_ids = sorted(set(all_team_ids + list(self.df['TEAM_ID'].unique())))
        assert len(all_team_ids) == 30
        team_id_dict = {tid: i for i, tid in enumerate(sorted(all_team_ids))}
        year_str = self.updates_df['SEASON_YEAR'].astype(str).str[2:]

        def id_to_str(x):
            return x.replace(team_id_dict).apply(lambda y: f'{y:02d}')

        self.updates_df['SEASON_TEAM'] = (year_str + id_to_str(self.updates_df['TEAM_ID'])).astype(int)
        self.updates_df['SEASON_OPPONENT'] = (year_str + id_to_str(self.updates_df['OPPONENT_ID'])).astype(int)

    @timeit
    def _fill_nans(self):
        """fill nans for players where we don't have information about their previous games"""
        mean_cols = [[c for c in self.updates_df if c.endswith(f'_mean{n}')] for n in MEAN_WINDOWS]
        base_mean_cols = [c.split('_mean')[0] for c in mean_cols[0]]
        salary_cols = [c for c in self.player_stats if c.endswith('_SALARY')]

        # filling values by matching salaries and positions
        for match in [salary_cols, salary_cols[0], salary_cols[1], salary_cols[2]]:
            if not self.updates_df[mean_cols[0]].isnull().any().any():
                break
            match_cols_to_mean_values = self.player_stats.groupby(match)[base_mean_cols].mean()
            match_cols_to_mean_values.reset_index(inplace=True)
            self.updates_df = self.updates_df.merge(match_cols_to_mean_values, on=match, how='left',
                                                    suffixes=(None, '_x'))
            fill_values = self.updates_df[[c + '_x' for c in base_mean_cols]].values
            for mc in mean_cols:
                fill_values2 = pd.DataFrame(fill_values, columns=mc, index=self.updates_df.index)
                self.updates_df[mc] = self.updates_df[mc].fillna(fill_values2)
            self.updates_df.drop(columns=[c for c in self.updates_df if c.endswith('_x')], inplace=True)

        # filling the rest with 0
        for mc in mean_cols:
            self.updates_df[mc] = self.updates_df[mc].fillna(0)

        # filling todays games stats with 0
        today_inds = self.updates_df['DATE'] == get_current_date()
        self.updates_df.loc[today_inds, COUNTING_STATS] = self.updates_df.loc[today_inds, COUNTING_STATS].fillna(0)

    @timeit
    def _drop_nans(self):
        check_cols = ['PLAYER_ID', 'TEAM_ID', 'DATE', 'YEARS_SINCE_DRAFT', 'DAYS_REST', 'MIN_mean5',
                      'START_POSITION_CALCULATED'] + COUNTING_STATS
        check_cols = sorted(set(check_cols) & set(self.updates_df.columns))
        self.updates_df.dropna(subset=check_cols, inplace=True)

    @timeit
    def _save_features(self):
        if self.df is None or self.df.empty:
            self.df = self.updates_df
        else:
            self.df = self.df.loc[~self.df['GAME_ID'].isin(self.updates_df['GAME_ID'].unique())]
            self.df = pd.concat([self.df, self.updates_df], axis=0)

        # setting the df order
        first_cols = ['DATE', 'PLAYER_ID', 'NAME', 'GAME_ID', 'TEAM_ID', 'TEAM', 'OPPONENT_ID', 'OPPONENT']
        cols = first_cols + [c for c in self.df.columns if c not in first_cols]
        self.df = self.df[cols]

        # dropping columns that end with "_x"
        self.df.drop(columns=[c for c in self.df.columns if c.endswith('_x')], inplace=True)

        self.df.sort_values(['DATE', 'TEAM_ID', 'PLAYER_ID'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        downcast_df(self.df)

        self._assert_valid_df(self.df)

        logging.info(f'saving features to {self.filepath}')
        tmp_path = self.filepath + '.tmp'
        self.df.to_feather(tmp_path)
        os.replace(tmp_path, self.filepath)

    def call(self, players_list: Optional[pd.DataFrame] = None, add_upcoming_games: bool = True):
        """this is where we call all the methods to calculate features"""
        self._load_stats(add_upcoming_games)
        self._calc_filler_features(self.player_stats)
        self._normalize_counting_stats()
        self._filter_stats_dfs()
        self.init_df(add_upcoming_games)
        if self.updates_df is None or self.updates_df.empty:
            return

        self._add_opponent_id()
        self._calc_time_features()
        self._add_data_from_nbacom_players_list(players_list=players_list)
        self._add_days_rest()
        self._calc_mean_stats()
        self._calc_positions()
        self._combine_year_team_ids()

        self._fill_nans()
        self._drop_nans()
        self._save_features()

    def _load_stats(self, add_upcoming_games: bool = True):
        """load stats using StatsLoaders"""
        player_stats = StatsLoaderPlayers()
        self.player_stats = player_stats.load_saved_data()
        if not add_upcoming_games:
            self.player_stats = self.player_stats.loc[self.player_stats['DATE'] < get_current_date()]
        self.team_stats = StatsLoaderTeams().load_saved_data()

        self.player_stats = self.player_stats.assign(EPOCH=dates_to_epoch(self.player_stats['DATE']),
                                                     STARTER=self.player_stats['START_POSITION'].notnull())

    def _normalize_counting_stats(self):
        self.player_stats = self.player_stats.merge(self.team_stats[['GAME_ID', 'TEAM_ID', 'MIN']],
                                                    on=['GAME_ID', 'TEAM_ID'], how='left', suffixes=(None, '_x'))
        normalize_value = self.player_stats['MIN_x'] / 48.
        self.player_stats[COUNTING_STATS] = self.player_stats[COUNTING_STATS].div(normalize_value, axis=0)
        self.player_stats.drop(columns=['MIN_x'], inplace=True)

    def _filter_stats_dfs(self):
        """remove players who we could've known that they would be out before the game started"""

        # Keep only players who have a non-null NAME in at least one row (vectorized, avoids per-player .query())
        has_any_name = self.player_stats.groupby('PLAYER_ID')['NAME'].transform(lambda s: s.notnull().any())
        self.player_stats = self.player_stats.loc[has_any_name].reset_index(drop=True)

        self.player_stats = self.player_stats.loc[~self.player_stats['IS_OUT'].fillna(False)]
        self.player_stats.drop_duplicates(['PLAYER_ID', 'GAME_ID'], inplace=True)

        self.player_stats['LEFT_MIDGAME2'] = (self.player_stats['LEFT_MIDGAME'] | self.player_stats['EJECTED']) & \
                                             (self.player_stats['MIN'] < 30)
        self.player_stats['LEFT_MIDGAME2'] |= ((self.player_stats['MIN'] == 0) & (self.player_stats['PF'] > 0))

        comments_lower = self.player_stats['COMMENT'].str.lower()
        out_words = ['DND', 'NWT', 'suspension', 'protocol', 'personal', 'rest', 'injury']
        drop_inds = pd.concat([comments_lower.str.contains(w.lower()) for w in out_words], axis=1).any(axis=1)
        drop_inds.loc[self.player_stats['STARTER']] = False
        self.player_stats = self.player_stats.loc[~drop_inds]

        # removing players who clearly sat for a day off, and not because the coach just didn't want them to play
        check_players = self.player_stats['PLAYER_ID'].unique().tolist()
        while True:
            minute_minimums = \
                self.player_stats.query(f'PLAYER_ID.isin({check_players}) & (~LEFT_MIDGAME2)').groupby(['PLAYER_ID']).\
                    rolling(10, closed='left', min_periods=5)['MIN'].min().reset_index().\
                    set_index('level_1')['MIN'].sort_index()
            remove = (self.player_stats.loc[minute_minimums.index, 'MIN'] == 0) & (minute_minimums > 15.)
            if not remove.any():
                break
            check_players = self.player_stats.loc[remove.index].loc[remove, 'PLAYER_ID'].unique().tolist()
            self.player_stats.drop(index=remove.loc[remove].index, inplace=True)
            self.player_stats.reset_index(drop=True, inplace=True)

        # removing games where a star player sat
        is_dnpcd = (self.player_stats['COMMENT'].str.lower().str.contains('coach') |
                    ((self.player_stats['MIN'] == 0) & (self.player_stats['PF'] == 0)))
        remove = self.player_stats['NAME'].isin(SUPERSTARS) & is_dnpcd
        self.player_stats = self.player_stats.loc[~remove]
        self.player_stats.reset_index(drop=True, inplace=True)

        # removing games where the coach gave a starter player a day off
        w = 5
        is_dnpcd = (self.player_stats['COMMENT'].str.lower().str.contains('coach') |
                    ((self.player_stats['MIN'] == 0) & (self.player_stats['PF'] == 0)))
        n_starts = self.player_stats.groupby('PLAYER_ID').rolling(w, closed='left', min_periods=w)[
            'STARTER'].mean().reset_index().set_index('level_1')['STARTER'].sort_index()
        remove = (n_starts == 1.) & is_dnpcd.values
        self.player_stats = self.player_stats.loc[~remove]
        self.player_stats.reset_index(drop=True, inplace=True)

        # ##########################################################################
        # from utils import FixPlayerNames
        # possibly_left_midgame = pd.read_csv('/home/danieli/Desktop/possibly_list_midgame.csv')#.drop(columns='Unnamed: 0')
        # possibly_left_midgame['DATE'] = pd.to_datetime(possibly_left_midgame['DATE']).dt.date
        # # possibly_left_midgame2 = possibly_left_midgame.copy()
        # # possibly_left_midgame2['DATE'] = possibly_left_midgame['DATE'] - pd.Timedelta(days=1)
        # # possibly_left_midgame = pd.concat([possibly_left_midgame, possibly_left_midgame2], axis=0)
        # possibly_left_midgame['NAME'] = FixPlayerNames().fix_names(possibly_left_midgame['NAME'])
        # possibly_left_midgame.sort_values(['ESPN_GAME_ID', 'DATE', 'NAME'], inplace=True)
        # possibly_left_midgame.drop_duplicates(['NAME', 'ESPN_GAME_ID'], keep='first', inplace=True)
        # possibly_left_midgame.drop_duplicates(['NAME', 'DATE'], keep='first', inplace=True)
        #
        # check_players = self.player_stats['PLAYER_ID'].unique().tolist()
        # while True:
        #     minute_minimums = \
        #         self.player_stats.query(f'PLAYER_ID.isin({check_players}) & (~LEFT_MIDGAME2)').groupby(['PLAYER_ID']). \
        #             rolling(5, closed='left', min_periods=3)['MIN'].min().reset_index(). \
        #             set_index('level_1')['MIN'].sort_index()
        #     merged = self.player_stats.merge(possibly_left_midgame, on=['DATE', 'NAME'], how='left', indicator='xxx')
        #     merged['minute_min'] = minute_minimums
        #     merged = merged.query('(~LEFT_MIDGAME2) & (xxx=="both") & (((MIN < 0.67 * minute_min) & (MIN > 0)) | TEXT.str.contains("left the game"))')
        #     for i in range(len(merged)):
        #         print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        #         print(f"{tuple(merged.iloc[i][['NAME', 'DATE']].values)},")
        #         print(int(merged.iloc[i]['ESPN_GAME_ID']))
        #         print(merged.iloc[i]['TEXT'])
        #     merged = merged.loc
        # ##########################################################################


        # ##########################################################
        # check_players = self.player_stats['PLAYER_ID'].unique().tolist()
        # while True:
        #     minute_minimums = \
        #         self.player_stats.query(f'PLAYER_ID.isin({check_players}) & (~LEFT_MIDGAME2)').groupby(['PLAYER_ID']).\
        #             rolling(10, closed='left', min_periods=5)['MIN'].min().reset_index().\
        #             set_index('level_1')['MIN'].sort_index()
        #     remove = (self.player_stats.loc[minute_minimums.index, 'MIN'] != 0) & \
        #              (self.player_stats.loc[minute_minimums.index, 'MIN'] < 15.) & \
        #              (minute_minimums > 30.)
        #     if not remove.any():
        #         break
        #     check_players = self.player_stats.loc[remove[remove].index, 'PLAYER_ID'].unique().tolist()
        #     self.player_stats.drop(index=remove.loc[remove].index, inplace=True)
        #     self.player_stats.reset_index(drop=True, inplace=True)
        # ##########################################################

        self.player_stats.drop(columns='LEFT_MIDGAME2', inplace=True)

    @staticmethod
    def _assert_valid_df(df: pd.DataFrame):
        assert_valid_df_all(df, allow_less_games_2014=False)
        assert_valid_df_players(df)

        group_cols = ['DATE', 'TEAM_ID']
        assert (df.groupby(group_cols)['STARTER'].sum() == 5).all(), 'STARTER column has incorrect info!!!'
        spc = 'START_POSITION_CALCULATED'
        groups = df.query(f'{spc}.isin({START_POSITIONS})').groupby(group_cols)
        assert (groups.size() == 5).all(), f"there is a team/date where {spc} doesn't have 5 starters!!!"
        assert (groups[spc].apply(set) == set(START_POSITIONS)).all(), \
            f"there is at least one team/date that doesn't have the correct {spc}!!!"

    def load_df(self, columns: Optional[List[str]] = None):
        if os.path.isfile(self.filepath):
            if columns is not None:
                add_cols = set(self.SORT_COLS + ['TEAM', 'SEASON_YEAR', 'NAME', 'STARTER'])
                columns = set(columns).union(add_cols) if columns is not None else add_cols
                columns = sorted(set(columns).intersection(set(get_column_names_from_feather_file(self.filepath))))

            self.df = pd.read_feather(self.filepath, columns=columns)
            if 'DATE' in self.df.columns:
                self.df['DATE'] = pd.to_datetime(self.df['DATE']).dt.date
            self.df.sort_values(self.SORT_COLS, inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            downcast_df(self.df)
            if 'DATE' in self.df.columns:
                self.latest_date_with_data = get_latest_date_with_data(self.df)
            self.df['NAME'] = self.df['NAME'].replace(FIX_PLAYER_NAMES)
            self.df['TEAM'] = self.df['TEAM'].replace(FIX_TEAM_NAMES)
            self.df.drop_duplicates(['PLAYER_ID', 'DATE', 'GAME_ID', 'TEAM_ID'], inplace=True)
            self._assert_valid_df(self.df)
        return self.df

    def init_df(self, add_upcoming_games: bool = True):
        self.load_df()

        if self.df.empty:
            self.updates_df = self.player_stats[COPY_FROM_PLAYER_STATS].copy()
        else:
            # we'll want to calculate the features for the games that we didn't save properly before, and all games
            # that came after
            all_zero_gids = self.df.groupby('GAME_ID')['MIN'].max() == 0
            if all_zero_gids.any():
                all_zero_gids = list(all_zero_gids.loc[all_zero_gids].index)
                dates_after_zero_gid = \
                        self.player_stats['DATE'] >= self.df.loc[self.df['GAME_ID'].isin(all_zero_gids), 'DATE'].min()
            else:
                dates_after_zero_gid = pd.Series(False, index=self.player_stats.index)

            new_gids = ~self.player_stats['GAME_ID'].isin(self.df['GAME_ID'].unique())
            if add_upcoming_games:
                new_gids |= self.player_stats['DATE'] == get_current_date()

            self.df = self.df.loc[self.df['DATE'] != get_current_date()]
            self.updates_df = self.player_stats.loc[new_gids | dates_after_zero_gid, COPY_FROM_PLAYER_STATS]

        if self.updates_df.empty:
            return

        downcast_df(self.updates_df)
        self.updates_df = self.updates_df.reset_index().rename(columns={'index': 'index_in_stats_df'})
        self.updates_df = self.updates_df.loc[self.updates_df['NAME'].notnull()]


if __name__ == '__main__':
    define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO,
                  filename=None, print_to_console=True, filemode='a')
    cf = CalcFeatures(filepath='/home/danieli/PycharmProjects/FantasyBasketballPredictions/features/features.feather')
    cf.call(players_list=None, add_upcoming_games=True)