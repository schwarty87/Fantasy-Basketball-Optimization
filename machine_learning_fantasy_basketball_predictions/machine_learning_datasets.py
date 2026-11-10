import datetime
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import Sampler, Subset

from constants import START_POSITIONS, UNKNOWN_PLAYER_ID, IGNORE_VAL, TEAM_NAME_TO_ID, POSITION_TO_NUM
from utils import get_current_date, get_current_time
from machine_learning.constants import METADATA
from machine_learning.enums import DataSplitType
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined, \
    AbstractMachineLearningDataset
from machine_learning.machine_learning_datasets import MachineLearningDatasetSampleWeightsTabular
from machine_learning_fantasy_basketball_predictions.params.machine_learning_dataset_params import \
    MachineLearningDatasetSampleWeightsTabularFantasyBasketballPredictionsParams, \
    MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams
from machine_learning.machine_learning_datasets import AbstractCategoricalEncodingMachineLearningDataset,\
    MachineLearningDatasetTabularCategoricalEncoding
from machine_learning_fantasy_basketball_predictions.params.params import FeaturesParamsFantasyBasketballPredictions
from machine_learning.registries import datasets_registry
from machine_learning_fantasy_basketball_predictions.samplers import ByDateBatchSampler


class AbstractCategoricalEncodingMachineLearningDatasetFantasyBasketballPredictions(AbstractCategoricalEncodingMachineLearningDataset):
    pass


@datasets_registry.register
class MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictions(MachineLearningDatasetTabularCategoricalEncoding,
                                                                                   AbstractCategoricalEncodingMachineLearningDatasetFantasyBasketballPredictions):
    pass


@datasets_registry.register
class MachineLearningDatasetSampleWeightsTabularFantasyBasketballPredictions(MachineLearningDatasetSampleWeightsTabular):
    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        self._features_df[self.params.feature_names] = self._features_df[self.params.feature_names].round()
        super()._transform_features(keep_mask=keep_mask)

        if self.params.extra_weight_on_certain_class is not None:
            self._features_df.loc[self._features_df['PLAYER_ID'] == self.params.extra_weight_on_certain_class, self.params.feature_names] *= 2


@datasets_registry.register
class MachineLearningDatasetCombinedFantasyBasketballPredictions(MachineLearningDatasetCombined):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_index_to_date: np.ndarray = None
        self._build_date_map()

    def _init_machine_learning_datasets(
        self, features_df: pd.DataFrame,
        dataset_name_to_scalers: Optional[Dict[str, StandardScaler]] = None,
        dataset_name_to_categorical_encoder: Optional[Dict[str, OrdinalEncoder]] = None):
        super()._init_machine_learning_datasets(features_df=features_df,
                                                dataset_name_to_scalers=dataset_name_to_scalers,
                                                dataset_name_to_categorical_encoder=dataset_name_to_categorical_encoder
                                                )

    def _build_date_map(self):
        """
        Creates a 1D numpy array mapping every original dataset index to its date.
        This handles the sample_per_game logic internally.
        """
        logging.info("Building index-to-date map for batch sampler...")
        if METADATA not in self.machine_learning_datasets_stacked.datasets:
            logging.warning(f"'{METADATA}' dataset not found. Cannot build date map for batching.")
            return

        metadata_dataset = self.machine_learning_datasets_stacked.datasets[METADATA]
        metadata_column_names = metadata_dataset.params.feature_names

        try:
            date_col_idx = metadata_column_names.index('DATE')
        except ValueError:
            logging.error(f"'DATE' column not found in '{METADATA}' dataset. Day-based batching will fail.")
            raise

        all_dates_numpy = metadata_dataset.get_features_numpy()

        if self.params.sample_per_game:
            if all_dates_numpy.ndim != 3:
                raise ValueError(
                    f"Metadata has wrong shape for sample_per_game=True. Expected 3D, got {all_dates_numpy.shape}")
            # Get date from the first player of each game
            self._original_index_to_date = all_dates_numpy[:, 0, date_col_idx]
        else:
            if all_dates_numpy.ndim != 2:
                raise ValueError(
                    f"Metadata has wrong shape for sample_per_game=False. Expected 2D, got {all_dates_numpy.shape}")
            self._original_index_to_date = all_dates_numpy[:, date_col_idx]

    def _get_start_positions_list(self) -> List[str]:
        return START_POSITIONS + [f'BN{n}' for n in range(1, self.params.num_bench_players + 1)]

    def _load_features(self) -> pd.DataFrame:
        df = super()._load_features()

        # # TODO DELETE!!!!
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # for c in ['YAHOO_POSITION', 'YAHOO_SALARY']:
        #     mean_isnull = df.groupby('DATE')[c].apply(lambda x: x.isnull().mean())
        #     remove_dates = mean_isnull.loc[mean_isnull > 0.1].index
        #     df = df.loc[~df['DATE'].isin(remove_dates)]
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        if 'YAHOO_POSITION' in df.columns:
            df['YAHOO_POSITION'] = df['YAHOO_POSITION'].map(POSITION_TO_NUM).fillna(IGNORE_VAL)

        # if self.params.all_games_every_day:
        #     team_ids = set(TEAM_NAME_TO_ID.values())
        #     num_players_per_team = 5 + self.params.num_bench_players
        #     dummy_df_all = list()
        #     for year, year_df in df.groupby('SEASON_YEAR'):
        #         max_game_id = year_df['GAME_ID'].max()
        #         dummy_df_year = list()
        #         for date, date_df in year_df.groupby('DATE'):
        #             missing_team_ids = team_ids - set(date_df['TEAM_ID'])
        #             if not missing_team_ids:
        #                 continue
        #             game_ids = range(max_game_id + 1, max_game_id + 1 + len(missing_team_ids) // 2)
        #             dummy_df = pd.DataFrame({'DATE': date,
        #                                      'TEAM_ID': np.repeat(list(missing_team_ids), num_players_per_team),
        #                                      'GAME_ID': np.repeat(game_ids, num_players_per_team * 2),
        #                                      'SEASON_PROGRESS': date_df['SEASON_PROGRESS'].iloc[0]
        #                                      })
        #             max_game_id = dummy_df['GAME_ID'].max()
        #             dummy_df_year.append(dummy_df)
        #         dummy_df_year = pd.concat(dummy_df_year, axis=0).reset_index(drop=True)
        #         dummy_df_year['SEASON_YEAR'] = year
        #
        #         # adding SEASON_YEAR column
        #         if 'SEASON_YEAR' in df.columns:
        #             team_to_season_team = year_df.set_index('TEAM_ID')['SEASON_TEAM'].drop_duplicates().to_dict()
        #             dummy_df_year['SEASON_TEAM'] = dummy_df_year['TEAM_ID'].map(team_to_season_team)
        #
        #         dummy_df_all.append(dummy_df_year)
        #     dummy_df_all = pd.concat(dummy_df_all, axis=0).reset_index(drop=True)
        #     num_games_in_dummy_df = dummy_df_all['GAME_ID'].nunique()
        #     start_positions = self._get_start_positions_list()
        #     team_id_to_team_name = df.set_index('TEAM_ID')['TEAM'].drop_duplicates().to_dict()
        #     dummy_df_all = (
        #         dummy_df_all.assign(PLAYER_ID=UNKNOWN_PLAYER_ID,
        #                             NAME='UNKOWN',
        #                             TEAM=dummy_df_all['TEAM_ID'].map(team_id_to_team_name),
        #                             START_POSITION_CALCULATED = np.tile(start_positions, num_games_in_dummy_df * 2),
        #                             STARTER=np.tile([True] * 5 + [False] * self.params.num_bench_players,
        #                                             num_games_in_dummy_df * 2),
        #                             HOME=np.tile([True] * num_players_per_team + [False] * num_players_per_team,
        #                                          num_games_in_dummy_df),
        #                             EJECTED=False,
        #                             LEFT_MIDGAME=False,
        #                             IS_DUMMY=True,
        #                             )
        #     )
        #
        #     if 'YAHOO_SALARY' in df.columns:
        #         dummy_df_all = dummy_df_all.assign(YAHOO_SALARY=100)
        #     if 'YAHOO_POSITION' in df.columns:
        #         dummy_df_all = dummy_df_all.assign(YAHOO_POSITION='PG')
        #
        #     # adding stats columns
        #     fill_stats_cols = [c for c in df if c not in dummy_df_all and pd.api.types.is_numeric_dtype(df[c])]
        #     dummy_df_all = pd.concat([dummy_df_all,
        #                               pd.DataFrame(IGNORE_VAL, columns=fill_stats_cols, index=dummy_df_all.index)
        #                               ], axis=1)
        #     dummy_df_all = dummy_df_all.astype(df.dtypes.to_dict())
        #
        #     df['IS_DUMMY'] = False
        #     df = pd.concat([df, dummy_df_all], axis=0)
        #     df.sort_values(['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID'], inplace=True)

        return df

    def _sort_features(self, features_df: pd.DataFrame):
        if self.params.sample_per_game:
            features_df.drop_duplicates(['GAME_ID', 'HOME', 'START_POSITION_CALCULATED'], inplace=True)
            n_players_per_game = (5 + self.params.num_bench_players) * 2
            n_unique_game_ids = features_df['GAME_ID'].nunique()
            start_positions = self._get_start_positions_list()
            game_ids = features_df[['DATE', 'GAME_ID']].sort_values(['DATE', 'GAME_ID']).\
                drop_duplicates()['GAME_ID'].values
            features = pd.DataFrame().assign(GAME_ID=np.repeat(game_ids, n_players_per_game),
                                             HOME=np.tile(np.hstack(
                                                 [np.zeros(n_players_per_game // 2),
                                                  np.ones(n_players_per_game // 2)]),
                                                 n_unique_game_ids),
                                             START_POSITION_CALCULATED=np.tile(start_positions, n_unique_game_ids * 2)
                                             )
            features_df['HOME'] = features_df['HOME'].astype(np.float32)
            features_df = features.merge(features_df, on=['GAME_ID', 'HOME', 'START_POSITION_CALCULATED'], how='left')

            features_df['PLAYER_ID'] = features_df['PLAYER_ID'].fillna(UNKNOWN_PLAYER_ID)
            features_df['DATE'] = features_df['DATE'].replace({IGNORE_VAL: np.nan}).ffill()
            features_df['SEASON_YEAR'] = features_df['SEASON_YEAR'].replace({IGNORE_VAL: np.nan}).ffill()
            features_df['TEAM_ID'] = features_df['TEAM_ID'].replace({IGNORE_VAL: np.nan}).ffill()
            features_df['TEAM'] = features_df['TEAM'].replace({IGNORE_VAL: np.nan}).ffill()
            features_df = features_df.fillna(IGNORE_VAL)
        else:
            features_df.sort_values(['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID'], inplace=True)

        return features_df

    @staticmethod
    def _add_additional_columns(features_df: pd.DataFrame, params: FeaturesParamsFantasyBasketballPredictions
                                ) -> pd.DataFrame:
        features_df = MachineLearningDatasetCombined._add_additional_columns(features_df=features_df, params=params)

        missing_columns = sorted(set(params.all_feature_names_list) - set(features_df.columns))

        if 'split_type' not in features_df.columns:
            datetimes = pd.to_datetime(features_df['DATE'])
            years, months, days = datetimes.dt.year, datetimes.dt.month, datetimes.dt.day
            dates = pd.concat([years, months, days], axis=1)
            dates.columns = ['year', 'month', 'day']
            u_dates = dates.drop_duplicates().copy()
            tt = np.array([DataSplitType.TRAIN] * u_dates.shape[0])
            tt[2::4] = DataSplitType.VALIDATION
            tt[3::4] = DataSplitType.TEST
            u_dates['split_type'] = tt
            dates = dates.merge(u_dates, on=['year', 'month', 'day'], how='left')
            features_df['split_type'] = dates['split_type'].values

            # today's data should always be in train (so that we could use it for sample weights)
            today_inds = features_df['DATE'] == get_current_date()
            features_df.loc[today_inds, 'split_type'] = DataSplitType.TRAIN

            # putting the last month of data in the train dataset. this should help learn the more recent games better
            if params.last_month_in_train:
                month_ago = (get_current_time() - datetime.timedelta(days=30)).date()
                features_df.loc[features_df['DATE'] >= month_ago, 'split_type'] = DataSplitType.TRAIN

        if 'PLAYER_SEASON' in missing_columns:
            player_season_str = features_df['PLAYER_ID'].astype(int).astype(str) + \
                                (features_df['SEASON_YEAR'].astype(int) - 2000).astype(str)
            features_df['PLAYER_SEASON'] = player_season_str.astype(np.int32)

        features_df['NOT_IGNORE'] = ~(features_df['EJECTED'] | features_df['LEFT_MIDGAME'])
        features_df['NOT_IGNORE'] = features_df['NOT_IGNORE'].astype(int).replace({0: IGNORE_VAL})

        # adding mean ratios
        for c in missing_columns:
            if c.endswith('_mean_ratio'):
                base_name = c.replace('_mean_ratio', '')
                mean_name = base_name + '_mean5'
                features_df[c] = features_df[base_name] / features_df[mean_name]
                features_df[c].clip(upper=10, inplace=True)
                features_df.loc[(features_df[mean_name] == 0) & (features_df[base_name] == 0), c] = 1
                features_df.loc[(features_df[mean_name] == 0) & (features_df[base_name] != 0), c] = 10

        # adding lagged data
        lag_missing_cols = [c for c in missing_columns if '_t-' in c]
        lags = [int(c.split('_t-')[-1]) for c in lag_missing_cols]
        lag_stats = [c.split('_t-')[0] for c in lag_missing_cols]
        all_lagged_features = list()
        prev_lag_features_df = None
        for lag in sorted(set(lags)):
            inds = [ii for ii, v in enumerate(lags) if v == lag]
            stats = [lag_stats[ii] for ii in inds]
            lagged_features = features_df.groupby(['PLAYER_ID', 'STARTER'])[stats].shift(lag)

            if lag == 1:
                lagged_features2 = features_df.groupby(['PLAYER_ID'])[stats].shift(lag)
                starter_inds = features_df['STARTER'] == True
                lagged_features2.loc[starter_inds] = lagged_features2.loc[starter_inds] * 1.25
                lagged_features2.loc[~starter_inds] = lagged_features2.loc[~starter_inds] * 0.75
                lagged_features = lagged_features.fillna(lagged_features2)
            else:
                if prev_lag_features_df is not None:
                    available_stats = [s for s in stats if s in prev_lag_features_df.columns]
                    lagged_features[available_stats] = lagged_features[available_stats].fillna(prev_lag_features_df[available_stats])

            # If a player has no prior games in history (e.g. debut), fill with the role/position
            # baseline rather than zeros or naive un-grouped bfill (which leaks stats across players)
            role_group_cols = [c for c in ['START_POSITION_CALCULATED', 'STARTER'] if c in features_df.columns]
            if role_group_cols:
                role_means = features_df.groupby(role_group_cols)[stats].transform('mean')
                lagged_features = lagged_features.fillna(role_means)
            lagged_features = lagged_features.fillna(0.0)

            prev_lag_features_df = lagged_features.copy()

            lagged_features_renamed = lagged_features.rename(columns={stat: f'{stat}_t-{lag}' for stat in stats})
            all_lagged_features.append(lagged_features_renamed)
        features_df = pd.concat([features_df] + all_lagged_features, axis=1)

        return features_df

    def _postprocess_datasets(self, machine_learning_datasets: Dict[str, AbstractMachineLearningDataset]):
        """
        if sample_per_game, reshape the features to have a shape (n_games, n_players_per_game, n_features)
        convert the features to a numpy array of type float32
        """
        if self.params.sample_per_game:
            n_players_per_game = (5 + self.params.num_bench_players) * 2
            for dataset_name in machine_learning_datasets.keys():
                features = machine_learning_datasets[dataset_name].get_features_numpy()
                n_games = features.shape[0] // n_players_per_game
                new_shape = (n_games, n_players_per_game, machine_learning_datasets[dataset_name].num_features())
                machine_learning_datasets[dataset_name].set_features_numpy(np.reshape(features, new_shape))
            all_ngames = [mld.get_features_numpy().shape[0] for mld in machine_learning_datasets.values()]
            if not len(np.unique(all_ngames)) == 1:
                raise ValueError('not all datasets have the same number of games!')

    def _get_data_loader_batch_kwargs(self, batch_size: int | None = None, shuffle: bool = True,
                                      machine_learning_datasets_stacked_subset: Subset | None = None,
                                      ) -> Dict[str, int | Sampler]:
        num_days_per_batch = self.training_params.num_days_per_batch
        if num_days_per_batch and num_days_per_batch > 0 and self._original_index_to_date is not None:
            # Use the custom ByDateBatchSampler
            return {'batch_sampler': ByDateBatchSampler(subset_dataset=machine_learning_datasets_stacked_subset,
                                                        original_index_to_date_map=self._original_index_to_date,
                                                        num_days_per_batch=num_days_per_batch,
                                                        shuffle=shuffle
                                                        )
                    }
        return super()._get_data_loader_batch_kwargs(batch_size=batch_size, shuffle=shuffle,
                                                     machine_learning_datasets_stacked_subset=machine_learning_datasets_stacked_subset
                                                     )