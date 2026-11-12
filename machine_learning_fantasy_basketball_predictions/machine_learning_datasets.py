import datetime
import logging
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, List, Union, Sequence, Any

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import Sampler, Subset, DataLoader

from constants import START_POSITIONS, UNKNOWN_PLAYER_ID, IGNORE_VAL, TEAM_NAME_TO_ID, POSITION_TO_NUM
from utils import get_current_date, get_current_time
from machine_learning.constants import METADATA
from machine_learning.enums import DataSplitType
from machine_learning.machine_learning_datasets import (
    MachineLearningDatasetCombined,
    AbstractMachineLearningDataset,
    MachineLearningDatasetSampleWeightsTabular,
    AbstractCategoricalEncodingMachineLearningDataset,
    MachineLearningDatasetTabularCategoricalEncoding,
    dates_to_epoch_days,
)
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

        if 'YAHOO_POSITION' in df.columns:
            df['YAHOO_POSITION'] = df['YAHOO_POSITION'].map(POSITION_TO_NUM).fillna(IGNORE_VAL)

        df = self._add_player_history_columns(df, self.params)

        return df

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

        is_real_player = features_df['PLAYER_ID'].fillna(UNKNOWN_PLAYER_ID) != UNKNOWN_PLAYER_ID
        not_ignore = is_real_player & ~(features_df['EJECTED'].fillna(False).astype(bool) | features_df['LEFT_MIDGAME'].fillna(False).astype(bool))
        features_df['NOT_IGNORE'] = np.where(not_ignore, 1, IGNORE_VAL)
        if 'IS_REAL_PLAYER' in missing_columns or 'IS_REAL_PLAYER' not in features_df.columns:
            features_df['IS_REAL_PLAYER'] = np.where(is_real_player, 1, IGNORE_VAL)

        # Idempotent fallback in case features were not loaded via _load_features (e.g. mocked tests)
        features_df = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(features_df, params)

        return features_df

    @classmethod
    def _add_player_history_columns(cls, df: pd.DataFrame, params: FeaturesParamsFantasyBasketballPredictions) -> pd.DataFrame:
        if df.empty:
            return df

        sort_cols = [c for c in ['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID'] if c in df.columns]
        if len(sort_cols) >= 2:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        # 1. Mean ratios (e.g. PTS_mean_ratio = PTS / PTS_mean5)
        for c in params.all_feature_names_list:
            if c.endswith('_mean_ratio') and c not in df.columns:
                base_name = c.replace('_mean_ratio', '')
                mean_name = base_name + '_mean5'
                if base_name in df.columns and mean_name in df.columns:
                    ratio = df[base_name] / df[mean_name]
                    ratio = ratio.clip(upper=10.0)
                    ratio.loc[(df[mean_name] == 0) & (df[base_name] == 0)] = 1.0
                    ratio.loc[(df[mean_name] == 0) & (df[base_name] != 0)] = 10.0
                    df[c] = ratio.fillna(1.0).astype(np.float32)

        # 3. Lagged features (e.g. *_t-{lag})
        lag_cols = [c for c in params.all_feature_names_list if '_t-' in c and c not in df.columns]
        if lag_cols and 'PLAYER_ID' in df.columns:
            lags = [int(c.split('_t-')[-1]) for c in lag_cols]
            lag_stats = [c.split('_t-')[0] for c in lag_cols]
            all_lagged = []
            prev_lag_df = None
            for lag in sorted(set(lags)):
                inds = [ii for ii, v in enumerate(lags) if v == lag]
                stats = list(dict.fromkeys([lag_stats[ii] for ii in inds if lag_stats[ii] in df.columns]))
                if not stats:
                    continue
                stats_df = df[stats].copy()
                for s in stats:
                    if pd.api.types.is_bool_dtype(stats_df[s]):
                        stats_df[s] = stats_df[s].astype(np.float32)

                lagged = stats_df.groupby(df['PLAYER_ID']).shift(lag)
                if lag == 1 and 'STARTER' in df.columns:
                    lagged_starter = stats_df.groupby([df['PLAYER_ID'], df['STARTER']]).shift(1)
                    lagged = lagged_starter.fillna(lagged)

                if prev_lag_df is not None:
                    avail = [s for s in stats if s in prev_lag_df.columns]
                    lagged[avail] = lagged[avail].fillna(prev_lag_df[avail])

                role_cols = [c for c in ['START_POSITION_CALCULATED', 'STARTER'] if c in df.columns]
                if role_cols:
                    role_means = stats_df.groupby([df[c] for c in role_cols]).transform('mean')
                    lagged = lagged.fillna(role_means)
                lagged = lagged.fillna(0.0)
                prev_lag_df = lagged.copy()

                renamed = lagged.rename(columns={s: f'{s}_t-{lag}' for s in stats})
                all_lagged.append(renamed)

            if all_lagged:
                df = pd.concat([df] + all_lagged, axis=1)

        return df

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

    def create_date_filtered_data_loader(
        self,
        target_dates: Union[datetime.date, str, Sequence[Any]],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a DataLoader containing only samples for the specified target date(s)."""
        if self._original_index_to_date is None:
            raise ValueError("Cannot create date-filtered DataLoader because _original_index_to_date is None.")

        target_days = np.round(dates_to_epoch_days(target_dates)).astype(np.int64)
        dataset_days = np.round(self._original_index_to_date).astype(np.int64)
        matching_indices = np.where(np.isin(dataset_days, target_days))[0].tolist()

        filtered_indices = self._remove_unused_indices_by_keep_mask(matching_indices)
        subset = Subset(self.machine_learning_datasets_stacked, indices=filtered_indices)

        effective_batch_size = batch_size or getattr(self.training_params, 'batch_size', 1)
        data_loader_batch_kwargs = self._get_data_loader_batch_kwargs(
            batch_size=effective_batch_size,
            shuffle=False,
            machine_learning_datasets_stacked_subset=subset,
        )
        return DataLoader(
            subset,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **data_loader_batch_kwargs,
        )
