import os
import time
import logging
import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
import lightning.pytorch as pl
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Type, Tuple
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from data_collectors.stats_loaders import StatsLoaderPlayers
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions, sharpe, DATA_DIR, convert_date_col_to_date
from machine_learning.constants import IGNORE
from machine_learning.enums import DataSplitType
from machine_learning.run_model import RunModel
from machine_learning.architecture import ModelArchitecture
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning.lightning_data_module import MachineLearningDataModule
from machine_learning.lightning_model import MachineLearningLightningModule
from machine_learning.utils import models_dir, get_next_model_id
from machine_learning_fantasy_basketball_predictions.constants import PROJECT_NAME
from machine_learning_fantasy_basketball_predictions.params.params import \
    MachineLearningParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import \
    MachineLearningDatasetCombinedFantasyBasketballPredictions
from machine_learning.callbacks import CallbacksManager
from machine_learning_fantasy_basketball_predictions.callbacks import CallbacksManagerFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.lightning_data_module import (
    FantasyBasketballDataModule,
)
from machine_learning_fantasy_basketball_predictions.lightning_model import (
    FantasyBasketballLightningModule,
)
from machine_learning_fantasy_basketball_predictions.params.layers_params import (
    update_calculate_fantasy_points_params,
)


class RunModelFantasyBasketballPredictions(RunModel):
    def __init__(self, params: MachineLearningParamsFantasyBasketballPredictions, project_name: str = PROJECT_NAME,
                 description: Optional[str] = None, use_lightning: bool = False):
        super().__init__(params=params, project_name=project_name, description=description, use_lightning=use_lightning)

    def update_categorical_params(self):
        super().update_categorical_params()
        update_calculate_fantasy_points_params(
            params=self.params,
            dataset_combined=getattr(self, 'dataset_combined', None),
            data_loaders=getattr(self, 'data_loaders', None),
        )

    def _data_class(self):
        if self.use_lightning:
            return FantasyBasketballDataModule
        return MachineLearningDatasetCombinedFantasyBasketballPredictions

    def _model_class(self):
        if self.use_lightning:
            return FantasyBasketballLightningModule
        return ModelArchitecture

    @staticmethod
    def _machine_learning_datasets_combined_class() -> Type[MachineLearningDatasetCombined]:
        return MachineLearningDatasetCombinedFantasyBasketballPredictions

    @staticmethod
    def _callbacks_combined_class() -> Type[CallbacksManager]:
        return CallbacksManagerFantasyBasketballPredictions

    @staticmethod
    def _load_contests_df() -> pd.DataFrame:
        contests_path = os.path.join(DATA_DIR, "dfs_contests", "dfs_contest_results2.feather")
        if not os.path.isfile(contests_path):
            raise FileNotFoundError(
                f"DFS contest results file not found at: '{contests_path}'. "
                f"Please ensure 'dfs_contest_results2.feather' exists in '{os.path.join(DATA_DIR, 'dfs_contests')}'."
            )
        contests_df = pd.read_feather(contests_path)
        contests_df = contests_df.loc[contests_df['score'] > 150]
        contests_df.rename(columns={'date': 'DATE'}, inplace=True)
        contests_df['DATE'] = pd.to_datetime(contests_df['DATE']).dt.date
        date_per_contest = contests_df.groupby('contestId')['DATE'].max()
        contests_df = \
            contests_df.drop(columns='DATE').merge(date_per_contest.reset_index(), on='contestId', how='left')
        return contests_df[['contestId', 'score', 'DATE', 'entry_fee']]

    @classmethod
    def _process_predictions_df(cls, predictions_df: pd.DataFrame, contests_df: pd.DataFrame,
                                model_dir: Optional[str] = None) -> pd.DataFrame:
        del model_dir
        collect_cols = [c for c in predictions_df.columns if c.startswith('CollectTruePredicted')]
        if collect_cols:
            predictions_df = predictions_df[collect_cols].copy()
            predictions_df.rename(columns={c: '_'.join(c.split('_')[1:]) for c in predictions_df.columns}, inplace=True)
        else:
            predictions_df = predictions_df.copy()

        if 'NOT_IGNORE' in predictions_df.columns:
            predictions_df = predictions_df.loc[predictions_df['NOT_IGNORE'].astype(bool)].drop(columns='NOT_IGNORE')

        true_cols = [c for c in predictions_df if '_true' in c]
        if true_cols:
            predictions_df = predictions_df.loc[(predictions_df[true_cols] != IGNORE).all(axis=1)]

        predictions_df['DATE'] = convert_date_col_to_date(predictions_df['DATE'])
        predictions_df = predictions_df.loc[predictions_df['DATE'].isin(contests_df['DATE'].unique())]

        # adding yahoo points if needed
        add_missing_points_columns_to_predictions(predictions_df=predictions_df)

        needs_raw = 'YAHOO_POINTS_raw' not in predictions_df.columns
        missing_cols = [c for c in ('YAHOO_SALARY', 'YAHOO_POSITION') if c not in predictions_df.columns]
        if needs_raw or len(missing_cols) > 0:
            player_stats = StatsLoaderPlayers().load_saved_data()
            cols_to_pull = ['PLAYER_ID', 'DATE']
            if needs_raw and 'YAHOO_POINTS' in player_stats.columns:
                cols_to_pull.append('YAHOO_POINTS')
            cols_to_pull += missing_cols

            raw_df = player_stats[cols_to_pull].copy()
            if 'YAHOO_POINTS' in raw_df.columns:
                raw_df.rename(columns={'YAHOO_POINTS': 'YAHOO_POINTS_raw'}, inplace=True)
            raw_df['DATE'] = convert_date_col_to_date(raw_df['DATE'])
            raw_df['PLAYER_ID'] = raw_df['PLAYER_ID'].astype(int)
            raw_df.drop_duplicates(['PLAYER_ID', 'DATE'], inplace=True)

            predictions_df['PLAYER_ID'] = predictions_df['PLAYER_ID'].astype(int)
            predictions_df = predictions_df.merge(raw_df, how='inner', on=['PLAYER_ID', 'DATE'])

        if 'YAHOO_POINTS_raw' not in predictions_df.columns:
            raise ValueError("YAHOO_POINTS_raw is missing from predictions_df")

        predictions_df = predictions_df.loc[(predictions_df['YAHOO_SALARY'] > 0) &
                                            (predictions_df['YAHOO_SALARY'].notnull())]

        predictions_df['PLAYER_ID'] = predictions_df['PLAYER_ID'].astype(int)

        return predictions_df

    def train(self) -> None:
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        return super().train(
            trainer_cls=pl.Trainer,
            torch_save=torch.save,
            precision=precision,
        )

    def predict(self, data_split_type: DataSplitType = DataSplitType.TRAIN, **kwargs) -> pd.DataFrame:
        return super().predict(data_split_type=data_split_type, trainer_cls=pl.Trainer, **kwargs)

    @classmethod
    def _calculate_prediction_correlations(cls, eval_results: pd.DataFrame, predict_results: pd.DataFrame,
                                           model_dir: Optional[str] = None) -> pd.DataFrame:
        del model_dir
        if predict_results is None or predict_results.empty:
            return eval_results

        preds = predict_results.copy()
        collect_cols = [c for c in preds.columns if c.startswith('CollectTruePredicted')]
        if collect_cols:
            preds = preds[collect_cols].copy()
            preds.rename(columns={c: '_'.join(c.split('_')[1:]) for c in preds.columns}, inplace=True)
        try:
            add_missing_points_columns_to_predictions(preds)
        except Exception:
            pass

        true_col = 'YAHOO_POINTS_true' if 'YAHOO_POINTS_true' in preds.columns else 'YAHOO_POINTS'
        if 'YAHOO_POINTS_pred' not in preds.columns or true_col not in preds.columns:
            return eval_results

        if 'NOT_IGNORE' in preds.columns:
            preds = preds.loc[preds['NOT_IGNORE'].astype(bool)]

        split_series = preds['split_type'].astype(str) if 'split_type' in preds.columns else pd.Series([], dtype=str)
        eval_split_series = eval_results['split_type'].astype(str) if 'split_type' in eval_results.columns else None

        for split_type in DataSplitType.all_not_none_split_types():
            split_name = split_type.name
            split_val = str(split_type.value)
            valid_splits = {split_name, split_val, split_name.lower(), split_name.upper()}
            split_mask = split_series.isin(valid_splits)
            split_df = preds[split_mask]
            valid = split_df[split_df[true_col].notna() & split_df['YAHOO_POINTS_pred'].notna()]
            if len(valid) < 5:
                continue

            p_corr, _ = pearsonr(valid['YAHOO_POINTS_pred'], valid[true_col])
            s_corr, _ = spearmanr(valid['YAHOO_POINTS_pred'], valid[true_col])

            daily_spearmans = []
            if 'DATE' in valid.columns:
                for _, day_df in valid.groupby('DATE'):
                    if len(day_df) >= 10 and day_df[true_col].nunique() > 1 and day_df['YAHOO_POINTS_pred'].nunique() > 1:
                        d_corr, _ = spearmanr(day_df['YAHOO_POINTS_pred'], day_df[true_col])
                        if not np.isnan(d_corr):
                            daily_spearmans.append(d_corr)

            daily_spearman_mean = float(np.mean(daily_spearmans)) if daily_spearmans else float('nan')

            if eval_split_series is not None and not eval_results.empty:
                target_mask = eval_split_series.isin(valid_splits)
                idx = eval_results[target_mask].index[0] if target_mask.any() else split_name
            else:
                idx = split_name

            eval_results.loc[idx, 'pearson_corr'] = p_corr
            eval_results.loc[idx, 'spearman_corr'] = s_corr
            eval_results.loc[idx, 'daily_spearman_corr'] = daily_spearman_mean

            logging.info(
                f"[{split_name}] Points Correlation: Pearson={p_corr:.4f}, "
                f"Spearman={s_corr:.4f}, Mean Daily Spearman={daily_spearman_mean:.4f} (over {len(daily_spearmans)} days)"
            )

        return eval_results

    def _add_extra_metrics_to_eval_results(self, eval_results: pd.DataFrame, predict_results: pd.DataFrame
                                           ) -> pd.DataFrame:
        eval_results = self._calculate_prediction_correlations(
            eval_results=eval_results, predict_results=predict_results, model_dir=self._model_dir)

        try:
            contests_df = self._load_contests_df()
        except FileNotFoundError as e:
            logging.warning(f"Could not load DFS contest results: {e}. Skipping extra DFS contest metrics.")
            return eval_results

        predict_results = self._process_predictions_df(
            predictions_df=predict_results, contests_df=contests_df, model_dir=self._model_dir)

        find_best_team = FindBestSetOfTeamsLinearProgrammingYahoo(points_col='YAHOO_POINTS_pred',
                                                                  num_teams_in_set=1,
                                                                  verbose=False)

        true_points_col = 'YAHOO_POINTS_raw'
        score_per_day = dict()
        for curr_date, day_df in tqdm(predict_results.groupby('DATE'), desc='finding best predicted team per day'):
            if day_df.shape[0] < 50:
                continue
            player_to_true_points = day_df.set_index('PLAYER_ID')[true_points_col].to_dict()
            predicted_best_team = find_best_team.run(df=day_df).iloc[0]
            chosen_players = predicted_best_team[find_best_team.positions()]

            has_true_points = np.array([v in player_to_true_points.keys() for v in chosen_players.values])
            if not all(has_true_points):
                players_without_true_points = chosen_players.values[~has_true_points]
                raise ValueError(f"There are players that we don't have true points for on date {curr_date}: "
                                 f"{players_without_true_points}")
            score_per_day[curr_date] = chosen_players.map(player_to_true_points).sum()
        score_per_day = pd.Series(score_per_day).to_frame().reset_index().rename(
            columns={'index': 'DATE', 0: 'predicted_score'})

        dfs_contests_df = contests_df.merge(score_per_day, on='DATE', how='inner')
        dfs_contests_df['predicted_is_better'] = dfs_contests_df['predicted_score'] > dfs_contests_df['score']
        percentile_per_contest = dfs_contests_df.groupby('contestId', as_index=False).agg(
            {'predicted_is_better': 'mean', 'predicted_score': 'first'})
        percentile_per_contest = percentile_per_contest.merge(dfs_contests_df[['contestId', 'DATE', 'entry_fee']].drop_duplicates(),
                                                              on='contestId', how='left')
        percentile_per_contest.rename(columns={'predicted_is_better': 'percentile'}, inplace=True)
        percentile_per_contest = percentile_per_contest.merge(
            predict_results[['DATE', 'split_type']].drop_duplicates(), on='DATE', how='left')
        percentile_per_contest['profit_5050'] = np.where(percentile_per_contest['percentile'] > 0.5, 0.8, -1)
        percentile_per_contest['profit_top10'] = np.where(percentile_per_contest['percentile'] > 0.9, 2, -1)

        for ppc, df_name in zip([percentile_per_contest, percentile_per_contest.query('entry_fee.notnull() & (entry_fee > 0)')], ['all', 'paid']):
            eval_results[f'portfolio_value_5050_{df_name}'] = [list()] * eval_results.shape[0]
            eval_results[f'portfolio_value_top10_{df_name}'] = [list()] * eval_results.shape[0]
            for raw_split, group_df in ppc.groupby('split_type', sort=False):
                split_key = raw_split.name if hasattr(raw_split, 'name') else str(raw_split)
                if split_key not in eval_results.index and split_key.upper() in eval_results.index:
                    split_key = split_key.upper()
                one_per_date = group_df.groupby('DATE').agg({'profit_5050': 'mean', 'profit_top10': 'mean',
                                                             'percentile': 'mean', 'predicted_score': 'mean'})
                one_per_date[f'portfolio_value_5050'] = one_per_date['profit_5050'].cumsum()
                one_per_date['portfolio_value_top10'] = one_per_date['profit_top10'].cumsum()
                eval_results.at[split_key, f'portfolio_value_5050_{df_name}'] = one_per_date['portfolio_value_5050'].values.tolist()
                eval_results.at[split_key, f'portfolio_value_top10_{df_name}'] = one_per_date['portfolio_value_top10'].values.tolist()
                eval_results.loc[split_key, f'mean_percentile_{df_name}'] = one_per_date['percentile'].mean()
                eval_results.loc[split_key, f'mean_score_{df_name}'] = one_per_date['predicted_score'].mean()
                eval_results.loc[split_key, f'top10_pct_{df_name}'] = (one_per_date['percentile'] >= 0.9).mean()
                eval_results.loc[split_key, f'sharpe_5050_{df_name}'] = sharpe(one_per_date['profit_5050'])
                eval_results.loc[split_key, f'sharpe_top10_{df_name}'] = sharpe(one_per_date['profit_top10'])

        return eval_results

    def run_full_training_session(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        eval_results, predict_results = super().run_full_training_session()

        # saving dfs results to tensorboard
        try:
            for split_type in DataSplitType.all_not_none_split_types():
                split_key = split_type.name if split_type.name in eval_results.index else str(split_type.value)
                if split_key not in eval_results.index:
                    continue
                writer = SummaryWriter(os.path.join(self._tensorboard_dir(), split_type.name.lower()))

                for suffix in ['all', 'paid']:
                    col_5050 = f'portfolio_value_5050_{suffix}'
                    col_top10 = f'portfolio_value_top10_{suffix}'
                    if col_5050 in eval_results.columns and col_top10 in eval_results.columns:
                        portfolio_value_5050 = eval_results.loc[split_key, col_5050]
                        portfolio_value_top10 = eval_results.loc[split_key, col_top10]
                        if hasattr(portfolio_value_5050, '__iter__') and hasattr(portfolio_value_top10, '__iter__'):
                            for step, (curr_5050, curr_top10) in enumerate(zip(portfolio_value_5050, portfolio_value_top10)):
                                writer.add_scalar(tag=f'DFS_Profit/portfolio_5050_{suffix}',
                                                  scalar_value=float(curr_5050), global_step=step)
                                writer.add_scalar(tag=f'DFS_Profit/portfolio_top10_{suffix}',
                                                  scalar_value=float(curr_top10), global_step=step)

                for m in ['mean_percentile', 'mean_score', 'top10_pct', 'sharpe_5050', 'sharpe_top10']:
                    for c in [cc for cc in eval_results.columns if cc.startswith(m)]:
                        val = eval_results.loc[split_type.name, c]
                        if pd.notna(val):
                            writer.add_scalar(tag=f'DFS_Contests/{c}',
                                              scalar_value=float(val),
                                              global_step=0)
                writer.flush()
                writer.close()
        except Exception as e:
            logging.warning(f"Could not save DFS results to tensorboard: {e}")

        return eval_results, predict_results
