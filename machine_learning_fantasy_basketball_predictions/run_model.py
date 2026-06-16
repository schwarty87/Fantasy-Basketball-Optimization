import os
import time
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Type, Tuple
from data_collectors.stats_loaders import StatsLoaderPlayers
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions, sharpe, DATA_DIR
from machine_learning.constants import IGNORE
from machine_learning.enums import DataSplitType
from machine_learning.run_model import RunModel
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning_fantasy_basketball_predictions.constants import PROJECT_NAME
from machine_learning_fantasy_basketball_predictions.params.params import \
    MachineLearningParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import \
    MachineLearningDatasetCombinedFantasyBasketballPredictions
from machine_learning.callbacks import CallbacksManager
from machine_learning_fantasy_basketball_predictions.callbacks import CallbacksManagerFantasyBasketballPredictions


class RunModelFantasyBasketballPredictions(RunModel):
    def __init__(self, params: MachineLearningParamsFantasyBasketballPredictions, project_name: str = PROJECT_NAME,
                 description: Optional[str] = None):
        super().__init__(params=params, project_name=project_name, description=description)

    @staticmethod
    def _callbacks_combined_class() -> Type[CallbacksManager]:
        return CallbacksManagerFantasyBasketballPredictions

    @staticmethod
    def _machine_learning_datasets_combined_class() -> Type[MachineLearningDatasetCombined]:
        return MachineLearningDatasetCombinedFantasyBasketballPredictions

    @staticmethod
    def _load_contests_df() -> pd.DataFrame:
        contests_df = pd.read_feather(os.path.join(DATA_DIR, "dfs_contests", "dfs_contest_results2.feather"))
        contests_df = contests_df.loc[contests_df['score'] > 150]
        contests_df.rename(columns={'date': 'DATE'}, inplace=True)
        contests_df['DATE'] = pd.to_datetime(contests_df['DATE']).dt.date
        date_per_contest = contests_df.groupby('contestId')['DATE'].max()
        contests_df = \
            contests_df.drop(columns='DATE').merge(date_per_contest.reset_index(), on='contestId', how='left')
        return contests_df[['contestId', 'score', 'DATE', 'entry_fee']]

    @staticmethod
    def _process_predictions_df(predictions_df: pd.DataFrame, contests_df: pd.DataFrame) -> pd.DataFrame:
        predictions_df = predictions_df[[c for c in predictions_df.columns if c.startswith('CollectTruePredicted')]]
        predictions_df.rename(columns={c: '_'.join(c.split('_')[1:]) for c in predictions_df.columns}, inplace=True)

        if 'NOT_IGNORE' in predictions_df.columns:
            predictions_df = predictions_df.loc[predictions_df['NOT_IGNORE'].astype(bool)].drop(columns='NOT_IGNORE')

        true_cols = [c for c in predictions_df if '_true' in c]
        predictions_df = predictions_df.loc[(predictions_df[true_cols] != IGNORE).all(axis=1)]
        not_date = [not isinstance(d, datetime.date) for d in predictions_df['DATE']]
        if any(not_date):
            predictions_df.loc[not_date, 'DATE'] = pd.to_datetime(predictions_df.loc[not_date, 'DATE'], unit='s').dt.date
        predictions_df = predictions_df.loc[predictions_df['DATE'].isin(contests_df['DATE'].unique())]

        # adding yahoo points if needed
        add_missing_points_columns_to_predictions(predictions_df=predictions_df)

        missing_cols = [c for c in ('YAHOO_SALARY', 'YAHOO_POSITION') if c not in predictions_df.columns]
        if len(missing_cols) > 0:
            player_stats = StatsLoaderPlayers().load_saved_data()
            predictions_df = predictions_df.merge(player_stats[['PLAYER_ID', 'DATE'] + missing_cols], how='inner',
                                                  on=['PLAYER_ID', 'DATE'])

        predictions_df = predictions_df.loc[(predictions_df['YAHOO_SALARY'] > 0) &
                                            (predictions_df['YAHOO_SALARY'].notnull())]

        predictions_df['PLAYER_ID'] = predictions_df['PLAYER_ID'].astype(int)
        
        return predictions_df

    def _add_extra_metrics_to_eval_results(self, eval_results: pd.DataFrame, predict_results: pd.DataFrame
                                           ) -> pd.DataFrame:
        contests_df = self._load_contests_df()
        predict_results = self._process_predictions_df(predictions_df=predict_results, contests_df=contests_df)

        find_best_team = FindBestSetOfTeamsLinearProgrammingYahoo(points_col='YAHOO_POINTS_pred',
                                                                  num_teams_in_set=1,
                                                                  verbose=False)

        score_per_day = dict()
        for curr_date, day_df in tqdm(predict_results.groupby('DATE'), desc='finding best predicted team per day'):
            if day_df.shape[0] < 50:
                continue
            player_to_true_points = day_df.set_index('PLAYER_ID')['YAHOO_POINTS_true'].to_dict()
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
            for split_type, group_df in ppc.groupby('split_type', sort=False):
                one_per_date = group_df.groupby('DATE').agg({'profit_5050': 'mean', 'profit_top10': 'mean',
                                                             'percentile': 'mean', 'predicted_score': 'mean'})
                one_per_date[f'portfolio_value_5050'] = one_per_date['profit_5050'].cumsum()
                one_per_date['portfolio_value_top10'] = one_per_date['profit_top10'].cumsum()
                eval_results.at[split_type, f'portfolio_value_5050_{df_name}'] = one_per_date['portfolio_value_5050'].values.tolist()
                eval_results.at[split_type, f'portfolio_value_top10_{df_name}'] = one_per_date['portfolio_value_top10'].values.tolist()
                eval_results.loc[split_type, f'mean_percentile_{df_name}'] = one_per_date['percentile'].mean()
                eval_results.loc[split_type, f'mean_score_{df_name}'] = one_per_date['predicted_score'].mean()
                eval_results.loc[split_type, f'top10_pct_{df_name}'] = (one_per_date['percentile'] >= 0.9).mean()
                eval_results.loc[split_type, f'sharpe_5050_{df_name}'] = sharpe(one_per_date['portfolio_value_5050'])

        return eval_results

    def run_full_training_session(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        eval_results, predict_results = super().run_full_training_session()

        # saving dfs results to tensorboard
        summary_writer = SummaryWriter(os.path.join(self._tensorboard_dir(), DataSplitType.TRAIN.name.lower()))
        for global_step, split_type in enumerate(DataSplitType.all_not_none_split_types()):
            for m in ['mean_percentile', 'mean_score', 'top10_pct', 'sharpe_5050']:
                for c in [cc for cc in eval_results.columns if cc.startswith(m)]:
                    summary_writer.add_scalar(tag=f'DFS_Results/{c}',
                                              scalar_value=eval_results.loc[split_type.name, c],
                                              global_step=global_step)
                    time.sleep(0.01)

            summary_writer2 = SummaryWriter(os.path.join(self._tensorboard_dir(), split_type.name.lower()))
            for suffix in ['all', 'paid']:
                portfolio_value_5050 = eval_results.loc[split_type.name, f'portfolio_value_5050_{suffix}']
                portfolio_value_top10 = eval_results.loc[split_type.name, f'portfolio_value_top10_{suffix}']
                for global_step2, (curr_5050, curr_top10) in enumerate(zip(portfolio_value_5050, portfolio_value_top10)):
                    summary_writer2.add_scalar(tag=f'DFS_Results/portfolio_value_5050_{split_type.name.lower()}_{suffix}',
                                               scalar_value=curr_5050, global_step=global_step2)
                    summary_writer2.add_scalar(tag=f'DFS_Results/portfolio_value_top10_{split_type.name.lower()}_{suffix}',
                                               scalar_value=curr_top10, global_step=global_step2)

        return eval_results, predict_results