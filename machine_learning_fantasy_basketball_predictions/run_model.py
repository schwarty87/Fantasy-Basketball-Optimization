import os
import logging
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
import lightning.pytorch as pl
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from typing import Type, Tuple, Any, Dict

from data_collectors.stats_loaders import StatsLoaderPlayers
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions, sharpe, DATA_DIR, convert_date_col_to_date
from machine_learning.constants import IGNORE
from machine_learning.enums import DataSplitType
from machine_learning.run_model import RunModel
from machine_learning.architecture import ModelArchitecture
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
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


def resolve_quantile_alphas(
    training_params: Any,
    default_cash: float = 0.35,
    default_gpp: float = 0.85,
) -> Tuple[float, float]:
    """Safely extract numeric ``(alpha_cash, alpha_gpp)`` from ``training_params`` (handles Mock objects)."""
    raw_cash = getattr(training_params, 'alpha_cash', default_cash)
    raw_gpp = getattr(training_params, 'alpha_gpp', default_gpp)
    alpha_cash = (
        float(raw_cash)
        if isinstance(raw_cash, (int, float, np.integer, np.floating)) and not isinstance(raw_cash, bool)
        else default_cash
    )
    alpha_gpp = (
        float(raw_gpp)
        if isinstance(raw_gpp, (int, float, np.integer, np.floating)) and not isinstance(raw_gpp, bool)
        else default_gpp
    )
    return alpha_cash, alpha_gpp


def add_blended_quantile_point_columns(
    predictions_df: pd.DataFrame,
    alpha_cash: float = 0.35,
    alpha_gpp: float = 0.85,
) -> Tuple[str, str]:
    """Add ``YAHOO_POINTS_cash_pred`` / ``YAHOO_POINTS_gpp_pred`` when quantile predictions exist.

    Returns ``(cash_points_col, gpp_points_col)``, falling back to ``'YAHOO_POINTS_pred'``
    for any contest type whose corresponding quantile column is absent.
    """
    for name, val in (('alpha_cash', alpha_cash), ('alpha_gpp', alpha_gpp)):
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")

    cash_col = 'YAHOO_POINTS_pred'
    gpp_col = 'YAHOO_POINTS_pred'
    if 'YAHOO_POINTS_pred' not in predictions_df.columns:
        return cash_col, gpp_col

    fp_pred = predictions_df['YAHOO_POINTS_pred'].astype(float)
    if 'YAHOO_POINTS_q20_pred' in predictions_df.columns:
        q20_pred = predictions_df['YAHOO_POINTS_q20_pred'].astype(float)
        predictions_df['YAHOO_POINTS_cash_pred'] = np.where(
            q20_pred.notna(),
            (1.0 - float(alpha_cash)) * fp_pred + float(alpha_cash) * q20_pred,
            fp_pred,
        )
        cash_col = 'YAHOO_POINTS_cash_pred'
    if 'YAHOO_POINTS_q80_pred' in predictions_df.columns:
        q80_pred = predictions_df['YAHOO_POINTS_q80_pred'].astype(float)
        predictions_df['YAHOO_POINTS_gpp_pred'] = np.where(
            q80_pred.notna(),
            (1.0 - float(alpha_gpp)) * fp_pred + float(alpha_gpp) * q80_pred,
            fp_pred,
        )
        gpp_col = 'YAHOO_POINTS_gpp_pred'
    return cash_col, gpp_col


class RunModelFantasyBasketballPredictions(RunModel):
    def __init__(self, params: MachineLearningParamsFantasyBasketballPredictions, project_name: str = PROJECT_NAME,
                 description: str | None = None, use_lightning: bool | None = None):
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

    @staticmethod
    def _normalize_predictions_df(predictions_df: pd.DataFrame) -> pd.DataFrame:
        collect_cols = [c for c in predictions_df.columns if c.startswith('CollectTruePredicted')]
        if collect_cols:
            normalized = predictions_df[collect_cols].copy()
            normalized.rename(columns={c: '_'.join(c.split('_')[1:]) for c in normalized.columns}, inplace=True)
        else:
            normalized = predictions_df.copy()
        try:
            add_missing_points_columns_to_predictions(predictions_df=normalized)
        except Exception:
            pass
        return normalized

    @staticmethod
    def _resolve_eval_split_key(eval_results: pd.DataFrame, raw_split: Any, eval_split_series: pd.Series | None = None) -> Any:
        split_name = raw_split.name if hasattr(raw_split, 'name') else str(raw_split)
        candidates = (split_name, split_name.lower(), split_name.upper())
        if eval_split_series is not None and not eval_results.empty:
            target_mask = eval_split_series.isin(candidates)
            if target_mask.any():
                return eval_results[target_mask].index[0]
        for cand in candidates:
            if cand in eval_results.index:
                return cand
        return split_name

    @classmethod
    def _process_predictions_df(
        cls,
        predictions_df: pd.DataFrame,
        contests_df: pd.DataFrame | None = None,
        raw_player_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        predictions_df = cls._normalize_predictions_df(predictions_df)

        if 'NOT_IGNORE' in predictions_df.columns:
            predictions_df = predictions_df.drop(columns='NOT_IGNORE')

        true_cols = [c for c in predictions_df if '_true' in c]
        if true_cols:
            predictions_df = predictions_df.loc[(predictions_df[true_cols] != IGNORE).all(axis=1)]

        predictions_df['DATE'] = convert_date_col_to_date(predictions_df['DATE'])
        if contests_df is not None:
            predictions_df = predictions_df.loc[predictions_df['DATE'].isin(contests_df['DATE'].unique())]

        # adding yahoo points if needed
        add_missing_points_columns_to_predictions(predictions_df=predictions_df)

        needs_raw = 'YAHOO_POINTS_raw' not in predictions_df.columns
        missing_cols = [c for c in ('YAHOO_SALARY', 'YAHOO_POSITION') if c not in predictions_df.columns]
        if needs_raw or len(missing_cols) > 0:
            player_stats = raw_player_df if raw_player_df is not None else StatsLoaderPlayers().load_saved_data()
            cols_to_pull = ['PLAYER_ID', 'DATE']
            if needs_raw:
                if 'YAHOO_POINTS' not in player_stats.columns:
                    raise ValueError("YAHOO_POINTS column missing from player stats!")
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
    def _calculate_prediction_correlations(cls, eval_results: pd.DataFrame, predict_results: pd.DataFrame) -> pd.DataFrame:
        if predict_results is None or predict_results.empty:
            return eval_results

        preds = cls._normalize_predictions_df(predict_results)

        true_col = 'YAHOO_POINTS_true' if 'YAHOO_POINTS_true' in preds.columns else 'YAHOO_POINTS'
        if 'YAHOO_POINTS_pred' not in preds.columns or true_col not in preds.columns:
            return eval_results

        if 'NOT_IGNORE' in preds.columns:
            preds = preds.loc[preds['NOT_IGNORE'] > 0]

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
            idx = cls._resolve_eval_split_key(eval_results, split_type, eval_split_series)

            eval_results.loc[idx, 'pearson_corr'] = p_corr
            eval_results.loc[idx, 'spearman_corr'] = s_corr
            eval_results.loc[idx, 'daily_spearman_corr'] = daily_spearman_mean

            logging.info(
                f"[{split_name}] Points Correlation: Pearson={p_corr:.4f}, "
                f"Spearman={s_corr:.4f}, Mean Daily Spearman={daily_spearman_mean:.4f} (over {len(daily_spearmans)} days)"
            )

        return eval_results

    @classmethod
    def _calculate_prediction_accuracy_metrics(
        cls,
        eval_results: pd.DataFrame,
        predict_results: pd.DataFrame,
        player_stats_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute physical accuracy, salary-tier biases, and dropped-player coverage per split."""
        if predict_results is None or predict_results.empty:
            return eval_results

        preds = cls._normalize_predictions_df(predict_results)
        if 'YAHOO_POINTS_pred' not in preds.columns or 'split_type' not in preds.columns:
            return eval_results

        if 'DATE' in preds.columns:
            preds['DATE'] = convert_date_col_to_date(preds['DATE'])

        real_mask = pd.Series(True, index=preds.index)
        if 'PLAYER_ID' in preds.columns:
            real_mask &= preds['PLAYER_ID'].fillna(0).astype(int) != 0
        if 'MIN_true' in preds.columns:
            real_mask &= preds['MIN_true'] != IGNORE
        preds_real = preds.loc[real_mask].copy()

        has_player_stats_raw = (
            player_stats_df is not None
            and {'PLAYER_ID', 'DATE', 'YAHOO_POINTS'}.issubset(player_stats_df.columns)
        )
        if 'YAHOO_POINTS_raw' not in preds_real.columns and has_player_stats_raw and {'PLAYER_ID', 'DATE'}.issubset(preds_real.columns):
            raw_pts = player_stats_df[['PLAYER_ID', 'DATE', 'YAHOO_POINTS']].copy()
            raw_pts['DATE'] = convert_date_col_to_date(raw_pts['DATE'])
            raw_pts['PLAYER_ID'] = raw_pts['PLAYER_ID'].astype(int)
            raw_pts = raw_pts.drop_duplicates(['PLAYER_ID', 'DATE']).rename(columns={'YAHOO_POINTS': 'YAHOO_POINTS_raw'})
            preds_real['PLAYER_ID'] = preds_real['PLAYER_ID'].astype(int)
            preds_real = preds_real.merge(raw_pts, on=['PLAYER_ID', 'DATE'], how='left')

        true_fp_col = next(
            (c for c in ('YAHOO_POINTS_raw', 'YAHOO_POINTS_true', 'YAHOO_POINTS') if c in preds_real.columns),
            None,
        )
        if true_fp_col is None:
            return eval_results

        ps_active = None
        if player_stats_df is not None and {'PLAYER_ID', 'DATE', 'YAHOO_SALARY'}.issubset(player_stats_df.columns):
            ps_mask = player_stats_df['YAHOO_SALARY'].notna() & (player_stats_df['YAHOO_SALARY'] >= 15)
            if 'IS_OUT' in player_stats_df.columns:
                ps_mask &= ~player_stats_df['IS_OUT'].fillna(False).astype(bool)
            if 'MIN' in player_stats_df.columns:
                ps_mask &= player_stats_df['MIN'].fillna(0) > 0
            ps_active = player_stats_df.loc[ps_mask, ['PLAYER_ID', 'DATE']].copy()
            ps_active['DATE'] = convert_date_col_to_date(ps_active['DATE'])
            ps_active['PLAYER_ID'] = ps_active['PLAYER_ID'].astype(int)

        split_series = preds_real['split_type'].astype(str)
        eval_split_series = eval_results['split_type'].astype(str) if 'split_type' in eval_results.columns else None

        for split_type in DataSplitType.all_not_none_split_types():
            valid_splits = {split_type.name, str(split_type.value), split_type.name.lower(), split_type.name.upper()}
            split_real = preds_real.loc[split_series.isin(valid_splits)]
            if split_real.empty:
                continue

            idx = cls._resolve_eval_split_key(eval_results, split_type, eval_split_series)

            kept = split_real
            if 'NOT_IGNORE' in kept.columns:
                kept = kept.loc[kept['NOT_IGNORE'] > 0]
            valid = kept.loc[kept['YAHOO_POINTS_pred'].notna() & kept[true_fp_col].notna()]
            if not valid.empty:
                err = valid['YAHOO_POINTS_pred'].astype(float) - valid[true_fp_col].astype(float)
                eval_results.loc[idx, 'fp_rmse_raw'] = float(np.sqrt(np.mean(err ** 2)))
                eval_results.loc[idx, 'fp_bias_raw'] = float(np.mean(err))

                if 'YAHOO_SALARY' in valid.columns:
                    sal = valid['YAHOO_SALARY'].astype(float)
                    tiers = {
                        'fp_bias_sal_le_15': (sal > 0) & (sal <= 15),
                        'fp_bias_sal_16_30': (sal > 15) & (sal <= 30),
                        'fp_bias_sal_gt_30': sal > 30,
                    }
                    for col_name, tier_mask in tiers.items():
                        if tier_mask.any():
                            eval_results.loc[idx, col_name] = float(err.loc[tier_mask].mean())

                    if {'MIN_pred', 'MIN_true'}.issubset(valid.columns):
                        sal_min_mask = (
                            (sal > 0)
                            & valid['MIN_pred'].notna()
                            & valid['MIN_true'].notna()
                            & (valid['MIN_true'] != IGNORE)
                        )
                        if sal_min_mask.any():
                            min_err = (
                                valid.loc[sal_min_mask, 'MIN_pred'].astype(float)
                                - valid.loc[sal_min_mask, 'MIN_true'].astype(float)
                            ).abs()
                            eval_results.loc[idx, 'min_mae_sal'] = float(min_err.mean())

            if ps_active is not None and {'PLAYER_ID', 'DATE'}.issubset(split_real.columns):
                split_dates = set(split_real['DATE'].dropna().unique())
                ps_split = ps_active.loc[ps_active['DATE'].isin(split_dates)]
                ps_keys = set(zip(ps_split['PLAYER_ID'], ps_split['DATE']))
                pred_keys = set(zip(split_real['PLAYER_ID'].astype(int), split_real['DATE']))
                eval_results.loc[idx, 'n_salaried_players'] = int(len(ps_keys))
                eval_results.loc[idx, 'n_dropped_salaried_players'] = int(len(ps_keys - pred_keys))

            # Evaluate quantile calibration against per-48 YAHOO_POINTS_true on split_real (retaining midgame exits).
            q_true_col = next(
                (c for c in ('YAHOO_POINTS_true', 'YAHOO_POINTS', 'YAHOO_POINTS_raw') if c in split_real.columns),
                None,
            )
            if q_true_col is not None and {'YAHOO_POINTS_q20_pred', 'YAHOO_POINTS_q80_pred'}.issubset(split_real.columns):
                valid_q = split_real.loc[
                    split_real['YAHOO_POINTS_q20_pred'].notna()
                    & split_real['YAHOO_POINTS_q80_pred'].notna()
                    & split_real[q_true_col].notna()
                    & (split_real[q_true_col].astype(float) != IGNORE)
                ]
                if not valid_q.empty:
                    y_q = valid_q[q_true_col].astype(float)
                    q20 = valid_q['YAHOO_POINTS_q20_pred'].astype(float)
                    q80 = valid_q['YAHOO_POINTS_q80_pred'].astype(float)
                    eval_results.loc[idx, 'q20_coverage'] = float(np.mean(y_q <= q20))
                    eval_results.loc[idx, 'q80_coverage'] = float(np.mean(y_q <= q80))
                    eval_results.loc[idx, 'quantile_crossing_rate'] = float(np.mean(q20 > q80))
                    if 'YAHOO_SALARY' in valid_q.columns:
                        sal_q = valid_q['YAHOO_SALARY'].astype(float)
                        sal_le_12 = (sal_q > 0) & (sal_q <= 12)
                        if sal_le_12.any():
                            eval_results.loc[idx, 'q20_coverage_sal_le_12'] = float(
                                np.mean(y_q.loc[sal_le_12] <= q20.loc[sal_le_12])
                            )
                            eval_results.loc[idx, 'q80_coverage_sal_le_12'] = float(
                                np.mean(y_q.loc[sal_le_12] <= q80.loc[sal_le_12])
                            )

        return eval_results

    @staticmethod
    def _solve_best_lineups_per_day(
        processed_predictions: pd.DataFrame,
        points_col: str,
        true_points_col: str = 'YAHOO_POINTS_raw',
    ) -> pd.DataFrame:
        find_best_team = FindBestSetOfTeamsLinearProgrammingYahoo(
            points_col=points_col,
            num_teams_in_set=1,
            verbose=False,
        )
        score_per_day = dict()
        for curr_date, day_df in tqdm(
            processed_predictions.groupby('DATE'),
            desc=f'finding best predicted team per day ({points_col})',
        ):
            if day_df.shape[0] < 50:
                continue
            player_to_true_points = day_df.set_index('PLAYER_ID')[true_points_col].to_dict()
            try:
                predicted_best_team = find_best_team.run(df=day_df).iloc[0]
            except ValueError:
                continue
            chosen_players = predicted_best_team[find_best_team.positions()]

            has_true_points = np.array([v in player_to_true_points.keys() for v in chosen_players.values])
            if not all(has_true_points):
                players_without_true_points = chosen_players.values[~has_true_points]
                raise ValueError(
                    f"There are players that we don't have true points for on date {curr_date}: "
                    f"{players_without_true_points}"
                )
            score_per_day[curr_date] = chosen_players.map(player_to_true_points).sum()
        return (
            pd.Series(score_per_day, dtype=float)
            .to_frame()
            .reset_index()
            .rename(columns={'index': 'DATE', 0: 'predicted_score'})
        )

    @classmethod
    def evaluate_predictions_df(
        cls,
        predict_results: pd.DataFrame,
        eval_results: pd.DataFrame | None = None,
        contests_df: pd.DataFrame | None = None,
        player_stats_df: pd.DataFrame | None = None,
        return_details: bool = False,
        alpha_cash: float = 0.35,
        alpha_gpp: float = 0.85,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        if eval_results is None:
            eval_results = pd.DataFrame(index=[s.name for s in DataSplitType.all_not_none_split_types()])
        else:
            eval_results = eval_results.copy()

        if contests_df is None:
            contests_df = cls._load_contests_df()

        normalized_preds = cls._normalize_predictions_df(predict_results)
        raw_df = player_stats_df
        if (
            raw_df is None
            and 'YAHOO_POINTS_raw' not in normalized_preds.columns
            and {'PLAYER_ID', 'DATE'}.issubset(normalized_preds.columns)
        ):
            raw_df = StatsLoaderPlayers().load_saved_data()

        eval_results = cls._calculate_prediction_correlations(
            eval_results=eval_results,
            predict_results=normalized_preds,
        )
        eval_results = cls._calculate_prediction_accuracy_metrics(
            eval_results=eval_results,
            predict_results=normalized_preds,
            player_stats_df=raw_df,
        )

        processed_predictions = cls._process_predictions_df(
            predictions_df=predict_results,
            raw_player_df=raw_df,
        )

        cash_col, gpp_col = add_blended_quantile_point_columns(
            processed_predictions, alpha_cash=alpha_cash, alpha_gpp=alpha_gpp
        )
        lineups_by_col: Dict[str, pd.DataFrame] = {
            'YAHOO_POINTS_pred': cls._solve_best_lineups_per_day(processed_predictions, 'YAHOO_POINTS_pred'),
        }

        def _get_or_solve(col: str) -> pd.DataFrame:
            if col in lineups_by_col:
                return lineups_by_col[col]
            base_vals = processed_predictions['YAHOO_POINTS_pred'].to_numpy(dtype=np.float64, na_value=np.nan)
            col_vals = processed_predictions[col].to_numpy(dtype=np.float64, na_value=np.nan)
            if np.array_equal(base_vals, col_vals, equal_nan=True):
                lineups_by_col[col] = lineups_by_col['YAHOO_POINTS_pred']
            else:
                lineups_by_col[col] = cls._solve_best_lineups_per_day(processed_predictions, col)
            return lineups_by_col[col]

        score_per_day = lineups_by_col['YAHOO_POINTS_pred']
        score_per_day_cash = _get_or_solve(cash_col)
        score_per_day_gpp = _get_or_solve(gpp_col)

        eval_split_series = eval_results['split_type'].astype(str) if 'split_type' in eval_results.columns else None
        all_dates_scores = pd.DataFrame(columns=['DATE', 'predicted_score', 'split_type'])
        if 'split_type' in processed_predictions.columns and not score_per_day.empty:
            all_dates_scores = score_per_day.merge(
                processed_predictions[['DATE', 'split_type']].drop_duplicates(),
                on='DATE',
                how='left',
            )
            if cash_col != 'YAHOO_POINTS_pred' and not score_per_day_cash.empty:
                all_dates_scores = all_dates_scores.merge(
                    score_per_day_cash.rename(columns={'predicted_score': 'predicted_score_cash'}),
                    on='DATE',
                    how='left',
                )
            if gpp_col != 'YAHOO_POINTS_pred' and not score_per_day_gpp.empty:
                all_dates_scores = all_dates_scores.merge(
                    score_per_day_gpp.rename(columns={'predicted_score': 'predicted_score_gpp'}),
                    on='DATE',
                    how='left',
                )
            for raw_split, grp in all_dates_scores.groupby('split_type', sort=False):
                split_key = cls._resolve_eval_split_key(eval_results, raw_split, eval_split_series)
                eval_results.loc[split_key, 'mean_lineup_score_all_dates'] = grp['predicted_score'].mean()
                eval_results.loc[split_key, 'n_lineup_dates'] = len(grp)
                if 'predicted_score_cash' in grp.columns:
                    eval_results.loc[split_key, 'mean_lineup_score_cash_all_dates'] = grp['predicted_score_cash'].mean()
                if 'predicted_score_gpp' in grp.columns:
                    eval_results.loc[split_key, 'mean_lineup_score_gpp_all_dates'] = grp['predicted_score_gpp'].mean()

        dfs_contests_df = contests_df.merge(score_per_day, on='DATE', how='inner')
        dfs_contests_df['predicted_is_better'] = dfs_contests_df['predicted_score'] > dfs_contests_df['score']
        if cash_col != 'YAHOO_POINTS_pred':
            dfs_contests_df = dfs_contests_df.merge(
                score_per_day_cash.rename(columns={'predicted_score': 'predicted_score_cash'}),
                on='DATE',
                how='left',
            )
            dfs_contests_df['predicted_is_better_cash'] = (
                dfs_contests_df['predicted_score_cash'] > dfs_contests_df['score']
            )
        if gpp_col != 'YAHOO_POINTS_pred':
            dfs_contests_df = dfs_contests_df.merge(
                score_per_day_gpp.rename(columns={'predicted_score': 'predicted_score_gpp'}),
                on='DATE',
                how='left',
            )
            dfs_contests_df['predicted_is_better_gpp'] = (
                dfs_contests_df['predicted_score_gpp'] > dfs_contests_df['score']
            )

        agg_spec: Dict[str, str] = {'predicted_is_better': 'mean', 'predicted_score': 'first'}
        if 'predicted_is_better_cash' in dfs_contests_df.columns:
            agg_spec['predicted_is_better_cash'] = 'mean'
            agg_spec['predicted_score_cash'] = 'first'
        if 'predicted_is_better_gpp' in dfs_contests_df.columns:
            agg_spec['predicted_is_better_gpp'] = 'mean'
            agg_spec['predicted_score_gpp'] = 'first'

        percentile_per_contest = dfs_contests_df.groupby('contestId', as_index=False).agg(agg_spec)
        percentile_per_contest = percentile_per_contest.merge(
            dfs_contests_df[['contestId', 'DATE', 'entry_fee']].drop_duplicates(),
            on='contestId',
            how='left',
        )
        percentile_per_contest.rename(
            columns={
                'predicted_is_better': 'percentile',
                'predicted_is_better_cash': 'percentile_cash',
                'predicted_is_better_gpp': 'percentile_gpp',
            },
            inplace=True,
        )
        percentile_per_contest = percentile_per_contest.merge(
            processed_predictions[['DATE', 'split_type']].drop_duplicates(),
            on='DATE',
            how='left',
        )
        cash_pct_col = 'percentile_cash' if 'percentile_cash' in percentile_per_contest.columns else 'percentile'
        gpp_pct_col = 'percentile_gpp' if 'percentile_gpp' in percentile_per_contest.columns else 'percentile'
        percentile_per_contest['is_top10'] = (percentile_per_contest[gpp_pct_col] > 0.9).astype(float)
        percentile_per_contest['profit_5050'] = np.where(percentile_per_contest[cash_pct_col] > 0.5, 0.8, -1.0)
        percentile_per_contest['profit_top10'] = np.where(percentile_per_contest[gpp_pct_col] > 0.9, 2.0, -1.0)
        percentile_per_contest['profit_top10_9x'] = np.where(percentile_per_contest[gpp_pct_col] > 0.9, 8.0, -1.0)

        paid_mask = percentile_per_contest['entry_fee'].notnull() & (percentile_per_contest['entry_fee'] > 0)
        for ppc, df_name in zip([percentile_per_contest, percentile_per_contest.loc[paid_mask]], ['all', 'paid']):
            for portfolio_col in (
                f'portfolio_value_5050_{df_name}',
                f'portfolio_value_top10_{df_name}',
                f'portfolio_value_top10_9x_{df_name}',
            ):
                eval_results[portfolio_col] = [list()] * eval_results.shape[0]
            for raw_split, group_df in ppc.groupby('split_type', sort=False):
                split_key = cls._resolve_eval_split_key(eval_results, raw_split, eval_split_series)
                date_agg: Dict[str, str] = {
                    'profit_5050': 'mean',
                    'profit_top10': 'mean',
                    'profit_top10_9x': 'mean',
                    'is_top10': 'mean',
                    'percentile': 'mean',
                    'predicted_score': 'mean',
                }
                for opt_col in ('percentile_cash', 'percentile_gpp', 'predicted_score_cash', 'predicted_score_gpp'):
                    if opt_col in group_df.columns:
                        date_agg[opt_col] = 'mean'
                one_per_date = group_df.groupby('DATE').agg(date_agg)
                one_per_date['portfolio_value_5050'] = one_per_date['profit_5050'].cumsum()
                one_per_date['portfolio_value_top10'] = one_per_date['profit_top10'].cumsum()
                one_per_date['portfolio_value_top10_9x'] = one_per_date['profit_top10_9x'].cumsum()

                eval_results.at[split_key, f'portfolio_value_5050_{df_name}'] = one_per_date['portfolio_value_5050'].values.tolist()
                eval_results.at[split_key, f'portfolio_value_top10_{df_name}'] = one_per_date['portfolio_value_top10'].values.tolist()
                eval_results.at[split_key, f'portfolio_value_top10_9x_{df_name}'] = one_per_date['portfolio_value_top10_9x'].values.tolist()

                eval_results.loc[split_key, f'mean_percentile_{df_name}'] = float(one_per_date['percentile'].mean())
                eval_results.loc[split_key, f'mean_score_{df_name}'] = float(one_per_date['predicted_score'].mean())
                eval_results.loc[split_key, f'top10_pct_{df_name}'] = float(one_per_date['is_top10'].mean())
                eval_results.loc[split_key, f'mean_profit_5050_{df_name}'] = float(one_per_date['profit_5050'].mean())
                eval_results.loc[split_key, f'mean_profit_top10_{df_name}'] = float(one_per_date['profit_top10'].mean())
                eval_results.loc[split_key, f'mean_profit_top10_9x_{df_name}'] = float(one_per_date['profit_top10_9x'].mean())
                eval_results.loc[split_key, f'sharpe_5050_{df_name}'] = sharpe(one_per_date['profit_5050'])
                eval_results.loc[split_key, f'sharpe_top10_{df_name}'] = sharpe(one_per_date['profit_top10'])
                eval_results.loc[split_key, f'sharpe_top10_9x_{df_name}'] = sharpe(one_per_date['profit_top10_9x'])
                eval_results.loc[split_key, f'mean_percentile_contests_{df_name}'] = float(group_df['percentile'].mean())
                eval_results.loc[split_key, f'top10_pct_contests_{df_name}'] = float(group_df['is_top10'].mean())
                eval_results.loc[split_key, f'n_contests_{df_name}'] = int(len(group_df))
                eval_results.loc[split_key, f'n_contest_dates_{df_name}'] = int(len(one_per_date))
                if 'percentile_cash' in one_per_date.columns:
                    eval_results.loc[split_key, f'mean_percentile_cash_{df_name}'] = float(
                        one_per_date['percentile_cash'].mean()
                    )
                    eval_results.loc[split_key, f'mean_score_cash_{df_name}'] = float(
                        one_per_date['predicted_score_cash'].mean()
                    )
                if 'percentile_gpp' in one_per_date.columns:
                    eval_results.loc[split_key, f'mean_percentile_gpp_{df_name}'] = float(
                        one_per_date['percentile_gpp'].mean()
                    )
                    eval_results.loc[split_key, f'mean_score_gpp_{df_name}'] = float(
                        one_per_date['predicted_score_gpp'].mean()
                    )

        if return_details:
            return eval_results, {
                'all_dates_scores': all_dates_scores,
                'percentile_per_contest': percentile_per_contest,
            }
        return eval_results

    def _add_extra_metrics_to_eval_results(self, eval_results: pd.DataFrame, predict_results: pd.DataFrame
                                           ) -> pd.DataFrame:
        try:
            contests_df = self._load_contests_df()
        except FileNotFoundError as e:
            logging.warning(f"Could not load DFS contest results: {e}. Skipping extra DFS contest metrics.")
            return self._calculate_prediction_correlations(
                eval_results=eval_results, predict_results=predict_results
            )

        tp = getattr(getattr(self, 'params', None), 'training_params', None)
        alpha_cash, alpha_gpp = resolve_quantile_alphas(tp)

        return self.evaluate_predictions_df(
            predict_results=predict_results,
            eval_results=eval_results,
            contests_df=contests_df,
            return_details=False,
            alpha_cash=alpha_cash,
            alpha_gpp=alpha_gpp,
        )

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
                    col_top10_9x = f'portfolio_value_top10_9x_{suffix}'
                    if col_5050 in eval_results.columns and col_top10 in eval_results.columns:
                        portfolio_value_5050 = eval_results.loc[split_key, col_5050]
                        portfolio_value_top10 = eval_results.loc[split_key, col_top10]
                        portfolio_value_top10_9x = (
                            eval_results.loc[split_key, col_top10_9x]
                            if col_top10_9x in eval_results.columns
                            else []
                        )
                        if hasattr(portfolio_value_5050, '__iter__') and hasattr(portfolio_value_top10, '__iter__'):
                            for step, (curr_5050, curr_top10) in enumerate(zip(portfolio_value_5050, portfolio_value_top10)):
                                writer.add_scalar(tag=f'DFS_Profit/portfolio_5050_{suffix}',
                                                  scalar_value=float(curr_5050), global_step=step)
                                writer.add_scalar(tag=f'DFS_Profit/portfolio_top10_{suffix}',
                                                  scalar_value=float(curr_top10), global_step=step)
                        if hasattr(portfolio_value_top10_9x, '__iter__'):
                            for step, curr_top10_9x in enumerate(portfolio_value_top10_9x):
                                writer.add_scalar(tag=f'DFS_Profit/portfolio_top10_9x_{suffix}',
                                                  scalar_value=float(curr_top10_9x), global_step=step)

                scalar_prefixes = (
                    'mean_percentile', 'mean_score', 'mean_lineup_score', 'top10_pct',
                    'sharpe_5050', 'sharpe_top10', 'mean_profit_', 'n_contest', 'n_lineup_dates',
                    'fp_rmse_raw', 'fp_bias_', 'min_mae_sal', 'n_salaried_players', 'n_dropped_salaried_players',
                    'q20_coverage', 'q80_coverage', 'quantile_crossing_rate',
                )
                for m in scalar_prefixes:
                    for c in [cc for cc in eval_results.columns if cc.startswith(m)]:
                        val = eval_results.loc[split_key, c]
                        if isinstance(val, (int, float, np.integer, np.floating)) and pd.notna(val):
                            writer.add_scalar(tag=f'DFS_Contests/{c}',
                                              scalar_value=float(val),
                                              global_step=0)
                writer.flush()
                writer.close()
        except Exception as e:
            logging.warning(f"Could not save DFS results to tensorboard: {e}")

        return eval_results, predict_results
