import os
import pickle
import datetime
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union

from machine_learning.constants import DATA_DIR, METADATA, IGNORE
from machine_learning.architecture import ModelArchitecture
from machine_learning.metrics.combined_metrics import LossesCombined, MetricsCombined
from machine_learning.lightning_model import MachineLearningLightningModule
from machine_learning_fantasy_basketball_predictions.params.params import (
    MachineLearningParamsFantasyBasketballPredictions,
)
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions


def _convert_date_col_to_date(date_series: pd.Series) -> pd.Series:
    """Robustly converts date series from seconds, days, strings, or datetimes to datetime.date."""
    if date_series.empty:
        return date_series
    first_val = date_series.dropna().iloc[0] if not date_series.dropna().empty else None
    if isinstance(first_val, datetime.date) and not isinstance(first_val, datetime.datetime):
        return date_series
    if pd.api.types.is_numeric_dtype(date_series):
        unit = "s" if float(first_val or 0) > 1e7 else "D"
        return pd.to_datetime(date_series, unit=unit).dt.date
    return pd.to_datetime(date_series).dt.date


class FantasyBasketballLightningModule(MachineLearningLightningModule):
    """
    Fantasy Basketball specific LightningModule inheriting core lifecycle from
    MachineLearningLightningModule and adding basketball prediction formatting & unscaling.
    """

    def __init__(
        self,
        params: Optional[MachineLearningParamsFantasyBasketballPredictions] = None,
        scaler_dir: Optional[str] = None,
        model: Optional[ModelArchitecture] = None,
        loss_function: Optional[LossesCombined] = None,
        metrics: Optional[MetricsCombined] = None,
        **kwargs,
    ):
        super().__init__(
            params=params,
            scaler_dir=scaler_dir,
            model=model,
            loss_function=loss_function,
            metrics=metrics,
            **kwargs,
        )
        self._target_scaler = None

    @classmethod
    def _init_params_from_dict(cls, kwargs: dict) -> Optional[MachineLearningParamsFantasyBasketballPredictions]:
        if kwargs:
            try:
                return MachineLearningParamsFantasyBasketballPredictions.init_from_dict(kwargs)
            except Exception:
                pass
        return None

    def _load_target_scaler(self):
        """Loads the fitted StandardScaler for true_label if not already cached."""
        if self._target_scaler is None and self.scaler_dir is not None:
            scaler_path = os.path.join(self.scaler_dir, "scalers", "true_label.pkl")
            if not os.path.isfile(scaler_path):
                scaler_path = os.path.join(self.scaler_dir, "true_label.pkl")
            if os.path.isfile(scaler_path):
                with open(scaler_path, "rb") as f:
                    self._target_scaler = pickle.load(f)
        return self._target_scaler

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        _, all_tensors = self._prepare_all_tensors(batch)
        result = {
            "outputs_pred": all_tensors["outputs_pred"].detach().cpu(),
        }
        if "true_label" in all_tensors:
            result["true_label"] = all_tensors["true_label"].detach().cpu()
        if METADATA in all_tensors:
            result[METADATA] = all_tensors[METADATA].detach().cpu()
        if "keep_mask" in all_tensors:
            result["keep_mask"] = all_tensors["keep_mask"].detach().cpu()
        return result

    def format_predictions(
        self, predict_batches: List[Dict[str, torch.Tensor]]
    ) -> pd.DataFrame:
        """
        Converts the list of batch outputs from trainer.predict into a fully formatted,
        unscaled DataFrame compatible with downstream pipelines (e.g. run_full_pipeline.py).
        """
        all_preds = []
        all_trues = []
        all_metas = []
        all_masks = []

        for batch in predict_batches:
            preds = batch["outputs_pred"].numpy()
            if preds.ndim == 3:
                preds = preds.reshape(-1, preds.shape[-1])
            all_preds.append(preds)

            if "true_label" in batch and batch["true_label"].numel() > 0:
                trues = batch["true_label"].numpy()
                if trues.ndim == 3:
                    trues = trues.reshape(-1, trues.shape[-1])
                all_trues.append(trues)

            if METADATA in batch and batch[METADATA].numel() > 0:
                meta = batch[METADATA].numpy()
                if meta.ndim == 3:
                    meta = meta.reshape(-1, meta.shape[-1])
                all_metas.append(meta)

            if "keep_mask" in batch and batch["keep_mask"].numel() > 0:
                mask = batch["keep_mask"].numpy()
                if mask.ndim == 3:
                    mask = mask.reshape(-1, mask.shape[-1])
                all_masks.append(mask)

        all_preds = np.vstack(all_preds) if all_preds else np.empty((0, 0))
        all_trues = np.vstack(all_trues) if all_trues else np.empty((0, 0))
        all_metas = np.vstack(all_metas) if all_metas else np.empty((0, 0))
        all_masks = np.vstack(all_masks) if all_masks else np.empty((0, 0))

        # Unscale targets if fitted scaler is available
        scaler = self._load_target_scaler()
        if scaler:
            if all_preds.size > 0:
                all_preds = scaler.inverse_transform(all_preds)
            if all_trues.size > 0:
                all_trues = np.where(all_trues != IGNORE, scaler.inverse_transform(all_trues), IGNORE)

        target_stats = self.params.features_params.feature_name_to_machine_learning_dataset_params[
            "true_label"
        ].feature_names
        meta_cols = self.params.features_params.feature_name_to_machine_learning_dataset_params[
            METADATA
        ].feature_names

        dfs = []
        if all_preds.size > 0:
            dfs.append(pd.DataFrame(all_preds, columns=[f"{c}_pred" for c in target_stats]))
        if all_trues.size > 0:
            dfs.append(pd.DataFrame(all_trues, columns=[f"{c}_true" for c in target_stats]))
        if all_metas.size > 0:
            dfs.append(pd.DataFrame(all_metas, columns=meta_cols))
        if all_masks.size > 0:
            dfs.append(pd.DataFrame(all_masks, columns=["NOT_IGNORE"]))

        results_df = pd.concat(dfs, axis=1)

        # Convert epoch float days back to datetime.date
        if "DATE" in results_df.columns:
            results_df["DATE"] = _convert_date_col_to_date(results_df["DATE"])

        # Ensure fantasy points calculations are added
        add_missing_points_columns_to_predictions(results_df)

        return results_df
