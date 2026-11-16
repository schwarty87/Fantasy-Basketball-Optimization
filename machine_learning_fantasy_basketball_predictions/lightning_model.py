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
from machine_learning.params.machine_learning_dataset_params import MachineLearningDatasetTabularScalingParams
from machine_learning_fantasy_basketball_predictions.params.params import (
    MachineLearningParamsFantasyBasketballPredictions,
)
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions, convert_date_col_to_date

_convert_date_col_to_date = convert_date_col_to_date


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
        if "stage1_alloc" in all_tensors:
            result["stage1_alloc"] = all_tensors["stage1_alloc"].detach().cpu()
        if "stage1_label" in all_tensors:
            result["stage1_label"] = all_tensors["stage1_label"].detach().cpu()
        if "true_label" in all_tensors:
            result["true_label"] = all_tensors["true_label"].detach().cpu()
        if METADATA in all_tensors:
            result[METADATA] = all_tensors[METADATA].detach().cpu()
        if "keep_mask" in all_tensors:
            result["keep_mask"] = all_tensors["keep_mask"].detach().cpu()
        return result

    @staticmethod
    def _stack_batch_tensor(predict_batches: List[Dict[str, torch.Tensor]], key: str) -> np.ndarray:
        chunks = []
        for batch in predict_batches:
            if key in batch and batch[key].numel() > 0:
                arr = batch[key].numpy()
                if arr.ndim == 3:
                    arr = arr.reshape(-1, arr.shape[-1])
                chunks.append(arr)
        return np.vstack(chunks) if chunks else np.empty((0, 0))

    def format_predictions(
        self, predict_batches: List[Dict[str, torch.Tensor]]
    ) -> pd.DataFrame:
        """
        Converts the list of batch outputs from trainer.predict into a fully formatted,
        unscaled DataFrame compatible with downstream pipelines (e.g. run_full_pipeline.py).
        """
        all_preds = self._stack_batch_tensor(predict_batches, "outputs_pred")
        all_trues = self._stack_batch_tensor(predict_batches, "true_label")
        all_metas = self._stack_batch_tensor(predict_batches, METADATA)
        all_masks = self._stack_batch_tensor(predict_batches, "keep_mask")
        all_stage1_preds = self._stack_batch_tensor(predict_batches, "stage1_alloc")
        all_stage1_trues = self._stack_batch_tensor(predict_batches, "stage1_label")

        # Unscale targets if fitted scaler is available
        scaler = self._load_target_scaler()
        if (
            isinstance(
                self.params.features_params.feature_name_to_machine_learning_dataset_params.get("true_label"),
                MachineLearningDatasetTabularScalingParams,
            )
            and self.scaler_dir is not None
            and scaler is None
        ):
            raise FileNotFoundError(f"Target scaler true_label.pkl not found under {self.scaler_dir}")

        if scaler:
            if all_preds.size > 0:
                all_preds = scaler.inverse_transform(all_preds)
            if all_trues.size > 0:
                should_unscale_true = all_trues != IGNORE
                if all_masks.size > 0 and all_masks.shape[0] == all_trues.shape[0]:
                    should_unscale_true = should_unscale_true & (all_masks > 0)
                all_trues = np.where(should_unscale_true, scaler.inverse_transform(all_trues), all_trues)

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
        if all_stage1_preds.size > 0:
            dfs.append(pd.DataFrame(all_stage1_preds, columns=["MIN_ALLOC_pred", "PLAYS_ALLOC_pred"]))
        if all_stage1_trues.size > 0:
            dfs.append(pd.DataFrame(all_stage1_trues, columns=["MIN_SHARE_true", "PLAYS_SHARE_true"]))
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
