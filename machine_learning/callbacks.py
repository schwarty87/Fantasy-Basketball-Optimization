import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle as pkl
import torch
from abc import ABC
from torch.optim.optimizer import Optimizer
from typing import Dict, Tuple, Type, Optional, Union, List
from machine_learning.constants import METADATA, IGNORE
from machine_learning.enums import ModelMode
from machine_learning.machine_learning_datasets import MachineLearningDatasetMetadata
from machine_learning.params.abstract_params import AbstractParams
from machine_learning.params.callbacks_params import TrackBestValueParams, ModelCheckpointParams, EarlyStoppingParams, \
    LogModelResultsParams, CollectTruePredictedOutputsParams, ReduceLearningRateOnPlateauParams
from machine_learning.registries import callbacks_registry
from machine_learning.utils import tensor_to_df, dict_to_pretty_dict


class AbstractCallback(ABC):
    def __init__(self, params: AbstractParams, model: Optional[torch.nn.Module] = None,
                 optimizer: Optional[Optimizer] = None, model_dir: Optional[str] = None):
        self.params = params
        self._model = model
        self._optimizer = optimizer
        self._model_dir = model_dir

    def on_epoch_begin(self, epoch_num):
        """Should be called at the beginning of each TRAINING epoch. A training epoch is counted one pass through all
        training batches, and one pass through all eval batches"""
        pass

    def on_epoch_end(self, epoch_num, metric_name_to_result: Dict[str, Union[float, Dict]]):
        """
        Should be called at the end of each TRAINING epoch. This should only be called after we've passed through all
        training batches and all eval batches
        :param epoch_num: The epoch number
        :param metric_name_to_result: A dictionary of metric_name: metric_value. For each metric we should have a value
                                     for metric_name and eval_metric_name
        """
        pass

    def on_train_epoch_end(self, epoch_num, metric_name_to_result: Dict[str, float]):
        """
        Should be called after one pass through the TRAINING data during training
        :param epoch_num: The epoch number
        :param metric_name_to_result: A dictionary of metric_name: metric_value. For each metric we should have a value
                                     for metric_name and eval_metric_name
        """
        pass

    def on_train_begin(self):
        """Should be called when TRAINING starts"""
        pass

    def on_train_end(self):
        """Should be called when TRAINING ends"""
        pass

    def on_train_batch_begin(self, batch_num):
        """Should be called at the beginning of each TRAINING batch"""
        pass

    def on_train_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        """Should be called at the end of each TRAINING batch"""
        pass

    def on_eval_begin(self):
        """Should be called when EVAL starts"""
        pass

    def on_eval_end(self, metric_name_to_result: Optional[Dict[str, float]] = None):
        """Should be called when EVAL ends"""
        pass

    def on_eval_batch_begin(self, batch_num):
        """Should be called at the beginning of each EVAL batch"""
        pass

    def on_eval_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        """Should be called at the end of each EVAL batch"""
        pass

    def on_predict_begin(self):
        """Should be called at the beginning of a PREDICTION run"""
        pass

    def on_predict_end(self, metric_name_to_results_df: Optional[Dict[str, pd.DataFrame]] = None
                       ) -> Optional[pd.DataFrame]:
        """Should be called at the end of a PREDICTION run"""
        return None

    def on_predict_batch_begin(self, batch_num):
        """Should be called at the beginning of each PREDICTION batch"""
        pass

    def on_predict_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        """Should be called at the end of each PREDICTION batch"""
        pass

    def reset(self):
        pass

    def stop_training(self) -> bool:
        return False


def training_callbacks() -> List[Type[AbstractCallback]]:
    """These callbacks should only be used during training"""
    return [ModelCheckpoint, EarlyStopping, LogModelResults, ReduceLearningRateOnPlateau]


class CallbacksManager:
    def __init__(self, callback_params: Tuple[AbstractParams], model: torch.nn.Module,
                 optimizer: Optional[Optimizer] = None, model_dir: Optional[str] = None):
        self.callbacks = list()
        for params in callback_params:
            callback_key = params.key_to_class_for_registry
            callback_class = callbacks_registry.get(callback_key)
            self.callbacks.append(callback_class(params=params, model=model, optimizer=optimizer, model_dir=model_dir))

    def stop_training(self) -> bool:
        return any(callback.stop_training() for callback in self.callbacks)

    def _dispatch(self, method_name: str, *args, **kwargs):
        for callback in self.callbacks:
            getattr(callback, method_name)(*args, **kwargs)

    def on_epoch_begin(self, epoch_num: int):
        self._dispatch('on_epoch_begin', epoch_num)

    def on_epoch_end(self, epoch_num: int, metric_name_to_result: Dict[str, Union[float, Dict]]):
        self._dispatch('on_epoch_end', epoch_num=epoch_num, metric_name_to_result=metric_name_to_result)

    def on_train_epoch_end(self, epoch_num, metric_name_to_result: Dict[str, float]):
        self._dispatch('on_train_epoch_end', epoch_num=epoch_num, metric_name_to_result=metric_name_to_result)

    def on_train_begin(self):
        self._dispatch('on_train_begin')

    def on_train_end(self):
        self._dispatch('on_train_end')

    def on_train_batch_begin(self, batch_num: int):
        self._dispatch('on_train_batch_begin', batch_num=batch_num)

    def on_train_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        self._dispatch('on_train_batch_end', batch_num=batch_num, all_tensors=all_tensors)

    def on_eval_begin(self):
        self._dispatch('on_eval_begin')

    def on_eval_end(self, metric_name_to_result: Optional[Dict[str, float]] = None):
        self._dispatch('on_eval_end', metric_name_to_result=metric_name_to_result)

    def on_eval_batch_begin(self, batch_num):
        self._dispatch('on_eval_batch_begin', batch_num=batch_num)

    def on_eval_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        self._dispatch('on_eval_batch_end', batch_num=batch_num, all_tensors=all_tensors)

    def on_predict_begin(self):
        self._dispatch('on_predict_begin')

    def on_predict_end(self, metric_name_to_results_df: Optional[Dict[str, pd.DataFrame]] = None
                       ) -> Dict[str, Optional[pd.DataFrame]]:
        """
        :param metric_name_to_results_df: A dict of each metric name to a dataframe with its results
        :return: results_df: A dataframe with a row for each sample, and all the true/predicted data, along with
                             metrics/losses results and metadata
        """
        return {type(callback.params).__name__: callback.on_predict_end(metric_name_to_results_df=metric_name_to_results_df)
                for callback in self.callbacks}

    def on_predict_batch_begin(self, batch_num):
        self._dispatch('on_predict_batch_begin', batch_num=batch_num)

    def on_predict_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        self._dispatch('on_predict_batch_end', batch_num=batch_num, all_tensors=all_tensors)

    def on_batch_begin(self, training_mode: ModelMode, batch_num: int):
        hook_map = {
            ModelMode.TRAIN: 'on_train_batch_begin',
            ModelMode.EVAL: 'on_eval_batch_begin',
            ModelMode.PREDICT: 'on_predict_batch_begin',
        }
        hook = hook_map.get(training_mode)
        if hook:
            self._dispatch(hook, batch_num=batch_num)

    def on_batch_end(self, training_mode: ModelMode, batch_num: int, all_tensors: Dict[str, torch.Tensor]):
        hook_map = {
            ModelMode.TRAIN: 'on_train_batch_end',
            ModelMode.EVAL: 'on_eval_batch_end',
            ModelMode.PREDICT: 'on_predict_batch_end',
        }
        hook = hook_map.get(training_mode)
        if hook:
            self._dispatch(hook, batch_num=batch_num, all_tensors=all_tensors)


class TrackBestValue(AbstractCallback, ABC):
    """
    This is a callback that is meant to be the parent of any callback that relies on finding the best loss value
    (like ModelCheckpoint and EarlyStopping)
    """
    def __init__(self, params: TrackBestValueParams, model: Optional[torch.nn.Module] = None,
                 optimizer: Optional[Optimizer] = None, model_dir: Optional[str] = None):
        super().__init__(params=params, model=model, optimizer=optimizer, model_dir=model_dir)
        self.reset()

    def _is_better(self, value: float):
        if self.params.track_min_or_max:
            return value < self.best_value
        else:
            return value > self.best_value

    def _on_found_better(self, epoch_num: int, value: float):
        raise NotImplementedError

    def _on_current_value_isnt_better(self):
        pass

    def on_epoch_end(self, epoch_num: int, metric_name_to_result: Dict[str, Union[float, Dict]]):
        value = metric_name_to_result[self.params.value_to_track]
        if self._is_better(value):
            self.best_epoch = epoch_num
            self.best_value = value
            self._on_found_better(epoch_num=epoch_num, value=value)
        else:
            self._on_current_value_isnt_better()

    def reset(self):
        super().reset()
        self.best_epoch = None
        self.best_value = np.inf if self.params.track_min_or_max else -np.inf


@callbacks_registry.register
class ModelCheckpoint(TrackBestValue):
    def __init__(self, params: ModelCheckpointParams, model: torch.nn.Module, model_dir: Optional[str] = None,
                 **kwargs):
        del kwargs
        super().__init__(params=params, model=model, model_dir=model_dir)
        self._save_path = os.path.join(model_dir, 'model.pt')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        self.reset()

    def _on_found_better(self, **kwargs):
        del kwargs
        torch.save(self._model, self._save_path)

    def on_train_end(self):
        """Load the last saved model's state into the current model"""
        loaded_model = self.load_model(self._save_path)
        self._model.load_state_dict(loaded_model.state_dict())

    @staticmethod
    def load_model(model_path: str) -> torch.nn.Module:
        return torch.load(model_path, weights_only=False)


@callbacks_registry.register
class EarlyStopping(TrackBestValue):
    def __init__(self, params: EarlyStoppingParams, **kwargs):
        del kwargs
        super().__init__(params=params)
        self.reset()

    def stop_training(self) -> bool:
        return self.count >= self.params.patience

    def _on_found_better(self, epoch_num, value):
        self.count = 0

    def _on_current_value_isnt_better(self):
        self.count += 1

    def reset(self):
        super().reset()
        self.count = 0


@callbacks_registry.register
class CollectTruePredictedOutputs(AbstractCallback):
    """Collect the model outputs"""
    def __init__(self, params: CollectTruePredictedOutputsParams, **kwargs):
        self.model_dir = kwargs.get('model_dir', None)
        del kwargs
        super().__init__(params=params)
        self.reset()

    @staticmethod
    def _reshape_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor

    def _convert_dict_of_tensors_to_df(self, tensors_dict: Dict[str, torch.Tensor],
                                       column_names_dict: Dict[str, List[str]], prefix: Optional[str] = None
                                       ) -> pd.DataFrame:
        dfs = list()
        tensors_dict = {k: v for k, v in tensors_dict.items() if k in column_names_dict.keys()}
        for tensor_name, curr_tensor in tensors_dict.items():
            reshaped_tensor = self._reshape_tensor(curr_tensor, name=tensor_name)
            column_names = column_names_dict.get(tensor_name)
            if tensor_name == METADATA:
                df = MachineLearningDatasetMetadata.tensor_to_df(reshaped_tensor, column_names=column_names)
            else:
                df = tensor_to_df(reshaped_tensor, column_names=column_names)
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        if prefix is not None:
            df.rename(columns={c: f'{prefix}_{c}' for c in df.columns}, inplace=True)
        return df.loc[:, ~df.columns.duplicated()]

    def tensors_to_df(self, all_tensors, tensor_name, prefix):
        all_tensors_dict = {n: all_tensors[n] for n in tensor_name}
        df = self._convert_dict_of_tensors_to_df(
            tensors_dict=all_tensors_dict, prefix=prefix, column_names_dict=self.params.column_names)
        return df.loc[:, ~df.columns.duplicated()]

    def on_predict_batch_end(self, batch_num, all_tensors: Dict[str, torch.Tensor]):
        df = self._convert_dict_of_tensors_to_df(tensors_dict=all_tensors, column_names_dict=self.params.column_names)
        self.results.append(df)

    def _resolve_scaler_dir(self) -> Optional[str]:
        """Resolve scaler directory from callback params or fallback to model_dir/scalers."""
        scaler_dir = self.params.scaler_dir
        if (scaler_dir is None or not os.path.isdir(scaler_dir)) and self.model_dir is not None:
            candidate = os.path.join(self.model_dir, 'scalers')
            if os.path.isdir(candidate):
                return candidate
        return scaler_dir

    def _unscale_results(self, results_df: pd.DataFrame, scaler_dir: str):
        """Unscale standardized target and prediction columns back to physical units."""
        for tensor_name in self.params.all_tensor_names:
            scaler_path = os.path.join(scaler_dir, f'{tensor_name}.pkl')
            if os.path.isfile(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pkl.load(f)
                for tt in ('true', 'pred'):
                    cols = [f'{c}_{tt}' for c in scaler.feature_names_in_]
                    num_in_df = len(set(cols).intersection(set(results_df.columns)))
                    if num_in_df > 0:
                        if num_in_df < len(cols):
                            raise ValueError(f'not all scaler features are in results_df {tt}!')
                        results_df[cols] = np.where(
                            results_df[cols] != IGNORE,
                            scaler.inverse_transform(results_df[cols].values),
                            IGNORE,
                        )
            elif self.params.scaler_dir is not None and tensor_name == 'true_label':
                logging.warning(
                    f"Scaler file '{scaler_path}' not found in scaler_dir '{scaler_dir}'. Predictions not unscaled."
                )

    @staticmethod
    def _normalize_date_column(results_df: pd.DataFrame):
        """Standardize the DATE column into datetime.date objects."""
        if 'DATE' not in results_df.columns or results_df.empty:
            return
        if pd.api.types.is_numeric_dtype(results_df['DATE']):
            first_val = results_df['DATE'].dropna().iloc[0] if not results_df['DATE'].dropna().empty else None
            unit = 's' if float(first_val or 0) > 1e7 else 'D'
            results_df['DATE'] = pd.to_datetime(results_df['DATE'], unit=unit).dt.date
        elif not isinstance(results_df['DATE'].iloc[0], datetime.date):
            results_df['DATE'] = pd.to_datetime(results_df['DATE']).dt.date

    @staticmethod
    def _merge_metric_results(
        results_df: pd.DataFrame, metric_name_to_results_df: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Concatenate auxiliary metric results dataframes onto the predictions dataframe."""
        if metric_name_to_results_df is None:
            return results_df
        dfs_to_concat = [results_df]
        for v in metric_name_to_results_df.values():
            if isinstance(v, (pd.DataFrame, pd.Series)):
                dfs_to_concat.append(v)
        return pd.concat(dfs_to_concat, axis=1)

    def on_predict_end(self, metric_name_to_results_df: Optional[Dict[str, pd.DataFrame]] = None) -> Optional[pd.DataFrame]:
        results_df = pd.concat(self.results, axis=0).reset_index(drop=True)

        scaler_dir = self._resolve_scaler_dir()
        if scaler_dir is not None:
            self._unscale_results(results_df, scaler_dir)

        self._normalize_date_column(results_df)
        results_df = self._merge_metric_results(results_df, metric_name_to_results_df)

        if self.params.save_path is not None:
            results_df.to_feather(self.params.save_path)

        sort_cols = [c for c in ['DATE', 'PLAYER_ID'] if c in results_df.columns]
        return results_df.sort_values(sort_cols) if sort_cols else results_df

    def reset(self):
        super().reset()
        self.results = list()


@callbacks_registry.register
class ReduceLearningRateOnPlateau(AbstractCallback):
    def __init__(self, params: ReduceLearningRateOnPlateauParams, optimizer: Optional[Optimizer] = None, **kwargs):
        del kwargs
        super().__init__(params=params, optimizer=optimizer)
        self.reset()

        self._learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                   patience=params.patience,
                                                                                   factor=params.factor,
                                                                                   threshold=params.threshold,
                                                                                   cooldown=params.cooldown,
                                                                                   min_lr=params.min_lr,
                                                                                   # verbose=params.verbose
                                                                                   )

    def on_epoch_end(self, epoch_num, metric_name_to_result: Dict[str, float], **kwargs):
        del kwargs
        self._learning_rate_scheduler.step(metric_name_to_result[self.params.value_to_track])


@callbacks_registry.register
class LogModelResults(AbstractCallback):
    def on_epoch_end(self, epoch_num: int, metric_name_to_result: Dict[str, Union[float, Dict]]):
        if metric_name_to_result is None:
            return
        metric_name_to_result_pretty = dict_to_pretty_dict(metric_name_to_result)
        metric_name_to_result_pretty_str = json.dumps(metric_name_to_result_pretty).replace('"', '')
        logging.info(f'epoch: {epoch_num}: {metric_name_to_result_pretty_str}')

