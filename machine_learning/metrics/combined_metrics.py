from abc import ABC
from typing import Optional, List, Dict, Union, Type, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from machine_learning.enums import Reduction
from machine_learning.metrics.abstract_metrics import LossFunctionAndWeight, AbstractMetric, get_column_names, \
    fix_y_size
from machine_learning.metrics.metrics import MAEMetric, MAPEMetric, BinaryCrossEntropyLoss, ROCAUC, \
    BinaryClassificationMetrics, MSELoss, QuantileRegressionLoss
from machine_learning.params.abstract_params import AbstractMetricParams, AbstractParams
from machine_learning.params.metrics_params import MAEMetricParams, MAPEMetricParams, BinaryCrossEntropyParams, \
    ROCAUCParams, BinaryClassificationMetricsParams, MSELossParams, QuantileRegressionLossParams
from machine_learning.utils import variable_is_number
from machine_learning.registries import metrics_registry


class MetricsCombined(torch.nn.Module, ABC):
    def __init__(self, all_params: Tuple[AbstractMetricParams, ...], device: torch.device,
                 name: Optional[str] = None):
        super().__init__()
        self._all_params = all_params
        self._name = name

        self._init_classes()
        self.to(device=device)

    def _init_classes(self):
        self.all_classes = list()
        for metric_params in self._all_params:
            metrics_key = metric_params.key_to_class_for_registry
            metric_class = metrics_registry.get(metrics_key)
            self.all_classes.append(metric_class(params=metric_params))

    @staticmethod
    def get_tensors_from_all_tensors(all_tensors: Dict[str, torch.Tensor], tensor_names: Optional[Union[str, List[str]]]
                                     ) -> Optional[torch.Tensor]:
        if tensor_names is not None:
            if not isinstance(tensor_names, list):
                tensor_names = [tensor_names]
            tensors = [all_tensors[n] for n in tensor_names]
            return fix_y_size(tensors, tensor_names)
        return None

    def _get_tensors_for_metric_calculation(self, all_tensors: Dict[str, torch.Tensor], params: AbstractMetricParams
                                            ) -> Dict[str, torch.Tensor]:
        name_to_tensor_names = {'y_true': params.tensor_names_true,
                                'y_pred': params.tensor_names_pred,
                                'keep_mask': params.tensor_names_keep_mask,
                                'sample_weights': params.tensor_names_sample_weights}
        return {name: self.get_tensors_from_all_tensors(all_tensors, tensor_names) for name, tensor_names in
                name_to_tensor_names.items()}

    def forward(self, all_tensors: Dict[str, torch.Tensor]):
        """
        Calculate the loss values for all losses and apply weights
        :param all_tensors: a dict of tensors, including true and predicted values
        """
        for metric_class in self.all_classes:
            tensors_for_metric_calculation = self._get_tensors_for_metric_calculation(all_tensors, metric_class.params)
            metric_class(**tensors_for_metric_calculation)

    def _get_all_metric_instances(self) -> List[AbstractMetric]:
        return self.all_classes

    def summarize(self):
        for metric_instance in self._get_all_metric_instances():
            metric_instance.summarize()

    def reset(self, device: Optional[torch.device] = None):
        for metric_instance in self._get_all_metric_instances():
            metric_instance.reset()
        if device is not None:
            self.to(device=device)

    def save_tensorboard_logs(self, epoch: int, summary_writer: SummaryWriter):
        for metric_instance in self._get_all_metric_instances():
            metric_instance.save_tensorboard_logs(epoch=epoch, summary_writer=summary_writer)
        summary_writer.flush()

    def get_logs(self) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """Get a dict of the model metrics/losses results"""
        metric_name_to_results = dict()
        for metric_instance in self._get_all_metric_instances():
            name_for_logs = metric_instance.params.name_for_logs
            results = metric_instance.epoch_results
            if variable_is_number(results) or isinstance(results, pd.DataFrame):
                metric_name_to_results[name_for_logs] = results
            elif isinstance(results, np.ndarray):
                column_names = get_column_names(column_names=metric_instance.params.column_names,
                                                name_for_logs=name_for_logs, results_len=len(results),
                                                add_name_for_logs_prefix=True)
                metric_name_to_results[name_for_logs] = dict(zip(column_names, results))
            elif isinstance(results, dict) and all([variable_is_number(v) for v in results.values()]):
                metric_name_to_results.update(results)
            else:
                raise TypeError('epoch_results has to be either a number, a numpy array or a dictionary with numbers!')
        return metric_name_to_results

    def set_reduction(self, reduction: Reduction):
        for metric_instance in self._get_all_metric_instances():
            metric_instance.set_reduction(reduction)

    def set_reduction_dim(self, reduction_dim: Optional[int] = None):
        for metric_instance in self._get_all_metric_instances():
            metric_instance.set_reduction_dim(reduction_dim)

    def to(self,  *args, **kwargs):
        for metric_class in self.all_classes:
            metric_class.to(*args, **kwargs)


class LossesCombined(MetricsCombined):
    def _init_classes(self):
        self.all_classes = list()
        for loss_params in self._all_params:
            loss_key = loss_params.key_to_class_for_registry
            loss_class = metrics_registry.get(loss_key)
            loss_class_instance = loss_class(params=loss_params)
            self.all_classes.append(LossFunctionAndWeight(loss_function=loss_class_instance, weight=loss_params.weight))

    def forward(self, all_tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate the loss values for all losses and apply weights
        :param all_tensors: a dict of all tensors, including predicted and true labels
        :return: the total loss value(s)
        """
        total_loss = list()
        for loss_function_and_weight in self.all_classes:
            loss_function = loss_function_and_weight.loss_function
            weight = loss_function_and_weight.weight
            tensors_for_loss_calculation = self._get_tensors_for_metric_calculation(all_tensors, loss_function.params)
            loss_val = loss_function(**tensors_for_loss_calculation)
            total_loss.append(loss_val * weight)
        return torch.sum(torch.stack(total_loss, dim=0), dim=0, dtype=torch.float64)

    def _get_all_metric_instances(self) -> List[AbstractMetric]:
        return [loss_function_and_weight.loss_function for loss_function_and_weight in self.all_classes]

    def _calculate_total_loss(self) -> float:
        losses = [loss_function_and_weight.loss_function.epoch_results * loss_function_and_weight.weight
                  for loss_function_and_weight in self.all_classes]
        if isinstance(losses[0], (float, np.float32)):
            return np.nansum(losses)
        return pd.concat(losses, axis=1).sum(axis=1).to_frame().rename(columns={0: self._name})

    def save_tensorboard_logs(self, epoch: int, summary_writer: Optional[SummaryWriter] = None):
        """
        The loss may be a linear combination of a few different losses. In that case we'll want to save the value of
        each of the individual losses, and in addition save the value of the total loss.
        """
        if len(self.all_classes) > 1:
            super().save_tensorboard_logs(epoch=epoch, summary_writer=summary_writer)
        if summary_writer is not None:
            summary_writer.add_scalar(tag=self._name, scalar_value=self._calculate_total_loss(), global_step=epoch)

    def get_logs(self) -> Dict[str, Union[float, np.ndarray, Dict, pd.DataFrame]]:
        logs = {self._name: self._calculate_total_loss()}

        # if there's more than one loss, we'll save the results from each one separately in addition to the total loss
        if len(self.all_classes) > 1:
            logs.update(super().get_logs())
        return logs

    def to(self,  *args, **kwargs):
        for loss_class in self.all_classes:
            loss_class.loss_function.to(*args, **kwargs)

