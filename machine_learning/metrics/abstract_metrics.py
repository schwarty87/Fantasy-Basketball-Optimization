from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, List, Union

import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from machine_learning.enums import Reduction
from machine_learning.params.abstract_params import AbstractMetricParams
from machine_learning.utils import variable_is_number


def get_column_names(column_names: Optional[str], name_for_logs: str, results_len: int,
                     add_name_for_logs_prefix: bool) -> List[str]:
    column_names = column_names if column_names is not None else list(range(results_len))
    if add_name_for_logs_prefix:
        column_names = [f'{name_for_logs}_{n}' for n in column_names]
    return column_names


def fix_y_size(y: Union[torch.Tensor, List[torch.Tensor]], tensor_names: Union[str, List[str]]) -> torch.Tensor:
    if isinstance(tensor_names, list):
        if len(tensor_names) == 1:
            return y[0]
        else:
            return torch.concat(y, dim=-1)
    return y


class AbstractMetric(torch.nn.Module, ABC):
    def __init__(self, params: AbstractMetricParams):
        """
        Abstract class for calculating loss and metrics
        :param params: the params for calculating the values
        """
        super().__init__()
        self.params = params
        self._is_mean_reduction = Reduction.is_mean_reduction(params.reduction)
        self._reduction_sum_function = partial(torch.nansum, dim=params.reduction_dim, dtype=torch.float64)
        self.reduces_to_single_value = None
        self.reset()

    def forward(self, y_pred: Union[torch.Tensor, List[torch.Tensor]],
                y_true: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
                keep_mask: Optional[torch.Tensor] = None, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the loss/metric values for the given batch.
        :param y_pred: The predicted tensors
        :param y_true: The true labels tensors
        :param keep_mask: A mask, the same size as y_true, showing which values to use in the loss calculation
        :param sample_weights: The weights for each sample
        :return: The loss/metric value(s) for the current batch
        """
        errors_unreduced = self._error_unreduced(y_true, y_pred)
        errors_weighted = self._apply_weights(errors_unreduced, sample_weights)
        errors_masked = self._mask_errors(errors_weighted, keep_mask)
        errors_postprocessed, keep_mask_postprocessed, sample_weights_postprocessed = \
            self._postprocess_errors(errors_masked, keep_mask, sample_weights)
        errors_sums, num_samples = self._reduce_errors_and_count_samples(errors_postprocessed,
                                                                         keep_mask=keep_mask_postprocessed,
                                                                         sample_weights=sample_weights_postprocessed)
        self._add_batch_results_to_epoch_results(errors_sums, num_samples)
        errors_reduced = errors_sums / num_samples if self._is_mean_reduction else errors_sums
        return errors_reduced

    @abstractmethod
    def _error_unreduced(self, y_true, y_pred):
        raise NotImplementedError

    @staticmethod
    def _apply_weights(errors: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if sample_weights is None:
            return errors
        while errors.dim() > sample_weights.dim():
            sample_weights = sample_weights.unsqueeze(-1)
        return torch.mul(errors, sample_weights)

    @staticmethod
    def _mask_errors(errors: torch.Tensor, keep_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if keep_mask is None:
            return errors
        errors = torch.where(keep_mask.to(bool), errors, torch.nan)
        return errors

    def _postprocess_errors(self, errors: torch.Tensor, keep_mask: torch.Tensor,
                            sample_weights: Optional[torch.Tensor] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.params.accumulate and errors.ndim == 3:
            # TODO maybe move this to only run for basketball projects, and only when a sample contains all players in a game
            errors = torch.reshape(errors, (errors.shape[0] * errors.shape[1], -1))
            keep_mask = torch.reshape(keep_mask, (keep_mask.shape[0] * keep_mask.shape[1], -1))
            if sample_weights is not None:
                sample_weights = torch.reshape(sample_weights, (sample_weights.shape[0] * sample_weights.shape[1], -1))
        return errors, keep_mask, sample_weights

    def _reduce_errors_and_count_samples(self, errors: torch.Tensor, keep_mask: Optional[torch.Tensor] = None,
                                         sample_weights: Optional[torch.Tensor] = None
                                         ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # whether we want a mean or sum reduction in the end, we'll calculate the sums of the errors and mask, and only
        # divide (if needed) later on. This way we could also add the values to the epoch results
        if Reduction.reduction_is_none(self.params.reduction):
            return errors, None
        errors_sum = self._reduction_sum_function(errors)
        if self._is_mean_reduction:
            if sample_weights is not None:
                values_to_reduce = sample_weights.clone()
            elif keep_mask is not None:
                keep_mask = self._reshape_keep_mask(keep_mask)
                values_to_reduce = keep_mask.clone()
            else:
                # if we don't use a mask then we use all values in the error tensor
                values_to_reduce = torch.ones_like(errors)

            # the errors are masked by keep_mask in _mask_errors(), so by removing all values where error is nan, we
            # also mask values_to_reduce with keep_mask
            values_to_reduce = torch.where(torch.isnan(errors), 0., values_to_reduce)
            num_samples = self._reduction_sum_function(values_to_reduce)
        else:
            num_samples = None
        return errors_sum, num_samples

    def _add_batch_results_to_epoch_results(self, batch_error_sums: torch.Tensor,
                                            batch_num_samples: Optional[torch.Tensor]):
        """
        Add the results from the batch to the epoch results
        :param batch_error_sums: The errors after applying the sum reduction
        :param batch_num_samples: The number of samples that were used to calculate each value in batch_error_sums.
               Should be the same size as batch_error_sums
        """
        if self.params.accumulate:
            self._epoch_error_sums = torch.concat([self._epoch_error_sums, batch_error_sums])
            if batch_num_samples is not None:
                self._epoch_num_samples = torch.concat([self._epoch_num_samples, batch_num_samples])
        else:
            self._epoch_error_sums = self._epoch_error_sums + batch_error_sums
            if batch_num_samples is not None:
                self._epoch_num_samples = self._epoch_num_samples + batch_num_samples

    @staticmethod
    def _reshape_keep_mask(keep_mask: torch.Tensor) -> torch.Tensor:
        return keep_mask

    def _summarize_accumulate(self, results: torch.Tensor):
        # if we're accumulating, then we expect reduction to be NONE or reduction_dim to be -1 (value per sample)
        if results.ndim > 1 and results.shape[1] == len(self.params.column_names):
            # value for each column
            column_names = get_column_names(column_names=self.params.column_names,
                                            name_for_logs=self.params.name_for_logs, results_len=results[0].shape[-1],
                                            add_name_for_logs_prefix=True)
        else:
            # single value per sample
            column_names = [self.params.name_for_logs]
            if self._is_mean_reduction:
                results = results / self._epoch_num_samples
        self.epoch_results = pd.DataFrame(results.cpu().detach().numpy(), columns=column_names)

    def _summarize_sum(self, results: torch.Tensor):
        if self._is_mean_reduction:
            results = results / self._epoch_num_samples
        self.epoch_results = results.cpu().detach().numpy()
        if self.reduces_to_single_value:
            self.epoch_results = self.epoch_results[0]

    def summarize(self):
        results = self._epoch_error_sums
        if self.params.accumulate:
            self._summarize_accumulate(results=results)
        else:
            self._summarize_sum(results=results)

    def reset(self):
        self.epoch_results = None

        # reset buffer values
        # notice that register_buffer creates the tensor on the cpu by default. If we want to run on gpu we'll have to
        # move them to the gpu after running reset()
        if self.params.accumulate:
            base_buffer_values = torch.empty(size=(0,), requires_grad=False, dtype=torch.float32)
        elif Reduction.reduction_is_none(self.params.reduction) or (self.params.column_names is None) or \
                (self.params.reduction_dim is None):
            base_buffer_values = torch.tensor([0.], requires_grad=False, dtype=torch.float32)
        else:
            base_buffer_values = torch.zeros(len(self.params.column_names), requires_grad=False, dtype=torch.float32)
        self.register_buffer('_epoch_error_sums', base_buffer_values)
        if self._is_mean_reduction:
            self.register_buffer('_epoch_num_samples', torch.clone(base_buffer_values))
        self.reduces_to_single_value = Reduction.reduces_to_a_single_value(member=self.params.reduction,
                                                                           reduction_dim=self.params.reduction_dim)

    def save_tensorboard_logs(self, epoch: int, summary_writer: Optional[SummaryWriter] = None):
        if variable_is_number(self.epoch_results):
            if summary_writer is not None and self.epoch_results is not None:
                summary_writer.add_scalar(tag=self.params.name_for_logs, scalar_value=self.epoch_results,
                                          global_step=epoch)
        elif isinstance(self.epoch_results, np.ndarray):
            # if we save a few values (for example, one value per column) then we'll save a bar chart and save it as an
            # image instead of creating multiple graphs (one for each column)
            column_names = get_column_names(column_names=self.params.column_names,
                                            name_for_logs=self.params.name_for_logs,
                                            results_len=len(self.epoch_results),
                                            add_name_for_logs_prefix=False)
            if len(self.epoch_results) == len(column_names):
                mpl_interactive_mode = copy.deepcopy(matplotlib.rcParams['interactive'])
                matplotlib.rcParams['interactive'] = False
                fig, ax = plt.subplots()
                ax.bar(x=range(len(self.epoch_results)), height=self.epoch_results, tick_label=column_names)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
                ax.set_title(self.params.name_for_logs)
                plt.tight_layout()
                summary_writer.add_figure(tag=self.params.name_for_logs, figure=fig, global_step=epoch)
                plt.close('all')
                matplotlib.rcParams['interactive'] = mpl_interactive_mode

                if summary_writer is not None:
                    for n, r in zip(column_names, self.epoch_results):
                        summary_writer.add_scalar(tag=f'{self.params.name_for_logs}/{n}', scalar_value=r,
                                                  global_step=epoch)
            else:
                raise ValueError("the length of params.column_names doesn't match the length of epoch_results!")
        else:
            raise TypeError('self.epoch_results has to be either a number or numpy array!')


@dataclass
class LossFunctionAndWeight:
    loss_function: AbstractMetric
    weight: float


