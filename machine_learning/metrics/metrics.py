import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from sklearn.metrics import roc_auc_score
from machine_learning.metrics.abstract_metrics import AbstractMetric
from machine_learning.registries import metrics_registry


@metrics_registry.register
class MAEMetric(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        return torch.abs(y_true - y_pred)


@metrics_registry.register
class MAPEMetric(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        abs_diff = torch.abs(y_true - y_pred)
        return abs_diff / torch.clamp(torch.abs(y_true), min=1e-8)


@metrics_registry.register
class BinaryCrossEntropyLoss(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        if self.params.logits:
            y_pred = 1 / (1 + torch.exp(-1 * y_pred))
        return - (y_true * (torch.log(y_pred)) + (1 - y_true) * torch.log((1 - y_pred)))


@metrics_registry.register
class ROCAUC(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        return torch.stack([y_true, y_pred], dim=-1)

    @staticmethod
    def _apply_weights(errors: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return errors

    def summarize(self):
        y_true, y_pred = torch.tensor_split(self._epoch_error_sums, sections=2, dim=1)
        y_true = y_true.cpu().detach().numpy().flatten()
        y_pred = y_pred.cpu().detach().numpy().flatten()
        is_finite_inds = np.isfinite(y_true) & np.isfinite(y_pred)
        self.epoch_results = roc_auc_score(y_true[is_finite_inds], y_pred[is_finite_inds])


@metrics_registry.register
class BinaryClassificationMetrics(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        true_positive = y_true == 1
        true_negative = y_true == 0
        pred_positive = y_pred > (0.5 if not self.params.logits else 0)
        pred_negative = torch.logical_not(pred_positive)

        TP = torch.logical_and(true_positive, pred_positive)
        FP = torch.logical_and(true_negative, pred_positive)
        TN = torch.logical_and(true_negative, pred_negative)
        FN = torch.logical_and(true_positive, pred_negative)

        return torch.stack([TP, FP, TN, FN], dim=-1)

    @staticmethod
    def _apply_weights(errors: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return errors

    def _reduce_errors_and_count_samples(self, errors: torch.Tensor, *args, **kwargs):
        del args
        del kwargs
        return torch.stack([torch.sum(t) for t in torch.tensor_split(errors, sections=4, dim=-1)]), None

    def summarize(self):
        TP, FP, TN, FN = self._epoch_error_sums
        results = {'sensitivity': TP / (TP + FN),
                   'specificity': TN / (TN + FP),
                   'precision': TP / (TP + FP),
                   'accuracy': (TP + TN) / (TP + FP + TN + FN),
                   'f1_score': 2 * TP / (2 * TP + FP + FN)
                   }
        self.epoch_results = {k: v.item() for k, v in results.items()}

    def save_tensorboard_logs(self, epoch: int, summary_writer: Optional[SummaryWriter] = None):
        if summary_writer is not None:
            for metric_name, metric_value in self.epoch_results.items():
                summary_writer.add_scalar(tag=metric_name, scalar_value=metric_value, global_step=epoch)


@metrics_registry.register
class MSELoss(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        error = (y_true - y_pred) ** 2
        return error


@metrics_registry.register
class QuantileRegressionLoss(AbstractMetric):
    def _error_unreduced(self, y_true, y_pred):
        e = y_true - y_pred
        return torch.max(self.params.quantile * e, (self.params.quantile - 1) * e)


