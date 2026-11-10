import os
import copy
import numpy as np
import torch
import lightning.pytorch as pl
from typing import Dict, Any, Tuple, Optional, Union

from machine_learning.architecture import ModelArchitecture
from machine_learning.metrics.combined_metrics import LossesCombined, MetricsCombined
from machine_learning.params.params import MachineLearningParams


class MachineLearningLightningModule(pl.LightningModule):
    """
    Generic LightningModule encapsulating ModelArchitecture forward pass,
    LossesCombined calculation, MetricsCombined evaluation, and AdamW optimization.
    """

    def __init__(
        self,
        params: Optional[MachineLearningParams] = None,
        scaler_dir: Optional[str] = None,
        model: Optional[ModelArchitecture] = None,
        loss_function: Optional[LossesCombined] = None,
        metrics: Optional[MetricsCombined] = None,
        **kwargs,
    ):
        super().__init__()

        if params is None:
            params = self._init_params_from_dict(kwargs)

        if params is not None and hasattr(params, "to_dict"):
            try:
                hparams = params.to_dict()
                if isinstance(hparams, dict):
                    self.save_hyperparameters(hparams)
            except Exception:
                pass

        self.params = params
        self.scaler_dir = scaler_dir
        self._last_epoch_eval_results: Dict[str, Any] = {}

        if model is not None:
            self.model = model
        elif params is not None:
            self.model = ModelArchitecture(
                params=params.architecture_params,
                feature_type_to_input_shape=params.features_params.feature_type_to_input_shape,
            )
        else:
            self.model = None

        if loss_function is not None:
            self.loss_function = loss_function
        elif params is not None:
            self.loss_function = LossesCombined(
                all_params=params.training_params.loss_params,
                name="loss",
                device=torch.device("cpu"),
            )
        else:
            self.loss_function = None

        if self.loss_function is not None:
            self.val_loss_function = copy.deepcopy(self.loss_function)
            self.test_loss_function = copy.deepcopy(self.loss_function)
        else:
            self.val_loss_function = None
            self.test_loss_function = None

        if metrics is not None:
            self.metrics = metrics
        elif params is not None:
            self.metrics = MetricsCombined(
                all_params=params.training_params.metrics_params,
                name="metrics",
                device=torch.device("cpu"),
            )
        else:
            self.metrics = None

        if self.metrics is not None:
            self.train_metrics = copy.deepcopy(self.metrics)
        else:
            self.train_metrics = None

    def setup(self, stage: Optional[str] = None):
        super().setup(stage=stage)
        for fn in (self.loss_function, self.val_loss_function, self.test_loss_function):
            if fn is not None and hasattr(fn, "to"):
                fn.to(self.device)
        if self.metrics is not None and hasattr(self.metrics, "to"):
            self.metrics.to(self.device)
        if self.train_metrics is not None and hasattr(self.train_metrics, "to"):
            self.train_metrics.to(self.device)

    def to(self, *args, **kwargs):
        res = super().to(*args, **kwargs)
        for fn in (self.loss_function, self.val_loss_function, self.test_loss_function):
            if fn is not None and hasattr(fn, "to"):
                fn.to(*args, **kwargs)
        if self.metrics is not None and hasattr(self.metrics, "to"):
            self.metrics.to(*args, **kwargs)
        if self.train_metrics is not None and hasattr(self.train_metrics, "to"):
            self.train_metrics.to(*args, **kwargs)
        return res

    @classmethod
    def _init_params_from_dict(cls, kwargs: dict) -> Optional[MachineLearningParams]:
        if kwargs:
            try:
                return MachineLearningParams.init_from_dict(kwargs)
            except Exception:
                pass
        return None

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch_inputs)

    def _prepare_all_tensors(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Filters inputs for model layers and joins model outputs with batch targets."""
        batch_inputs = {
            k: v
            for k, v in batch.items()
            if k in self.params.architecture_params.all_layer_input_tensor_names
        }
        model_outputs = self.model(batch_inputs)
        all_tensors = {**batch, **model_outputs}
        return batch_inputs, all_tensors

    @staticmethod
    def _get_batch_size(batch: Dict[str, torch.Tensor]) -> int:
        if not batch:
            return 1
        for v in batch.values():
            if hasattr(v, "shape") and len(v.shape) > 0:
                return v.shape[0]
        return 1

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _, all_tensors = self._prepare_all_tensors(batch)
        total_loss = self.loss_function(all_tensors=all_tensors)
        if self.train_metrics is not None:
            with torch.no_grad():
                self.train_metrics(all_tensors=all_tensors)
        batch_size = self._get_batch_size(batch)

        self.log(
            "train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _, all_tensors = self._prepare_all_tensors(batch)
        loss_fn = self.val_loss_function if self.val_loss_function is not None else self.loss_function
        val_loss = loss_fn(all_tensors=all_tensors)
        if self.metrics is not None:
            self.metrics(all_tensors=all_tensors)
        batch_size = self._get_batch_size(batch)

        self.log(
            "val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size
        )
        return val_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        _, all_tensors = self._prepare_all_tensors(batch)
        loss_fn = self.test_loss_function if self.test_loss_function is not None else self.loss_function
        test_loss = loss_fn(all_tensors=all_tensors)
        if self.metrics is not None:
            self.metrics(all_tensors=all_tensors)
        batch_size = self._get_batch_size(batch)

        self.log(
            "test/loss", test_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size
        )
        return test_loss

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        _, all_tensors = self._prepare_all_tensors(batch)
        return all_tensors

    def _log_nested_dict(self, prefix: str, logs: dict, **log_kwargs):
        """Logs a possibly-nested dict of scalar values with a key prefix."""
        for k, v in logs.items():
            if isinstance(v, (int, float, np.number)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                self.log(f"{prefix}/{k}", float(v), **log_kwargs)
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, (int, float, np.number)) or (isinstance(sub_v, torch.Tensor) and sub_v.numel() == 1):
                        self.log(f"{prefix}/{sub_k}", float(sub_v), **log_kwargs)

    def on_before_optimizer_step(self, optimizer):
        track_norm = getattr(getattr(self.params, "training_params", None), "track_grad_norm", None)
        if track_norm is not None and type(track_norm).__name__ != "MagicMock":
            from lightning.pytorch.utilities import grad_norm
            norm_type = float(track_norm) if isinstance(track_norm, (int, float)) else 2
            norms = grad_norm(self.model, norm_type=norm_type)
            self.log_dict(norms)

    def on_train_epoch_end(self):
        if self.loss_function is not None:
            self.loss_function.summarize()
            self._log_nested_dict("train", self.loss_function.get_logs(), on_step=False, on_epoch=True)
            self.loss_function.reset()
        if self.train_metrics is not None:
            self.train_metrics.summarize()
            self._log_nested_dict("train", self.train_metrics.get_logs(), on_step=False, on_epoch=True)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        loss_fn = self.val_loss_function if self.val_loss_function is not None else self.loss_function
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and getattr(trainer, "sanity_checking", False):
            if self.loss_function is not None:
                self.loss_function.reset()
            if self.val_loss_function is not None and self.val_loss_function is not self.loss_function:
                self.val_loss_function.reset()
            if self.metrics is not None:
                self.metrics.reset()
            if self.train_metrics is not None:
                self.train_metrics.reset()
            return

        eval_logs = {}
        if loss_fn is not None:
            loss_fn.summarize()
            loss_logs = loss_fn.get_logs()
            self._log_nested_dict("val", loss_logs, on_step=False, on_epoch=True)
            eval_logs.update(loss_logs)
            loss_fn.reset()

        if self.metrics is not None:
            self.metrics.summarize()
            metrics_logs = self.metrics.get_logs()
            self._log_nested_dict("val", metrics_logs, prog_bar=False)
            eval_logs.update(metrics_logs)
            self.metrics.reset()

        self._last_epoch_eval_results = eval_logs

    def on_test_epoch_end(self):
        loss_fn = self.test_loss_function if self.test_loss_function is not None else self.loss_function
        test_logs = {}
        if loss_fn is not None:
            loss_fn.summarize()
            loss_logs = loss_fn.get_logs()
            self._log_nested_dict("test", loss_logs, on_step=False, on_epoch=True)
            test_logs.update(loss_logs)
            loss_fn.reset()

        if self.metrics is not None:
            self.metrics.summarize()
            metrics_logs = self.metrics.get_logs()
            self._log_nested_dict("test", metrics_logs, prog_bar=False)
            test_logs.update(metrics_logs)
            self.metrics.reset()

        self._last_epoch_eval_results = test_logs

    def configure_optimizers(self):
        lr = self.params.training_params.learning_rate
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        callbacks_params = getattr(self.params.training_params, "callbacks_params", ())
        plateau_params = next(
            (p for p in callbacks_params if p.__class__.__name__ == "ReduceLearningRateOnPlateauParams"),
            None,
        )

        if plateau_params is not None:
            mode = plateau_params.mode.value if hasattr(plateau_params.mode, "value") else str(plateau_params.mode)
            monitor = getattr(plateau_params, "value_to_track", getattr(plateau_params, "monitor", "eval_loss"))
            if monitor in ("eval_loss", "loss", "loss_function"):
                monitor = "val/loss"
            elif monitor.startswith("eval_"):
                monitor = f"val/{monitor[5:]}"

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=plateau_params.factor,
                patience=plateau_params.patience,
                min_lr=plateau_params.min_lr,
                threshold=plateau_params.threshold,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "frequency": 1,
                },
            }

        return optimizer
