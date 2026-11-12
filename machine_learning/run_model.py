import os
import contextlib
import copy
import logging
import torch
import torch.nn as nn
import numpy as np
import traceback
import pandas as pd
from typing import Type, Dict, Optional, Union, Tuple, Any, List
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from machine_learning.constants import METADATA
from machine_learning.architecture import ModelArchitecture
from machine_learning.callbacks import CallbacksManager, training_callbacks, ModelCheckpoint
from machine_learning.enums import ModelMode, DataSplitType
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning.metrics.combined_metrics import LossesCombined, MetricsCombined
from machine_learning.lightning_data_module import MachineLearningDataModule
from machine_learning.lightning_model import MachineLearningLightningModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ModelCheckpoint as PLModelCheckpoint,
    EarlyStopping as PLEarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
from machine_learning.params.abstract_params import AbstractMetricParams, AbstractLossParams, \
    AbstractMetricsAndCallbacksParams
from machine_learning.params.callbacks_params import ModelCheckpointParams
from machine_learning.params.layers_params import EmbeddingsLayerParams
from machine_learning.params.params import MachineLearningParams, FeaturesParams
from machine_learning.registries import callbacks_registry
from machine_learning.utils import RANDOM_SEED, models_dir, get_next_model_id

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

OPTIMIZER_TYPE = Optional[torch.optim.Optimizer]


class RunModel:
    def __init__(self, params: MachineLearningParams, project_name: str, description: Optional[str] = None,
                 use_lightning: bool = False):
        self.params = params
        self.project_name = project_name
        self._description = description
        self.use_lightning = use_lightning

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info('running on gpu')
        else:
            self.device = torch.device("cpu")
            logging.info('running on cpu')

        self._init_models_dir()
        if not os.path.isdir(self._models_dir):
            os.makedirs(self._models_dir, exist_ok=True)
        self._init_model_id()
        self._model_dir = os.path.join(self._models_dir, str(self.model_id))
        os.makedirs(self._model_dir, exist_ok=True)
        if self.params.continue_model_id is not None:
            self._continue_model_dir = os.path.join(self._models_dir, str(self.params.continue_model_id))
        else:
            self._continue_model_dir = None

        self.create_data_loaders()
        self.update_categorical_params()
        self._create_model()

        if not self.use_lightning:
            self._loss_function = self._define_loss_function()
            self._metrics = self._define_metrics()
        else:
            model = getattr(self, "model", None)
            self._loss_function = getattr(model, "loss_function", None)
            self._metrics = getattr(model, "metrics", None)
            self.trainer: Optional[Any] = None

        all_params = [getattr(self.params.training_params, "loss_params", ()),
                      getattr(self.params.training_params, "metrics_params", ()),
                      getattr(self.params.training_params, "callbacks_params", ())]
        self._all_tensor_names_for_losses_metrics_callbacks = \
            list(set(sum([curr_params.all_tensor_names for p in all_params for curr_params in p if hasattr(curr_params, "all_tensor_names")], [])))

    @property
    def lightning_module(self):
        return self.model

    @lightning_module.setter
    def lightning_module(self, val):
        self.model = val

    def _data_class(self):
        if self.use_lightning:
            return MachineLearningDataModule
        return MachineLearningDatasetCombined

    def _model_class(self):
        if self.use_lightning:
            return MachineLearningLightningModule
        return ModelArchitecture

    def _init_models_dir(self):
        self._models_dir = models_dir(self.project_name)

    def _init_model_id(self):
        if self.params.continue_model_id is not None:
            self.model_id = self.params.continue_model_id
        else:
            self.model_id = get_next_model_id(self.project_name)

    def update_categorical_params(self):
        # TODO deal with this!
        for p in self.params.architecture_params.layer_params_list:
            if isinstance(p, EmbeddingsLayerParams):
                categorical_dataset = \
                    MachineLearningDatasetCombined.get_single_dataset_from_data_loaders(
                        data_loaders=self.data_loaders,
                        data_split_type=DataSplitType.TRAIN,
                        feature_type=p.input_tensor_name
                    )
                num_unique_values_per_feature = categorical_dataset.num_unique_values_per_feature()
                update_dict = {'num_embeddings': num_unique_values_per_feature.astype(int).tolist()}
                p.update_after_loading_data(update_dict)

    def create_data_loaders(self, scalers: Optional[Dict[str, StandardScaler]] = None,
                            label_encoders: Optional[OrdinalEncoder] = None):
        data_cls = self._data_class()
        if self.use_lightning:
            self.data_module = data_cls(
                params=self.params,
                project_name=self.project_name,
                model_dir=self._model_dir,
            )
            self.data_module.setup(stage="fit")
            self.data_loaders = self.data_module.data_loaders
            self.dataset_combined = getattr(self.data_module, 'dataset_combined', None)
        else:
            if self.params.continue_model_id is not None:
                scalers, label_encoders = data_cls.load_transformers(self._continue_model_dir)
                if isinstance(getattr(self.params, "features_params", None), FeaturesParams):
                    MachineLearningDatasetCombined.assert_transformers_complete(
                        self.params.features_params, scalers, label_encoders
                    )
            self.dataset_combined = data_cls(
                params=self.params.features_params,
                training_params=self.params.training_params,
                project_name=self.project_name,
                dataset_name_to_scalers=scalers,
                dataset_name_to_categorical_encoder=label_encoders)
            self.data_loaders = self.dataset_combined.create_data_loaders(
                batch_size=self.params.training_params.batch_size,
                num_workers=self.params.training_params.dataloader_num_workers)

    def _find_checkpoint_path(self, model_dir: Optional[str]) -> Optional[str]:
        if not model_dir or not os.path.isdir(model_dir):
            return None
        last_ckpt = os.path.join(model_dir, "last.ckpt")
        if os.path.isfile(last_ckpt):
            return last_ckpt
        ckpts = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".ckpt")]
        if ckpts:
            return sorted(ckpts)[-1]
        model_pt = os.path.join(model_dir, "model.pt")
        if os.path.isfile(model_pt):
            return model_pt
        return None

    def _load_lightning_checkpoint_weights(self):
        ckpt_path = self._find_checkpoint_path(self._continue_model_dir)
        if ckpt_path is None:
            logging.warning(f"No checkpoint found in continue_model_dir: {self._continue_model_dir}")
            return

        logging.info(f"Loading weights from checkpoint: {ckpt_path}")
        try:
            if ckpt_path.endswith(".ckpt"):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt)
                self.model.load_state_dict(state_dict, strict=False)
            elif ckpt_path.endswith(".pt"):
                loaded_model = ModelCheckpoint.load_model(ckpt_path)
                if hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
                    self.model.model.load_state_dict(loaded_model.state_dict())
                else:
                    self.model.model = loaded_model
        except Exception as e:
            logging.exception(f"Could not load checkpoint weights from {ckpt_path}: {e}")
            raise

    def _create_model(self):
        model_cls = self._model_class()
        if self.use_lightning:
            try:
                scaler_dir = (
                    self._continue_model_dir
                    if (self.params.continue_model_id is not None and self._continue_model_dir and os.path.isdir(self._continue_model_dir))
                    else self._model_dir
                )
                self.model = model_cls(params=self.params, scaler_dir=scaler_dir)
            except Exception as e:
                logging.warning(f"Could not create lightning model: {e}")
                self.model = None
            if self.model is not None and self.params.continue_model_id is not None and self._continue_model_dir:
                self._load_lightning_checkpoint_weights()
        else:
            if self.params.continue_model_id is not None:
                self.model = ModelCheckpoint.load_model(os.path.join(self._continue_model_dir, 'model.pt'))
                if self.model is not None and hasattr(self.model, "modules"):
                    for m in self.model.modules():
                        if isinstance(m, nn.Embedding) and m.padding_idx is None and m.num_embeddings > 0:
                            with torch.no_grad():
                                pad_row = m.weight.new_zeros((1, m.embedding_dim))
                                m.weight = nn.Parameter(
                                    torch.cat([m.weight.detach(), pad_row], dim=0),
                                    requires_grad=m.weight.requires_grad,
                                )
                            m.num_embeddings = m.weight.shape[0]
                            m.padding_idx = m.num_embeddings - 1
            else:
                try:
                    self.model = model_cls(
                        self.params.architecture_params,
                        feature_type_to_input_shape=self.params.features_params.feature_type_to_input_shape,
                    )
                except Exception as e:
                    logging.warning(f"Could not create model architecture: {e}")
                    self.model = None

            if self.model is not None and hasattr(self.model, "to"):
                self.model.to(self.device)

        self._initial_epoch = 0

    def _create_model_architecture(self):
        return self._create_model()

    @staticmethod
    def _machine_learning_datasets_combined_class() -> Type[MachineLearningDatasetCombined]:
        return MachineLearningDatasetCombined

    def _model_architecture_class(self):
        return self._model_class()

    @staticmethod
    def _losses_combined_class() -> Type[LossesCombined]:
        return LossesCombined

    @staticmethod
    def _metrics_combined_class() -> Type[MetricsCombined]:
        return MetricsCombined

    @staticmethod
    def _callbacks_combined_class() -> Type[CallbacksManager]:
        return CallbacksManager

    @staticmethod
    def _resolve_lightning_monitor_and_mode(p: Any) -> Tuple[str, str]:
        """Translate legacy metric tracking names (e.g. 'eval_loss', 'eval_mae') to Lightning logger keys ('val/loss', 'val/mae')."""
        monitor = getattr(p, "value_to_track", "eval_loss")
        if monitor in ("eval_loss", "loss", "loss_function"):
            monitor = "val/loss"
        elif monitor.startswith("eval_"):
            monitor = f"val/{monitor[5:]}"
        mode = "min" if getattr(p, "track_min_or_max", True) else "max"
        return monitor, mode

    def _build_lightning_callbacks(self) -> List[Any]:
        callbacks = [LearningRateMonitor(logging_interval="epoch")]
        callbacks_params = getattr(self.params.training_params, "callbacks_params", ())

        has_checkpoint = False
        has_early_stopping = False
        for p in callbacks_params:
            class_name = p.__class__.__name__
            if "ModelCheckpoint" in class_name:
                has_checkpoint = True
                monitor, mode = self._resolve_lightning_monitor_and_mode(p)
                callbacks.append(
                    PLModelCheckpoint(
                        dirpath=self._model_dir,
                        filename="model_{epoch:02d}_{val_loss:.3f}",
                        monitor=monitor,
                        mode=mode,
                        save_top_k=1,
                        save_last=True,
                    )
                )
            elif "EarlyStopping" in class_name:
                has_early_stopping = True
                monitor, mode = self._resolve_lightning_monitor_and_mode(p)
                patience = getattr(p, "patience", 10)
                callbacks.append(
                    PLEarlyStopping(
                        monitor=monitor,
                        patience=patience,
                        mode=mode,
                        verbose=True,
                    )
                )

        if not has_checkpoint:
            callbacks.append(
                PLModelCheckpoint(
                    dirpath=self._model_dir,
                    filename="model_{epoch:02d}_{val_loss:.3f}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                )
            )
        if not has_early_stopping:
            callbacks.append(
                PLEarlyStopping(
                    monitor="val/loss",
                    patience=10,
                    mode="min",
                    verbose=True,
                )
            )

        if getattr(self.params.training_params, "use_swa", False) is True:
            try:
                from lightning.pytorch.callbacks import StochasticWeightAveraging
                swa_lrs = getattr(self.params.training_params, "swa_lrs", 1e-3)
                callbacks.append(StochasticWeightAveraging(swa_lrs=swa_lrs))
            except Exception as e:
                logging.warning(f"Could not initialize SWA callback: {e}")

        if getattr(self.params.training_params, "use_rich", False) is True:
            try:
                from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
                callbacks.append(RichProgressBar())
                callbacks.append(RichModelSummary(max_depth=2))
            except Exception as e:
                logging.warning(f"Could not initialize Rich callbacks: {e}")

        return callbacks

    def _define_optimizer(self) -> torch.optim.Optimizer:
        lr = self.params.training_params.learning_rate
        weight_decay = getattr(self.params.training_params, "weight_decay", 1e-4)
        if weight_decay and weight_decay > 0 and self.model is not None:
            decay_params = []
            no_decay_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or "bias" in name or "norm" in name.lower() or "embedding" in name.lower():
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            param_groups = []
            if decay_params:
                param_groups.append({"params": decay_params, "weight_decay": weight_decay})
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
            return torch.optim.AdamW(param_groups if param_groups else self.model.parameters(), lr=lr)
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _should_update_optimizer(self, batch_num: int) -> bool:
        if not self.model.training:
            return False
        if self.params.training_params.gradient_accumulation_num_batches is None:
            return True
        return ((batch_num + 1) % self.params.training_params.gradient_accumulation_num_batches == 0) or \
               ((batch_num + 1) == len(self.data_loaders[DataSplitType.TRAIN]))

    def _run_batch(self, training_mode: ModelMode, batch_num: int, batch_data: Dict[str, torch.Tensor],
                   callbacks: CallbacksManager, metrics: MetricsCombined, loss_function: LossesCombined,
                   optimizer: OPTIMIZER_TYPE = None):
        with torch.no_grad():
            callbacks.on_batch_begin(training_mode=training_mode, batch_num=batch_num)

        batch_inputs = dict()
        all_tensors = dict()
        for tensor_name, tensor in batch_data.items():
            if tensor_name in self.params.architecture_params.all_layer_input_tensor_names:
                batch_inputs[tensor_name] = tensor
            if tensor_name in self._all_tensor_names_for_losses_metrics_callbacks:
                all_tensors[tensor_name] = tensor
        self._machine_learning_datasets_combined_class().move_batch_data_to_device(batch_inputs, self.device)
        self._machine_learning_datasets_combined_class().move_batch_data_to_device(all_tensors, self.device,
                                                                                   dont_convert=[METADATA])

        if torch.cuda.is_available():
            context_manager = torch.cuda.amp.autocast(dtype=self.params.training_params.training_dtype)
        else:
            context_manager = contextlib.nullcontext()

        with context_manager:
            model_outputs = self.model(batch_inputs)
            all_tensors.update(model_outputs)
            loss = loss_function(all_tensors=all_tensors)

        if self.model.training:
            # TODO consider using torch.cuda.amp.GradScaler as shown in
            # https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
            loss.backward()

            if self._should_update_optimizer(batch_num=batch_num):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            metrics(all_tensors=all_tensors)
            callbacks.on_batch_end(training_mode=training_mode, batch_num=batch_num, all_tensors=all_tensors)

    def _run_epoch(self, training_mode: ModelMode, data_split_type: DataSplitType,
                   loss_function: LossesCombined, metrics: MetricsCombined, callbacks: CallbacksManager,
                   optimizer: OPTIMIZER_TYPE = None, profiler: Optional[torch.profiler.profile] = None,
                   data_loader: Optional[DataLoader] = None) -> \
        Dict[str, Union[float, Dict, pd.DataFrame]]:
        loss_function.reset(device=self.device)
        metrics.reset(device=self.device)

        if self.model.training:
            optimizer.zero_grad(set_to_none=True)

        loader = data_loader if data_loader is not None else self.data_loaders[data_split_type]
        for batch_num, batch_data in enumerate(loader):
            if (self.params.training_params.num_batches_per_epoch is not None) and \
                (batch_num >= self.params.training_params.num_batches_per_epoch):
                break
            if profiler is not None:
                profiler.step()
            self._run_batch(training_mode=training_mode, batch_num=batch_num, batch_data=batch_data,
                            callbacks=callbacks, metrics=metrics, loss_function=loss_function, optimizer=optimizer)

        with torch.no_grad():
            loss_function.summarize()
            loss_results = loss_function.get_logs()
            metrics.summarize()
            metrics_results = metrics.get_logs()
            logs = {**loss_results, **metrics_results}

        return logs

    def _save_tensorboard_logs(self, epoch_num: int, summary_writer: SummaryWriter):
        if self.use_lightning:
            eval_results = getattr(self.model, "_last_epoch_eval_results", {})
            for k, v in eval_results.items():
                if isinstance(v, (int, float, np.number)):
                    summary_writer.add_scalar(k, float(v), global_step=epoch_num or 0)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, np.number)):
                            summary_writer.add_scalar(f"{k}/{sub_k}", float(sub_v), global_step=epoch_num or 0)
        else:
            self._metrics.save_tensorboard_logs(epoch=epoch_num, summary_writer=summary_writer)
            self._loss_function.save_tensorboard_logs(epoch=epoch_num, summary_writer=summary_writer)

    def _save_data_to_db(self, epochs: int):
        """After training, save the model data to the db"""
        if not self.params.training_params.save_to_db:
            return
        model_dir = os.path.join(self._models_dir, str(self.model_id))
        self.params.save_to_json(os.path.join(model_dir, 'params.json'))

    def _define_loss_function(self, loss_params: Optional[Tuple[AbstractLossParams, ...]] = None, name='loss') -> LossesCombined:
        if loss_params is None:
            loss_params = self.params.training_params.loss_params
        return self._losses_combined_class()(all_params=loss_params, name=name, device=self.device)

    def _define_metrics(self, metrics_params: Optional[Tuple[AbstractMetricParams, ...]] = None, name='metrics') -> MetricsCombined:
        if metrics_params is None:
            metrics_params = self.params.training_params.metrics_params
        return self._metrics_combined_class()(all_params=metrics_params, name=name, device=self.device)

    def _define_callbacks_manager(self, is_training: bool, optimizer: OPTIMIZER_TYPE = None) -> CallbacksManager:
        callbacks_params = copy.deepcopy(self.params.training_params.callbacks_params)
        callbacks_combined_class = self._callbacks_combined_class()
        if not is_training:
            callbacks_params = tuple([p for p in callbacks_params if callbacks_registry.get(p.key_to_class_for_registry)
                                not in training_callbacks()])
        return callbacks_combined_class(
            callback_params=callbacks_params, model=self.model, optimizer=optimizer, model_dir=self._model_dir)

    def _get_dataloader(self, data_split_type: DataSplitType) -> Optional[DataLoader]:
        if hasattr(self, "data_module") and hasattr(self.data_module, "data_loaders") and self.data_module.data_loaders:
            return self.data_module.data_loaders.get(data_split_type)
        if hasattr(self, "data_loaders") and self.data_loaders:
            return self.data_loaders.get(data_split_type)
        return None

    def _define_callbacks(self, is_training: bool = True, optimizer: OPTIMIZER_TYPE = None):
        """Unified callback creator that returns Lightning callbacks or legacy CallbacksManager based on use_lightning."""
        if self.use_lightning:
            return self._build_lightning_callbacks()
        return self._define_callbacks_manager(is_training=is_training, optimizer=optimizer)

    def _stop_training(self, callbacks: CallbacksManager, epoch_num: int) -> bool:
        return callbacks.stop_training() or \
               ((self.params.training_params.num_epochs is not None) and
                (epoch_num >= (self.params.training_params.num_epochs - 1)))

    def train(self, trainer_cls=None, torch_save=None, precision=None):
        # NOTE: model_id / _model_dir are initialized once in __init__. Re-deriving them here would make
        # get_next_model_id() see the dir __init__ just created and return model_id + 1.
        seed = getattr(self.params.training_params, "seed", None)
        if seed is not None and type(seed).__name__ != "MagicMock":
            if self.use_lightning:
                pl.seed_everything(seed, workers=True)
            else:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

        # Shared pre-training persistence of transformers and params
        if hasattr(self, "data_module") and hasattr(self.data_module, "save_transformers"):
            self.data_module.save_transformers(save_dir=self._model_dir)
        elif hasattr(self, "data_loaders"):
            MachineLearningDatasetCombined.save_transformers(data_loader=self.data_loaders, save_dir=self._model_dir)

        self.params.save_to_json(os.path.join(self._model_dir, "params.json"))

        if self.use_lightning:
            if trainer_cls is None:
                trainer_cls = pl.Trainer
            if torch_save is None:
                torch_save = torch.save

            callbacks = self._define_callbacks(is_training=True)
            logger = TensorBoardLogger(
                save_dir=self._model_dir,
                name="tensorboard_logs",
                version="",
            )
            if precision is None:
                precision = "16-mixed" if torch.cuda.is_available() else "32-true"

            trainer_kwargs = {
                "accelerator": "auto",
                "devices": 1,
                "precision": precision,
                "max_epochs": self.params.training_params.num_epochs or 50,
                "callbacks": callbacks,
                "logger": logger,
                "accumulate_grad_batches": self.params.training_params.gradient_accumulation_num_batches or 1,
                "log_every_n_steps": 10,
            }

            limit_batches = getattr(self.params.training_params, "num_batches_per_epoch", None)
            if limit_batches is not None and type(limit_batches).__name__ != "MagicMock":
                trainer_kwargs["limit_train_batches"] = limit_batches

            if getattr(self.params.training_params, "run_profiler", False) is True:
                try:
                    from lightning.pytorch.profilers import PyTorchProfiler
                    profiler_dir = os.path.join(self._tensorboard_dir(), 'profiler')
                    trainer_kwargs["profiler"] = PyTorchProfiler(dirpath=profiler_dir, filename='perf_logs')
                except Exception as e:
                    logging.warning(f"Could not initialize PyTorchProfiler: {e}")

            grad_clip = getattr(self.params.training_params, "gradient_clip_val", None)
            if grad_clip is not None and type(grad_clip).__name__ != "MagicMock":
                trainer_kwargs["gradient_clip_val"] = grad_clip
                trainer_kwargs["gradient_clip_algorithm"] = getattr(self.params.training_params, "gradient_clip_algorithm", "norm")

            if getattr(self.params.training_params, "fast_dev_run", False) is True:
                trainer_kwargs["fast_dev_run"] = True

            overfit = getattr(self.params.training_params, "overfit_batches", 0.0)
            if overfit and type(overfit).__name__ != "MagicMock":
                trainer_kwargs["overfit_batches"] = overfit

            self.trainer = trainer_cls(**trainer_kwargs)

            ckpt_path = None
            if self.params.continue_model_id is not None and getattr(self, "_continue_model_dir", None):
                found_ckpt = self._find_checkpoint_path(self._continue_model_dir)
                if found_ckpt and found_ckpt.endswith(".ckpt"):
                    ckpt_path = found_ckpt

            fit_kwargs = {"model": self.model}
            if hasattr(self, "data_module"):
                fit_kwargs["datamodule"] = self.data_module
            elif hasattr(self, "data_loaders"):
                fit_kwargs["train_dataloaders"] = self.data_loaders.get(DataSplitType.TRAIN)
                fit_kwargs["val_dataloaders"] = self.data_loaders.get(DataSplitType.VALIDATION)

            if ckpt_path is not None:
                fit_kwargs["ckpt_path"] = ckpt_path

            if getattr(self.params.training_params, "compile_model", False) is True and hasattr(torch, "compile"):
                try:
                    if hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
                        self.model.model = torch.compile(self.model.model)
                    else:
                        self.model = torch.compile(self.model)
                    logging.info("Compiled model using torch.compile()")
                except Exception as e:
                    logging.warning(f"Could not torch.compile model: {e}")

            if (getattr(self.params.training_params, "auto_lr_find", False) is True or
                getattr(self.params.training_params, "auto_scale_batch_size", False) is True):
                try:
                    from lightning.pytorch.tuner import Tuner
                    tuner = Tuner(self.trainer)
                    if getattr(self.params.training_params, "auto_lr_find", False) is True:
                        lr_finder = tuner.lr_find(**fit_kwargs)
                        if lr_finder and lr_finder.suggestion():
                            suggested_lr = lr_finder.suggestion()
                            logging.info(f"Auto-LR finder suggested learning rate: {suggested_lr}")
                            if hasattr(self.model, "hparams") and "learning_rate" in self.model.hparams:
                                self.model.hparams.learning_rate = suggested_lr
                            self.params.training_params.learning_rate = suggested_lr

                    if getattr(self.params.training_params, "auto_scale_batch_size", False) is True:
                        tuner.scale_batch_size(**fit_kwargs, mode="binsearch")
                except Exception as e:
                    logging.warning(f"Could not run Lightning Tuner: {e}")

            self.trainer.fit(**fit_kwargs)

            # Restore best checkpoint weights into self.model if available
            best_ckpt = getattr(getattr(self.trainer, "checkpoint_callback", None), "best_model_path", None)
            if isinstance(best_ckpt, str) and os.path.isfile(best_ckpt):
                try:
                    ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                    state_dict = ckpt.get("state_dict", ckpt)
                    self.model.load_state_dict(state_dict, strict=False)
                    logging.info(f"Restored best model weights from checkpoint: {best_ckpt}")
                except Exception as e:
                    logging.warning(f"Could not restore best model weights from {best_ckpt}: {e}")

            try:
                torch_save(getattr(self.model, "model", self.model), os.path.join(self._model_dir, "model.pt"))
            except Exception as e:
                logging.warning(f"Could not save legacy model.pt: {e}")

            return

        optimizer = self._define_optimizer()
        callbacks = self._define_callbacks(is_training=True, optimizer=optimizer)

        tensorboard_dir = self._tensorboard_dir()
        summary_writer_train = SummaryWriter(os.path.join(tensorboard_dir, DataSplitType.TRAIN.name.lower()))
        summary_writer_validation = SummaryWriter(os.path.join(tensorboard_dir, DataSplitType.VALIDATION.name.lower()))

        epoch_num = self._initial_epoch
        callbacks.on_train_begin()

        prof = None
        if self.params.training_params.run_profiler:
            tensorboard_profiler_dir = os.path.join(self._tensorboard_dir(), 'profiler')
            prof = torch.profiler.profile(schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=2),
                                          on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_profiler_dir),
                                          record_shapes=False, profile_memory=True, with_stack=False)
            prof.start()

        logging.info(f'starting training for model {self.model_id}...')
        try:
            while True:
                # training epoch
                self.model.train()
                callbacks.on_epoch_begin(epoch_num=epoch_num)
                metric_name_to_result_train = self._run_epoch(training_mode=ModelMode.TRAIN,
                                                              callbacks=callbacks,
                                                              data_split_type=DataSplitType.TRAIN,
                                                              optimizer=optimizer,
                                                              loss_function=self._loss_function,
                                                              metrics=self._metrics,
                                                              profiler=prof)
                callbacks.on_train_epoch_end(epoch_num=epoch_num, metric_name_to_result=metric_name_to_result_train)
                self._save_tensorboard_logs(epoch_num=epoch_num, summary_writer=summary_writer_train)

                # eval
                metric_name_to_result_eval = self.eval(data_split_type=DataSplitType.VALIDATION, epoch_num=epoch_num,
                                                       summary_writer=summary_writer_validation, callbacks=callbacks)

                metric_name_to_result_eval = {f'eval_{k}': v for k, v in metric_name_to_result_eval.items()}
                metric_name_to_result = {**metric_name_to_result_train, **metric_name_to_result_eval}

                callbacks.on_epoch_end(epoch_num=epoch_num, metric_name_to_result=metric_name_to_result)

                if self._stop_training(callbacks=callbacks, epoch_num=epoch_num) or getattr(self.params.training_params, "fast_dev_run", False):
                    break

                epoch_num = epoch_num + 1

            if prof is not None:
                prof.stop()

            callbacks.on_train_end()
        except Exception as e:
            logging.exception(f'Error during model training for model {self.model_id}: {e}')
            raise
        finally:
            model_checkpoint = [c for c in callbacks.callbacks if isinstance(c, ModelCheckpoint)]
            epochs = epoch_num if len(model_checkpoint) == 0 else model_checkpoint[0].best_epoch
            self._save_data_to_db(epochs=epochs)

    def eval(self, data_split_type: DataSplitType, epoch_num: Optional[int] = None,
             summary_writer: Optional[SummaryWriter] = None, callbacks: Optional[CallbacksManager] = None) -> \
        Dict[str, Union[float, Dict]]:
        """
        Perform an evaluation run on the data. This means we'll perform a forward pass of all the samples in the
        "data_split_type" dataset and calculate the total loss/metrics.
        If called during training, we should send summary_writer, epoch_num in from the training loop.
        If called on its own then all inputs should be left empty except data_split_type.
        :param data_split_type: The data type to use
        :param summary_writer: The SummaryWriter to use for writing tensorboard logs. If None, don't save to tensorboard
        :param callbacks: An instance of CallbacksCombined. This holds the callbacks information.
                          If None, create it
        :param epoch_num: The epoch number (should only have a value during training)
        :return: A dict of metric/loss name to value
        """
        if self.use_lightning:
            if self.trainer is None:
                precision = "16-mixed" if torch.cuda.is_available() else "32-true"
                self.trainer = pl.Trainer(accelerator="auto", devices=1, precision=precision, logger=False)

            loader = self._get_dataloader(data_split_type)
            if loader is None:
                return {}

            # Suppress TensorBoard logging during standalone offline evaluation (e.g. eval_all_data)
            orig_loggers = getattr(self.trainer, "_loggers", None)
            if summary_writer is None and hasattr(self.trainer, "_loggers"):
                self.trainer._loggers = []
            try:
                self.trainer.validate(model=self.model, dataloaders=loader, verbose=False)
            finally:
                if orig_loggers is not None and hasattr(self.trainer, "_loggers"):
                    self.trainer._loggers = orig_loggers

            metric_name_to_result_eval = copy.deepcopy(getattr(self.model, "_last_epoch_eval_results", {}))
        else:
            if callbacks is None:
                callbacks = self._define_callbacks(is_training=False)

            self.model.eval()
            callbacks.on_eval_begin()
            with torch.no_grad():
                metric_name_to_result_eval = self._run_epoch(training_mode=ModelMode.EVAL, callbacks=callbacks,
                                                             data_split_type=data_split_type, metrics=self._metrics,
                                                             loss_function=self._loss_function)
            callbacks.on_eval_end(metric_name_to_result=metric_name_to_result_eval)

        if summary_writer is not None:
            self._save_tensorboard_logs(epoch_num=epoch_num, summary_writer=summary_writer)
        return metric_name_to_result_eval

    def eval_all_data(self) -> pd.DataFrame:
        """Run an eval over all data split types, and save all results in a dataframe"""
        eval_results = dict()
        for data_split_type in DataSplitType.all_not_none_split_types():
            curr_eval_results = self.eval(data_split_type=data_split_type)
            metric_names = list(curr_eval_results.keys())
            for metric_name in metric_names:
                if isinstance(curr_eval_results[metric_name], dict):
                    curr_eval_results.update(curr_eval_results.pop(metric_name))
            eval_results[data_split_type.name] = curr_eval_results
        return pd.DataFrame(eval_results).T

    @staticmethod
    def _change_metrics_params_tuple_to_per_sample(metrics_params: Tuple[Union[AbstractMetricParams,
                                                                                     AbstractLossParams], ...]) -> \
        Tuple[Union[AbstractMetricParams, AbstractLossParams], ...]:
        return tuple(map(AbstractMetricParams.change_reduction_to_per_sample, metrics_params))

    @staticmethod
    def _format_generic_predictions(predict_batches: List[Any]) -> pd.DataFrame:
        all_dfs = []
        for batch in predict_batches:
            if isinstance(batch, pd.DataFrame):
                all_dfs.append(batch)
            elif isinstance(batch, dict):
                batch_dict = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        arr = v.detach().cpu().numpy()
                        if arr.ndim == 1:
                            batch_dict[k] = arr
                        elif arr.ndim == 2 and arr.shape[1] == 1:
                            batch_dict[k] = arr[:, 0]
                if batch_dict:
                    all_dfs.append(pd.DataFrame(batch_dict))
        if all_dfs:
            return pd.concat(all_dfs, axis=0, ignore_index=True)
        return pd.DataFrame()

    def predict(self, data_split_type: DataSplitType, loss_params: Optional[Tuple[AbstractLossParams, ...]] = None,
                metrics_params: Optional[Tuple[AbstractMetricParams, ...]] = None, trainer_cls=None,
                data_loader: Optional[DataLoader] = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Perform a prediction run on the data. This means we'll perform a forward pass of all the samples in the
        "data_split_type" dataset and collect all the results for each sample.
        :param data_split_type: The data type to use
        :param loss_params: The params for the loss function. If None, use the data from
                            self.params.training_params.loss_params
        :param metrics_params: The params for the metrics. If None, use the data from
                                self.params.training_params.metrics_params
        :param data_loader: Optional custom DataLoader to evaluate (e.g. single-date evaluation).
                            If provided, overrides default data loaders.
        :return: A df with all the results for each sample
        """
        if self.use_lightning:
            if trainer_cls is None:
                trainer_cls = pl.Trainer
            if self.trainer is None:
                precision = "16-mixed" if torch.cuda.is_available() else "32-true"
                self.trainer = trainer_cls(accelerator="auto", devices=1, precision=precision, logger=False)

            if data_loader is not None:
                batches = self.trainer.predict(model=self.model, dataloaders=data_loader)
            elif hasattr(self, "data_module"):
                self.data_module.predict_data_split_type = data_split_type
                batches = self.trainer.predict(model=self.model, datamodule=self.data_module)
            else:
                loader = self._get_dataloader(data_split_type)
                batches = self.trainer.predict(model=self.model, dataloaders=loader)

            if hasattr(self.model, "format_predictions"):
                df = self.model.format_predictions(batches)
            else:
                df = self._format_generic_predictions(batches)

            df['split_type'] = data_split_type.name
            return df

        callbacks = self._define_callbacks(is_training=False)
        loss_function = \
            self._define_loss_function(loss_params=loss_params) if loss_params is not None else self._loss_function
        metrics = self._metrics if metrics_params is None else self._define_metrics(
            metrics_params=tuple((m for m in metrics_params if m.run_in_predict)))

        self.model.eval()
        callbacks.on_predict_begin()
        with torch.no_grad():
            metric_name_to_results_df = self._run_epoch(training_mode=ModelMode.PREDICT, callbacks=callbacks,
                                                        data_split_type=data_split_type,
                                                        metrics=metrics, loss_function=loss_function,
                                                        data_loader=data_loader)
            callback_name_to_results = callbacks.on_predict_end(metric_name_to_results_df=metric_name_to_results_df)
            results_dfs = list()
            for callback_name, callback_results_df in callback_name_to_results.items():
                if callback_results_df is None:
                    continue
                callback_results_df.columns = list(map(lambda column_name: f'{callback_name}_{column_name}',
                                                       callback_results_df.columns))
                results_dfs.append(callback_results_df)
        if len(results_dfs) > 0:
            df = pd.concat(results_dfs, axis=0)
            df['split_type'] = data_split_type.name
            return df

    def predict_all_data(self, loss_params: Optional[Tuple[AbstractLossParams, ...]] = None,
                         metrics_params: Optional[Tuple[AbstractMetricParams, ...]] = None, **kwargs) -> pd.DataFrame:
        predict_results = list()
        for data_split_type in DataSplitType.all_not_none_split_types():
            curr_predict_results = self.predict(data_split_type=data_split_type, loss_params=loss_params,
                                                metrics_params=metrics_params, **kwargs)
            if curr_predict_results is not None and not curr_predict_results.empty:
                predict_results.append(curr_predict_results)
        if predict_results:
            return pd.concat(predict_results, axis=0, ignore_index=True)
        return pd.DataFrame()

    def _save_hyperparameters_metrics_to_tensorboard(self, eval_results: pd.DataFrame):
        hyperparameters = {'learning_rate': self.params.training_params.learning_rate,
                           'batch_size': self.params.training_params.batch_size
                           }

        if self.use_lightning and getattr(self, "trainer", None) is not None and getattr(self.trainer, "logger", None) is not None:
            try:
                val_metrics = {}
                for split_type in DataSplitType.all_not_none_split_types():
                    if split_type.name in eval_results.index:
                        split_row = eval_results.loc[split_type.name].to_dict()
                        for k, v in split_row.items():
                            if isinstance(v, (int, float, np.number)) and not (isinstance(v, float) and np.isnan(v)):
                                val_metrics[f"{split_type.name.lower()}/{k}"] = float(v)
                self.trainer.logger.log_hyperparams(params=hyperparameters, metrics=val_metrics)
                return
            except Exception as e:
                logging.warning(f"Could not log hyperparams via Lightning logger: {e}")

        for split_type in DataSplitType.all_not_none_split_types():
            if split_type.name not in eval_results.index:
                continue
            summary_writer = SummaryWriter(os.path.join(self._tensorboard_dir(), split_type.name.lower()))
            metrics = eval_results.loc[split_type.name].to_dict()
            scalar_metrics = {
                k: float(v) if isinstance(v, (int, float, np.number)) else v
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.number, bool)) and not (isinstance(v, float) and np.isnan(v))
            }
            scalar_metrics['split_type'] = split_type.value
            try:
                summary_writer.add_hparams(run_name=str(self.model_id), hparam_dict=hyperparameters, metric_dict=scalar_metrics)
            except Exception as e:
                logging.warning(f"Could not save hparams to tensorboard: {e}")

    def _tensorboard_dir(self):
        return os.path.join(self._model_dir, 'tensorboard_logs')

    def _save_eval_results_to_feather(self, eval_results: pd.DataFrame):
        save_path = os.path.join(self._model_dir, 'eval_results.arrow')
        eval_results.reset_index().rename(columns={'index': 'split_type'}).to_feather(save_path)

    def _add_extra_metrics_to_eval_results(self, eval_results: pd.DataFrame, predict_results: pd.DataFrame
                                           ) -> pd.DataFrame:
        return eval_results

    def run_full_training_session(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.train()
        eval_results = self.eval_all_data()

        # updating params to save a value per sample
        metrics_params = self._change_metrics_params_tuple_to_per_sample(self.params.training_params.metrics_params)
        loss_params = self._change_metrics_params_tuple_to_per_sample(self.params.training_params.loss_params)
        predict_results = self.predict_all_data(loss_params=loss_params, metrics_params=metrics_params)

        eval_results = self._add_extra_metrics_to_eval_results(eval_results=eval_results,
                                                               predict_results=predict_results)

        self._save_eval_results_to_feather(eval_results)
        if self.params.training_params.save_hyperparameters_to_tensorboard:
            self._save_hyperparameters_metrics_to_tensorboard(eval_results=eval_results)

        print(eval_results[[c for c in eval_results if 'column' not in c]].to_string())

        return eval_results, predict_results


