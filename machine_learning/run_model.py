import os
import contextlib
import copy
import logging
import torch
import traceback
import pandas as pd
from typing import Type, Dict, Optional, Union, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from machine_learning.constants import METADATA
from machine_learning.architecture import ModelArchitecture
from machine_learning.callbacks import CallbacksManager, training_callbacks, ModelCheckpoint
from machine_learning.enums import ModelMode, DataSplitType
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning.metrics.combined_metrics import LossesCombined, MetricsCombined
from machine_learning.params.abstract_params import AbstractMetricParams, AbstractLossParams, \
    AbstractMetricsAndCallbacksParams
from machine_learning.params.callbacks_params import ModelCheckpointParams
from machine_learning.params.layers_params import EmbeddingsLayerParams
from machine_learning.params.params import MachineLearningParams
from machine_learning.registries import callbacks_registry
from machine_learning.utils import RANDOM_SEED, models_dir, get_next_model_id

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

OPTIMIZER_TYPE = Optional[torch.optim.Optimizer]


class RunModel:
    def __init__(self, params: MachineLearningParams, project_name: str, description: Optional[str] = None):
        self.params = params
        self.project_name = project_name
        self._description = description

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info('running on gpu')
        else:
            self.device = torch.device("cpu")
            logging.info('running on cpu')

        self._init_models_dir()
        if not os.path.isdir(self._models_dir):
            os.mkdir(self._models_dir)
        self._init_model_id()
        self._model_dir = os.path.join(self._models_dir, str(self.model_id))
        if self.params.continue_model_id is not None:
            self._continue_model_dir = os.path.join(self._models_dir, str(self.params.continue_model_id))
        else:
            self._continue_model_dir = None

        self.create_data_loaders()
        self.update_categorical_params()
        self._create_model_architecture()

        self._loss_function = self._define_loss_function()
        self._metrics = self._define_metrics()

        all_params = [self.params.training_params.loss_params,
                      self.params.training_params.metrics_params,
                      self.params.training_params.callbacks_params]
        self._all_tensor_names_for_losses_metrics_callbacks = \
            list(set(sum([curr_params.all_tensor_names for p in all_params for curr_params in p], [])))

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
        if self.params.continue_model_id is not None:
            scalers, label_encoders = \
                self._machine_learning_datasets_combined_class().load_transformers(self._continue_model_dir)
        machine_learning_datasets_combined = self._machine_learning_datasets_combined_class()(
            params=self.params.features_params,
            training_params=self.params.training_params,
            project_name=self.project_name,
            dataset_name_to_scalers=scalers,
            dataset_name_to_categorical_encoder=label_encoders)
        self.data_loaders = machine_learning_datasets_combined.create_data_loaders(
            batch_size=self.params.training_params.batch_size,
            num_workers=self.params.training_params.dataloader_num_workers)

    @staticmethod
    def _machine_learning_datasets_combined_class() -> Type[MachineLearningDatasetCombined]:
        return MachineLearningDatasetCombined

    @staticmethod
    def _model_architecture_class() -> Type[ModelArchitecture]:
        return ModelArchitecture

    @staticmethod
    def _losses_combined_class() -> Type[LossesCombined]:
        return LossesCombined

    @staticmethod
    def _metrics_combined_class() -> Type[MetricsCombined]:
        return MetricsCombined

    @staticmethod
    def _callbacks_combined_class() -> Type[CallbacksManager]:
        return CallbacksManager

    def _create_model_architecture(self):
        if self.params.continue_model_id is not None:
            self.model = ModelCheckpoint.load_model(os.path.join(self._continue_model_dir, 'model.pt'))
            self._initial_epoch = 0
        else:
            model_class = self._model_architecture_class()
            self.model = model_class(self.params.architecture_params,
                                     feature_type_to_input_shape=self.params.features_params.feature_type_to_input_shape
                                     )
            self._initial_epoch = 0
        self.model.to(self.device)

    def _define_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=self.params.training_params.learning_rate)

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
                   optimizer: OPTIMIZER_TYPE = None, profiler: Optional[torch.profiler.profile] = None) -> \
        Dict[str, Union[float, Dict, pd.DataFrame]]:
        loss_function.reset(device=self.device)
        metrics.reset(device=self.device)

        if self.model.training:
            optimizer.zero_grad(set_to_none=True)

        for batch_num, batch_data in enumerate(self.data_loaders[data_split_type]):
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

    def _update_model_checkpoint_save_dir(self, callbacks_params: Tuple[AbstractMetricsAndCallbacksParams]):
        for callback_params in callbacks_params:
            if isinstance(callback_params, ModelCheckpointParams):
                callback_params.save_dir = self._model_dir
                break

    def _define_callbacks_manager(self, is_training: bool, optimizer: OPTIMIZER_TYPE = None) -> CallbacksManager:
        callbacks_params = copy.deepcopy(self.params.training_params.callbacks_params)
        callbacks_combined_class = self._callbacks_combined_class()
        if not is_training:
            callbacks_params = tuple([p for p in callbacks_params if callbacks_registry.get(p.key_to_class_for_registry)
                                not in training_callbacks()])
        return callbacks_combined_class(
            callback_params=callbacks_params, model=self.model, optimizer=optimizer, model_dir=self._model_dir)

    def _stop_training(self, callbacks: CallbacksManager, epoch_num: int) -> bool:
        return callbacks.stop_training() or \
               ((self.params.training_params.num_epochs is not None) and
                (epoch_num >= (self.params.training_params.num_epochs - 1)))

    def train(self):
        self._init_models_dir()
        self._init_model_id()
        MachineLearningDatasetCombined.save_transformers(data_loader=self.data_loaders, save_dir=self._model_dir)

        optimizer = self._define_optimizer()
        callbacks = self._define_callbacks_manager(is_training=True, optimizer=optimizer)

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

                if self._stop_training(callbacks=callbacks, epoch_num=epoch_num):
                    break

                epoch_num = epoch_num + 1

            if prof is not None:
                prof.stop()

            callbacks.on_train_end()
        except Exception as e:
            logging.warning(f'There was an error during model training {e}!!!')
            traceback.print_exc()
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
        if callbacks is None:
            callbacks = self._define_callbacks_manager(is_training=False)

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

    def predict(self, data_split_type: DataSplitType, loss_params: Optional[Tuple[AbstractLossParams, ...]] = None,
                metrics_params: Optional[Tuple[AbstractMetricParams, ...]] = None) -> Optional[pd.DataFrame]:
        """
        Perform a prediction run on the data. This means we'll perform a forward pass of all the samples in the
        "data_split_type" dataset and collect all the results for each sample.
        :param data_split_type: The data type to use
        :param loss_params: The params for the loss function. If None, use the data from
                            self.params.training_params.loss_params
        :param metrics_params: The params for the metrics. If None, use the data from
                               self.params.training_params.metrics_params
        :return: A df with all the results for each sample
        """
        callbacks = self._define_callbacks_manager(is_training=False)
        loss_function = \
            self._define_loss_function(loss_params=loss_params) if loss_params is not None else self._loss_function
        metrics = self._metrics if metrics_params is None else self._define_metrics(
            metrics_params=tuple((m for m in metrics_params if m.run_in_predict)))

        self.model.eval()
        callbacks.on_predict_begin()
        with torch.no_grad():
            metric_name_to_results_df = self._run_epoch(training_mode=ModelMode.PREDICT, callbacks=callbacks,
                                                        data_split_type=data_split_type,
                                                        metrics=metrics, loss_function=loss_function)
            callback_name_to_results = callbacks.on_predict_end(metric_name_to_results_df=metric_name_to_results_df)
            results_dfs = list()
            for callback_name, callback_results_df in callback_name_to_results.items():
                if callback_results_df is None:
                    continue
                callback_results_df.columns = list(map(lambda column_name: f'{callback_name}_{column_name}',
                                                       callback_results_df.columns))
                results_dfs.append(callback_results_df)
        if len(results_dfs) > 0:
            return pd.concat(results_dfs, axis=0)

    def predict_all_data(self, loss_params: Optional[Tuple[AbstractLossParams, ...]] = None,
                         metrics_params: Optional[Tuple[AbstractMetricParams, ...]] = None) -> pd.DataFrame:
        predict_results = list()
        for data_split_type in DataSplitType.all_not_none_split_types():
            curr_predict_results = self.predict(data_split_type=data_split_type, loss_params=loss_params,
                                                metrics_params=metrics_params)
            predict_results.append(curr_predict_results)
        return pd.concat(predict_results, axis=0)

    def _save_hyperparameters_metrics_to_tensorboard(self, eval_results: pd.DataFrame):
        hyperparameters = {'learning_rate': self.params.training_params.learning_rate,
                           'batch_size': self.params.training_params.batch_size
                           }
        for split_type in DataSplitType.all_not_none_split_types():
            summary_writer = SummaryWriter(os.path.join(self._tensorboard_dir(), split_type.name.lower()))
            metrics = eval_results.loc[split_type.name].to_dict()
            metrics['split_type'] = split_type.value
            summary_writer.add_hparams(run_name=str(self.model_id), hparam_dict=hyperparameters, metric_dict=metrics)

    def _tensorboard_dir(self):
        return os.path.join(self._model_dir, 'tensorboard_logs')

    def _save_eval_results_to_feather(self, eval_results: pd.DataFrame):
        save_path = os.path.join(self._models_dir, str(self.model_id), 'eval_results.arrow')
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


