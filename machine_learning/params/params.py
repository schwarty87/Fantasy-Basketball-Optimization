import copy
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Type, Tuple, Union
from machine_learning.params.abstract_params import AbstractParams, AbstractMetricParams, AbstractLossParams, \
    LayerParams, AbstractMachineLearningDatasetParams
from machine_learning.params.callbacks_params import ModelCheckpointParams, \
    EarlyStoppingParams, ReduceLearningRateOnPlateauParams, LogModelResultsParams, CollectTruePredictedOutputsParams
from machine_learning.enums import ImputationType
from machine_learning.params.layers_params import TCNParams, TransformerEncoderParams, TabularEncodingParams, \
    TimeseriesReductionParams, DenseLayersParams, EmbeddingsLayerParams, TSMixerParams, ReshapeLayerParams, \
    TemporalFusionTransformerParams, CombineTensorsParams, TransformerEncoderPytorchParams, SplitAttentionCombineParams, \
    TransformerEncoderCustomParams, SplitAttentionCombineWrapperParams
from machine_learning.params.metrics_params import MAEMetricParams, BinaryCrossEntropyParams, ROCAUCParams, \
    BinaryClassificationMetricsParams, MSELossParams, QuantileRegressionLossParams, MAPEMetricParams
from machine_learning.params.machine_learning_dataset_params import MachineLearningDatasetTimeseriesParams, \
    MachineLearningDatasetTabularParams, MachineLearningDatasetTabularScalingParams, \
    MachineLearningDatasetTabularCategoricalEncodingParams, MachineLearningDatasetTimeseriesScalingParams, \
    MachineLearningDatasetMetadataParams, MachineLearningDatasetKeepMaskTabularParams, \
    MachineLearningDatasetSampleWeightsTabularParams
# from machine_learning.params.params_registry import register_params
from machine_learning.registries import params_registry


@params_registry.register
@dataclass
class ArchitectureParams(AbstractParams):
    """
    This class holds all parameters needed for creating a neural network model architecture.
    The architecture can be made up of multiple "sub_models" (such as TCN, transformer...), each one with its own set
    of parameters
    """
    layer_params_list: List[LayerParams] = field(default_factory=list)
    output_tensor_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        all_layer_input_tensor_names = [p.input_tensor_name for p in self.layer_params_list]
        self.all_layer_input_tensor_names = list()
        for n in all_layer_input_tensor_names:
            if isinstance(n, list):
                self.all_layer_input_tensor_names += n
            elif isinstance(n, dict):
                self.all_layer_input_tensor_names += list(n.values())
            else:
                self.all_layer_input_tensor_names.append(n)


def _convert_dictionary_keys_from_enum_to_str(nested_dict: Dict):
    return_dict = dict()
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            value = _convert_dictionary_keys_from_enum_to_str(value)
        if not isinstance(key, str):
            key = key.name
        return_dict[key] = value
    return return_dict


@params_registry.register
@dataclass(kw_only=True)
class FeaturesParams(AbstractParams):
    """
    This class holds all parameters needed for loading and processing the features to be used in the neural network.
    """
    # features path to use when loading the features
    features_path: Optional[str] = None

    feature_name_to_machine_learning_dataset_params: Optional[Dict[str, AbstractMachineLearningDatasetParams]] = None

    # the number of steps between samples (for example, if stride==2 then we'll take samples 0, 2, 4...)
    stride: int = 1

    impute_types: Optional[Tuple[ImputationType, ...]] = (ImputationType.LINEAR,)

    convert_to_binary_column_names_to_threshold: Optional[Dict[str, float]] = None
    binary_closing_opening_kernel_size: Optional[int] = None

    # Added just so the fields appear when we call dataclasses.fields. These values will be overwritten in the post_init
    all_feature_names_list: List[str] | None = None
    feature_type_to_input_shape: Dict[str, Tuple[int, ...]] | None = None

    def __post_init__(self):
        self.all_feature_names_list = list()
        feature_name_to_machine_learning_dataset_params = (self.feature_name_to_machine_learning_dataset_params if
                                                           self.feature_name_to_machine_learning_dataset_params
                                                           is not None else dict())
        for feature_type, features_params in feature_name_to_machine_learning_dataset_params.items():
            self.all_feature_names_list += features_params.feature_names
        self.all_feature_names_list = sorted(set(self.all_feature_names_list))

        # saving the size of each input tensor to help with building the model architecture
        self.feature_type_to_input_shape = dict()
        for curr_type, curr_params in feature_name_to_machine_learning_dataset_params.items():
            curr_shape = len(curr_params.feature_names),
            if isinstance(curr_params, MachineLearningDatasetTimeseriesParams):
                curr_shape = curr_shape + (curr_params.window_size // curr_params.sampling_rate,)
            self.feature_type_to_input_shape[curr_type] = curr_shape

        self.feature_type_to_input_shape = self.feature_type_to_input_shape if len(
            self.feature_type_to_input_shape) > 0 else None


@params_registry.register
@dataclass
class TrainingParams(AbstractParams):
    """This class holds all parameters needed for training a neural network model."""
    learning_rate: float = 0.01
    weight_decay: float = 1e-2
    batch_size: int | None = 256
    training_dtype: torch.dtype = torch.float32
    save_to_db: bool = True
    dataloader_num_workers: int = 0
    num_epochs: Optional[int] = None
    save_hyperparameters_to_tensorboard: bool = False
    run_profiler: bool = False
    gradient_accumulation_num_batches: Optional[int] = None

    # the parameters for the loss function
    loss_params: Union[AbstractLossParams, Tuple[AbstractLossParams, ...]] = field(default_factory=tuple)

    # the parameters for the metrics to use
    metrics_params: Union[AbstractMetricParams, Tuple[AbstractMetricParams, ...]] = \
        field(default_factory=tuple)

    # the parameters for the callbacks to use
    callbacks_params: Union[AbstractParams, Tuple[AbstractParams, ...]] = \
        field(default_factory=lambda: (ModelCheckpointParams(), EarlyStoppingParams()))

    num_batches_per_epoch: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"
    fast_dev_run: bool = False
    overfit_batches: Union[int, float] = 0.0
    auto_lr_find: bool = False
    auto_scale_batch_size: bool = False
    use_swa: bool = False
    swa_lrs: float = 1e-3
    use_rich: bool = False
    compile_model: bool = False
    seed: Optional[int] = 42
    track_grad_norm: Optional[Union[int, float, str]] = None

    def __post_init__(self):
        if isinstance(self.loss_params, AbstractLossParams):
            self.loss_params = (self.loss_params,)
        if isinstance(self.callbacks_params, AbstractLossParams):
            self.callbacks_params = (self.callbacks_params,)
        if isinstance(self.metrics_params, AbstractMetricParams):
            self.metrics_params = (self.metrics_params,)
        if self.batch_size:
            if self.batch_size < 0:
                raise ValueError('batch_size must be a positive value!')
            if int(self.batch_size) != self.batch_size:
                raise ValueError('batch_size must be an integer!')
        if self.learning_rate <= 0:
            raise ValueError('learning_rate must be a positive value!')
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError('weight_decay must be non-negative!')
        if self.num_epochs is not None and self.num_epochs <= 0:
            raise ValueError('num_epochs must be a positive value!')


    @staticmethod
    def _convert_list_of_classes_to_dict(classes: Union[AbstractParams, Tuple[AbstractParams, ...]]
                                         ) -> Dict[str, AbstractParams]:
        """
        Create a dict of (class_name + int) to the class instance.
        We add the integer because we may have a few instances of the same class. In that case we'd have multiple
        identical key names, and we'd lose information
        """
        if isinstance(classes, tuple) and len(classes) == 0:
            return {}
        if not isinstance(classes, tuple):
            classes = (classes,)
        return {f'{type(p).__name__}_{i}': p for i, p in enumerate(classes)}

    def to_dict(self, dict_factory: Optional[Callable] = None) -> Dict:
        return_dict = super().to_dict()
        return_dict['training_dtype'] = str(self.training_dtype)
        return return_dict

    @classmethod
    def init_from_dict(cls: dataclass, params_dict: Dict) -> AbstractParams:
        params_dict_with_classes = copy.deepcopy(params_dict)
        params_dict_with_classes['training_dtype'] = eval(params_dict['training_dtype'])
        return super().init_from_dict(params_dict=params_dict_with_classes)


@params_registry.register
@dataclass
class MachineLearningParams(AbstractParams):
    """
    This class holds all parameters needed to fully create and run a model. It holds a few of its own params, and an
    instance of FeatureParams, ArchitectureParams, TrainingParams
    """
    architecture_params: ArchitectureParams = field(default_factory=ArchitectureParams)
    training_params: TrainingParams = field(default_factory=TrainingParams)
    features_params: FeaturesParams = field(default_factory=FeaturesParams)

    continue_model_id: Optional[int] = None

    @staticmethod
    def update_from_previous_version(params):
        return params
