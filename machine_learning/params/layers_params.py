import dataclasses
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Tuple
from machine_learning.params.abstract_params import LayerParams
from machine_learning.enums import ActivationFunction
from machine_learning.registries import params_registry

@params_registry.register
@dataclass
class CombineTensorsParams(LayerParams):
    dim: int = -1

    # one of "sum"/"concat"
    combine_func_str: str = 'concat'


@params_registry.register
@dataclass
class TCNParams(LayerParams):
    """
    This class holds all parameters needed for building the temporal convolutional network layer in the neural
    network model.
    """
    # number of channels to use in each of the TCN layers. The number of layers is derived from the length of the list
    num_channels: List[int] = field(default_factory=lambda: [32])

    # the kernel size to use in the convolutional layers
    kernel_size: int = 3

    # dropout rate to use in the TCN layers
    dropout_rate: float = 0.

    # the number of times to call (conv -> relu -> dropout) in each tcn block (usually 2)
    num_sub_blocks: int = 2

    # the dilations to use in each TCN block.
    # If None, we'll use 2 ** range(len(num_channels))
    # If int we'll use that value for each block
    # If list, contains the dilation size for each block (should be the same length as num_channels)
    dilation_sizes: Optional[Union[int, List[int]]] = None


@params_registry.register
@dataclass
class TransformerEncoderParams(LayerParams):
    """This class holds all parameters needed for building the transformer encoder layer in the neural network model."""
    version = 0

    # number of encoder layers to use
    n_encoders: int = 1

    # number of heads to use in the multihead attention layer
    n_heads: int = 4

    # dropout rate to use in the transformer layers
    dropout_rate: float = 0.


@params_registry.register
@dataclass
class TabularEncodingParams(LayerParams):
    activation: ActivationFunction = ActivationFunction.LINEAR

    # If True, have a single node for each variable and then combine. If False, run all variable through a dense layer
    # with num_variables nodes
    encode_separately: bool = True


@params_registry.register
@dataclass
class TimeseriesReductionParams(LayerParams):
    """Params for the layer that reshapes the timeseries data"""

    # output size of the temporal dimension
    temporal_output_size: Optional[int] = None

    # output size of the features dimension
    channels_output_size: Optional[int] = None

    # whether to flatten the output to be 2D
    flatten: bool = False


@params_registry.register
@dataclass
class DenseLayersParams(LayerParams):
    """Params for the layer that reshapes the timeseries data"""

    # the number of layers will be inferred from this
    num_nodes_per_layer: List[int] = field(default_factory=list)

    activation: ActivationFunction = ActivationFunction.RELU

    activation_on_last_layer: ActivationFunction = ActivationFunction.LINEAR

    use_batchnorm: bool = False

    dropout_rate: Optional[float] = None


@params_registry.register
@dataclass
class EmbeddingsLayerParams(LayerParams):

    # the number of unique values in each categorical feature
    num_embeddings: Optional[List[int]] = None

    # the dimensions for each one of the embeddings
    embeddings_dims: List[int] = field(default_factory=list)

    # kwargs for creating CombineLayerParams
    combine_params_kwargs: Dict[str, Any] = field(default_factory=lambda: {'dim': -1, 'combine_func_str': 'concat'})

    max_norm: Optional[float] = None

    def update_after_loading_data(self, updated_params: Dict[str, Any]):
        self.update_params_from_dict(updated_params)


@params_registry.register
@dataclass
class ReshapeLayerParams(LayerParams):
    # The desired shape, without the batch dimension
    shape: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        super().__post_init__()

        if self.shape is None:
            raise ValueError("shape can't be None!")


@params_registry.register
@dataclass
class TSMixerParams(LayerParams):
    # Number of mixer layers
    num_mixer_layers: Optional[int] = None

    # Number of time steps to predict
    prediction_length: Optional[int] = None

    # Number of hidden channels in the feature mixing MLP
    feature_mixing_hidden_channels: Optional[int] = None

    # Number of output feature. If None, will equal the number of features in the timeseries inputs
    num_out_features: Optional[int] = None

    dropout_rate: float = 0.

    # Dropout rate of last layer. If None, will equal self.dropout
    dropout_last_layer: Optional[float] = None

    time_dimension: int = -1

    def __post_init__(self):
        super().__post_init__()

        if self.dropout_last_layer is None:
            self.dropout_last_layer = self.dropout_rate

        if self.num_mixer_layers is None:
            raise ValueError("num_mixer_layers can't be None!")
        if self.prediction_length is None:
            raise ValueError("prediction_length can't be None!")
        if self.feature_mixing_hidden_channels is None:
            raise ValueError("feat_mixing_hidden_channels can't be None!")


@params_registry.register
@dataclass
class TransformerEncoderPytorchParams(LayerParams):
    """
    Params for creating a pytorch TransformerEncoder Layer. This includes the params for the
    TransformerEncoderLayer that is used to create TransformerEncoder
    """

    # TransformerEncoderLayer params
    d_model: int = 32
    nhead: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: ActivationFunction = ActivationFunction.RELU
    layer_norm_eps: float = 1e-5
    batch_first: bool = True
    norm_first: bool = False
    bias: bool = True
    device: Optional[Union[torch.device, str]] = None
    dtype: Optional[Union[str, torch.dtype]] = None

    # TransformerEncoder params
    num_layers: int = 1
    norm: Optional[torch.nn.Module] = None
    enable_nested_tensor: bool = True
    mask_check: bool = True

    def __post_init__(self):
        super().__post_init__()

        # if norm_first is True and enable_nested_tensor is True, we'll get a warning in the torch TransformerEncoder
        # module:
        # UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because
        # encoder_layer.norm_first was True
        if self.norm_first:
            self.enable_nested_tensor = False

        encoder_keys = ['num_layers', 'norm', 'enable_nested_tensor', 'mask_check']
        all_dict = dataclasses.asdict(self)

        self.transformer_encoder_kwargs = {k: v for k, v in all_dict.items() if k in encoder_keys}

        self.transformer_encoder_layer_kwargs = dataclasses.asdict(self)
        self.transformer_encoder_layer_kwargs['activation'] = \
            ActivationFunction.member_to_function(self.transformer_encoder_layer_kwargs['activation'])
        for k in encoder_keys + ['input_tensor_name', 'output_tensor_name', 'key_to_class_for_registry']:
            del self.transformer_encoder_layer_kwargs[k]


@params_registry.register
@dataclass
class SplitAttentionCombineWrapperParams(LayerParams):
    d_model: int = 32
    n_heads: int = 8
    ffdim: int = 32
    dropout_rate: float = 0.
    norm_at_beginning: bool = True
    n_bench_players: int = 6
    activation: str = 'relu'
    num_layers: int = 2

    def __post_init__(self):
        super().__post_init__()
        self.ffdim = max(self.ffdim, self.d_model)

        self.single_layer_kwargs = dataclasses.asdict(self)
        del self.single_layer_kwargs['num_layers']


@params_registry.register
@dataclass
class SplitAttentionCombineParams(LayerParams):
    d_model: int = 32
    n_heads: int = 8
    ffdim: int = 32
    dropout_rate: float = 0.
    norm_at_beginning: bool = True
    n_bench_players: int = 6
    activation: str = 'relu'

    def __post_init__(self):
        super().__post_init__()

        self.ffdim = max(self.ffdim, self.d_model)


@params_registry.register
@dataclass
class TransformerEncoderCustomParams(LayerParams):
    d_model: int = 32
    n_heads: int = 8
    ffdim: int = 32
    d_v: Optional[int] = None
    dropout_rate: float = 0.
    # some papers report that they get better results without layernorm on the input of the first encoder layer
    # https://arxiv.org/pdf/2106.11959v2.pdf
    norm_at_beginning: bool = True
    activation: str = 'relu'

    def __post_init__(self):
        super().__post_init__()

        if self.d_v is None:
            self.d_v = self.d_model


@params_registry.register
@dataclass
class TemporalFusionTransformerParams(LayerParams):
    task_type: str = 'regression'
    output_quantiles: Optional[List[float]] = field(default_factory=lambda: [0.5])
    categorical_cardinalities: Dict[str, List[int]] = \
        field(default_factory=lambda: {k: list() for k in ['static', 'historical', 'future']})
    attention_heads: int = 4
    dropout: float = 0.
    lstm_layers: int = 1
    target_window_start: Optional[int] = None
    state_size: int = 64
    output_size: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()

        if self.output_size is None:
            self.output_size = self.state_size

        if self.task_type == 'regression':
            if self.output_quantiles is None:
                raise ValueError("output_quantiles can't be None if task_type is regression!")
            self.num_outputs = len(self.output_quantiles)
        elif self.task_type == 'classification':
            if self.output_quantiles is not None:
                raise ValueError("output_quantiles must be None if task_type is classification!")
            self.num_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")
