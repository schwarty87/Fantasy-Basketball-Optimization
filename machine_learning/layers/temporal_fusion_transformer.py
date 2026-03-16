"""
Based on the code in https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py
Link to original paper: https://arxiv.org/abs/1912.09363
blog post with an explanation: https://www.playtika-blog.com/playtika-ai/multi-horizon-forecasting-using-temporal-fusion-transformers-a-comprehensive-overview-part-1/
"""

import copy
import math
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Dict, Tuple, Optional
from machine_learning.layers.abstract_layer import AbstractLayer, LayersInputOutputShape
from machine_learning.params.layers_params import TemporalFusionTransformerParams
from machine_learning.registries import layers_registry


INPUT_TENSORS_BASE_NAMES = ["historical_timeseries", "future_timeseries", 'static_feats']
FEATURE_TYPES = ['numeric', 'categorical']
INPUT_TENSOR_NAMES = [f'{base_name}_{input_type}' for base_name in INPUT_TENSORS_BASE_NAMES for input_type in
                      FEATURE_TYPES]
STATIC_ENCODER_LAYERS_NAMES = ['selection', 'enrichment', 'sequential_cell_init', 'sequential_state_init']


class TimeDistributed(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/base_blocks.py

    This module can wrap any given module and stacks the time dimension with the batch dimension of the inputs
    before applying the module.
    Borrowed from this fruitful `discussion thread
    <https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4>`_.

    Parameters
    ----------
    module : nn.Module
        The wrapped module.
    batch_first: bool
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    return_reshaped: bool
        A boolean indicating whether to return the output in the corresponding original shape or not.
    """

    def __init__(self, module: nn.Module, batch_first: Optional[bool] = None, return_reshaped: bool = True):
        super(TimeDistributed, self).__init__()
        self.module: nn.Module = module  # the wrapped module
        self.batch_first: Optional[bool] = batch_first  # indicates the dimensions order of the sequential data.
        self.return_reshaped: bool = return_reshaped

        if (batch_first is not None) != return_reshaped:
            raise ValueError(f'batch_first has to be {"a bool" if return_reshaped else "None"} when return_reshaped is '
                             f'{return_reshaped}!')

    def forward(self, x):

        # in case the incoming tensor is a two-dimensional tensor - infer no temporal information is involved,
        # and simply apply the module
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and time-steps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * time-steps, input_size)
        # apply the module on each time-step separately
        y = self.module(x_reshape)

        # reshaping the module output as sequential tensor (if required)
        if self.return_reshaped:
            if self.batch_first:
                y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, time-steps, output_size)
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # (time-steps, samples, output_size)

        return y


class NullTransform(nn.Module):
    """ Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/base_blocks.py """

    def __init__(self):
        super(NullTransform, self).__init__()

    @staticmethod
    def forward(empty_input: torch.tensor):
        return []


class GatedLinearUnit(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    This module is also known as  **GLU** - Formulated in:
    `Dauphin, Yann N., et al. "Language modeling with gated convolutional networks."
    International conference on machine learning. PMLR, 2017
    <https://arxiv.org/abs/1612.08083>`_.

    The output of the layer is a linear projection (X * W + b) modulated by the gates **sigmoid** (X * V + c).
    These gates multiply each element of the matrix X * W + b and control the information passed on in the hierarchy.
    This unit is a simplified gating mechanism for non-deterministic gates that reduce the vanishing gradient problem,
    by having linear units coupled to the gates. This retains the non-linear capabilities of the layer while allowing
    the gradient to propagate through the linear unit without scaling.

    Parameters
    ----------
    input_dim: int
        The embedding size of the input.
    """

    def __init__(self, input_dim: int):
        super(GatedLinearUnit, self).__init__()

        # Two dimension-preserving dense layers
        self.fully_connected1 = nn.Linear(input_dim, input_dim)
        self.fully_connected2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fully_connected1(x))
        x = self.fully_connected2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py
    Link to original paper: https://arxiv.org/abs/1912.09363

    This module, known as **GRN**, takes in a primary input (x) and an optional context vector (c).
    It uses a ``GatedLinearUnit`` for controlling the extent to which the module will contribute to the original input
    (x), potentially skipping over the layer entirely as the GLU outputs could be all close to zero, by that suppressing
    the non-linear contribution.
    In cases where no context vector is used, the GRN simply treats the context input as zero.
    During training, dropout is applied before the gating layer.

    Parameters
    ----------
    input_dim: int
        The embedding width/dimension of the input.
    hidden_dim: int
        The intermediate embedding width.
    output_dim: int
        The embedding width of the output tensors.
    dropout: Optional[float]
        The dropout rate associated with the component.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: Optional[float] = 0.05,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # =================================================
        # Input conditioning components (Eq.4 in the original paper)
        # =================================================
        # for using direct residual connection the dimension of the input must match the output dimension.
        # otherwise, we'll need to project the input for creating this residual connection
        self.project_residual: bool = self.input_dim != self.output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_dim, self.output_dim), batch_first=batch_first)

        # A linear layer for projecting the primary input (acts across time if necessary)
        self.fully_connected1 = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim), batch_first=batch_first)

        # In case we expect context input, an additional linear layer will project the context
        if self.context_dim is not None:
            self.context_projection = TimeDistributed(nn.Linear(self.context_dim, self.hidden_dim, bias=False),
                                                      batch_first=batch_first)
        # non-linearity to be applied on the sum of the projections
        self.elu1 = nn.ELU()

        # ============================================================
        # Further projection components (Eq.3 in the original paper)
        # ============================================================
        # additional projection on top of the non-linearity
        self.fully_connected2 = TimeDistributed(nn.Linear(self.hidden_dim, self.output_dim), batch_first=batch_first)

        # ============================================================
        # Output gating components (Eq.2 in the original paper)
        # ============================================================
        self.dropout = nn.Dropout(self.dropout)
        self.gate = TimeDistributed(GatedLinearUnit(self.output_dim), batch_first=batch_first)
        self.layer_norm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)

    def forward(self, x, context=None):

        # compute residual (for skipping) if necessary
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x
        # ===========================
        # Compute Eq.4
        # ===========================
        x = self.fully_connected1(x)
        if context is not None:
            context = self.context_projection(context)
            x = x + context

        # compute eta_2 (according to paper)
        x = self.elu1(x)

        # ===========================
        # Compute Eq.3
        # ===========================
        # compute eta_1 (according to paper)
        x = self.fully_connected2(x)

        # ===========================
        # Compute Eq.2
        # ===========================
        x = self.dropout(x)
        x = self.gate(x)
        # perform skipping using the residual
        x = x + residual
        # apply normalization layer
        x = self.layer_norm(x)

        return x


class VariableSelectionNetwork(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    This module is designed to handle the fact that the relevant and specific contribution of each input variable
    to the output is typically unknown. This module enables instance-wise variable selection, and is applied to
    both the static covariates and time-dependent covariates.

    Beyond providing insights into which variables are the most significant ones for the prediction problem,
    variable selection also allows the model to remove any unnecessary noisy inputs which could negatively impact
    performance.

    Parameters
    ----------
    input_dim: int
        The attribute/embedding dimension of the input, associated with the ``state_size`` of th model.
    num_inputs: int
        The number of input variables, including both numeric and categorical inputs for the relevant channel.
    hidden_dim: int
        The embedding width of the output.
    dropout: float
        The dropout rate associated with ``GatedResidualNetwork`` objects composing this object.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context_dim = context_dim

        # A GRN to apply on the flat concatenation of the input representation (all inputs together),
        # possibly provided with context information
        self.flattened_grn = GatedResidualNetwork(input_dim=self.num_inputs * self.input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  output_dim=self.num_inputs,
                                                  dropout=self.dropout,
                                                  context_dim=self.context_dim,
                                                  batch_first=batch_first)
        # activation for transforming the GRN output to weights
        self.softmax = nn.Softmax(dim=1)

        # In addition, each input variable (after transformed to its wide representation) goes through its own GRN.
        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(input_dim=self.input_dim,
                                     hidden_dim=self.hidden_dim,
                                     output_dim=self.hidden_dim,
                                     dropout=self.dropout,
                                     batch_first=batch_first))

    def forward(self, flattened_embedding, context=None):
        # ===========================================================================
        # Infer variable selection weights - using the flattened representation GRN
        # ===========================================================================
        # the flattened embedding should be of shape [(num_samples * num_temporal_steps) x (num_inputs x input_dim)]
        # where in our case input_dim represents the model_dim or the state_size.
        # in the case of static variables selection, num_temporal_steps is disregarded and can be thought of as 1.
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        # After that step "sparse_weights" is of shape [(num_samples * num_temporal_steps) x num_inputs x 1]

        # Before weighting the variables - apply a GRN on each transformed input
        processed_inputs = []
        for i in range(self.num_inputs):
            # select slice of embedding belonging to a single input - and apply the variable-specific GRN
            # (the slice is taken from the flattened concatenated embedding)
            processed_inputs.append(
                self.single_variable_grns[i](flattened_embedding[..., (i * self.input_dim): (i + 1) * self.input_dim]))
        # each element in the resulting list is of size: [(num_samples * num_temporal_steps) x state_size],
        # and each element corresponds to a single input variable

        # combine the outputs of the single-var GRNs (along an additional axis)
        processed_inputs = torch.stack(processed_inputs, dim=-1)
        # Dimensions:
        # processed_inputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        # weigh them by multiplying with the weights tensor viewed as
        # [(num_samples * num_temporal_steps) x 1 x num_inputs]
        # so that the weight given to each input variable (for each time-step/observation) multiplies the entire state
        # vector representing the specific input variable on this specific time-step
        outputs = processed_inputs * sparse_weights.transpose(1, 2)
        # Dimensions:
        # outputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

        # and finally sum up - for creating a weighted sum representation of width state_size for every time-step
        outputs = outputs.sum(axis=-1)
        # Dimensions:
        # outputs: [(num_samples * num_temporal_steps) x state_size]

        return outputs, sparse_weights


class InputChannelEmbedding(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    A module to handle the transformation/embedding of an input channel composed of numeric tensors and categorical
    tensors.
    It holds a NumericInputTransformation module for handling the embedding of the numeric inputs,
    and a CategoricalInputTransformation module for handling the embedding of the categorical inputs.

    Parameters
    ----------
    state_size : int
        The state size of the model, which determines the embedding dimension/width of each input variable.
    num_numeric : int
        The number of numeric input variables associated with the input channel.
    categorical_cardinalities: List[int]
        The number of categories associated with each of the categorical input variables.
    time_distribute: Optional[bool]
        A boolean indicating whether to wrap the composing transformations using the ``TimeDistributed`` module.
    """

    def __init__(self, state_size: int, num_numeric: int, categorical_cardinalities: List[int],
                 time_distribute: Optional[bool] = False):
        super(InputChannelEmbedding, self).__init__()

        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = len(categorical_cardinalities)
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute

        if (num_numeric + self.num_categorical) < 1:
            raise ValueError(f"""At least a single input variable (either numeric or categorical) should be included
            as part of the input channel.
            According to the provided configuration:
            num_numeric + num_categorical = {num_numeric} + {self.num_categorical} =
            {num_numeric + self.num_categorical} < 1
            """)

        if self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_inputs=num_numeric, state_size=state_size), return_reshaped=False)
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(state_size=state_size, cardinalities=categorical_cardinalities),
                return_reshaped=False)
        else:
            self.numeric_transform = NumericInputTransformation(num_inputs=num_numeric, state_size=state_size)
            self.categorical_transform = CategoricalInputTransformation(state_size=state_size,
                                                                        cardinalities=categorical_cardinalities)

        # in case some input types are not expected there is no need in the type specific transformation.
        # instead the "transformation" will return an empty list
        if num_numeric == 0:
            self.numeric_transform = NullTransform()
        if self.num_categorical == 0:
            self.categorical_transform = NullTransform()

    def forward(self, x_numeric, x_categorical) -> torch.tensor:
        batch_shape = x_numeric.shape if x_numeric.nelement() > 0 else x_categorical.shape

        processed_numeric = self.numeric_transform(x_numeric)
        processed_categorical = self.categorical_transform(x_categorical)
        # Both of the returned values, "processed_numeric" and "processed_categorical" are lists,
        # with "num_numeric" elements and "num_categorical" respectively - each element in these lists corresponds
        # to a single input variable, and is represented by its embedding, shaped as:
        # [(num_samples * num_temporal_steps) x state_size]
        # (for the static input channel, num_temporal_steps is irrelevant and can be treated as 1

        # the resulting embeddings for all the input variables are concatenated to a flattened representation
        merged_transformations = torch.cat(processed_numeric + processed_categorical, dim=1)
        # Dimensions:
        # merged_transformations: [(num_samples * num_temporal_steps) x (state_size * total_input_variables)]
        # total_input_variables stands for the amount of all input variables in the specific input channel, i.e.
        # num_numeric + num_categorical

        # for temporal data we return the resulting tensor to its 3-dimensional shape
        if self.time_distribute:
            merged_transformations = merged_transformations.view(batch_shape[0], batch_shape[1], -1)
            # In that case:
            # merged_transformations: [num_samples x num_temporal_steps x (state_size * total_input_variables)]

        return merged_transformations


class NumericInputTransformation(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    A module for transforming/embeddings the set of numeric input variables from a single input channel.
    Each input variable will be projected using a dedicated linear layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Parameters
    ----------
    num_inputs : int
        The number of numeric input variables associated with this module.
    state_size : int
        The state size of the model, which determines the embedding dimension/width.
    """

    def __init__(self, num_inputs: int, state_size: int):
        super(NumericInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size

        self.numeric_projection_layers = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        # every input variable is projected using its dedicated linear layer,
        # the results are stored as a list
        projections = []
        for i, numeric_projection_layer in enumerate(self.numeric_projection_layers):
            projections.append(numeric_projection_layer(x[:, [i]]))

        return projections


class CategoricalInputTransformation(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    A module for transforming/embeddings the set of categorical input variables from a single input channel.
    Each input variable will be projected using a dedicated embedding layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Parameters
    ----------
    state_size : int
        The state size of the model, which determines the embedding dimension/width.
    cardinalities: List[int]
        The cardinality (number of categories) associated with each of the input variables.
    """

    def __init__(self, state_size: int, cardinalities: List[int]):
        super(CategoricalInputTransformation, self).__init__()
        self.state_size = state_size
        self.cardinalities = cardinalities

        # layers for processing the categorical inputs
        self.categorical_embedding_layers = nn.ModuleList()
        for cardinality in self.cardinalities:
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(self, x: torch.tensor) -> List[torch.tensor]:
        # every input variable is projected using its dedicated embedding layer,
        # the results are stored as a list
        embeddings = []
        for i, categorical_embedding_layer in enumerate(self.categorical_embedding_layers):
            embeddings.append(categorical_embedding_layer(x[:, i]))

        return embeddings


class GateAddNorm(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    This module encapsulates an operation performed multiple times across the TemporalFusionTransformer model.
    The composite operation includes:
    a. A *Dropout* layer.
    b. Gating using a ``GatedLinearUnit``.
    c. A residual connection to an "earlier" signal from the forward pass of the parent model.
    d. Layer normalization.

    Parameters
    ----------
    input_dim: int
        The dimension associated with the expected input of this module.
    dropout: Optional[float]
        The dropout rate associated with the component.
    """

    def __init__(self, input_dim: int, dropout: Optional[float] = None):
        super(GateAddNorm, self).__init__()
        self.dropout_rate = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.gate = TimeDistributed(GatedLinearUnit(input_dim), batch_first=True)
        self.layer_norm = TimeDistributed(nn.LayerNorm(input_dim), batch_first=True)

    def forward(self, x, residual=None):
        if self.dropout_rate:
            x = self.dropout_layer(x)
        x = self.gate(x)
        # perform skipping
        if residual is not None:
            x = x + residual
        # apply normalization layer
        x = self.layer_norm(x)

        return x


class InterpretableMultiHeadAttention(nn.Module):
    """
    Taken from https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py
    Implements section 4.4 from the temporal fusion transoformers paper: https://arxiv.org/abs/1912.09363
    Original attention paper (attention is all you need): https://arxiv.org/pdf/1706.03762v7.pdf

    The mechanism implemented in this module is used to learn long-term relationships across different time-steps.
    It is a modified version of multi-head attention, for enhancing explainability. On this modification,
    as opposed to traditional versions of multi-head attention, the "values" signal is shared for all the heads -
    and additive aggregation is employed across all the heads.
    According to the paper, each head can learn different temporal patterns, while attending to a common set of
    input features which can be interpreted as  a simple ensemble over attention weights into a combined matrix, which,
    compared to the original multi-head attention matrix, yields an increased representation capacity in an efficient
    way.

    Parameters
    ----------
    embed_dim: int
        The dimensions associated with the ``state_size`` of th model, corresponding to the input as well as the output.
    num_heads: int
        The number of attention heads composing the Multi-head attention component.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.d_model = embed_dim  # the state_size (model_size) corresponding to the input and output dimension
        self.num_heads = num_heads  # the number of attention heads
        self.all_heads_dim = embed_dim * num_heads  # the width of the projection for the keys and queries

        self.w_q = nn.Linear(embed_dim, self.all_heads_dim)  # multi-head projection for the queries
        self.w_k = nn.Linear(embed_dim, self.all_heads_dim)  # multi-head projection for the keys
        self.w_v = nn.Linear(embed_dim, embed_dim)  # a single, shared, projection for the values

        # the last layer is used for final linear mapping (corresponds to W_H in the paper)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        num_samples = q.size(0)

        # Dimensions:
        # queries tensor - q: [num_samples x num_future_steps x state_size]
        # keys tensor - k: [num_samples x (num_total_steps) x state_size]
        # values tensor - v: [num_samples x (num_total_steps) x state_size]

        # perform linear operation and split into h heads
        q_proj = self.w_q(q).view(num_samples, -1, self.num_heads, self.d_model)
        k_proj = self.w_k(k).view(num_samples, -1, self.num_heads, self.d_model)
        v_proj = self.w_v(v).repeat(1, 1, self.num_heads).view(num_samples, -1, self.num_heads, self.d_model)

        # transpose to get the following shapes
        q_proj = q_proj.transpose(1, 2)  # (num_samples x num_future_steps x num_heads x state_size)
        k_proj = k_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        v_proj = v_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)

        # calculate attention using function we will define next
        attn_outputs_all_heads, attn_scores_all_heads = self.attention(q_proj, k_proj, v_proj, mask)
        # Dimensions:
        # attn_scores_all_heads: [num_samples x num_heads x num_future_steps x num_total_steps]
        # attn_outputs_all_heads: [num_samples x num_heads x num_future_steps x state_size]

        # take average along heads
        attention_scores = attn_scores_all_heads.mean(dim=1)
        attention_outputs = attn_outputs_all_heads.mean(dim=1)
        # Dimensions:
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        # attention_outputs: [num_samples x num_future_steps x state_size]

        # weigh attention outputs
        output = self.out(attention_outputs)
        # output: [num_samples x num_future_steps x state_size]

        return output, attention_outputs, attention_scores

    def attention(self, q, k, v, mask=None):
        # Applying the scaled dot product
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        # Dimensions:
        # attention_scores: [num_samples x num_heads x num_future_steps x num_total_steps]

        # Decoder masking is applied to the multi-head attention layer to ensure that each temporal dimension can only
        # attend to features preceding it
        if mask is not None:
            # the mask is broadcast along the batch(dim=0) and heads(dim=1) dimensions,
            # where the mask==True, the scores are "cancelled" by setting a very small value
            new_val = -1e9 if attention_scores.dtype == torch.float32 else -5e4
            attention_scores = attention_scores.masked_fill(mask, new_val)

        # still part of the scaled dot-product attention (dimensions are kept)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # matrix multiplication is performed on the last two-dimensions to retrieve attention outputs
        attention_outputs = torch.matmul(attention_scores, v)
        # Dimensions:
        # attention_outputs: [num_samples x num_heads x num_future_steps x state_size]

        return attention_outputs, attention_scores


@layers_registry.register
class TemporalFusionTransformer(AbstractLayer):
    """
    Based on the code in https://github.com/PlaytikaOSS/tft-torch/blob/main/tft_torch/tft.py

    This class implements the Temporal Fusion Transformer model described in the paper
    `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    <https://arxiv.org/abs/1912.09363>`_.

    Parameters
    ----------
    params: TemporalFusionTransformerParams
        The parameters that describe both the expected structure of the input of the model, and the architectural
        specification of the model.
    input_shape:
        A dict of feature name to the expected input shape (not including batch size).
        The dict keys should be a subset of INPUT_TENSOR_NAMES
    """

    def __init__(self, params: TemporalFusionTransformerParams,
                 input_shape: Dict[str, Tuple[int, ...]]):

        if len(set(input_shape.keys()) - set(INPUT_TENSOR_NAMES)) > 0:
            raise NameError(f'The keys of input_shape have to be a subset of {INPUT_TENSOR_NAMES}. '
                            f'Got {set(input_shape.keys()) - set(INPUT_TENSOR_NAMES)}')

        super().__init__(params=params, input_shape=input_shape)

        self.historical_timeseries_representative_key = \
            'historical_timeseries_numeric' if 'historical_timeseries_numeric' in input_shape.keys() else \
                'historical_timeseries_categorical'
        self.future_timeseries_representative_key = \
            'future_timeseries_numeric' if 'future_timeseries_numeric' in input_shape.keys() else \
                'future_timeseries_categorical'

        # ============
        # model props
        # ============
        self.target_window_start_index = \
            (self.params.target_window_start - 1) if self.params.target_window_start is not None else 0
        self.output_quantiles = self.params.output_quantiles
        self.num_outputs = self.params.num_outputs
        state_size = self.params.state_size
        dropout = self.params.dropout

        input_category_to_input_type_to_num_features = \
            {k.split('_')[0]: {input_type: input_shape.get(f'{k}_{input_type}', (0,))[0] for input_type in FEATURE_TYPES}
             for k in INPUT_TENSORS_BASE_NAMES}

        # =====================
        # Input Transformation
        # =====================
        self._init_input_transforms(feature_name_to_num_numerics={k: v['numeric'] for k, v in
                                                                  input_category_to_input_type_to_num_features.items()})

        # =============================
        # Variable Selection Networks
        # =============================
        self._init_selection_layers(state_size, dropout,
                                    feature_name_to_num_inputs={k: sum(v.values()) for k, v in
                                                                input_category_to_input_type_to_num_features.items()})

        # =============================
        # static covariate encoders
        # =============================
        static_covariate_encoder = GatedResidualNetwork(input_dim=state_size, hidden_dim=state_size,
                                                        output_dim=state_size, dropout=dropout)
        self.static_encoder_layers = {k: copy.deepcopy(static_covariate_encoder) for k in STATIC_ENCODER_LAYERS_NAMES}
        self.static_encoder_layers = nn.ModuleDict(self.static_encoder_layers)

        # ============================================================
        # Locality Enhancement with Sequence-to-Sequence processing
        # ============================================================
        self.lstms = {k: nn.LSTM(input_size=state_size, hidden_size=state_size, num_layers=self.params.lstm_layers,
                                 dropout=dropout, batch_first=True)
                      for k in ['past', 'future']}
        self.lstms = nn.ModuleDict(self.lstms)
        self.post_lstm_gating = GateAddNorm(input_dim=state_size, dropout=dropout)

        # ============================================================
        # Static enrichment
        # ============================================================
        self.static_enrichment_grn = GatedResidualNetwork(input_dim=state_size, hidden_dim=state_size, dropout=dropout,
                                                          output_dim=state_size, context_dim=state_size)

        # ============================================================
        # Temporal Self-Attention
        # ============================================================
        self.multihead_attn = \
            InterpretableMultiHeadAttention(embed_dim=state_size, num_heads=self.params.attention_heads)
        self.post_attention_gating = GateAddNorm(input_dim=state_size, dropout=dropout)

        # ============================================================
        # Position-wise feed forward
        # ============================================================
        self.pos_wise_ff_grn = GatedResidualNetwork(input_dim=state_size, hidden_dim=state_size, output_dim=state_size,
                                                    dropout=dropout)
        self.pos_wise_ff_gating = GateAddNorm(input_dim=state_size, dropout=None)

        # ============================================================
        # Output layer
        # ============================================================
        # self.output_layer = nn.Linear(state_size, self.num_outputs)
        self.output_layer = nn.Conv1d(in_channels=input_shape['future_timeseries_numeric'][1], out_channels=1,
                                      kernel_size=1)
        self.output_layer2 = nn.Linear(state_size, self.params.output_size)

    def _init_input_transforms(self, feature_name_to_num_numerics: Dict[str, int]):
        self.input_transforms = {name: InputChannelEmbedding(state_size=self.params.state_size, num_numeric=num_numeric,
                                                             time_distribute=name != 'static',
                                                             categorical_cardinalities=self.params.categorical_cardinalities[name])
                                 for name, num_numeric in feature_name_to_num_numerics.items()
                                 }
        self.input_transforms = nn.ModuleDict(self.input_transforms)

    def _init_selection_layers(self, state_size: int, dropout: float, feature_name_to_num_inputs: Dict[str, int]):
        self.selection_layers = {name: VariableSelectionNetwork(input_dim=state_size, hidden_dim=state_size,
                                                                dropout=dropout,
                                                                context_dim=None if name == 'static' else state_size,
                                                                num_inputs=num_inputs)
                                 for name, num_inputs in feature_name_to_num_inputs.items()
                                 }
        self.selection_layers = nn.ModuleDict(self.selection_layers)

    def apply_temporal_selection(self, temporal_representation: torch.tensor,
                                 static_selection_signal: torch.tensor,
                                 temporal_selection_module: VariableSelectionNetwork
                                 ) -> Tuple[torch.tensor, torch.tensor]:
        num_samples, num_temporal_steps, _ = temporal_representation.shape

        # replicate the selection signal along the time dimension
        time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                             time_steps=num_temporal_steps)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]
        # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]

        # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
        temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # Dimensions:
        # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        # applying the selection module across time
        temporal_selection_output, temporal_selection_weights = temporal_selection_module(
            flattened_embedding=temporal_flattened_embedding, context=time_distributed_context)
        # Dimensions:
        # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
        # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]

        # Reshape the selection outputs and selection weights - to represent the temporal dimension separately
        temporal_selection_output = temporal_selection_output.view(num_samples, num_temporal_steps, -1)
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).view(num_samples, num_temporal_steps, -1)
        # Dimensions:
        # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
        # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]

        return temporal_selection_output, temporal_selection_weights

    @staticmethod
    def replicate_along_time(static_signal: torch.tensor, time_steps: int) -> torch.tensor:
        """
        This method gets as an input a static_signal (non-temporal tensor) [num_samples x num_features],
        and replicates it along time for 'time_steps' times,
        creating a tensor of [num_samples x time_steps x num_features]

        Args:
            static_signal: the non-temporal tensor for which the replication is required.
            time_steps: the number of time steps according to which the replication is required.

        Returns:
            torch.tensor: the time-wise replicated tensor
        """
        time_distributed_signal = static_signal.unsqueeze(1).repeat(1, time_steps, 1)
        return time_distributed_signal

    @staticmethod
    def stack_time_steps_along_batch(temporal_signal: torch.tensor) -> torch.tensor:
        """
        This method gets as an input a temporal signal [num_samples x time_steps x num_features]
        and stacks the batch dimension and the temporal dimension on the same axis (dim=0).

        The last dimension (features dimension) is kept as is, but the rest is stacked along dim=0.
        """
        return temporal_signal.view(-1, temporal_signal.size(-1))

    def transform_inputs(self, batch: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        """
        This method processes the batch and transform each input channel
        (historical_timeseries, future_timeseries, static) separately to eventually return the learned embedding for
        each of the input channels

        each feature is embedded to a vector of state_size dimension:
        - numeric features will be projected using a linear layer
        - categorical features will be embedded using an embedding layer

        eventually the embedding for all the features will be concatenated together on the last dimension of the tensor
        (i.e. dim=1 for the static features, dim=2 for the temporal data).

        """
        empty_tensor = torch.empty((0, 0))

        return \
            {base_name.split('_')[0]:
                 self.input_transforms[base_name.split('_')[0]](x_numeric=batch.get(f'{base_name}_numeric', empty_tensor),
                                                                x_categorical=batch.get(f'{base_name}_categorical', empty_tensor))
             for base_name in INPUT_TENSORS_BASE_NAMES}

    def get_static_encoders(self, selected_static: torch.tensor) -> Dict[str, torch.tensor]:
        """
        This method processes the variable selection results for the static data, yielding signals which are designed
        to allow better integration of the information from static metadata.
        Each of the resulting signals is generated using a separate GRN, and is eventually wired into various locations
        in the temporal fusion decoder, for allowing static variables to play an important role in processing.

        c_selection will be used for temporal variable selection
        c_seq_hidden & c_seq_cell will be used both for local processing of temporal features
        c_enrichment will be used for enriching temporal features with static information.
        """
        return {k: layer(selected_static) for k, layer in self.static_encoder_layers.items()}

    def apply_sequential_processing(self, selected_historical: torch.tensor, selected_future: torch.tensor,
                                    c_seq_hidden: torch.tensor, c_seq_cell: torch.tensor) -> torch.tensor:
        """
        This part of the model is designated to mimic a sequence-to-sequence layer which will be used for local
        processing.
        On that part the historical ("observed") information will be fed into a recurrent layer called "Encoder" and
        the future information ("known") will be fed into a recurrent layer called "Decoder".
        This will generate a set of uniform temporal features which will serve as inputs into the temporal fusion
        decoder itself.
        To allow static metadata to influence local processing, we use "c_seq_hidden" and "c_seq_cell" context vectors
        from the static covariate encoders to initialize the hidden state and the cell state respectively.
        The output of the recurrent layers is gated and fused with a residual connection to the input of this block.
        """

        # concatenate the historical (observed) temporal signal with the futuristic (known) temporal signal, along the
        # time dimension
        lstm_input = torch.cat([selected_historical, selected_future], dim=1)

        # the historical temporal signal is fed into the first recurrent module
        # using the static metadata as initial hidden and cell state
        # (initial cell and hidden states are replicated for feeding to each layer in the stack)
        past_lstm_output, hidden = self.lstms['past'](selected_historical,
                                                      (c_seq_hidden.unsqueeze(0).repeat(self.params.lstm_layers, 1, 1),
                                                       c_seq_cell.unsqueeze(0).repeat(self.params.lstm_layers, 1, 1)))

        # the future (known) temporal signal is fed into the second recurrent module
        # using the latest (hidden,cell) state of the first recurrent module
        # for setting the initial (hidden,cell) state.
        future_lstm_output, _ = self.lstms['future'](selected_future, hidden)

        # concatenate the historical recurrent output with the futuristic recurrent output, along the time dimension
        lstm_output = torch.cat([past_lstm_output, future_lstm_output], dim=1)

        # perform gating to the recurrent output signal, using a residual connection to input of this block
        gated_lstm_output = self.post_lstm_gating(lstm_output, residual=lstm_input)
        return gated_lstm_output

    def apply_static_enrichment(self, gated_lstm_output: torch.tensor,
                                static_enrichment_signal: torch.tensor) -> torch.tensor:
        """
        This static enrichment stage enhances temporal features with static metadata using a GRN.
        The static enrichment signal is an output of a static covariate encoder, and the GRN is shared across time.
        """
        num_samples, num_temporal_steps, _ = gated_lstm_output.shape

        # replicate the selection signal along time
        time_distributed_context = self.replicate_along_time(static_signal=static_enrichment_signal,
                                                             time_steps=num_temporal_steps)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]

        # for applying the same GRN module on all time-steps, we stack the time dimension with the batch dimension
        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # Dimensions:
        # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        # applying the GRN using the static enrichment signal as context data
        enriched_sequence = self.static_enrichment_grn(flattened_gated_lstm_output,
                                                       context=time_distributed_context)
        # Dimensions:
        # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]

        # reshape back to represent temporal dimension separately
        enriched_sequence = enriched_sequence.view(num_samples, -1, self.params.state_size)
        # Dimensions:
        # enriched_sequence: [num_samples x num_temporal_steps x state_size]

        return enriched_sequence

    def apply_self_attention(self, enriched_sequence: torch.tensor,
                             num_historical_steps: int,
                             num_future_steps: int):
        # create a mask - so that future steps will be exposed (able to attend) only to preceding steps
        output_sequence_length = num_future_steps - self.target_window_start_index
        mask = torch.cat([torch.zeros(output_sequence_length,
                                      num_historical_steps + self.target_window_start_index,
                                      device=enriched_sequence.device),
                          torch.triu(torch.ones(output_sequence_length, output_sequence_length,
                                                device=enriched_sequence.device),
                                     diagonal=1)], dim=1)
        # Dimensions:
        # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]

        # apply the InterpretableMultiHeadAttention mechanism
        post_attention, attention_outputs, attention_scores = self.multihead_attn(
            q=enriched_sequence[:, (num_historical_steps + self.target_window_start_index):, :],  # query
            k=enriched_sequence,  # keys
            v=enriched_sequence,  # values
            mask=mask.bool())
        # Dimensions:
        # post_attention: [num_samples x num_future_steps x state_size]
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]

        # Apply gating with a residual connection to the input of this stage.
        # Because the output of the attention layer is only for the future time-steps,
        # the residual connection is only to the future time-steps of the temporal input signal
        gated_post_attention = self.post_attention_gating(
            x=post_attention,
            residual=enriched_sequence[:, (num_historical_steps + self.target_window_start_index):, :])
        # Dimensions:
        # gated_post_attention: [num_samples x num_future_steps x state_size]

        return gated_post_attention, attention_scores

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        The batch should have a dict with where the keys are a subset of INPUT_TENSOR_NAMES
        """
        if len(set(batch.keys()) - set(INPUT_TENSOR_NAMES)) > 0:
            raise NameError(f'The keys of batch have to be a subset of {INPUT_TENSOR_NAMES}. '
                            f'Got {set(batch.keys()) - set(INPUT_TENSOR_NAMES)}')

        # added by yoni, the timeseries data should be channels last
        for k in batch.keys():
            if batch[k].dim() == 3:
                batch[k] = batch[k].transpose(2, 1)

        # infer batch structure
        num_samples, num_historical_steps, _ = batch[self.historical_timeseries_representative_key].shape
        num_future_steps = batch[self.future_timeseries_representative_key].shape[1]
        # define output_sequence_length : num_future_steps - self.target_window_start_idx

        # =========== Transform all input channels ==============
        feature_type_to_transformed_inputs = self.transform_inputs(batch)
        # Dimensions:
        # static_rep: [num_samples x (total_num_static_inputs * state_size)]
        # historical_timeseries_rep: [num_samples x num_historical_steps x (total_num_historical_inputs * state_size)]
        # future_timeseries_rep: [num_samples x num_future_steps x (total_num_future_inputs * state_size)]

        # =========== Static Variables Selection ==============
        selected_static, static_weights = self.selection_layers['static'](feature_type_to_transformed_inputs['static'])
        # Dimensions:
        # selected_static: [num_samples x state_size]
        # static_weights: [num_samples x num_static_inputs x 1]

        # =========== Static Covariate Encoding ==============
        encoded_static = self.get_static_encoders(selected_static)
        # each of the static encoders signals is of shape: [num_samples x state_size]

        # =========== Historical/Future variables selection ==============
        selected = dict()
        selection_weights = dict()
        for feature_category in ['historical', 'future']:
            selected[feature_category], selection_weights[feature_category] = self.apply_temporal_selection(
                temporal_representation=feature_type_to_transformed_inputs[feature_category],
                static_selection_signal=encoded_static['selection'],
                temporal_selection_module=self.selection_layers[feature_category])
            # Dimensions:
            # selected_(historical/future): [num_samples x num_(historical/future)_steps x state_size]
            # (historical/future)_selection_weights:
            # [num_samples x num_(historical/future)_steps x total_num_(historical/future)_inputs]

        # =========== Locality Enhancement - Sequential Processing ==============
        gated_lstm_output = self.apply_sequential_processing(selected_historical=selected['historical'],
                                                             selected_future=selected['future'],
                                                             c_seq_hidden=encoded_static['sequential_state_init'],
                                                             c_seq_cell=encoded_static['sequential_cell_init'])
        # Dimensions:
        # gated_lstm_output : [num_samples x (num_historical_steps + num_future_steps) x state_size]

        # =========== Static enrichment ==============
        enriched_sequence = self.apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                         static_enrichment_signal=encoded_static['enrichment'])
        # Dimensions:
        # enriched_sequence: [num_samples x (num_historical_steps + num_future_steps) x state_size]

        # =========== self-attention ==============
        gated_post_attention, attention_scores = self.apply_self_attention(enriched_sequence=enriched_sequence,
                                                                           num_historical_steps=num_historical_steps,
                                                                           num_future_steps=num_future_steps)
        # Dimensions:
        # attention_scores: [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
        # gated_post_attention: [num_samples x output_sequence_length x state_size]

        # =========== position-wise feed-forward ==============
        # Applying an additional non-linear processing to the outputs of the self-attention layer using a GRN,
        # where its weights are shared across the entire layer
        post_poswise_ff_grn = self.pos_wise_ff_grn(gated_post_attention)
        # Also applying a gated residual connection skipping over the
        # attention block (using sequential processing output), providing a direct path to the sequence-to-sequence
        # layer, yielding a simpler model if additional complexity is not required
        gated_poswise_ff = self.pos_wise_ff_gating(
            post_poswise_ff_grn,
            residual=gated_lstm_output[:, (num_historical_steps + self.target_window_start_index):, :])
        # Dimensions:
        # gated_poswise_ff: [num_samples x output_sequence_length x state_size]

        output = self.output_layer(gated_poswise_ff)[:, 0, :]
        output = self.output_layer2(output)
        return output

    def output_shape(self) -> LayersInputOutputShape:
        return (self.params.output_size,)
