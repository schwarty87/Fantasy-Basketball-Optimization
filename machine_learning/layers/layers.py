import torch
from torch import nn
from typing import List, Tuple, Optional, Union, Dict
from machine_learning.layers.abstract_layer import AbstractLayer, LayersInputOutputShape
from machine_learning.params.layers_params import (TabularEncodingParams, TimeseriesReductionParams, DenseLayersParams, \
    EmbeddingsLayerParams, ReshapeLayerParams, CombineTensorsParams, TransformerEncoderPytorchParams, \
    SplitAttentionCombineParams, TransformerEncoderCustomParams, SplitAttentionCombineWrapperParams)
from machine_learning.enums import ActivationFunction
from machine_learning.registries import layers_registry


@layers_registry.register
class TabularEncodingLayer(AbstractLayer):
    def __init__(self, params: TabularEncodingParams, input_shape: Tuple[int, ...]):
        """Each tabular input is sent through a nn.Linear layer. The output should be the same size as the input"""
        super().__init__(params=params, input_shape=input_shape)

        if params.encode_separately:
            self.linear_layers = nn.ModuleList([nn.Linear(1, 1) for _ in range(input_shape[-1])])
        else:
            self.linear_layers = nn.Linear(input_shape[-1], input_shape[-1])
        self.activation = ActivationFunction.member_to_function(params.activation)

    def forward(self, x):
        if self.params.encode_separately:
            x_split = torch.split(x, split_size_or_sections=1, dim=-1)
            linear_output = [linear_layer(curr_x) for curr_x, linear_layer in zip(x_split, self.linear_layers)]
            linear_output = torch.concat(linear_output, dim=-1)
        else:
            linear_output = self.linear_layers(x)

        if self.activation is not None:
            return self.activation(linear_output)
        return linear_output

    def output_shape(self) -> Tuple[int]:
        return self.input_shape


@layers_registry.register
class TimeseriesReductionLayer(AbstractLayer):
    def __init__(self, params: TimeseriesReductionParams, input_shape: Tuple[int, int]):
        """
        Reshape the timeseries data
        :param params: an instance of TimeseriesReductionParams
        :param input_shape: number of channels in the input timeseries
        """
        super().__init__(params=params, input_shape=input_shape)

        if params.temporal_output_size is not None:
            self.temporal_reduction = nn.Linear(input_shape[1], params.temporal_output_size)
            output_length = params.temporal_output_size
        else:
            output_length = input_shape[1]

        if params.channels_output_size is not None:
            # same as nn.Linear, but works on dimension 1 and not dimension -1
            self.channels_reduction = nn.Conv1d(input_shape[0], params.channels_output_size, kernel_size=1)
            num_outputs = params.channels_output_size
        else:
            num_outputs = input_shape[0]

        if params.flatten:
            self._output_shape = (num_outputs * output_length,)
        else:
            self._output_shape = (num_outputs, output_length)

    def forward(self, x):
        # x input shape: [batch_size, num_channels, num_time_steps]
        if self.params.temporal_output_size is not None:
            x = self.temporal_reduction(x)      # [batch_size, num_channels, temporal_output_size]
        if self.params.channels_output_size is not None:
            x = self.channels_reduction(x)      # [batch_size, num_output_channels, temporal_output_size]
        if self.params.flatten:
            x = nn.Flatten()(x)           # [batch_size, num_output_channels * temporal_output_size]
        return x

    def output_shape(self) -> LayersInputOutputShape:
        return self._output_shape


@layers_registry.register
class DenseLayers(AbstractLayer):
    def __init__(self, params: DenseLayersParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)
        self.layers = nn.ModuleList()
        in_features = input_shape[-1]
        for out_features in params.num_nodes_per_layer[:-1]:
            self.layers += self._single_dense_layer(in_features=in_features, out_features=out_features,
                                               activation=params.activation, use_batchnorm=params.use_batchnorm,
                                               dropout_rate=params.dropout_rate)
            in_features = out_features

        # last layer
        self.layers += self._single_dense_layer(in_features=in_features, out_features=params.num_nodes_per_layer[-1],
                                           activation=params.activation_on_last_layer, use_batchnorm=False,
                                           dropout_rate=None)

    @staticmethod
    def _single_dense_layer(in_features: int, out_features: int, activation: ActivationFunction,
                            use_batchnorm: bool, dropout_rate: Optional[float] = None) -> List[nn.Module]:
        layers_for_single_dense = [nn.Linear(in_features=in_features, out_features=out_features)]
        activation_function = ActivationFunction.member_to_function(activation)
        if activation_function is not None:
            layers_for_single_dense.append(activation_function)
        if use_batchnorm:
            # layers_for_single_dense.append(BatchNorm1d(num_features=out_features))
            layers_for_single_dense.append(nn.LayerNorm(normalized_shape=[out_features]))
        if dropout_rate is not None:
            layers_for_single_dense.append(nn.Dropout(dropout_rate))
        return layers_for_single_dense

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def output_shape(self) -> Tuple[int]:
        return self.input_shape[:-1] + (self.params.num_nodes_per_layer[-1],)


@layers_registry.register
class EmbeddingsLayer(AbstractLayer):
    def __init__(self, params: EmbeddingsLayerParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)
        self.embedding_layers = nn.ModuleList()
        for num_embeddings, embedding_dim in zip(params.num_embeddings, params.embeddings_dims):
            pad_idx = (num_embeddings - 1) if num_embeddings > 0 else None
            self.embedding_layers.append(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                                      max_norm=self.params.max_norm, padding_idx=pad_idx))

    def forward(self, x) -> Dict[str, torch.Tensor]:
        x_split = torch.split(x, 1, dim=-1)
        embedding_outputs = dict()
        for i, (features, embedding_layer) in enumerate(zip(x_split, self.embedding_layers)):
            output_name = self.params.output_tensor_name[i]
            embedding_outputs[output_name] = embedding_layer(features[..., 0])
        return embedding_outputs

    def output_shape(self) -> List[Tuple[int, ...]]:
        return [self.input_shape[:-1] + (d,) for d in self.params.embeddings_dims]


@layers_registry.register
class ReshapeLayer(AbstractLayer):
    def __init__(self, params: ReshapeLayerParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)

        self.shape = (-1,) + params.shape

    def forward(self, x) -> torch.Tensor:
        return torch.reshape(x, self.shape)

    def output_shape(self) -> Tuple[int]:
        return self.params.shape


@layers_registry.register
class CombineTensors(AbstractLayer):
    def __init__(self, params: CombineTensorsParams, input_shape: List[Tuple[int, ...]]):
        super().__init__(params=params, input_shape=input_shape)
        if self.params.combine_func_str == 'concat':
            self.combine_func = self._concat
        elif self.params.combine_func_str == 'sum':
            self.combine_func = self._sum

            if len(set([s[-1] for s in input_shape])) != 1:
                raise ValueError('when combining tensors using sum, all tensors have to have the same dimension!')

        elif self.params.combine_func_str == 'stack':
            self.combine_func = self._stack
            # Strict validation: shapes must be identical to stack
            if not all(s == input_shape[0] for s in input_shape[1:]):
                raise ValueError(
                    f'when combining tensors using stack, all tensors must have the same shape! Got: {input_shape}')
        else:
            raise ValueError(f'combine_func_str has to be one of "concat", "sum", or "stack". Got: {self.params.combine_func_str}')


    def _concat(self, x: List[torch.Tensor]):
        return torch.concat(x, dim=self.params.dim)

    @staticmethod
    def _sum(x: List[torch.Tensor]):
        return torch.sum(torch.stack(x, dim=0), dim=0)

    def _stack(self, x: List[torch.Tensor]):
        # Stacks along the new dimension specified by params.dim
        return torch.stack(x, dim=self.params.dim)

    def forward(self, x: Union[Dict[str, torch.Tensor], List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, dict):
            x = [x[n] for n in self.params.input_tensor_name]
        return self.combine_func(x)

    def output_shape(self) -> Tuple[int]:
        if self.params.combine_func_str == 'concat':
            dim = (len(self.input_shape[0]) if self.params.dim == -1 else self.params.dim) - 1
            output_shape = list(self.input_shape[0])
            output_shape[dim] = sum([curr_shape[dim] for curr_shape in self.input_shape])
            return tuple(output_shape)
        elif self.params.combine_func_str == 'sum':
            return self.input_shape[0]
        elif self.params.combine_func_str == 'stack':
            # Calculate insertion index relative to feature dimensions (excluding batch)
            # params.dim is 0-indexed including batch.
            base_shape = list(self.input_shape[0])
            dim_idx = self.params.dim

            if dim_idx > 0:
                dim_idx -= 1  # Adjust for missing batch dim in output_shape
            elif dim_idx < 0:
                # Handle negative indexing (e.g. -1 means end)
                # If shape is (F,), len=1. stack at -1 -> index 1.
                dim_idx = len(base_shape) + dim_idx + 1

            # Insert the new dimension (size = number of tensors being stacked)
            base_shape.insert(dim_idx, len(self.input_shape))
            return tuple(base_shape)

        raise ValueError('combine_func_str has to be one of "concat/"sum"!')


@layers_registry.register
class TransformerEncoderPytorch(AbstractLayer):
    def __init__(self, params: TransformerEncoderPytorchParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)

        encoder_layer = nn.TransformerEncoderLayer(**params.transformer_encoder_layer_kwargs)
        self.layer = nn.TransformerEncoder(encoder_layer=encoder_layer, **params.transformer_encoder_kwargs)

    def forward(self, x):
        return self.layer(x)

    def output_shape(self) -> LayersInputOutputShape:
        return self.input_shape


@layers_registry.register
class SplitAttentionCombineWrapper(AbstractLayer):
    def __init__(self, params: SplitAttentionCombineWrapperParams, input_shape: Tuple[int, ...]):
        super(SplitAttentionCombineWrapper, self).__init__(params=params, input_shape=input_shape)
        single_layer_params = SplitAttentionCombineParams(**params.single_layer_kwargs)
        self.layers = nn.ModuleList([SplitAttentionCombine(params=single_layer_params, input_shape=input_shape)
                                     for _ in range(params.num_layers)]
                                    )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def output_shape(self) -> LayersInputOutputShape:
        return self.input_shape


@layers_registry.register
class SplitAttentionCombine(AbstractLayer):
    def __init__(self, params: SplitAttentionCombineParams, input_shape: Tuple[int, ...]):
        super(SplitAttentionCombine, self).__init__(params=params, input_shape=input_shape)
        transformer_encoder_params = TransformerEncoderCustomParams(input_tensor_name='', output_tensor_name=None,
                                                                    d_model=params.d_model, n_heads=params.n_heads,
                                                                    ffdim=params.ffdim, dropout_rate=params.dropout_rate,
                                                                    norm_at_beginning=params.norm_at_beginning,
                                                                    activation=params.activation)
        transformer_encoder_input_shape = (input_shape[0] // 2, input_shape[-1])
        self.transformer_encoder_within = TransformerEncoderCustom(params=transformer_encoder_params,
                                                                   input_shape=transformer_encoder_input_shape)
        self.transformer_encoder_between = TransformerEncoderCustom(params=transformer_encoder_params,
                                                                    input_shape=transformer_encoder_input_shape)

        self.fusion_layer = nn.Linear(params.d_model * 2, params.d_model)
        self.norm = nn.LayerNorm(params.d_model)

        # self.linear1 = nn.Conv1d(input_shape[0], input_shape[0] // 2, kernel_size=1)
        # self.linear2 = nn.Conv1d(input_shape[0], input_shape[0] // 2, kernel_size=1)
        # # self.linear3 = nn.Conv1d(input_shape[0] * 2, input_shape[0], kernel_size=1)

    def forward(self, x: torch.Tensor):
        x1, x2 = torch.split(x, 5 + self.params.n_bench_players, dim=1)
        within1 = self.transformer_encoder_within(x1)
        within2 = self.transformer_encoder_within(x2)
        between1 = self.transformer_encoder_between(x1, x2)
        between2 = self.transformer_encoder_between(x2, x1)

        fused1 = torch.cat([within1, between1], dim=-1)
        fused2 = torch.cat([within2, between2], dim=-1)
        out1 = self.fusion_layer(fused1)
        out1 = self.norm(x1 + out1)
        out2 = self.fusion_layer(fused2)
        out2 = self.norm(x2 + out2)
        out_combined = torch.cat([out1, out2], dim=1)

        return out_combined

    def output_shape(self) -> LayersInputOutputShape:
        return self.input_shape


@layers_registry.register
class TransformerEncoderCustom(AbstractLayer):
    def __init__(self, params: TransformerEncoderCustomParams, input_shape: Tuple[int, ...]):
        super(TransformerEncoderCustom, self).__init__(params, input_shape=input_shape)
        self.attn_heads = list()

        # --- Attention Block ---
        self.norm1 = nn.LayerNorm(normalized_shape=params.d_model)
        self.attn = nn.MultiheadAttention(embed_dim=params.d_model,
                                       num_heads=params.n_heads,
                                       dropout=params.dropout_rate,
                                       batch_first=True)
        self.dropout = nn.Dropout(p=params.dropout_rate)

        # --- Feed-Forward Block ---
        self.norm2 = nn.LayerNorm(normalized_shape=params.d_model)
        activation_layer = (
            ActivationFunction.member_to_function(params.activation)
            if isinstance(params.activation, ActivationFunction)
            else (
                ActivationFunction.member_to_function(ActivationFunction[params.activation.upper()])
                if isinstance(params.activation, str) and hasattr(ActivationFunction, params.activation.upper())
                else nn.ReLU()
            )
        )
        if activation_layer is None:
            activation_layer = nn.Identity()
        self.feedforward = nn.Sequential(nn.Linear(in_features=params.d_model, out_features=params.ffdim),
                                         activation_layer,
                                         nn.Dropout(p=params.dropout_rate),
                                         nn.Linear(in_features=params.ffdim, out_features=params.d_model)
                                         )

    def forward(self, query, key=None, value=None):
        x = query
        if key is None:
            key = query
        if self.params.norm_at_beginning:
            query = self.norm1(query)
            key = self.norm1(key)
        if value is None:
            value = key
        elif self.params.norm_at_beginning:
            value = self.norm1(value)

        # attention
        attn_output, attn_output_weights = self.attn(query, key, value)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output + x
        if not self.params.norm_at_beginning:
            attn_output = self.norm1(attn_output)

        # feed-forward
        ff_output = attn_output
        if self.params.norm_at_beginning:
            ff_output = self.norm2(ff_output)
        ff_output = self.feedforward(ff_output)
        ff_output = self.dropout(ff_output)
        ff_output = ff_output + attn_output
        if not self.params.norm_at_beginning:
            ff_output = self.norm2(ff_output)
        return ff_output

    def output_shape(self) -> LayersInputOutputShape:
        return self.input_shape


# #############################################################################################
# class TransformerEncoderCustom(AbstractLayer):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#
#     TransformerEncoderLayer can handle either traditional torch.tensor inputs,
#     or Nested Tensor inputs.  Derived classes are expected to similarly accept
#     both input formats.  (Not all combinations of inputs are currently
#     supported by TransformerEncoderLayer while Nested Tensor is in prototype
#     state.)
#
#     If you are implementing a custom layer, you may derive it either from
#     the Module or TransformerEncoderLayer class.  If your custom layer
#     supports both torch.Tensors and Nested Tensors inputs, make its
#     implementation a derived class of TransformerEncoderLayer. If your custom
#     Layer supports only torch.Tensor inputs, derive its implementation from
#     Module.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of the intermediate layer, can be a string
#             ("relu" or "gelu") or a unary callable. Default: relu
#         layer_norm_eps: the eps value in layer normalization components (default=1e-5).
#         batch_first: If ``True``, then the input and output tensors are provided
#             as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
#         norm_first: if ``True``, layer norm is done prior to attention and feedforward
#             operations, respectively. Otherwise it's done after. Default: ``False`` (after).
#         bias: If set to ``False``, ``nn.Linear`` and ``LayerNorm`` layers will not learn an additive
#             bias. Default: ``True``.
#
#     Examples::
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         >>> src = torch.rand(10, 32, 512)
#         >>> out = encoder_layer(src)
#
#     Alternatively, when ``batch_first`` is ``True``:
#         >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
#         >>> src = torch.rand(32, 10, 512)
#         >>> out = encoder_layer(src)
#
#     Fast path:
#         forward() will use a special optimized implementation described in
#         `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
#         conditions are met:
#
#         - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
#           argument ``requires_grad``
#         - training is disabled (using ``.eval()``)
#         - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
#         - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
#         - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
#         - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
#           nor ``src_key_padding_mask`` is passed
#         - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
#           unless the caller has manually modified one without modifying the other)
#
#         If the optimized implementation is in use, a
#         `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
#         passed for ``src`` to represent padding more efficiently than using a padding
#         mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
#         returned, and an additional speedup proportional to the fraction of the input that
#         is padding can be expected.
#
#         .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
#          https://arxiv.org/abs/2205.14135
#
#     """
#     __constants__ = ['norm_first']
#
#     def __init__(self, params: TransformerEncoderCustomParams, input_shape: Tuple[int, ...]):
#         super(TransformerEncoderCustom, self).__init__(params, input_shape=input_shape)
#         factory_kwargs = {'device': None, 'dtype': None}
#         layer_norm_eps = 1e-5
#
#         self.self_attn = MultiheadAttention(params.d_model, params.n_heads, dropout=params.dropout_rate,
#                                             bias=True, batch_first=True, **factory_kwargs)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(params.d_model, params.ffdim, bias=True, **factory_kwargs)
#         self.dropout = Dropout(params.dropout_rate)
#         self.linear2 = nn.Linear(params.ffdim, params.d_model, bias=True, **factory_kwargs)
#
#         self.norm_first = params.norm_at_beginning
#         self.norm1 = LayerNorm(params.d_model, eps=layer_norm_eps, bias=True, **factory_kwargs)
#         self.norm2 = LayerNorm(params.d_model, eps=layer_norm_eps, bias=True, **factory_kwargs)
#         self.dropout1 = Dropout(params.dropout_rate)
#         self.dropout2 = Dropout(params.dropout_rate)
#
#         # Legacy string support for activation function.
#         activation = params.activation
#         if isinstance(activation, str):
#             activation = _get_activation_fn(activation)
#
#         # We can't test self.activation in forward() in TorchScript,
#         # so stash some information about it instead.
#         if activation is F.relu or isinstance(activation, ReLU):
#             self.activation_relu_or_gelu = 1
#         elif activation is F.gelu or isinstance(activation, GELU):
#             self.activation_relu_or_gelu = 2
#         else:
#             self.activation_relu_or_gelu = 0
#         self.activation = activation
#
#     def __setstate__(self, state):
#         super().__setstate__(state)
#         if not hasattr(self, 'activation'):
#             self.activation = F.relu
#
#     def forward(
#             self,
#             src: torch.Tensor,
#             src2: torch.Tensor = None,
#             src_mask: Optional[torch.Tensor] = None,
#             src_key_padding_mask: Optional[torch.Tensor] = None,
#             is_causal: bool = False) -> torch.Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#             is_causal: If specified, applies a causal mask as ``src mask``.
#                 Default: ``False``.
#                 Warning:
#                 ``is_causal`` provides a hint that ``src_mask`` is the
#                 causal mask. Providing incorrect hints can result in
#                 incorrect execution, including forward and backward
#                 compatibility.
#
#         Shape:
#             see the docs in Transformer class.
#         """
#         if src2 is None:
#             src2 = src
#
#         src_key_padding_mask = F._canonical_mask(
#             mask=src_key_padding_mask,
#             mask_name="src_key_padding_mask",
#             other_type=F._none_or_dtype(src_mask),
#             other_name="src_mask",
#             target_type=src.dtype
#         )
#
#         src_mask = F._canonical_mask(
#             mask=src_mask,
#             mask_name="src_mask",
#             other_type=None,
#             other_name="",
#             target_type=src.dtype,
#             check_other=False,
#         )
#
#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
#         why_not_sparsity_fast_path = ''
#         if not src.dim() == 3:
#             why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
#         elif self.training:
#             why_not_sparsity_fast_path = "training is enabled"
#         elif not self.self_attn.batch_first :
#             why_not_sparsity_fast_path = "self_attn.batch_first was not True"
#         elif not self.self_attn._qkv_same_embed_dim :
#             why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
#         elif not self.activation_relu_or_gelu:
#             why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
#         elif not (self.norm1.eps == self.norm2.eps):
#             why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
#         elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
#             why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
#         elif self.self_attn.num_heads % 2 == 1:
#             why_not_sparsity_fast_path = "num_head is odd"
#         elif torch.is_autocast_enabled():
#             why_not_sparsity_fast_path = "autocast is enabled"
#         if not why_not_sparsity_fast_path:
#             tensor_args = (
#                 src,
#                 self.self_attn.in_proj_weight,
#                 self.self_attn.in_proj_bias,
#                 self.self_attn.out_proj.weight,
#                 self.self_attn.out_proj.bias,
#                 self.norm1.weight,
#                 self.norm1.bias,
#                 self.norm2.weight,
#                 self.norm2.bias,
#                 self.linear1.weight,
#                 self.linear1.bias,
#                 self.linear2.weight,
#                 self.linear2.bias,
#             )
#
#             # We have to use list comprehensions below because TorchScript does not support
#             # generator expressions.
#             _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
#             if torch.overrides.has_torch_function(tensor_args):
#                 why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
#             elif not all((x.device.type in _supported_device_type) for x in tensor_args):
#                 why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
#                                               f"{_supported_device_type}")
#             elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
#                 why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
#                                               "input/output projection weights or biases requires_grad")
#
#             if not why_not_sparsity_fast_path:
#                 merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
#                 return torch._transformer_encoder_layer_fwd(
#                     src,
#                     self.self_attn.embed_dim,
#                     self.self_attn.num_heads,
#                     self.self_attn.in_proj_weight,
#                     self.self_attn.in_proj_bias,
#                     self.self_attn.out_proj.weight,
#                     self.self_attn.out_proj.bias,
#                     self.activation_relu_or_gelu == 2,
#                     self.norm_first,
#                     self.norm1.eps,
#                     self.norm1.weight,
#                     self.norm1.bias,
#                     self.norm2.weight,
#                     self.norm2.bias,
#                     self.linear1.weight,
#                     self.linear1.bias,
#                     self.linear2.weight,
#                     self.linear2.bias,
#                     merged_mask,
#                     mask_type,
#                 )
#
#
#         x = src
#         if self.norm_first:
#             x = x + self._ca_block(self.norm1(x), self.norm1(src2), src_mask, src_key_padding_mask, is_causal=is_causal)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._ca_block(x, src2, src_mask, src_key_padding_mask, is_causal=is_causal))
#             x = self.norm2(x + self._ff_block(x))
#
#         return x
#
#     # self-attention block
#     def _ca_block(self, x: torch.Tensor, y: torch.Tensor,
#                   attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
#         x = self.self_attn(x, y, y,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False, is_causal=is_causal)[0]
#         return self.dropout1(x)
#
#     # feed forward block
#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)
#
#     def output_shape(self) -> LayersInputOutputShape:
#         return self.input_shape
# #############################################################################################

