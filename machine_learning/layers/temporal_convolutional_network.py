"""Based on https://github.com/locuslab/TCN/blob/master/TCN/tcn.py"""

import torch
from typing import Optional, List, Union, Tuple
from torch.nn.utils import weight_norm
from machine_learning.params.layers_params import TCNParams
from machine_learning.layers.abstract_layer import AbstractLayer


class _CausalConv1D(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, stride: int = 1):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=self.padding,
                                                dilation=dilation))

        self._init_weights()

    def _init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out[:, :, :-self.padding].contiguous()


class _TemporalConvolutionalNetworkSubBlock(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, stride: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.conv = _CausalConv1D(n_inputs, n_outputs, kernel_size, stride=stride, dilation=dilation)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        # x input shape: [batch_size, channels_in, num_time_steps]

        x = self.conv(x)   # [batch_size, channels_out, num_time_steps]
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalConvolutionalNetworkBlock(torch.nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, stride: int = 1,
                 dropout: float = 0.0, num_sub_blocks: int = 2):
        super().__init__()
        sub_blocks = list()
        for i in range(num_sub_blocks):
            sub_blocks.append(_TemporalConvolutionalNetworkSubBlock(n_inputs=n_inputs if i == 0 else n_outputs,
                                                                    n_outputs=n_outputs, kernel_size=kernel_size,
                                                                    dilation=dilation, stride=stride, dropout=dropout)
                              )
        self.net = torch.nn.Sequential(*sub_blocks)

        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = torch.nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape: [batch_size, channels_in, num_time_steps]
        out = self.net(x)              # [batch_size, channels_in, num_time_steps]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)    # [batch_size, channels_out, num_time_steps]


class TemporalConvolutionalNetwork(AbstractLayer):
    def __init__(self, params: TCNParams, input_shape: Tuple[int, int]):
        super().__init__(params=params, input_shape=input_shape)
        layers = []
        in_channels = input_shape[0]
        for i, out_channels in enumerate(params.num_channels):
            dilation_size = self._calc_dilation_size(i, params.dilation_sizes)
            layers.append(TemporalConvolutionalNetworkBlock(in_channels, out_channels, params.kernel_size,
                                                            stride=1,
                                                            dilation=dilation_size,
                                                            dropout=params.dropout_rate,
                                                            num_sub_blocks=params.num_sub_blocks)
                          )
            in_channels = out_channels
        layers = torch.nn.ModuleList(layers)
        self.network = torch.nn.Sequential(*layers)

    @staticmethod
    def _calc_dilation_size(block_num: int, dilation_sizes: Optional[Union[int, List[int]]] = None) -> int:
        if dilation_sizes is None:
            return 2 ** block_num
        if isinstance(dilation_sizes, int):
            return dilation_sizes
        if isinstance(dilation_sizes, list):
            return dilation_sizes[block_num]
        raise ValueError('dilation_size has to be one of (None, int, List[int])!!!')

    def forward(self, x):
        return self.network(x)

    def output_shape(self) -> Tuple[int, int]:
        return self.params.num_channels[-1], self.input_shape[1]

