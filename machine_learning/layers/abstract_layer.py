from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict, List

import torch

from machine_learning.params.abstract_params import LayerParams
from machine_learning.enums import FeatureType


TUPLE_OF_INTS_TYPE = Tuple[int, ...]
LayersInputOutputShape = Union[Tuple[int, ...], Dict[Union[str, FeatureType], Tuple[int, ...]], List[Tuple[int, ...]]]


class AbstractLayer(torch.nn.Module, ABC):
    def __init__(self, params: LayerParams, input_shape: LayersInputOutputShape):
        """
        All model layers should inherit from this class
        :param params: params for the given layer
        :param input_shape: the shape of the input tensor (without batch_size)
        """
        super().__init__()
        self.params = params
        self.input_shape = input_shape
        self.input_tensor_name = params.input_tensor_name
        self.output_tensor_name = (
            params.output_tensor_name if params.output_tensor_name is not None else type(self).__name__)

    @abstractmethod
    def output_shape(self) -> LayersInputOutputShape:
        """The number of output channels/nodes"""
        raise NotImplementedError
