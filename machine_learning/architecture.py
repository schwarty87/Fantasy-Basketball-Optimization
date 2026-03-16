import copy
import torch
from torch.nn import Module, ModuleList
from typing import Dict, Union

from machine_learning.layers.abstract_layer import AbstractLayer, LayersInputOutputShape
from machine_learning.params.abstract_params import LayerParams
from machine_learning.params.params import ArchitectureParams
from machine_learning.registries import layers_registry


class ModelArchitecture(Module):
    """
    Define the model architecture
    """

    def __init__(self, params: ArchitectureParams,
                 feature_type_to_input_shape: Dict[str, LayersInputOutputShape]):
        super().__init__()
        self.params = params

        # initialize all layers that work on inputs, for all feature types
        name_to_output_shape = copy.deepcopy(feature_type_to_input_shape)
        self.layers = list()
        self.input_tensor_names_list = list()
        for layer_params in self.params.layer_params_list:
            input_names = layer_params.input_tensor_name
            if isinstance(input_names, str):
                input_shape = name_to_output_shape[input_names]
                self.input_tensor_names_list.append([input_names])
            elif isinstance(input_names, list):
                input_shape = [name_to_output_shape[n] for n in input_names]
                self.input_tensor_names_list.append(input_names)
            elif isinstance(input_names, dict):
                input_shape = {k: name_to_output_shape[v] for k, v in input_names.items()}
                self.input_tensor_names_list.append(list(input_names.values()))
            else:
                raise ValueError('input_names has to be either FeatureNames/str/List/Dict!')
            layer = self._init_layer(input_shape=input_shape, layer_params=layer_params)
            self.layers.append(layer)
            output_shape = layer.output_shape()
            if isinstance(output_shape, list):
                name_to_output_shape.update({n: s for n, s in zip(layer.params.output_tensor_name, output_shape)})
            else:
                name_to_output_shape[layer_params.output_tensor_name] = layer.output_shape()

        self.layers = ModuleList(self.layers)

    @staticmethod
    def _init_layer(input_shape: LayersInputOutputShape, layer_params: LayerParams) -> AbstractLayer:
        layer_key = layer_params.key_to_class_for_registry
        layer_class = layers_registry.get(layer_key)
        return layer_class(layer_params, input_shape=input_shape)

    @staticmethod
    def _get_inputs_from_previous_outputs_dict(layer: AbstractLayer, previous_layer_outputs: Dict[str, torch.Tensor]
                                               ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if isinstance(layer.input_tensor_name, str):
            return previous_layer_outputs[layer.input_tensor_name]
        if isinstance(layer.input_tensor_name, list):
            return {n: previous_layer_outputs[n] for n in layer.input_tensor_name}
        elif isinstance(layer.input_tensor_name, dict):
            return {k: previous_layer_outputs[v] for k, v in layer.input_tensor_name.items()}
        else:
            raise ValueError('input_tensor_name must be one of str/list/dict!')

    def _remove_unused_tensors(self, all_xs: Dict[str, torch.Tensor], layer_count: int):
        #TODO precalculate which tensors to delete in the init instead of calculating it here!
        remaining_input_tensor_names = sum(self.input_tensor_names_list[layer_count + 1:], [])
        for n in self.input_tensor_names_list[layer_count]:
            if n not in remaining_input_tensor_names and n not in self.params.output_tensor_names:
                del all_xs[n]

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.tensor:
        all_xs = x.copy()
        for layer_count, layer in enumerate(self.layers):
            inputs = self._get_inputs_from_previous_outputs_dict(layer, all_xs)
            layer_outputs = layer(inputs)
            if isinstance(layer.output_tensor_name, str):
                all_xs[layer.output_tensor_name] = layer_outputs
            elif isinstance(layer_outputs, list):
                all_xs.update(dict(zip(layer.output_tensor_name, layer_outputs)))
            elif isinstance(layer_outputs, dict):
                all_xs.update(layer_outputs)
            else:
                raise ValueError('layer_outputs has to be a tensor/list of tensors/dict of str: tensor!')

            # self._remove_unused_tensors(all_xs, layer_count=layer_count)
        return all_xs
