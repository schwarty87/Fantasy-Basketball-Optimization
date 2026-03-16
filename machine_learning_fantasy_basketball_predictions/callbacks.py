import torch
from typing import Type
from machine_learning.params.abstract_params import AbstractParams
from machine_learning.callbacks import CollectTruePredictedOutputs, AbstractCallback, \
    CollectTruePredictedOutputsParams, CallbacksManager
from machine_learning.registries import callbacks_registry


@callbacks_registry.register
class CollectTruePredictedOutputsFantasyBasketballPredictions(CollectTruePredictedOutputs):
    @staticmethod
    def _reshape_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim == 3:
            return tensor.view((tensor.shape[0] * tensor.shape[1], -1))
        return tensor


@callbacks_registry.register
class CallbacksManagerFantasyBasketballPredictions(CallbacksManager):
    pass