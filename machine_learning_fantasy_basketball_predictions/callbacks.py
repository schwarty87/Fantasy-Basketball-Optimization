import torch
from typing import Type
from machine_learning.params.abstract_params import AbstractParams
from machine_learning.callbacks import CollectTruePredictedOutputs, AbstractCallback, \
    CollectTruePredictedOutputsParams, CallbacksManager
from machine_learning.registries import callbacks_registry


@callbacks_registry.register
class CollectTruePredictedOutputsFantasyBasketballPredictions(CollectTruePredictedOutputs):
    """
    Fantasy Basketball specialization for collecting model predictions and true labels.

    In the daily fantasy pipeline, inputs and outputs often carry a 3D shape:
        (num_days_per_batch, num_players_per_day, num_features)
    This callback flattens the day and player dimensions into 2D:
        (num_days_per_batch * num_players_per_day, num_features)
    so they can be seamlessly converted to tabular pandas DataFrames with one row per player-game.
    """

    @staticmethod
    def _reshape_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Flatten 3D day-batched tensors (days, players, features) into 2D tabular rows."""
        if tensor.ndim == 3:
            return tensor.view((tensor.shape[0] * tensor.shape[1], -1))
        return tensor


@callbacks_registry.register
class CallbacksManagerFantasyBasketballPredictions(CallbacksManager):
    """Fantasy Basketball callbacks manager inheriting from core CallbacksManager."""
    pass