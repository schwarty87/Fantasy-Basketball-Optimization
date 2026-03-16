from dataclasses import dataclass, field
from typing import Optional, List, Dict

from machine_learning.params.abstract_params import AbstractMetricsAndCallbacksParams
from machine_learning.registries import params_registry


@params_registry.register
@dataclass
class TrackBestValueParams(AbstractMetricsAndCallbacksParams):
    track_min_or_max: bool = True

    # the value to track when searching for the "best" value
    value_to_track: str = 'eval_loss'


@params_registry.register
@dataclass
class ModelCheckpointParams(TrackBestValueParams):
    """Params for the ModelCheckpoint callback"""
    pass


@params_registry.register
@dataclass
class EarlyStoppingParams(TrackBestValueParams):
    """Params for the ModelCheckpoint callback"""

    patience: int = 10


@params_registry.register
@dataclass
class ReduceLearningRateOnPlateauParams(TrackBestValueParams):
    """Params to use in torch.optim.lr_scheduler.ReduceLROnPlateau"""

    # using the default values from the pytorch code
    patience: int = 10
    factor: float = 0.1
    threshold: float = 1e-4
    mode: str = 'min'             # either "min" or "max"
    threshold_mode: str = 'rel'   # either "rel" or "abs"
    cooldown: int = 0
    min_lr: float = 0.
    eps: float = 1e-8
    verbose: bool = True          # the default in the pytorch code is False, but we'll usually use True


@params_registry.register
@dataclass
class LogModelResultsParams(AbstractMetricsAndCallbacksParams):
    """Params for logging results"""

    save_to_tensorboard: bool = True


@params_registry.register
@dataclass
class CollectTruePredictedOutputsParams(AbstractMetricsAndCallbacksParams):
    """Params for collecting results"""

    # A dict of tensor name to the column names for each column in the tensor.
    # Each key should be found in either tensor_names_true/tensor_names_pred/metadata
    column_names: Dict[str, List[str]] = field(default_factory=lambda: dict())
    # If the outputs were scaled we'll unscale them using the scaler in the given path
    scaler_dir: Optional[str] = None
    save_path: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self.all_tensor_names += list(self.column_names.keys())
        for k, v in self.column_names.items():
            if k not in self.all_tensor_names:
                raise ValueError(f"all keys in column_names must be found in "
                                 f"tensor_names_true/tensor_names_pred/metadata!\n{k} is not found in any "
                                 f"of them!")


