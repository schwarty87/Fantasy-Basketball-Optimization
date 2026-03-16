import copy
import numpy as np
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from machine_learning.params.abstract_params import AbstractMachineLearningDatasetParams, AbstractParams
from machine_learning.registries import params_registry
from machine_learning.enums import SampleWeightType


@params_registry.register
@dataclass
class MachineLearningDatasetTabularParams(AbstractMachineLearningDatasetParams):
    # Shift the data by this number of samples
    shift: Optional[int] = None


@params_registry.register
@dataclass
class MachineLearningDatasetTabularScalingParams(MachineLearningDatasetTabularParams):
    pass


@params_registry.register
@dataclass
class MachineLearningDatasetTabularCategoricalEncodingParams(MachineLearningDatasetTabularParams):
    pass


@params_registry.register
@dataclass
class MachineLearningDatasetTimeseriesParams(AbstractMachineLearningDatasetParams):
    # the window size to use
    window_size: Optional[int] = None

    # Whether to apply the window size to the past or to the future
    use_past_or_future_values: bool = True

    apply_diff: bool = False

    # will only be used if apply_diff is True
    normalize_diffs_by_time_or_not_to_normalize: bool = False

    sampling_rate: int = 1

    channels_first_or_channels_last: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.window_size is None:
            raise ValueError("window_size can't be None!")


@params_registry.register
@dataclass
class MachineLearningDatasetTimeseriesScalingParams(MachineLearningDatasetTimeseriesParams):
    pass


@params_registry.register
@dataclass
class MachineLearningDatasetMetadataParams(MachineLearningDatasetTabularParams):
    dtype: Optional[np.dtype] = None


@params_registry.register
@dataclass
class MachineLearningDatasetKeepMaskTabularParams(MachineLearningDatasetTabularParams):
    remove_values_between: Optional[Tuple[float, float]] = None

    # only used when using binary classification
    remove_buffer_areas: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.remove_values_between is not None:
            if len(self.remove_values_between) == 2:
                self.remove_values_between = sorted(self.remove_values_between)
            else:
                raise ValueError('remove_values_between has to contain two values!')


@params_registry.register
@dataclass
class MachineLearningDatasetSampleWeightsTabularParams(MachineLearningDatasetTabularParams):
    sample_weight_type: Optional[SampleWeightType] = None
    extra_weight_on_certain_class: Optional[Dict[int, float]] = None
    weights_per_column_or_all_together: bool = True
    column_name_to_weight: Optional[Dict[str, float]] = None

    # The max weight that we could give to a class. This will run before the additional weights calculations
    max_class_weight: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if self.sample_weight_type is None:
            raise ValueError('sample_weight_type must not be None!')

