import torch
from enum import Enum, auto
from typing import Callable, Optional, List, Union, Tuple
from machine_learning.registries import enums_registry


@enums_registry.register
class DataSplitType(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TEST = auto()

    # there could be samples that we don't want to include in any of the train/validation/test sets. For example, can be
    # used if we want to compare results to a different model and the times don't overlap exactly
    NONE = auto()

    @classmethod
    def all_not_none_split_types(cls) -> List["DataSplitType"]:
        return [cls.TRAIN, cls.VALIDATION, cls.TEST]


@enums_registry.register
class ModelMode(Enum):
    TRAIN = auto()
    EVAL = auto()
    PREDICT = auto()


@enums_registry.register
class ImputationType(Enum):
    FILL_NA = auto()
    LINEAR = auto()
    NEAREST = auto()
    FILL_NA_BACKWARD = auto()
    NONE = auto()

    @classmethod
    def get_imputation_function(cls, imputation_type: "ImputationType"):
        if imputation_type == cls.FILL_NA:
            return lambda x: x.fillna(method='ffill', axis=0)
        if imputation_type == cls.FILL_NA_BACKWARD:
            return lambda x: x.fillna(method='bfill', axis=0)
        if imputation_type == cls.LINEAR:
            return lambda x: x.interpolate(method='linear', axis=0)
        if imputation_type == cls.NEAREST:
            return lambda x: x.interpolate(method='nearest', axis=0)
        if imputation_type == cls.NONE:
            return lambda x: x
        raise ValueError(f'imputation type must be one of {[c.name for c in cls]}')


@enums_registry.register
class FeatureType(Enum):
    TABULAR = auto()
    CATEGORICAL = auto()
    TIMESERIES = auto()
    TIMESERIES_FUTURE = auto()
    METADATA = auto()

    @classmethod
    def all_names(cls):
        return [c.name for c in cls]

    @classmethod
    def maybe_str_to_feature_type(cls, maybe_feature_type: str) -> Union["FeatureType", str]:
        return cls[maybe_feature_type] if maybe_feature_type in cls.all_names() else maybe_feature_type

    @staticmethod
    def maybe_feature_type_to_str(maybe_feature_type: Union["FeatureType", str]) -> str:
        return maybe_feature_type if isinstance(maybe_feature_type, str) else maybe_feature_type.name


@enums_registry.register
class SampleWeightType(Enum):
    BY_HISTOGRAM = auto()
    BY_VALUE = auto()
    BY_DIFFERENCE_FROM_VALUE = auto()
    BY_INVERSE_CLASS_FREQUENCY = auto()


@enums_registry.register
class _Exp(torch.nn.Module):
    @staticmethod
    def forward(x):
        return torch.exp(x)


@enums_registry.register
class ActivationFunction(Enum):
    LINEAR = auto()
    RELU = auto()
    SIGMOID = auto()
    EXP = auto()
    SOFTPLUS = auto()
    GELU = auto()
    SWISH = auto()

    @classmethod
    def member_to_function(cls, member: "ActivationFunction") -> Optional[Callable]:
        if member == cls.LINEAR:
            return None
        if member == cls.RELU:
            return torch.nn.ReLU()
        if member == cls.SIGMOID:
            return torch.nn.Sigmoid()
        if member == cls.EXP:
            return _Exp()
        if member == cls.SOFTPLUS:
            return torch.nn.Softplus()
        if member == cls.GELU:
            return torch.nn.GELU()
        if member == cls.SWISH:
            return torch.nn.SiLU()
        raise ValueError(f'Activation function "{member}" is not supported')


@enums_registry.register
class Reduction(Enum):
    NONE = auto()             # no reduction
    MEAN = auto()             # mean of all values
    SUM = auto()              # sum of all values

    @classmethod
    def is_mean_reduction(cls, member: "Reduction"):
        return member == cls.MEAN

    @classmethod
    def reduction_is_none(cls, member: "Reduction"):
        return member == cls.NONE

    @classmethod
    def reduces_to_a_single_value(cls, member: "Reduction",
                                  reduction_dim: Optional[Union[int, Tuple[int, ...]]]) -> bool:
        """
        Check if the reduction is going to reduce an array down to a single value or not.
        We assume that if reduction_dim is not None then we're not reducing to a single value (if we were we'd set
        reduction_dim to None)
        """
        return (not cls.reduction_is_none(member)) and (reduction_dim is None)


if __name__ == '__main__':
    aaa=1
