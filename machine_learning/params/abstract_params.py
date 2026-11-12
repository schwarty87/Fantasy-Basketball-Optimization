import copy
import importlib
import json
import logging
import numpy as np
from abc import ABC
from dataclasses import dataclass, asdict, fields
from enums import Enum
from typing import Optional, Callable, Dict, TypeVar, Union, Tuple, List, Any, Type

from machine_learning.enums import Reduction, FeatureType
from machine_learning.registries import enums_registry, params_registry

_ABSTRACT_PARAMS_TYPEVAR = TypeVar('_ABSTRACT_PARAMS_TYPEVAR', bound="AbstractParams")


def _class_to_string_path(cls) -> str:
    """ Get the class path """
    return f'{cls.__module__}.{type(cls).__name__}'


def _get_class_from_string_path(string_path: str) -> Union["AbstractParams", Enum]:
    """ Receives a path in the form xxx.yyy.zzz and returns the class zzz """
    split_path = string_path.split('.')
    mod = importlib.import_module('.'.join(split_path[:-1]))
    return getattr(mod, split_path[-1])


def _instantiate_class_from_string_path(path: str, values):
    curr_class = _get_class_from_string_path(path)
    if issubclass(curr_class, AbstractParams):
        assert isinstance(values, dict), 'values must be a dict if the class we want to instantiate is an instance of AbstractParams!'
        return curr_class.init_from_dict(values)
    elif issubclass(curr_class, Enum):
        assert isinstance(values, str), 'values must be a string if the class we want to instantiate is an Enum!'
        return curr_class[values]
    else:
        raise ValueError('path must point to a child of AbstractParams or an Enum!')


@dataclass
class AbstractParams(ABC):
    # The key of the class that these are the params for in the relevant registry
    key_to_class_for_registry: str | None = None

    def __post_init__(self):
        if self.key_to_class_for_registry is None:
            self.key_to_class_for_registry = self.__class__.__name__.replace("Params", "")

    def to_dict(self, dict_factory: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Return a nested dictionary of all the fields in the given instance.
        If a custom dict_factory is provided, it's used directly and overrides the rest of the conversion logic.
        Otherwise, the default serialization logic is applied, which handles nested param classes and enums with
        explicit type information via registries.
        :param dict_factory: An optional callable to be used by 'asdict'.
        :return: A dictionary representation of the dataclass instance.
        """

        def serialize_item(item: Any) -> Any:
            if isinstance(item, Enum):
                return {'_enum_name': item.__class__.__name__,
                        'value': item.name
                        }
            if isinstance(item, AbstractParams):
                return item.to_dict()
            if isinstance(item, list):
                return {'_type': "list",
                        'value': [serialize_item(i) for i in item]
                        }
            if isinstance(item, tuple):
                return {'_type': "tuple",
                        'value': tuple([serialize_item(i) for i in item])
                        }
            if isinstance(item, dict):
                return {k: serialize_item(v) for k, v in item.items()}
            return item

        if dict_factory is not None:
            return asdict(obj=self, dict_factory=dict_factory)

        return_dict = asdict(self)
        return_dict['_params_name'] = self.__class__.__name__

        for f in fields(self):
            value = getattr(self, f.name)
            return_dict[f.name] = serialize_item(value)

        return return_dict

    @classmethod
    def init_from_dict(cls: Type[_ABSTRACT_PARAMS_TYPEVAR], params_dict: Dict[str, Any]) -> _ABSTRACT_PARAMS_TYPEVAR:
        """Initialize the params class from a dictionary"""

        def deserialize_item(item: Any) -> Any:
            if isinstance(item, dict):
                if '_params_name' in item:
                    param_key = item['_params_name']
                    return params_registry.get(param_key).init_from_dict(item)
                if '_enum_name' in item:
                    enum_key = item['_enum_name']
                    return enums_registry.get(enum_key)[item['value']]
                if '_type' in item and 'value' in item:
                    collection_type = item['_type']
                    deserialized_value = [deserialize_item(i) for i in item['value']]
                    if collection_type == 'list':
                        return deserialized_value
                    elif collection_type == 'tuple':
                        return tuple(deserialized_value)
                    else:
                        raise ValueError('Collection type must be one of "list"/"tuple"!')
                return {k: deserialize_item(v) for k, v in item.items()}

            if isinstance(item, (list, tuple)):
                item_type = type(item)
                deserialized = [deserialize_item(i) for i in item]
                return item_type(deserialized)
            return item

        known_field_names = {f.name for f in fields(cls)}
        constructor_kwargs: Dict[str, Any] = {}
        for key, value in params_dict.items():
            if key in known_field_names:
                constructor_kwargs[key] = deserialize_item(value)
        return cls(**constructor_kwargs)

    @classmethod
    def load_params(cls, path: str) -> _ABSTRACT_PARAMS_TYPEVAR:
        with open(path, 'r') as f:
            params_dict = json.load(f)
        return cls.init_from_dict(params_dict)

    def save_to_json(self, path: str):
        with open(path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def update_params_from_dict(self, updates_dict: Dict):
        """
        updates specific params in a params class instance from a dictionary.
        """
        for param_name, param_value in updates_dict.items():
            if hasattr(self, param_name):
                setattr(self, param_name, param_value)
            else:
                error_message = f"no attribute named {param_name}!!!"
                logging.error(error_message)
                raise ValueError(error_message)


@dataclass
class AbstractMetricsAndCallbacksParams(AbstractParams, ABC):
    tensor_names_pred: Optional[Union[str, List[str]]] = None
    tensor_names_true: Optional[Union[str, List[str]]] = None

    def __post_init__(self):
        super().__post_init__()
        self.all_tensor_names = list()
        self._add_tensor_names_to_all_tensors_list(self.tensor_names_true)
        self._add_tensor_names_to_all_tensors_list(self.tensor_names_pred)

    def _add_tensor_names_to_all_tensors_list(self, tensor_names: Optional[Union[str, List[str]]]):
        if tensor_names is not None:
            if isinstance(tensor_names, list):
                self.all_tensor_names += tensor_names
            else:
                self.all_tensor_names.append(tensor_names)


@dataclass(kw_only=True)
class LayerParams(AbstractParams, ABC):
    input_tensor_name: Union[str, List[str], Dict[Union[str, FeatureType], Union[str, List[str]]]]

    output_tensor_name: Optional[Union[str, List[str]]] = None

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.input_tensor_name, str):
            self.input_tensor_names_list = [self.input_tensor_name]
        elif isinstance(self.input_tensor_name, list):
            self.input_tensor_names_list = copy.deepcopy(self.input_tensor_name)
        else:
            self.input_tensor_names_list = list(self.input_tensor_name.values())

    def update_after_loading_data(self, updated_params: Dict[str, Any]):
        pass


@dataclass
class AbstractMetricParams(AbstractMetricsAndCallbacksParams, ABC):
    """Params for all metrics and losses"""
    tensor_names_sample_weights: Optional[Union[str, List[str]]] = None
    tensor_names_keep_mask: Optional[Union[str, List[str]]] = None

    # the name to use in tensorboard and when printing logs
    name_for_logs: str = ''
    reduction: Reduction = Reduction.MEAN
    reduction_dim: Optional[Union[int, Tuple[int, ...]]] = None
    accumulate: bool = False
    column_names: Optional[List[str]] = None

    # run the metric when in predict mode (there are some metrics that shouldn't run in predict mode, like ROCAUC,
    # since they don't have a value per sample)
    run_in_predict: bool = True

    def __post_init__(self):
        super().__post_init__()
        self._add_tensor_names_to_all_tensors_list(self.tensor_names_sample_weights)
        self._add_tensor_names_to_all_tensors_list(self.tensor_names_keep_mask)


    @staticmethod
    def change_reduction_to_per_sample(class_instance: "AbstractMetricParams") -> "AbstractMetricParams":
        """
        Changes the given params to save a value per sample. Before running predict we should update the params using
        this method
        """
        class_instance = copy.deepcopy(class_instance)
        class_instance.accumulate = True
        class_instance.reduction = Reduction.MEAN
        class_instance.reduction_dim = -1
        return class_instance


@dataclass
class AbstractBinaryClassificationMetricParams(AbstractMetricParams, ABC):
    """ Params for all metrics for binary classification """
    reduction: Reduction = Reduction.NONE

    accumulate: bool = False

    run_in_predict: bool = False

    logits: bool = True


@dataclass
class AbstractLossParams(AbstractMetricParams, ABC):
    """Params for all losses"""
    weight: float = 1.


@dataclass(kw_only=True)
class AbstractMachineLearningDatasetParams(AbstractParams, ABC):
    feature_names: List[str]

    dtype: Optional[np.dtype] = np.dtype('float32')

    # replace values in the data, mainly used for keep_mask
    replace_values: Optional[Dict[Union[int, float], Union[int, float]]] = None

    dataset_name: Optional[str] = None

    # Some datasets do preprocessing, we want to exclude ignored values for the preprocessing only on the
    # `relevant_keep_mask_dataset_name` dataset
    relevant_keep_mask_dataset_name: Optional[str] = None
    masked_fill_value: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        self.feature_names = sorted(set(self.feature_names))

    def to_dict(self, dict_factory=None) -> Dict:
        """Convert all enums to strings when saving to dict"""
        return_dict = super().to_dict()
        if return_dict['dtype'] is not None:
            return_dict['dtype'] = self.dtype.name

        return return_dict

    @classmethod
    def init_from_dict(cls: dataclass, params_dict: Dict) -> AbstractParams:
        if params_dict['dtype'] is not None:
            params_dict['dtype'] = np.dtype(params_dict['dtype'])
        return super().init_from_dict(params_dict=params_dict)
