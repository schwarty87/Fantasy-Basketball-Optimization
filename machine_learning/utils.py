import numpy as np
import os
import pandas as pd
import re
import torch
from scipy.signal import convolve
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Dict, Union, Callable, Optional

RANDOM_SEED = 42


def variable_is_number(variable) -> bool:
    return type(variable) in (int, float) or isinstance(variable, np.number)


def tensor_to_df(tensor: torch.Tensor, column_names: Optional[List[str]] = None) -> pd.DataFrame:
    if tensor.ndimension() != 2:
        raise ValueError('the tensor has to be 2d in order to convert it to a dataframe!')
    if column_names is not None and len(column_names) != tensor.size(dim=1):
        raise ValueError('the tensor has to have the same amount of columns as the length of column_names!')
    df = pd.DataFrame(tensor.cpu().detach().numpy(), columns=column_names)
    return df


def dict_to_pretty_dict(dict_to_convert: Dict[str, Union[float, Dict]]) -> Dict[str, Union[float, Dict]]:
    """Convert the values of a nested dictionary of float values to strings for printing"""
    pretty_dict = dict()
    for k, v in dict_to_convert.items():
        if isinstance(v, dict):
            pretty_dict[k] = dict_to_pretty_dict(v)
        else:
            pretty_dict[k] = f'{v:.3f}' if abs(v) > 1 else f'{v:.3e}'
    return pretty_dict


def apply_function_to_nested_dict_recursively(dict_to_transform: Dict, fn: Callable,
                                              apply_fn_to_each_value_in_list: bool = True) -> Dict:
    """
    Apply a function to the values in a dict recursively
    :param dict_to_transform: Apply fn to the values of dict_to_transform
    :param fn: The function to apply to the values of dict_to_transform
    :param apply_fn_to_each_value_in_list: If the value is a list, should we apply fn to the list itself or to each
                                           value in the list
    :return: A dict with the same keys as dict_to_transform, but each value has fn applied to it
    """
    return_dict = dict()
    for k, curr_values in dict_to_transform.items():
        if isinstance(curr_values, dict):
            return_dict[k] = apply_function_to_nested_dict_recursively(curr_values, fn, apply_fn_to_each_value_in_list)
        elif isinstance(curr_values, list) and apply_fn_to_each_value_in_list:
            return_dict[k] = [fn(v) for v in curr_values]
        else:
            return_dict[k] = fn(curr_values)
    return return_dict


def filter_names_by_pattern(names: List[str], str_to_match: str, return_names_or_indices: bool = True
                            ) -> Union[List[int], List[str]]:
    """Filter a list of strings by a given pattern. We could choose to return the strings that match or their indices"""
    filtered_names_and_indices = [(i, n) for i, n in enumerate(names) if re.match(str_to_match, n)]
    return_ind = 1 if return_names_or_indices else 0
    return [n[return_ind] for n in filtered_names_and_indices]


def binary_erosion_1d(x: Union[pd.DataFrame, np.ndarray], kernel_size: int) -> np.ndarray:
    x_convolved = convolve(x, np.ones((kernel_size, 1)), mode='same')
    return (x_convolved == kernel_size).astype(int)


def binary_dilation_1d(x: Union[pd.DataFrame, np.ndarray], kernel_size: int) -> np.ndarray:
    x_convolved = convolve(x, np.ones((kernel_size, 1)), mode='same')
    return (x_convolved >= 1).astype(int)


def binary_opening_1d(x: Union[pd.DataFrame, np.ndarray], kernel_size: int) -> np.ndarray:
    x = binary_erosion_1d(x, kernel_size=kernel_size)
    x = binary_dilation_1d(x, kernel_size=kernel_size)
    return x


def binary_closing_1d(x: Union[pd.DataFrame, np.ndarray], kernel_size: int) -> np.ndarray:
    x = binary_dilation_1d(x, kernel_size=kernel_size)
    x = binary_erosion_1d(x, kernel_size=kernel_size)
    return x


def convert_time_to_previous_hour_datetime(profits_df: pd.DataFrame) -> pd.DataFrame:
    profits_df['time'] = pd.to_datetime(profits_df['time'] - 60 * 60, unit='s')
    return profits_df


def models_dir(project_name: str) -> str:
    return os.path.join(f'machine_learning_{project_name}', 'models')


def get_next_model_id(prpject_name: str) -> int:
    previous_models = os.listdir(models_dir(prpject_name))
    if len(previous_models) > 0:
        return max([int(m) for m in previous_models if m.isdigit()]) + 1
    return 0


class CustomOrdinalEncoder(OrdinalEncoder):
    """
    An extension of sklearn's OrdinalEncoder that can set unknown values separately for each column.

    This encoder behaves identically to the standard OrdinalEncoder, except when 'handle_unknown'
    is set to 'use_encoded_value'. In this case, it assigns unique unknown values for each feature
    column, equal to the number of known categories in that column.

    Notes:
    ------
    When 'handle_unknown' is not set to 'use_encoded_value', this encoder behaves exactly like the standard
    OrdinalEncoder.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs['handle_unknown'] is not None and kwargs['handle_unknown'] == 'use_encoded_value':
            self.replace_unknowns_with_num_categories = True
            self.unknown_value = -1
        else:
            self.replace_unknowns_with_num_categories = False
            self.unknown_value = None
        self.unknown_values_list = list()

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.replace_unknowns_with_num_categories:
            self.unknown_values_list = [len(categories) for categories in self.categories_]
        return self

    def transform(self, X):
        x_encoded = super().transform(X)

        # Replace -1 (unknown category) with the max category + 1
        if self.replace_unknowns_with_num_categories:
            x_encoded = np.where(x_encoded == self.unknown_value, self.unknown_values_list, x_encoded)

        return x_encoded


