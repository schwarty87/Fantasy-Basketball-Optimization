import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
from abc import ABC, abstractmethod
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import Dataset, DataLoader, Subset, StackDataset
from typing import Optional, Dict, List, Tuple, Union, Callable, Any
from calc_features import CalcFeatures
from machine_learning.constants import IGNORE
from machine_learning.enums import DataSplitType, FeatureType, ImputationType, SampleWeightType
from machine_learning.params.abstract_params import AbstractMachineLearningDatasetParams
from machine_learning.params.machine_learning_dataset_params import MachineLearningDatasetTabularParams, \
    MachineLearningDatasetTimeseriesParams, MachineLearningDatasetTabularScalingParams, \
    MachineLearningDatasetTabularCategoricalEncodingParams, MachineLearningDatasetTimeseriesScalingParams, \
    MachineLearningDatasetMetadataParams, MachineLearningDatasetKeepMaskTabularParams, \
    MachineLearningDatasetSampleWeightsTabularParams
from machine_learning.params.params import FeaturesParams, TrainingParams
from machine_learning.registries import datasets_registry
from machine_learning.utils import tensor_to_df, binary_opening_1d, binary_closing_1d, CustomOrdinalEncoder


def _reshape_keep_mask(keep_mask: np.ndarray, compare_arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    if (keep_mask.ndim == compare_arr.ndim - 1) & (keep_mask.shape == compare_arr.shape[:-1]):
        return keep_mask[:, np.newaxis]
    return keep_mask

class WeightByHist:
    def __init__(self, bins: Union[int, np.ndarray, list, str], max_hist_weight: Optional[float] = None):
        self.bins = bins
        self.max_hist_weight = max_hist_weight

    def fit(self, y: Union[np.ndarray, pd.Series]):
        if isinstance(y, pd.Series):
            y = y.values

        if isinstance(self.bins, str):
            if self.bins == 'one_per_integer':
                rounded_data = np.sort(np.unique(np.round(y[np.isfinite(y)])))
                self.bins = np.arange(np.min(rounded_data) - 0.5, np.max(rounded_data) + 1.5, 1)
            elif 'by_quantile' in self.bins:
                # TODO fill this! the number of quantiles should be in the string
                pass

        self.hist_values, self.hist_bins = np.histogram(y[~np.isnan(y)], bins=self.bins, density=True)
        max_hist_weight = np.max(self.hist_values) if self.max_hist_weight is None else self.max_hist_weight
        self.hist_values = max_hist_weight - self.hist_values
        self.hist_values[np.isnan(self.hist_values)] = 0
        self.hist_values = np.clip(self.hist_values, a_min=0.001, a_max=max_hist_weight)
        self.hist_bins[0] = -np.inf
        self.hist_bins[-1] = np.inf

    def transform(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        return np.array(pd.cut(y, bins=self.hist_bins, labels=self.hist_values, ordered=False))


def _calculate_scaler(scale_data: pd.DataFrame, keep_mask: Optional[np.ndarray] = None) -> StandardScaler:
    scaler = StandardScaler()
    if keep_mask is not None:
        keep_mask = _reshape_keep_mask(keep_mask, scale_data)
        scale_data = pd.DataFrame(np.where(keep_mask.astype(bool), scale_data, np.nan), columns=scale_data.columns,
                                  index=scale_data.index)
    scaler.fit(scale_data)
    return scaler


def get_train_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df['split_type'] == DataSplitType.TRAIN]


class AbstractMachineLearningDataset(Dataset, ABC):
    """
    All the MachineLearningDataset classes inherit from this class. Contains all the methods needed for processing the
    datasets
    """

    def __init__(self, params: AbstractMachineLearningDatasetParams, features_df: pd.DataFrame,
                 scaler: Optional[StandardScaler] = None,
                 categorical_encoder: Optional[OrdinalEncoder] = None,
                 process_features_in_class_init_or_not: bool = True, keep_mask: Optional[np.ndarray] = None):
        """
        :param params: an object containing all the details needed for creating and processing the features
        :param features_df: a pandas dataframe containing all the necessary features. The features should have been
                            filtered by data_split_type before being sent to here
        :param scaler: If we want to scale the data, the information will be stored here.
        :param categorical_encoder: If we want to encode the data, the information will be stored here.
        :param process_features_in_class_init_or_not: Whether to process the features or not. A reason for it to be
                                                      False is if we have multiple inheritance, and we'd like to call it
                                                      separately.
        :param keep_mask: a numpy array of 0/1 showing which values will be used during training
        """
        super().__init__()
        self.params = params
        self._features_df = features_df
        self.scaler = scaler
        self.categorical_encoder = categorical_encoder

        self._features_numpy: np.ndarray = np.empty(shape=0)

        if process_features_in_class_init_or_not:
            self._process_features(keep_mask)

    def _process_features(self, keep_mask: Optional[np.ndarray] = None):
        self._transform_features(keep_mask=keep_mask)
        self._keep_relevant_columns()
        self._add_additional_info()
        self._convert_features_dtype()
        self._convert_df_to_numpy()
        self._reshape_features()
        self._features_df = None

    @abstractmethod
    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        raise NotImplementedError

    def _keep_relevant_columns(self):
        self._features_df = self._features_df[self.params.feature_names]

    def _add_additional_info(self):
        pass

    def _convert_features_dtype(self, to_type: Optional[np.dtype] = None):
        """Convert the features df to a numpy array of type to_type"""
        if to_type is None:
            if self.params.dtype is None:
                return
            to_type = self.params.dtype
        self._features_df = self._features_df.astype(to_type)

    def _convert_df_to_numpy(self):
        self._features_numpy = self._features_df.values
        self._features_numpy = np.ascontiguousarray(self._features_numpy)

    def get_features_numpy(self) -> np.ndarray:
        return self._features_numpy

    def set_features_numpy(self, arr: np.ndarray):
        self._features_numpy = arr

    def num_features(self) -> int:
        return len(self.params.feature_names)

    def _reshape_features(self):
        pass

    def _get_train_data(self) -> pd.DataFrame:
        return get_train_data(df=self._features_df)

    @abstractmethod
    def save_transformers(self, save_dir: str):
        raise NotImplementedError

    def _create_dir_and_get_save_path(self, save_dir: str) -> str:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        return os.path.join(save_dir, f'{self.params.dataset_name}.pkl')

    @classmethod
    def load_categorical_encoder(cls, load_dir, dataset_name: str) -> Optional[OrdinalEncoder]:
        """
        Load saved categorical encoders. All datasets that deal with categorical features will override this method and
        implement loading the objects. We'll return None for datasets that don't use a categorical encoder.
        :param load_dir: the directory to load the categorical encoder object from
        :param dataset_name: the name of the current dataset
        :return: A scipy OrdinalEncoder object that was previously fitted and saved in load_dir
        """
        return

    @classmethod
    def load_scaler(cls, load_dir, dataset_name: str) -> Optional[StandardScaler]:
        return

    def __len__(self):
        return self._features_numpy.shape[0]

    def __getitem__(self, idx):
        """Take the idx row from the features numpy array"""
        return self._features_numpy[idx]


def _load_transformer(load_dir, transformer_name: str, dataset_name: str
                      ) -> Optional[Union[Dict[str, StandardScaler], OrdinalEncoder]]:
    """
    This function can be called from any machine_learning_dataset to load the previously fitted and saved transform
    object, that was saved in pkl format. As of now we support transforms of type StandardScaler, is saved as a dict of
    (feature_name, StandardScaler), or OrdinalEncoder for categorical features.
    :param load_dir: The directory where the transforms are saved
    :param transformer_name: the transformer name. We expect the saved object to be in a subfloder with this name
    :param dataset_name: The dataset name. We expect the saved filename to have this name.
    :return: The loaded transformer object if it exists, otherwise None
    """
    load_path = os.path.join(load_dir, transformer_name, f'{dataset_name}.pkl')
    if not os.path.isfile(load_path):
        return
    with open(load_path, 'rb') as f:
        loaded_object = pkl.load(f)
    return loaded_object


class AbstractScalingMachineLearningDataset(AbstractMachineLearningDataset, ABC):
    """This is an abstract dataset that all datasets that need scaling should inherit from """
    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        if self.num_features() == 0:
            return
        if self.scaler is None:
            train_data = self._get_train_data()[self.params.feature_names]
            keep_mask_train = None
            if keep_mask is not None:
                keep_mask_train = keep_mask[self._features_df['split_type'] == DataSplitType.TRAIN]
            self.scaler = _calculate_scaler(scale_data=train_data, keep_mask=keep_mask_train)
        self._features_df = self._scale_features(self.scaler, features_to_scale=self._features_df, keep_mask=keep_mask)

    @classmethod
    def load_scaler(cls, load_dir, dataset_name: str) -> Optional[StandardScaler]:
        return _load_transformer(load_dir, 'scalers', dataset_name)

    def save_transformers(self, save_dir: str):
        if self.scaler is not None:
            save_path = self._create_dir_and_get_save_path(os.path.join(save_dir, 'scalers'))
            with open(save_path, 'wb') as f:
                pkl.dump(self.scaler, f)

    @staticmethod
    def _scale_features(scaler: StandardScaler, features_to_scale: pd.DataFrame, keep_mask: Optional[np.ndarray] = None
                        ) -> pd.DataFrame:
        scaler_columns = scaler.feature_names_in_
        features_to_scale = features_to_scale.copy()
        scaled_features = pd.DataFrame(scaler.transform(features_to_scale[scaler_columns]),
                                       columns=scaler_columns, index=features_to_scale.index)
        if keep_mask is None:
            features_to_scale[scaler_columns] = scaled_features
        else:
            keep_mask = _reshape_keep_mask(keep_mask, scaled_features)
            features_to_scale[scaler_columns] = \
                np.where(keep_mask.astype(bool), scaled_features, features_to_scale[scaler_columns])
        return features_to_scale


class AbstractCategoricalEncodingMachineLearningDataset(AbstractMachineLearningDataset, ABC):
    """This is an abstract dataset that all categorical type datasets should inherit from"""

    def __init__(self, params: AbstractMachineLearningDatasetParams, features_df: pd.DataFrame,
                 categorical_encoder: Optional[OrdinalEncoder] = None, **kwargs):
        """
        :param params: an object containing all the details needed for creating and processing the features
        :param features_df: a pandas dataframe containing all the necessary features. The features should have been
                         filtered by data_split_type before being sent to here
        :param categorical_encoder: If we want to encode the data, the information will be stored here.
        """
        del kwargs
        self.categorical_encoder = categorical_encoder
        super().__init__(params=params, features_df=features_df, categorical_encoder=categorical_encoder)

    def _transform_features(self, **kwargs):
        del kwargs
        if self.categorical_encoder is None:
            self.categorical_encoder = self._calc_categorical_encoder()
        self._encode_features()

    @classmethod
    def load_categorical_encoder(cls, load_dir, dataset_name: str) -> Optional[OrdinalEncoder]:
        return _load_transformer(load_dir, 'categorical_encoder', dataset_name)

    def save_transformers(self, save_dir: str):
        if self.categorical_encoder is not None:
            save_path = self._create_dir_and_get_save_path(os.path.join(save_dir, 'categorical_encoder'))
            with open(save_path, 'wb') as f:
                pkl.dump(self.categorical_encoder, f)

    def _calc_categorical_encoder(self) -> OrdinalEncoder:
        train_data = self._get_train_data()
        encoder = CustomOrdinalEncoder(dtype=self.params.dtype,
                                       handle_unknown='use_encoded_value',
                                       unknown_value=-1 #train_data[self.params.feature_names].nunique().max()
                                       )
        encoder.fit(train_data[self.params.feature_names])
        return encoder

    def _encode_features(self):
        encoder_feature_names = self.categorical_encoder.feature_names_in_
        self._features_df[encoder_feature_names] = \
            self.categorical_encoder.transform(self._features_df[encoder_feature_names])

    def _convert_features_dtype(self, to_type=np.int32):
        super()._convert_features_dtype(to_type=to_type)

    def num_unique_values_per_feature(self) -> np.ndarray:
        if self.categorical_encoder is not None and hasattr(self.categorical_encoder, 'categories_'):
            return np.array([len(cats) + 1 for cats in self.categorical_encoder.categories_])
        features_flat = self._features_numpy.reshape(-1, self._features_numpy.shape[-1])
        return np.array([len(np.unique(features_flat[:, i])) + 1 for i in range(features_flat.shape[-1])])


@datasets_registry.register
class MachineLearningDatasetTimeseries(AbstractMachineLearningDataset):
    """Deals with all timeseries features"""

    def __init__(self, params: MachineLearningDatasetTimeseriesParams, features_df: pd.DataFrame, **kwargs):
        super().__init__(params=params, features_df=features_df, **kwargs)
        if params.use_past_or_future_values:
            self.add_to_index = np.arange(-params.window_size + 1, 1, params.sampling_rate)
        else:
            self.add_to_index = np.arange(1, params.window_size + 1, params.sampling_rate)

    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        # apply diff to timeseries data
        if self.params.apply_diff:
            self._features_df[self.params.feature_names] = self._features_df[self.params.feature_names].diff(axis=0)
            self._features_df.loc[self._features_df.index[0], self.params.feature_names] = 0

            if self.params.normalize_diffs_by_time_or_not_to_normalize:
                time_diffs = self._features_df['time'].diff() / 60.
                time_diffs.loc[time_diffs.index[0]] = 1.
                time_diffs = time_diffs.astype(self._features_df[self.params.feature_names].dtypes[0])
                self._features_df[self.params.feature_names] /= time_diffs.values[:, None]

    def _reshape_features(self):
        if self.params.channels_first_or_channels_last:
            self._features_numpy = self._features_numpy.T

    def save_transformers(self, save_dir: str):
        pass

    def __len__(self):
        return self._features_numpy.shape[-1] if self.params.channels_first_or_channels_last else self._features_numpy.shape[0]

    def __getitem__(self, idx):
        use_inds = idx + self.add_to_index
        if self.params.channels_first_or_channels_last:
            return self._features_numpy[:, use_inds]
        return self._features_numpy[use_inds]


@datasets_registry.register
class MachineLearningDatasetTimeseriesScaling(MachineLearningDatasetTimeseries, AbstractScalingMachineLearningDataset):
    def __init__(self, params: MachineLearningDatasetTimeseriesScalingParams, features_df: pd.DataFrame,
                 scaler: Optional[StandardScaler] = None, **kwargs):
        del kwargs
        super().__init__(params=params, features_df=features_df, scaler=scaler, process_features_in_class_init_or_not=False)
        self._process_features()

    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        # A bit of a weird implementation to assure that the features are transformed by both super classes.
        MachineLearningDatasetTimeseries._transform_features(self, keep_mask=keep_mask)
        AbstractScalingMachineLearningDataset._transform_features(self, keep_mask=keep_mask)

    def save_transformers(self, save_dir: str):
        # If we remove this method, calling self.save_transformers will only call save_transformers of the left-most
        # parent class (which in this case is MachineLearningDatasetTimeseries)
        AbstractScalingMachineLearningDataset.save_transformers(self, save_dir=save_dir)


@datasets_registry.register
class MachineLearningDatasetTabular(AbstractMachineLearningDataset):
    """Deals with all tabular features"""

    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        if self.params.shift is not None:
            self._features_df[self.params.feature_names] = \
                    self._features_df[self.params.feature_names].shift(self.params.shift).fillna(method='ffill').fillna(method='bfill')

    def save_transformers(self, save_dir: str):
        pass


@datasets_registry.register
class MachineLearningDatasetTabularScaling(MachineLearningDatasetTabular, AbstractScalingMachineLearningDataset):
    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        MachineLearningDatasetTabular._transform_features(self, keep_mask=keep_mask)
        AbstractScalingMachineLearningDataset._transform_features(self, keep_mask=keep_mask)

    def save_transformers(self, save_dir: str):
        AbstractScalingMachineLearningDataset.save_transformers(self, save_dir=save_dir)


@datasets_registry.register
class MachineLearningDatasetTabularCategoricalEncoding(MachineLearningDatasetTabular,
                                                       AbstractCategoricalEncodingMachineLearningDataset):
    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        MachineLearningDatasetTabular._transform_features(self, keep_mask=keep_mask)
        AbstractCategoricalEncodingMachineLearningDataset._transform_features(self, keep_mask=keep_mask)

    def save_transformers(self, save_dir: str):
        AbstractCategoricalEncodingMachineLearningDataset.save_transformers(self, save_dir=save_dir)


@datasets_registry.register
class MachineLearningDatasetMetadata(MachineLearningDatasetTabular):
    """Deals with all metadata"""

    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        if 'split_type' in self.params.feature_names:
            self._features_df['split_type'] = self._features_df['split_type'].apply(lambda x: x.value)

    def _convert_features_dtype(self, to_type: Optional[np.dtype] = None):
        if 'time' in self.params.feature_names:
            time_int = self._features_df['time'].astype(np.int64)
            if (self._features_df['time'] == time_int).all():
                self._features_df = self._features_df.assign(time=time_int)

        if 'DATE' in self.params.feature_names:
            # count days from epoch (ensuring datetime64[s] before dividing by seconds per day)
            dt_s = pd.to_datetime(self._features_df['DATE']).astype('datetime64[s]').astype(np.int64)
            self._features_df = self._features_df.assign(DATE=dt_s / 86400.0)

    @staticmethod
    def tensor_to_df(tensor: torch.Tensor, column_names: List[str]) -> pd.DataFrame:
        df = tensor_to_df(tensor=tensor, column_names=column_names)
        if 'split_type' in df.columns:
            df['split_type'] = df['split_type'].apply(lambda x: DataSplitType(x).name)
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], unit='D').dt.date
        return df


@datasets_registry.register
class MachineLearningDatasetKeepMaskTabular(MachineLearningDatasetTabular):
    @staticmethod
    def _features_are_binary(features_df: pd.DataFrame) -> bool:
        return (features_df.isin([True, False, 0, 1, IGNORE]) | features_df.isnull()).all(axis=None)

    def _additional_transforms(self, keep_mask: pd.DataFrame) -> pd.DataFrame:
        return keep_mask

    def _transform_features(self, **kwargs):
        del kwargs

        super()._transform_features()

        feature_names = self.params.feature_names
        keep_mask = (self._features_df[self.params.feature_names] != IGNORE).astype(int)

        is_binary = self._features_are_binary(self._features_df[feature_names])
        if is_binary:
            self._features_df[feature_names] = self._features_df[feature_names].astype(int)

            if self.params.remove_buffer_areas:
                is_zero = self._features_df[feature_names] == 0
                is_one = self._features_df[feature_names] == 1
                is_change = (is_zero & is_one.shift(-1)) | (is_one & is_zero.shift(-1))
                is_change |= is_change.shift(1)
                keep_mask = keep_mask.mask(is_change, 0)

        feature_names_not_classification = feature_names
        if is_binary:
            feature_names_not_classification = [c.replace('_classification', '') for c in
                                                feature_names_not_classification]

        if self.params.remove_values_between is not None:
            ignore = (self._features_df[feature_names_not_classification] > self.params.remove_values_between[0]) & \
                     (self._features_df[feature_names_not_classification] < self.params.remove_values_between[1])
            keep_mask = keep_mask.mask(ignore.values, 0)

        keep_mask = self._additional_transforms(keep_mask)

        self._features_df[feature_names] = keep_mask

    def _reshape_features(self):
        super()._reshape_features()
        self._features_numpy = np.any(self._features_numpy, axis=-1).astype(self._features_numpy.dtype)

    def get_use_indices(self) -> np.ndarray:
        """ Return an array of sample indices that will be used in training (whose values aren't all zeros) """
        use_sample = self._features_numpy.any(axis=1)
        return np.where(use_sample)[0]


@datasets_registry.register
class MachineLearningDatasetSampleWeightsTabular(MachineLearningDatasetTabular):
    def _add_additional_weights(self, weights: pd.DataFrame) -> pd.DataFrame:
        return weights

    @staticmethod
    def _calc_weight_per_class_dict(arr: Union[pd.Series, pd.DataFrame]) -> Dict[Union[float, int], float]:
        """
        Calculate the class weight for each value in arr, using inverse class frequencies (similar to 'balanced' weights
        in compute_class_weight)
        :param arr: The array with the values. The class weights will be calculated based on the frequencies of each
                    class in this array. Could be any shape.
        :return: a dictionary of class value to weight for the given class.
        """
        num_not_null = np.sum(arr.notnull())
        all_classes = np.unique(arr, return_counts=False)
        all_classes = all_classes[~np.isnan(all_classes)]
        num_classes = len(all_classes)
        counts = pd.Series({curr_class: np.sum(arr == curr_class) for curr_class in all_classes})
        return (num_not_null / (num_classes * counts)).to_dict()

    def _init_weights(self) -> pd.DataFrame:
        weights = pd.DataFrame(1, columns=self.params.feature_names, index=self._features_df.index)
        train_df = self._get_train_data()[self.params.feature_names]
        if self.params.sample_weight_type == SampleWeightType.BY_INVERSE_CLASS_FREQUENCY:
            if self.params.weights_per_column_or_all_together:
                for c in self.params.feature_names:
                    weights_per_class = self._calc_weight_per_class_dict(train_df[c])
                    weights[c] = weights[c].mask(self._features_df[c].isin(weights_per_class.keys()),
                                                 self._features_df[c].replace(weights_per_class))
            else:
                weights_per_class = self._calc_weight_per_class_dict(train_df)
                weights = weights.mask(self._features_df.isin(weights_per_class.keys()),
                                       self._features_df.replace(weights_per_class))
        elif self.params.sample_weight_type == SampleWeightType.BY_HISTOGRAM:
            for c in self.params.feature_names:
                vals = train_df[c].round().values
                if len(vals) == 0:
                    continue
                bins = np.arange(np.nanmin(vals) - 0.5, np.nanmax(vals) + 1.5, 1)
                weights_class = WeightByHist(bins=bins, max_hist_weight=self.params.max_class_weight)
                weights_class.fit(vals)
                weights[c] = weights_class.transform(self._features_df[c])
        elif self.params.sample_weight_type == SampleWeightType.BY_VALUE:
            df = pd.read_feather('/data/recs/research-projects/machine-learning/multidimensional-trends/models/868/predict_results.arrow')
            df.rename(columns=lambda x: x.replace('CollectTruePredictedOutputsParams_', ''), inplace=True)
            true_cols = [c for c in df if 'true' in c]
            pred_cols = [c.replace('true', 'pred') for c in true_cols]
            losses = ((df[true_cols] - df[pred_cols].values) ** 2).rename(columns=lambda x: x.replace('_true', ''))
            losses['time'] = df['time'].values
            weights = self._features_df[['time']].merge(losses, on='time', how='left')[self.params.feature_names]
            weights.ffill(axis=0, limit=10, inplace=True)

            # weights = self._features_df[self.params.feature_names].copy()
        else:
            raise NotImplementedError(f"Sample weight type not implemented {self.params.sample_weight_type.name}")

        return weights

    def _add_extra_weight_on_certain_class(self, weights: pd.DataFrame) -> pd.DataFrame:
        all_change_classes = self.params.extra_weight_on_certain_class.keys()
        multiply_by = \
            self._features_df[self.params.feature_names].replace(self.params.extra_weight_on_certain_class)
        mask = self._features_df[self.params.feature_names].isin(all_change_classes)
        return weights.mask(mask, weights * multiply_by)

    def _transform_features(self, keep_mask: Optional[np.ndarray] = None):
        if keep_mask is not None:
            keep_mask = _reshape_keep_mask(keep_mask, self._features_df[self.params.feature_names])
            self._features_df[self.params.feature_names] = \
                np.where(keep_mask == 0, np.nan, self._features_df[self.params.feature_names])
        self._features_df[self.params.feature_names] = \
            self._features_df[self.params.feature_names].replace({IGNORE: np.nan})

        super()._transform_features(keep_mask=keep_mask)

        weights = self._init_weights()
        if self.params.extra_weight_on_certain_class is not None:
            weights = self._add_extra_weight_on_certain_class(weights)
        weights = self._add_additional_weights(weights)

        train_weights = weights.loc[self._features_df['split_type'] == DataSplitType.TRAIN]
        train_weights = train_weights.mask(self._get_train_data()[self.params.feature_names].isnull(), np.nan)
        means = train_weights.mean(axis=0 if self.params.weights_per_column_or_all_together else None)

        self._features_df[self.params.feature_names] = (weights.fillna(0.) / means)

        if self.params.column_name_to_weight is not None:
            for k, v in self.params.column_name_to_weight.items():
                self._features_df[k] *= v


def _load_transformers(load_dir: str, transformer_type: str, load_function: Callable) -> Dict[str, Any]:
    dataset_name_to_transformer = dict()
    transformers_dir = os.path.join(load_dir, transformer_type)
    if os.path.isdir(transformers_dir):
        file_names = os.listdir(transformers_dir)
        for file_name in file_names:
            dataset_name = file_name.replace('.pkl', '')
            dataset_name_to_transformer[dataset_name] = load_function(load_dir, dataset_name=dataset_name)
    return dataset_name_to_transformer


class MachineLearningDatasetCombined:
    """
    Holds all the input/output data for running a neural network model. The data is all saved in a dictionary
    "machine_learning_datasets" that holds multiple MachineLearningDataset classes, each one for a different features
    type
    """

    def __init__(self, params: FeaturesParams, training_params: TrainingParams, project_name: str,
                 dataset_name_to_scalers: Optional[Dict[str, StandardScaler]] = None,
                 dataset_name_to_categorical_encoder: Optional[Dict[str, OrdinalEncoder]] = None):
        if 'DATE' not in params.all_feature_names_list or 'split_type' not in params.all_feature_names_list:
            raise ValueError("Parameters without `DATE` or `split_type` features are not supported")
        self.params = params
        self.training_params = training_params
        self.project_name = project_name    # TODO maybe delete this?

        features_df = self._load_and_preprocess_features()
        self.data_split_cycle_start_end_inds = self._get_data_split_cycle_start_end_inds(features_df)
        self._init_machine_learning_datasets(features_df, dataset_name_to_scalers=dataset_name_to_scalers,
                                             dataset_name_to_categorical_encoder=dataset_name_to_categorical_encoder)

    @staticmethod
    def _get_data_split_cycle_start_end_inds(features_df: pd.DataFrame) -> Dict[DataSplitType, List[Tuple[int, int]]]:
        """
        Calculate the start and end indices of each cycle for each DataSplitType
        :param features_df: A df with the split_type data. must have a column named 'split_type'
        :return: A dict of DataSplitType to a list of int pairs, where each pair corresponds with the start and end of
                 a cycle. The end indices are equal to one above the last index of the cycle, so a start/end indices
                 pair of (1, 101) would indicate that the cycle goes from indices 1 until 100.
        """
        split_types = features_df['split_type']
        start_inds = np.where(split_types != split_types.shift(1))[0].tolist()
        end_inds = [i for i in start_inds[1:]] + [len(split_types)]
        split_type_to_start_end_inds = {split_type: list() for split_type in DataSplitType.all_not_none_split_types()}
        for start_ind, end_ind in zip(start_inds, end_inds):
            split_type = split_types.iloc[start_ind]
            if split_type == DataSplitType.NONE:
                continue
            split_type_to_start_end_inds[split_type].append((start_ind, end_ind))
        return split_type_to_start_end_inds

    def _load_and_preprocess_features(self) -> pd.DataFrame:
        """
        Initialize the features df by loading and preprocessing it. These methods should work on all columns of the
        features df
        """
        features_df = self._load_features()
        features_df = self._convert_datetime_to_epoch_time(features_df)
        features_df = self._sort_features(features_df)
        features_df = self._impute_data(features_df, impute_types=self.params.impute_types)
        features_df = self._add_additional_columns(features_df=features_df, params=self.params)
        features_df = self._filter_features_df_columns(features_df)
        return features_df

    def _load_features(self) -> pd.DataFrame:
        """Load the features dataframe with the relevant columns"""
        return CalcFeatures(filepath=self.params.features_path).load_df(columns=self.params.all_feature_names_list)

    def _filter_features_df_columns(self, features_df: pd.DataFrame) -> pd.DataFrame:
        all_feature_names = sorted(set(self.params.all_feature_names_list))
        return features_df[all_feature_names]

    @staticmethod
    def _convert_datetime_to_epoch_time(features_df: pd.DataFrame) -> pd.DataFrame:
        if not is_datetime64_any_dtype(features_df['DATE']):
            # pandas 2.x infers datetime64[s] for date objects, so pin to ns before converting to int64
            epoch_ns = pd.to_datetime(features_df['DATE']).astype('datetime64[ns]').astype('int64')
            features_df = features_df.assign(EPOCH_DATE=epoch_ns * 1e-9)
            assert (pd.to_datetime(features_df['EPOCH_DATE'], unit='s') == features_df['DATE']).all(), \
                "The conversion to epoch time was unsuccessful!"
        return features_df

    def _sort_features(self, features_df: pd.DataFrame):
        return features_df.sort_values(['DATE', 'GAME_ID', 'HOME', 'PLAYER_ID'])

    @staticmethod
    def _impute_data(features_df: pd.DataFrame, impute_types: Optional[Tuple[ImputationType, ...]]) -> pd.DataFrame:
        """Impute the features data"""
        numeric_cols = [c for c in features_df.columns if is_numeric_dtype(features_df[c])]
        if impute_types is not None:
            for impute_type in impute_types:
                if impute_type == ImputationType.NONE:
                    continue
                imputation_function = ImputationType.get_imputation_function(impute_type)
                fill_cols = [c for c in numeric_cols if features_df[c].isnull().any()]
                features_df[fill_cols] = imputation_function(features_df[fill_cols])
        return features_df

    @staticmethod
    def _add_additional_columns(features_df: pd.DataFrame, params: FeaturesParams) -> pd.DataFrame:
        if params.convert_to_binary_column_names_to_threshold is None:
            return features_df

        column_names = list(params.convert_to_binary_column_names_to_threshold.keys())
        features_df_binary = \
            features_df[column_names].gt(params.convert_to_binary_column_names_to_threshold).astype(float)

        if params.binary_closing_opening_kernel_size is not None:
            features_df_binary = \
                binary_opening_1d(features_df_binary, kernel_size=params.binary_closing_opening_kernel_size)
            features_df_binary = \
                binary_closing_1d(features_df_binary, kernel_size=params.binary_closing_opening_kernel_size)
            features_df_binary = pd.DataFrame(features_df_binary, columns=column_names)

        features_df_binary.rename(columns=lambda c: c + '_classification', inplace=True)
        return pd.concat([features_df, features_df_binary], axis=1)

    def _init_machine_learning_datasets(
        self, features_df: pd.DataFrame,
        dataset_name_to_scalers: Optional[Dict[str, StandardScaler]] = None,
        dataset_name_to_categorical_encoder: Optional[Dict[str, OrdinalEncoder]] = None):
        """
        Initialize each one of the Dataset classes.
        If the transformers dict was received when initializing the class, we'll use it. Otherwise, we'll calculate
        them here.
        """
        machine_learning_datasets = dict()

        # first run keep masks, then the rest
        keep_mask_feature_names = \
            list(set([n.relevant_keep_mask_dataset_name for n in
                      self.params.feature_name_to_machine_learning_dataset_params.values() if
                      n.relevant_keep_mask_dataset_name is not None]))
        feature_names = keep_mask_feature_names + \
                        [n for n in self.params.feature_name_to_machine_learning_dataset_params.keys() if
                         n not in keep_mask_feature_names]
        for feature_name in feature_names:
            machine_learning_dataset_params = self.params.feature_name_to_machine_learning_dataset_params[feature_name]

            if machine_learning_dataset_params.dataset_name is None:
                machine_learning_dataset_params.dataset_name = FeatureType.maybe_feature_type_to_str(feature_name)
            scaler = dataset_name_to_scalers.get(feature_name) if dataset_name_to_scalers is not None else None
            categorical_encoder = dataset_name_to_categorical_encoder.get(feature_name) if \
                dataset_name_to_categorical_encoder is not None else None
            if machine_learning_dataset_params.relevant_keep_mask_dataset_name is not None:
                keep_mask = machine_learning_datasets[
                    machine_learning_dataset_params.relevant_keep_mask_dataset_name].get_features_numpy()
            else:
                keep_mask = None

            # get relevant columns so we don't copy all features each time
            required_cols = list(machine_learning_dataset_params.feature_names) + ['split_type']
            required_cols = list(set(required_cols).intersection(features_df.columns))
            dataset_df_subset = features_df[required_cols].copy()

            dataset_key = machine_learning_dataset_params.key_to_class_for_registry
            dataset_class = datasets_registry.get(dataset_key)
            machine_learning_datasets[feature_name] = \
                dataset_class(params=machine_learning_dataset_params,
                              features_df=dataset_df_subset,
                              scaler=scaler,
                              categorical_encoder=categorical_encoder,
                              keep_mask=keep_mask)
        self._postprocess_datasets(machine_learning_datasets=machine_learning_datasets)
        self.machine_learning_datasets_stacked = StackDataset(**machine_learning_datasets)

    def _postprocess_datasets(self, machine_learning_datasets: Dict[str, AbstractMachineLearningDataset]):
        pass

    def _remove_unused_indices_by_keep_mask(self, sampler_indices: List[int]) -> List[int]:
        """Removing indices that don't have any "used" values in keep_masks"""
        keep_mask_dataset_names = set([ds.params.relevant_keep_mask_dataset_name for ds in
                                       self.machine_learning_datasets_stacked.datasets.values() if
                                       ds.params.relevant_keep_mask_dataset_name is not None])
        sampler_indices = set(sampler_indices)
        for keep_mask_dataset_name in keep_mask_dataset_names:
            keep_mask_dataset = self.machine_learning_datasets_stacked.datasets[keep_mask_dataset_name]
            use_sample_inds = keep_mask_dataset.get_use_indices()
            sampler_indices = sampler_indices.intersection(set(use_sample_inds))
        sampler_indices = sorted(sampler_indices)
        return sampler_indices

    def _filter_sampler_indices_for_dataloader(self, cycle_start_end_inds: Optional[List[Tuple[int, int]]] = None
                                               ) -> List[int]:
        """
        Compute the indices for a given data loader. We will filter to not use indices that are too early or late,
        given the lookback/lookahead windows, and remove indices that don't have any "used" values in the keep_mask
        :param cycle_start_end_inds: the indices to use from the features df, in tuples of
               (cycle_start_index, cycle_end_index). If None, use all indices.
               Cycle_end_index corresponds to one above the last index in the cycle, so if a start/end pair equals
               (1, 101), it would indicate a cycle with indices from 1 until 100 (including)
        :return: The filtered indices
        """
        if cycle_start_end_inds is None:
            cycle_start_end_inds = [(0, len(list(self.machine_learning_datasets_stacked.datasets.values())[0]))]

        # when using timeseries data, start_ind is the last index of the first sample, since we use a lookback
        # window and would have empty values if we started before that
        windows_before = \
            [p.window_size for p in self.params.feature_name_to_machine_learning_dataset_params.values() if
             hasattr(p, 'window_size') and p.use_past_or_future_values]
        windows_after = [p.window_size for p in self.params.feature_name_to_machine_learning_dataset_params.values() if
                         hasattr(p, 'window_size') and not p.use_past_or_future_values]

        num_skip_at_start = max(windows_before) if len(windows_before) > 0 else 0
        num_skip_at_end = max(windows_after) if len(windows_after) > 0 else 0
        sampler_indices = np.concatenate([np.arange(s + num_skip_at_start, e - num_skip_at_end, self.params.stride)
                                          for s, e in cycle_start_end_inds]).tolist()
        if self.params.sample_per_game:
            n_players_per_game = (5 + self.params.num_bench_players) * 2
            sampler_indices = sorted(set([i // n_players_per_game for i in sampler_indices]))
        sampler_indices = self._remove_unused_indices_by_keep_mask(sampler_indices)

        return sampler_indices

    def _get_data_loader_batch_kwargs(self, batch_size: int | None = None, shuffle: bool = True,
                                      **kwargs) -> Dict[str, Any]:
        """
        Get the information for batching the data for the DataLoader.
        By default, we use the basic batch_size and shuffle
        The keys of the output dict must be fields that DataLoader accepts as arguments.
        """
        del kwargs
        if batch_size:
            return {'batch_size': batch_size,
                    'shuffle': shuffle
                    }
        raise ValueError("batch size/sampler not declared!")

    def _create_data_loader(self, batch_size: int, shuffle: bool, num_workers: int = 0,
                            cycle_start_end_inds: Optional[List[Tuple[int, int]]] = None) -> DataLoader:
        """
        Creates a single DataLoader object using the current class instance and a sampler
        :param batch_size: The batch size to use
        :param cycle_start_end_inds: the indices to use from the features df, in tuples of
               (cycle_start_index, cycle_end_index). If None, use all indices.
               Cycle_end_index corresponds to one above the last index in the cycle, so if a start/end pair equals
               (1, 101), it would indicate a cycle with indices from 1 until 100 (including)
        :param shuffle: Whether to use random indices (sampled from "indices") or sequential
        :param num_workers: the number of workers to use for loading data
        :return: A DataLoader instance with the relevant data, to be sent to the neural net
        """
        sampler_indices = self._filter_sampler_indices_for_dataloader(cycle_start_end_inds)
        machine_learning_datasets_stacked_subset = \
            Subset(self.machine_learning_datasets_stacked, indices=sampler_indices)
        data_loader_batch_kwargs = (
            self._get_data_loader_batch_kwargs(batch_size=batch_size,
                                               shuffle=shuffle,
                                               machine_learning_datasets_stacked_subset=machine_learning_datasets_stacked_subset
                                               )
        )
        return DataLoader(machine_learning_datasets_stacked_subset, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(), **data_loader_batch_kwargs)

    def create_data_loaders(self, batch_size: int, num_workers: int = 0) -> Dict[DataSplitType, DataLoader]:
        """
        Create all data loaders for all DataSplitTypes. self.data_loaders is a dict of DataSplitType to DataLoader.
        Each DataLoader holds an instance of DataForModelCombined, and we use a different sampler for each one depending
        on the data split type
        """
        #TODO possibly slice the feature_numpy of each dataloader, so it only holds the relevant data, which should
        # speed up sampling from it
        data_loaders = dict()
        for data_split_type, cycle_start_end_inds in self.data_split_cycle_start_end_inds.items():
            data_loaders[data_split_type] = self._create_data_loader(cycle_start_end_inds=cycle_start_end_inds,
                                                                     batch_size=batch_size,
                                                                     shuffle=data_split_type == DataSplitType.TRAIN,
                                                                     num_workers=num_workers
                                                                     )
        return data_loaders

    @staticmethod
    def _get_all_datasets_from_data_loader(data_loader: DataLoader, return_dict_or_list: bool = False
                                           ) -> Union[Dict[str, AbstractMachineLearningDataset],
                                                      List[AbstractMachineLearningDataset]]:
        datasets = data_loader.dataset.dataset.datasets
        if return_dict_or_list:
            return datasets
        return datasets.values()

    @classmethod
    def get_single_dataset_from_data_loaders(cls, data_loaders: Dict[DataSplitType, DataLoader],
                                             feature_type: str, data_split_type: DataSplitType
                                             ) -> AbstractMachineLearningDataset:
        feature_type = FeatureType.maybe_feature_type_to_str(feature_type)
        datasets = cls._get_all_datasets_from_data_loader(data_loader=data_loaders[data_split_type],
                                                          return_dict_or_list=True)
        return datasets[feature_type]

    @classmethod
    def save_transformers(cls, data_loader: Union[DataLoader, Dict[DataSplitType, DataLoader]], save_dir):
        """
        Save the transform data for all MachineLearningDataset classes that are held in the given dataloader
        """
        if isinstance(data_loader, dict):
            data_loader = data_loader[DataSplitType.TRAIN]
        datasets = cls._get_all_datasets_from_data_loader(data_loader=data_loader, return_dict_or_list=False)
        for dataset in datasets:
            dataset.save_transformers(save_dir=save_dir)

    @classmethod
    def load_transformers(cls, load_dir: str) -> Tuple[Dict[str, StandardScaler], OrdinalEncoder]:
        dataset_name_to_scaler = _load_transformers(
            load_dir, 'scalers', AbstractScalingMachineLearningDataset.load_scaler)
        dataset_name_to_categorical_encoder = _load_transformers(
            load_dir, 'categorical_encoder',
            AbstractCategoricalEncodingMachineLearningDataset.load_categorical_encoder)

        return dataset_name_to_scaler, dataset_name_to_categorical_encoder

    @staticmethod
    def assert_transformers_complete(features_params: FeaturesParams, scalers: Optional[Dict], encoders: Optional[Dict]) -> None:
        """Verify that all required scalers and categorical encoders exist when continuing from a model checkpoint."""
        if not hasattr(features_params, 'feature_name_to_machine_learning_dataset_params'):
            return
        scalers = scalers or {}
        encoders = encoders or {}
        missing_scalers = []
        missing_encoders = []
        for name, ds_params in features_params.feature_name_to_machine_learning_dataset_params.items():
            if isinstance(ds_params, (MachineLearningDatasetTabularScalingParams, MachineLearningDatasetTimeseriesScalingParams)):
                if name not in scalers or scalers[name] is None:
                    missing_scalers.append(name)
            elif isinstance(ds_params, MachineLearningDatasetTabularCategoricalEncodingParams):
                if name not in encoders or encoders[name] is None:
                    missing_encoders.append(name)
        if missing_scalers or missing_encoders:
            err_msgs = []
            if missing_scalers:
                err_msgs.append(f"Missing scalers for datasets: {missing_scalers}")
            if missing_encoders:
                err_msgs.append(f"Missing encoders for datasets: {missing_encoders}")
            raise FileNotFoundError("; ".join(err_msgs))

    @staticmethod
    def move_batch_data_to_device(batch_data: Dict[str, Union[Dict[str, torch.Tensor], torch.tensor]],
                                  device: torch.device, dont_convert: Optional[List[str]] = None):
        if dont_convert is None:
            dont_convert = list()
        for feature_type in batch_data.keys():
            if feature_type not in dont_convert:
                batch_data[feature_type] = batch_data[feature_type].to(device)
