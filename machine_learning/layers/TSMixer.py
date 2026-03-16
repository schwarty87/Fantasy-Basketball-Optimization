import torch
from typing import Dict, Optional
from machine_learning.params.layers_params import TSMixerParams
from machine_learning.layers.abstract_layer import AbstractLayer, LayersInputOutputShape
from machine_learning.enums import FeatureType
from machine_learning.registries import layers_registry


class TSBatchNorm2d(torch.nn.Module):

    def __init__(self):
        super(TSBatchNorm2d, self).__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)

        # Reshape input_data to (batch_size, 1, timepoints, features)
        x = x.unsqueeze(1)

        # Forward pass
        output = self.bn(x)

        # Reshape the output back to (batch_size, timepoints, features)
        output = output.squeeze(1)
        return output


class MLPTime(torch.nn.Module):
    def __init__(self, width_time: int, dropout: float):
        super().__init__()

        self.dense = torch.nn.Linear(in_features=width_time, out_features=width_time)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, features, time)

        # Apply MLP to time dimension
        y = self.dense(x)
        y = self.act(y)

        # Dropout
        y = self.dropout(y)

        return y


class MLPFeat(torch.nn.Module):
    def __init__(self, width_feats: int, width_feats_hidden: int, dropout: float):
        super().__init__()

        self.lin_1 = torch.nn.Linear(in_features=width_feats, out_features=width_feats_hidden)
        self.lin_2 = torch.nn.Linear(in_features=width_feats_hidden, out_features=width_feats)
        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.dropout_2 = torch.nn.Dropout(p=dropout)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)

        # Apply MLP to feat dimension
        y = self.lin_1(x)
        y = self.act(y)
        y = self.dropout_1(y)
        y = self.lin_2(y)
        y = self.dropout_2(y)

        return y


class TimeMixing(torch.nn.Module):
    def __init__(self, width_time: int, dropout: float):
        super().__init__()
        self.norm = TSBatchNorm2d()

        self.mlp_time = MLPTime(width_time=width_time, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)

        # Now rotate such that shape is (batch_size, features, time)
        y = torch.transpose(y, 1, 2)

        # Apply MLP to time dimension
        y = self.mlp_time(y)

        # Rotate back such that shape is (batch_size, time, features)
        y = torch.transpose(y, 1, 2)

        # Add residual connection
        return x + y


class FeatureMixing(torch.nn.Module):
    def __init__(self, width_feats: int, width_feats_hidden: int, dropout: float):
        super().__init__()
        self.norm = TSBatchNorm2d()

        self.mlp_feat = MLPFeat(width_feats=width_feats, width_feats_hidden=width_feats_hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)

        # Apply MLP to feat dimension
        y = self.mlp_feat(y)

        # Add residual connection
        return x + y


class MixingLayer(torch.nn.Module):
    def __init__(self, input_length: int, num_features: int, feature_mixing_hidden_channels: int, dropout: float,
                 out_features: Optional[int] = None):
        super().__init__()
        self.time_mixing = TimeMixing(width_time=input_length, dropout=dropout)
        self.feature_mixing = FeatureMixing(width_feats=num_features, width_feats_hidden=feature_mixing_hidden_channels,
                                            dropout=dropout)

        self.out_features = out_features
        if out_features is not None:
            self.out_dense = torch.nn.Linear(in_features=num_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.time_mixing(x)
        y = self.feature_mixing(y)
        if self.out_features is not None:
            y = self.out_dense(y)
        return y


class TemporalProjection(torch.nn.Module):
    def __init__(self, input_length: int, forecast_length: int):
        super().__init__()
        self.dense = torch.nn.Linear(in_features=input_length, out_features=forecast_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        # Now rotate such that shape is (batch_size, features, time=input_length)
        y = torch.transpose(x, 1, 2)

        # Apply linear projection -> shape is (batch_size, features, time=forecast_length)
        y = self.dense(y)

        # Rotate back such that shape is (batch_size, time=forecast_length, features)
        y = torch.transpose(y, 1, 2)
        return y


@layers_registry.register
class TSMixer(AbstractLayer):
    def __init__(self, params: TSMixerParams, input_shape: LayersInputOutputShape):
        super().__init__(params=params, input_shape=input_shape)

        self.has_tabular = FeatureType.TABULAR in params.input_tensor_name.keys()
        self.has_timeseries_future = FeatureType.TIMESERIES_FUTURE in params.input_tensor_name.keys()

        timeseries_time_len = input_shape[FeatureType.TIMESERIES][0 if params.time_dimension == 1 else 0]
        num_features = input_shape[FeatureType.TIMESERIES][1 if params.time_dimension == 1 else 0]
        self.temporal_projection_timeseries = TemporalProjection(input_length=timeseries_time_len,
                                                                 forecast_length=params.prediction_length)
        self.feature_mixing_timeseries = FeatureMixing(width_feats=num_features,
                                                       width_feats_hidden=params.feature_mixing_hidden_channels,
                                                       dropout=params.dropout_rate)

        if self.has_timeseries_future:
            input_shape_timeseries_future = input_shape[FeatureType.TIMESERIES_FUTURE]
            self.temporal_projection_timeseries_future = TemporalProjection(input_length=input_shape_timeseries_future[1],
                                                                            forecast_length=params.prediction_length)
            self.feature_mixing_timeseries_future = FeatureMixing(width_feats=input_shape_timeseries_future[0],
                                                                  width_feats_hidden=params.feature_mixing_hidden_channels,
                                                                  dropout=params.dropout_rate)
            num_features += input_shape_timeseries_future[0]

        input_shape_tabular = 0
        if self.has_tabular:
            input_shape_tabular = input_shape[FeatureType.TABULAR][0]
            self.feature_mixing_tabular = [FeatureMixing(width_feats=input_shape_tabular,
                                                         width_feats_hidden=input_shape_tabular,
                                                         dropout=params.dropout_rate)
                                           for _ in range(params.num_mixer_layers)]
            self.feature_mixing_tabular = torch.nn.ModuleList(self.feature_mixing_tabular)
            num_features += input_shape_tabular

        self.num_out_features = params.num_out_features if params.num_out_features is not None else num_features
        num_features_after_first = self.num_out_features + input_shape_tabular
        self.mixing_layers = list()
        for i in range(params.num_mixer_layers):
            num_features = num_features if i == 0 else num_features_after_first
            dropout = params.dropout_rate if i < params.num_mixer_layers - 1 else params.dropout_last_layer
            self.mixing_layers.append(MixingLayer(
                input_length=params.prediction_length,
                num_features=num_features,
                feature_mixing_hidden_channels=params.feature_mixing_hidden_channels,
                dropout=dropout,
                out_features=self.num_out_features))
        self.mixing_layers = torch.nn.ModuleList(self.mixing_layers)

    @staticmethod
    def _repeat_tabular(x: torch.Tensor, num_repeats: int):
        x = torch.unsqueeze(x, 1)
        return x.repeat([1, num_repeats, 1])

    def forward(self, x: Dict[FeatureType, torch.Tensor]):
        # Input x timeseries: (batch_size, features, time) or (batch_size, time, features)
        # (depending on the vlue of params.time_dimension)
        # Output size: (batch_size, features, prediction_length)

        if self.params.num_mixer_layers == 0:
            return x

        # x_timeseries: (batch_size, time, features)
        x_timeseries = x.get(FeatureType.TIMESERIES)
        if self.params.time_dimension in (-1, 2):
            x_timeseries = torch.transpose(x_timeseries, 2, 1)
        y = self.temporal_projection_timeseries(x_timeseries)
        y = self.feature_mixing_timeseries(y)

        if self.has_timeseries_future:
            # x_timeseries_future: (batch_size, time, features)
            x_timeseries_future = torch.transpose(x.get(FeatureType.TIMESERIES_FUTURE), 2, 1)
            x_timeseries_future = self.temporal_projection_timeseries_future(x_timeseries_future)
            x_timeseries_future = self.feature_mixing_timeseries_future(x_timeseries_future)
            y = torch.concat([y, x_timeseries_future], dim=-1)

        if self.has_tabular:
            x_tabular = self._repeat_tabular(x.get(FeatureType.TABULAR), num_repeats=self.params.prediction_length)

        for layer_num in range(self.params.num_mixer_layers):
            if self.has_tabular:
                mixed_tabular = self.feature_mixing_tabular[layer_num](x_tabular)
                y = torch.concat([y, mixed_tabular], dim=-1)
            y = self.mixing_layers[layer_num](y)

        # y: (batch_size, num_out_features, prediction_length)
        if self.params.time_dimension in (-1, 2):
            y = torch.transpose(y, 2, 1)

        return y

    def output_shape(self) -> LayersInputOutputShape:
        if self.params.time_dimension in (-1, 2):
            return self.num_out_features, self.params.prediction_length
        return self.params.prediction_length, self.num_out_features

