import os
import shutil
import logging
import torch
import numpy as np
from dataclasses import dataclass, is_dataclass, fields
from typing import Dict
from enums import Sites
from utils import define_logger, update_dir_name
from machine_learning.enums import SampleWeightType, ImputationType, ActivationFunction, FeatureType
from machine_learning.constants import METADATA
from machine_learning.utils import models_dir, get_next_model_id
from machine_learning.params.callbacks_params import ModelCheckpointParams, ReduceLearningRateOnPlateauParams, \
    LogModelResultsParams, EarlyStoppingParams
from machine_learning.params.machine_learning_dataset_params import MachineLearningDatasetTabularScalingParams, \
    MachineLearningDatasetTabularCategoricalEncodingParams, MachineLearningDatasetTabularParams, \
    MachineLearningDatasetSampleWeightsTabularParams, MachineLearningDatasetMetadataParams, \
    MachineLearningDatasetKeepMaskTabularParams
from machine_learning.params.layers_params import DenseLayersParams, EmbeddingsLayerParams, CombineTensorsParams, \
    TransformerEncoderPytorchParams, TSMixerParams, SplitAttentionCombineParams, SplitAttentionCombineWrapperParams
from machine_learning.params.metrics_params import BinaryCrossEntropyParams, MSELossParams, MAEMetricParams, \
    QuantileRegressionLossParams, MAPEMetricParams
from machine_learning_fantasy_basketball_predictions.params.params import FeaturesParamsFantasyBasketballPredictions,\
    MachineLearningParamsFantasyBasketballPredictions, TrainingParamsFantasyBasketballPredictions, \
    ArchitectureParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.constants import PROJECT_NAME
from machine_learning_fantasy_basketball_predictions.run_model import RunModelFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.params.layers_params import \
    CalculateFantasyPointsParams, ReshapeToSamplePerDayParams, OptimizeTeamParams
from machine_learning_fantasy_basketball_predictions.params.machine_learning_dataset_params import \
    MachineLearningDatasetSampleWeightsTabularFantasyBasketballPredictionsParams, \
    MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams
from machine_learning_fantasy_basketball_predictions.params.callbacks_params import \
    CollectTruePredictedOutputsParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.params.metrics_params import (OptimizeTeamsLossParams,
                                                                                   PairwiseRankingLossParams,
                                                                                   BradleyTerryLossParams)


def _compare_state_dicts(sd1: Dict, sd2: Dict, path: str):
    assert set(sd1.keys()) == set(sd2.keys()), f"Model state dict keys don't match at {path}"
    for key, tensor1 in sd1.items():
        tensor2 = sd2[key]
        assert torch.equal(tensor1, tensor2), f"Tensor mismatch in state_dict for key {key} at {path}"

def compare_dataclasses(obj1: dataclass, obj2: dataclass, path: str = ""):
    # 1. Handle the None case explicitly for clarity
    if obj1 is None:
        # assert obj2 is None, f'Mismatch at {path}: obj1 is None and obj2 is not ({type(obj2).__name__})'
        return
    elif obj2 is None:
        assert obj1 is None, f'Mismatch at {path}: obj2 is None and obj1 is not ({type(obj1).__name__})'
        return

    # 2. Compare types. This must come after the None check
    assert type(obj1) == type(obj2), f'Type mismatch at {path}: {type(obj1).__name__} vs {type(obj2).__name__}'

    # 3. Compare based on the object's type
    if is_dataclass(obj1):
        for field in fields(obj1):
            field_name = field.name
            new_path = f'{path}.{field_name}' if path else field_name
            compare_dataclasses(getattr(obj1, field_name), getattr(obj2, field_name), path=new_path)

    elif isinstance(obj1, (list, tuple)):
        assert len(obj1) == len(obj2), f'Sequence length mismatch at {path}: {len(obj1)} vs {len(obj2)}'
        for i, (item1, item2) in enumerate(zip(obj1, obj2)):
            compare_dataclasses(item1, item2, path=f"{path}[{i}]")

    elif isinstance(obj1, dict):
        assert set(obj1.keys()) == set(obj2.keys()), \
            f'Dict key mismatch at {path}: {set(obj1.keys())} vs {set(obj2.keys())}'
        for key in obj1.keys():
            compare_dataclasses(obj1[key], obj2[key], path=f"{path}[{key}]")

    elif isinstance(obj1, np.ndarray):
        assert np.array_equal(obj1, obj2), f'numpy array mismatch at {path}'

    elif isinstance(obj1, type):
        assert obj1 == obj2, f'Class type mismatch at {path}: {obj1.__name__} vs {obj2.__name__}'

    elif isinstance(obj1, torch.nn.Module):
        assert str(obj1) == str(obj2), f'Pytorch module architecture mismatch at {path}: {str(obj1)} vs {str(obj2)}'
        _compare_state_dicts(obj1.state_dict(), obj2.state_dict(), path=path)
    else:
        assert obj1 == obj2, f'Value mismatch at {path}: {obj1} vs {obj2}'


def create_features_params() -> FeaturesParamsFantasyBasketballPredictions:
    features_path = os.path.join(update_dir_name('features'), 'features.feather')
    features_dtype = np.dtype('float32')
    output_features = ['REB', 'STL', 'AST', '3PM', 'BLK', 'TOV', '2PM', 'FG_MISSED', 'FTM', 'FT_MISSED', 'MIN']
    # output_features = ['REB_mean_ratio', 'STL_mean_ratio', 'AST_mean_ratio', '3PM_mean_ratio', 'BLK_mean_ratio',
    #                    'TOV_mean_ratio', '2PM_mean_ratio', 'FG_MISSED_mean_ratio', 'FTM_mean_ratio', 'FT_MISSED_mean_ratio', 'MIN_mean_ratio']
    # output_features = ['3PM', '2PM', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']
    # output_features = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'MIN']
    feature_name_to_machine_learning_dataset_params = \
        {'tabular': MachineLearningDatasetTabularScalingParams(feature_names=['DAYS_REST', 'SEASON_PROGRESS'],
                                                               dtype=features_dtype),
         'lagged': MachineLearningDatasetTabularScalingParams(feature_names=[f'{stat}_t-{lag}' for stat in output_features for lag in range(1, 10)],
                                                               dtype=features_dtype),
         'positional_encoding_features':
             MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams(
                 feature_names=['HOME', 'START_POSITION_CALCULATED'],
                 dtype=features_dtype),
         'embedding_features':
             MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams(
                 feature_names=['PLAYER_ID', 'PLAYER_SEASON', 'SEASON_TEAM', 'SEASON_YEAR', 'TEAM_ID'],
                 dtype=features_dtype),
         'true_label':
             MachineLearningDatasetTabularScalingParams(feature_names=output_features,
                                                        relevant_keep_mask_dataset_name='keep_mask',
                                                        dtype=features_dtype),
         'sample_weights':
             MachineLearningDatasetSampleWeightsTabularFantasyBasketballPredictionsParams(feature_names=output_features,
                                                                                          dtype=features_dtype,
                                                                                          # sample_weight_type=SampleWeightType.BY_INVERSE_CLASS_FREQUENCY,
                                                                                          sample_weight_type=SampleWeightType.BY_HISTOGRAM,
                                                                                          weights_per_column_or_all_together=True,
                                                                                          max_class_weight=1,
                                                                                          # column_name_to_weight={k: abs(v) for k, v in SCORING[Sites.YAHOO].items()},
                                                                                          relevant_keep_mask_dataset_name='keep_mask'),
         METADATA: MachineLearningDatasetMetadataParams(feature_names=['DATE', 'split_type', 'PLAYER_ID', 'YAHOO_SALARY',
                                                                       'YAHOO_POSITION', 'YAHOO_POINTS'],
                                                        dtype=features_dtype),
         'keep_mask': MachineLearningDatasetKeepMaskTabularParams(feature_names=['NOT_IGNORE'], dtype=features_dtype)
         }
    return FeaturesParamsFantasyBasketballPredictions(features_path=features_path,
                                                      feature_name_to_machine_learning_dataset_params=feature_name_to_machine_learning_dataset_params,
                                                      num_bench_players=6,
                                                      impute_types=(ImputationType.NONE,),
                                                      sample_per_game=True,
                                                      all_games_every_day=True,
                                                      )


def create_architecture_params(features_params: FeaturesParamsFantasyBasketballPredictions, num_days_per_batch: int
                               ) -> ArchitectureParamsFantasyBasketballPredictions:
    output_feature_names = features_params.feature_name_to_machine_learning_dataset_params['true_label'].feature_names
    num_outputs = len(output_feature_names)

    # embeddings_dims = [12, 12, 4, 2, 2]
    embeddings_dims = [32, 12, 4, 4, 4]
    # embeddings_dims = [24, 24, 8, 4, 4]   # 'PLAYER_ID', 'PLAYER_SEASON', 'SEASON_TEAM', 'SEASON_YEAR', 'TEAM_ID'
    embeddings = EmbeddingsLayerParams(input_tensor_name='embedding_features',
                                       output_tensor_name=[f'embedded_{i}' for i in range(len(embeddings_dims))],
                                       embeddings_dims=embeddings_dims,
                                       combine_params_kwargs={'dim': -1, 'combine_func_str': 'concat'})
    tabular_features = CombineTensorsParams(input_tensor_name=['tabular', 'lagged'],
                                            output_tensor_name='all_tabular',
                                            combine_func_str='concat')
    num_tabular_features = len(features_params.feature_name_to_machine_learning_dataset_params['tabular'].feature_names)
    num_lagged_features = (
        len(set([f.split('_t-')[0] for f in
                 features_params.feature_name_to_machine_learning_dataset_params['lagged'].feature_names]
                )))
    tabular_dense = DenseLayersParams(input_tensor_name=tabular_features.output_tensor_name,
                                      output_tensor_name='embedded_tabular',
                                      num_nodes_per_layer=[num_tabular_features + num_lagged_features]
                                      )
    combined_tabular_embeddings = CombineTensorsParams(input_tensor_name=embeddings.output_tensor_name +
                                                                         [tabular_features.output_tensor_name],
                                                       output_tensor_name='combined_tabular_embeddings',
                                                       combine_func_str='concat',
                                                       dim=-1)

    positional_encodings_name = 'positional_encoding_features'
    num_positional_encoding_features = \
        len(features_params.feature_name_to_machine_learning_dataset_params[positional_encodings_name].feature_names)
    combined_embeddings_dim = (sum(embeddings.embeddings_dims) +
                               tabular_dense.num_nodes_per_layer[-1]
                               )
    # positional_encodings_embedding_dim = combined_embeddings_dim // 2
    positional_encodings_embedding_dim = [1, 11]  # HOME, START_POSITION_CALCULATED
    positional_encodings = EmbeddingsLayerParams(input_tensor_name='positional_encoding_features',
                                                 output_tensor_name=[f'embedded_positional_encoding_{i}' for i in
                                                                     range(num_positional_encoding_features)],
                                                 # embeddings_dims=[positional_encodings_embedding_dim] *
                                                 #                 num_positional_encoding_features,
                                                 embeddings_dims=positional_encodings_embedding_dim,
                                                 combine_params_kwargs={'dim': -1, 'combine_func_str': 'sum'},
                                                 max_norm=1.)
    with_positional_encoding = CombineTensorsParams(input_tensor_name=[combined_tabular_embeddings.output_tensor_name] +
                                                                      positional_encodings.output_tensor_name,
                                                    output_tensor_name='with_positional_encoding',
                                                    combine_func_str='concat',
                                                    dim=-1)
    n_heads = 8
    d_model = n_heads * 16
    inputs_encoder_params = DenseLayersParams(input_tensor_name=with_positional_encoding.output_tensor_name,
                                              output_tensor_name='transformer_inputs',
                                              num_nodes_per_layer=[d_model,],
                                              )

    # tsmixer_params = TSMixerParams(input_tensor_name={FeatureType.TIMESERIES: 'transformer_inputs'},
    #                                output_tensor_name='transformer_output',
    #                                num_mixer_layers=2,
    #                                prediction_length=22,
    #                                feature_mixing_hidden_channels=11,
    #                                time_dimension=1)

    transformer_encoder_params_1 = SplitAttentionCombineWrapperParams(input_tensor_name=inputs_encoder_params.output_tensor_name,
                                                                    output_tensor_name='transformer_output_1',
                                                                    d_model=d_model,
                                                                    n_heads=n_heads,
                                                                    ffdim=d_model * 4,
                                                                    dropout_rate=0.4,
                                                                    norm_at_beginning=True,
                                                                    n_bench_players=features_params.num_bench_players,
                                                                    num_layers=1
                                                                    )

    transformer_encoder_params = TransformerEncoderPytorchParams(input_tensor_name=transformer_encoder_params_1.output_tensor_name,
                                                                 # input_tensor_name=inputs_encoder_params.output_tensor_name, #inputs_encoder_params.output_tensor_name,
                                                                 output_tensor_name='transformer_output',
                                                                 d_model=d_model,
                                                                 nhead=n_heads,
                                                                 dim_feedforward=d_model * 4,
                                                                 dropout=0.4,
                                                                 norm_first=True,
                                                                 num_layers=2,
                                                                 activation=ActivationFunction.RELU,
                                                                 )

    output_dense = DenseLayersParams(input_tensor_name=transformer_encoder_params.output_tensor_name,
                                     #input_tensor_name=transformer_encoder_params_1.output_tensor_name,
                                     output_tensor_name='outputs_pred',
                                     num_nodes_per_layer=[num_outputs, num_outputs],
                                     use_batchnorm=True,
                                     activation=ActivationFunction.RELU)

    calculate_fantasy_points_params = CalculateFantasyPointsParams(input_tensor_name='outputs_pred',
                                                                   output_tensor_name='fantasy_points',
                                                                   site=Sites.YAHOO,
                                                                   stat_names=output_feature_names,
                                                                   )
    metadata_feature_names = features_params.feature_name_to_machine_learning_dataset_params[METADATA].feature_names
    fp_pred_reshape_to_sample_per_day_params = \
        ReshapeToSamplePerDayParams(input_tensor_name=[calculate_fantasy_points_params.output_tensor_name, METADATA],
                                    output_tensor_name='fantasy_points_pred_sample_per_day',
                                    sample_per_game_tensor_name=calculate_fantasy_points_params.output_tensor_name,
                                    metadata_name=METADATA,
                                    num_players_per_game=(5 + features_params.num_bench_players) * 2,
                                    date_ind_in_metadata=metadata_feature_names.index('DATE'),
                                    num_days_per_batch=num_days_per_batch
                                    )
    fp_true_reshape_to_sample_per_day_params = \
        ReshapeToSamplePerDayParams(input_tensor_name=[METADATA, METADATA],
                                    output_tensor_name='fantasy_points_true_sample_per_day',
                                    sample_per_game_tensor_name=METADATA,
                                    ind_in_sample_per_game_tensor=metadata_feature_names.index('YAHOO_POINTS'),
                                    metadata_name=METADATA,
                                    num_players_per_game=(5 + features_params.num_bench_players) * 2,
                                    date_ind_in_metadata=metadata_feature_names.index('DATE'),
                                    num_days_per_batch=num_days_per_batch
                                    )
    salary_reshape_to_sample_per_day_params = \
        ReshapeToSamplePerDayParams(input_tensor_name=[METADATA, METADATA],
                                    output_tensor_name='salary_sample_per_day',
                                    sample_per_game_tensor_name=METADATA,
                                    ind_in_sample_per_game_tensor=metadata_feature_names.index('YAHOO_SALARY'),
                                    metadata_name=METADATA,
                                    num_players_per_game=(5 + features_params.num_bench_players) * 2,
                                    date_ind_in_metadata=metadata_feature_names.index('DATE'),
                                    num_days_per_batch=num_days_per_batch
                                    )
    position_reshape_to_sample_per_day_params = \
        ReshapeToSamplePerDayParams(input_tensor_name=[METADATA, METADATA],
                                    output_tensor_name='position_sample_per_day',
                                    sample_per_game_tensor_name=METADATA,
                                    ind_in_sample_per_game_tensor=metadata_feature_names.index('YAHOO_POSITION'),
                                    metadata_name=METADATA,
                                    num_players_per_game=(5 + features_params.num_bench_players) * 2,
                                    date_ind_in_metadata=metadata_feature_names.index('DATE'),
                                    num_days_per_batch=num_days_per_batch
                                    )
    date_reshape_to_sample_per_day_params = \
        ReshapeToSamplePerDayParams(input_tensor_name=[METADATA, METADATA],
                                    output_tensor_name='date_sample_per_day',
                                    sample_per_game_tensor_name=METADATA,
                                    ind_in_sample_per_game_tensor=metadata_feature_names.index('DATE'),
                                    metadata_name=METADATA,
                                    num_players_per_game=(5 + features_params.num_bench_players) * 2,
                                    date_ind_in_metadata=metadata_feature_names.index('DATE'),
                                    num_days_per_batch=num_days_per_batch
                                    )
    keep_mask_reshape_to_sample_per_day_params = \
        ReshapeToSamplePerDayParams(input_tensor_name=['keep_mask', METADATA],
                                    output_tensor_name='keep_mask_sample_per_day',
                                    sample_per_game_tensor_name='keep_mask',
                                    metadata_name=METADATA,
                                    num_players_per_game=(5 + features_params.num_bench_players) * 2,
                                    date_ind_in_metadata=metadata_feature_names.index('DATE'),
                                    num_days_per_batch=num_days_per_batch
                                    )

    true_fp_and_salary_sample_per_day_params = (
        CombineTensorsParams(input_tensor_name=[fp_true_reshape_to_sample_per_day_params.output_tensor_name,
                                                salary_reshape_to_sample_per_day_params.output_tensor_name],
                             output_tensor_name='fantasy_points_salary_true_sample_per_day_combined',
                             dim=-1,
                             combine_func_str='stack'
                             ))

    optimize_teams_params = OptimizeTeamParams(input_tensor_name=[fp_pred_reshape_to_sample_per_day_params.output_tensor_name,
                                                                  fp_true_reshape_to_sample_per_day_params.output_tensor_name,
                                                                  salary_reshape_to_sample_per_day_params.output_tensor_name,
                                                                  position_reshape_to_sample_per_day_params.output_tensor_name,
                                                                  date_reshape_to_sample_per_day_params.output_tensor_name,
                                                                  ],
                                               output_tensor_name='optimize_teams_results',
                                               site=Sites.YAHOO,
                                               num_teams_in_set=1
                                               )

    # layer_params_list = [embeddings, tabular_dense, lagged_embedding, combined_tabular_embeddings,
    layer_params_list = [embeddings, tabular_features, tabular_dense, combined_tabular_embeddings,
                         positional_encodings, with_positional_encoding, inputs_encoder_params,
                         # tsmixer_params, output_dense]
                         transformer_encoder_params_1,
                         transformer_encoder_params,
                         output_dense, calculate_fantasy_points_params,
                         fp_pred_reshape_to_sample_per_day_params, fp_true_reshape_to_sample_per_day_params,
                         salary_reshape_to_sample_per_day_params, position_reshape_to_sample_per_day_params,
                         date_reshape_to_sample_per_day_params, keep_mask_reshape_to_sample_per_day_params,
                         true_fp_and_salary_sample_per_day_params,
                         optimize_teams_params
                         ]
    output_tensor_names = ['outputs_pred']
    return ArchitectureParamsFantasyBasketballPredictions(layer_params_list=layer_params_list,
                                                          output_tensor_names=output_tensor_names)


def create_training_params(features_params: FeaturesParamsFantasyBasketballPredictions, num_days_per_batch: int
                           ) -> TrainingParamsFantasyBasketballPredictions:
    batch_size = None #256
    lr = 0.001
    training_dtype = torch.float32
    dataloader_num_workers = 0

    model_id = get_next_model_id(PROJECT_NAME)
    true_label_dataset = features_params.feature_name_to_machine_learning_dataset_params['true_label']
    true_label_names = true_label_dataset.feature_names
    true_label_is_scaled = isinstance(true_label_dataset, MachineLearningDatasetTabularScalingParams)

    loss_params = MSELossParams(name_for_logs='mse',
                                column_names=true_label_names,
                                tensor_names_sample_weights='sample_weights',
                                tensor_names_pred='outputs_pred',
                                tensor_names_true='true_label',
                                tensor_names_keep_mask='keep_mask'
                                )
    # loss_params = QuantileRegressionLossParams(name_for_logs='quantile_loss',
    #                                            column_names=true_label_names,
    #                                            # tensor_names_sample_weights='sample_weights',
    #                                            tensor_names_pred='outputs_pred',
    #                                            tensor_names_true='true_label',
    #                                            tensor_names_keep_mask='keep_mask',
    #                                            quantile=0.3
    #                                            )

    optimize_teams_loss_params = OptimizeTeamsLossParams(tensor_names_true='fantasy_points_true_sample_per_day',
                                                         tensor_names_pred='optimize_teams_results',
                                                         weight=0., #0.002,
                                                         name_for_logs='optimize_teams_loss')

    pairwise_rank_loss_params = PairwiseRankingLossParams(tensor_names_true='fantasy_points_salary_true_sample_per_day_combined',
                                                          # tensor_names_true='fantasy_points_true_sample_per_day',
                                                          tensor_names_pred='fantasy_points_pred_sample_per_day',
                                                          tensor_names_keep_mask='keep_mask_sample_per_day',
                                                          weight=1.,
                                                          margin=0.05,
                                                          margin_is_pct_of_true_diffs=True,
                                                          use_true_diff_weights=True,
                                                          name_for_logs='pairwise_rank_loss')

    bradley_terry_rank_loss_params = BradleyTerryLossParams(tensor_names_true='fantasy_points_salary_true_sample_per_day_combined',
                                                            tensor_names_pred='fantasy_points_pred_sample_per_day',
                                                            tensor_names_keep_mask='keep_mask_sample_per_day',
                                                            name_for_logs='bradley_terry_loss',
                                                            margin=3,
                                                            margin_is_pct_of_true_diffs=False,
                                                            weight=5)

    save_predictions_path = os.path.join(models_dir(PROJECT_NAME), str(model_id), 'predictions.arrow')
    callbacks_params = (ModelCheckpointParams(),
                        EarlyStoppingParams(),
                        ReduceLearningRateOnPlateauParams(),
                        LogModelResultsParams(),
                        CollectTruePredictedOutputsParamsFantasyBasketballPredictions(
                            column_names={'true_label': [f'{c}_true' for c in true_label_names],
                                          'outputs_pred': [f'{c}_pred' for c in true_label_names],
                                          METADATA: features_params.feature_name_to_machine_learning_dataset_params[METADATA].feature_names,
                                          'keep_mask': ['NOT_IGNORE']},
                            scaler_dir=os.path.join(models_dir(PROJECT_NAME), str(model_id), 'scalers') if true_label_is_scaled else None
                        ),
                        )

    metrics_params = (MAEMetricParams(tensor_names_pred='outputs_pred',
                                      tensor_names_true='true_label',
                                      tensor_names_keep_mask='keep_mask',
                                      reduction_dim=(0, 1),
                                      name_for_logs='mae_metrics',
                                      column_names=true_label_names),
                      optimize_teams_loss_params
                      )

    return TrainingParamsFantasyBasketballPredictions(batch_size=batch_size,
                                                      num_days_per_batch=num_days_per_batch,
                                                      learning_rate=lr,
                                                      num_epochs=None,
                                                      training_dtype=training_dtype,
                                                      dataloader_num_workers=dataloader_num_workers,
                                                      loss_params=(loss_params,
                                                                   # optimize_teams_loss_params,
                                                                   # pairwise_rank_loss_params,
                                                                   bradley_terry_rank_loss_params
                                                                   ),
                                                      callbacks_params=callbacks_params,
                                                      metrics_params=metrics_params,
                                                      )


def create_params() -> MachineLearningParamsFantasyBasketballPredictions:
    num_days_per_batch = 4
    features_params = create_features_params()
    architecture_params = create_architecture_params(features_params, num_days_per_batch=num_days_per_batch)
    training_params = create_training_params(features_params, num_days_per_batch=num_days_per_batch)
    return MachineLearningParamsFantasyBasketballPredictions(features_params=features_params,
                                                             architecture_params=architecture_params,
                                                             training_params=training_params)


def test_params(params):
    # testing to_dict -> init_from_dict, to make sure they produce identical results
    params_dict = params.to_dict()
    params2 = MachineLearningParamsFantasyBasketballPredictions.init_from_dict(params_dict)
    compare_dataclasses(params, params2)

    # testing to_dict -> save -> load -> init_from_dict to make sure the final result is identical to the original
    test_path = 'garbage_params.json'
    params.save_to_json(path=test_path)
    params3 = MachineLearningParamsFantasyBasketballPredictions.load_params(test_path)
    compare_dataclasses(params, params3)
    os.remove(test_path)


if __name__ == '__main__':
    # remove directories with no model file
    project_models_dir = models_dir(PROJECT_NAME)
    for d in os.listdir(project_models_dir):
        curr_model_dir = os.path.join(project_models_dir, d)
        if not os.path.isfile(os.path.join(curr_model_dir, 'model.pt')):
            shutil.rmtree(curr_model_dir)

    define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO,
                  filename=None, print_to_console=True, filemode=None)
    
    p1 = MachineLearningParamsFantasyBasketballPredictions.load_params('C:/Users/schwa/PycharmProjects/Fantasy-Basketball-Predictions/machine_learning_fantasy_basketball_predictions/models/71/params.json')
    
    params = create_params()
    test_params(params)

    # aaa =  compare_dataclasses(params, p1)

    run_model = RunModelFantasyBasketballPredictions(params=params, description=None)
    eval_results, predict_results = run_model.run_full_training_session()

    predict_results.to_feather(os.path.join(models_dir(PROJECT_NAME), str(run_model.model_id), 'predict_results.feather'))
