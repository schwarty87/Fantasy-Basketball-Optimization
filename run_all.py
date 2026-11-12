import os
import shutil
import tempfile
import argparse
import logging
import torch
import numpy as np
from dataclasses import dataclass, is_dataclass, fields
from typing import Dict, Tuple, Optional
import pandas as pd
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

    feature_name_to_machine_learning_dataset_params = {
        'tabular': MachineLearningDatasetTabularScalingParams(
            feature_names=[
                'DAYS_REST', 'SEASON_PROGRESS',
            ] + [f'{stat}_mean5' for stat in output_features] + [f'{stat}_mean10' for stat in output_features],
            relevant_keep_mask_dataset_name='real_player_mask',
            masked_fill_value=0.0,
            dtype=features_dtype,
        ),
        'lagged': MachineLearningDatasetTabularScalingParams(
            feature_names=[f'{stat}_t-{lag}' for stat in output_features + ['STARTER'] for lag in range(1, 4)],
            relevant_keep_mask_dataset_name='real_player_mask',
            masked_fill_value=0.0,
            dtype=features_dtype,
        ),
        'positional_encoding_features': MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams(
            feature_names=['HOME', 'START_POSITION_CALCULATED'],
            dtype=features_dtype,
        ),
        'embedding_features': MachineLearningDatasetTabularCategoricalEncodingFantasyBasketballPredictionsParams(
            feature_names=['PLAYER_ID', 'PLAYER_SEASON', 'SEASON_OPPONENT', 'SEASON_TEAM', 'SEASON_YEAR', 'TEAM_ID'],
            dtype=features_dtype,
        ),
        'true_label': MachineLearningDatasetTabularScalingParams(
            feature_names=output_features,
            relevant_keep_mask_dataset_name='keep_mask',
            dtype=features_dtype,
        ),
        'sample_weights': MachineLearningDatasetSampleWeightsTabularFantasyBasketballPredictionsParams(
            feature_names=output_features,
            dtype=features_dtype,
            sample_weight_type=SampleWeightType.BY_HISTOGRAM,
            weights_per_column_or_all_together=True,
            max_class_weight=1,
            relevant_keep_mask_dataset_name='keep_mask',
        ),
        METADATA: MachineLearningDatasetMetadataParams(
            feature_names=['DATE', 'split_type', 'PLAYER_ID', 'YAHOO_SALARY', 'YAHOO_POSITION', 'YAHOO_POINTS'],
            dtype=features_dtype,
        ),
        'keep_mask': MachineLearningDatasetKeepMaskTabularParams(
            feature_names=['NOT_IGNORE'],
            dtype=features_dtype,
        ),
        'real_player_mask': MachineLearningDatasetKeepMaskTabularParams(
            feature_names=['IS_REAL_PLAYER'],
            dtype=features_dtype,
        ),
    }

    return FeaturesParamsFantasyBasketballPredictions(
        features_path=features_path,
        feature_name_to_machine_learning_dataset_params=feature_name_to_machine_learning_dataset_params,
        num_bench_players=6,
        impute_types=(ImputationType.NONE,),
        sample_per_game=True,
        all_games_every_day=True,
    )


def create_architecture_params(
    features_params: FeaturesParamsFantasyBasketballPredictions,
    num_days_per_batch: int,
    use_skip_connection: bool = False,
    d_model: int = 128,
    n_heads: int = 8,
) -> ArchitectureParamsFantasyBasketballPredictions:
    output_feature_names = features_params.feature_name_to_machine_learning_dataset_params['true_label'].feature_names
    num_outputs = len(output_feature_names)

    # 1. Categorical embeddings (mapped explicitly to prevent index mismatch)
    embedding_feature_dims = {
        'PLAYER_ID': 32,
        'PLAYER_SEASON': 12,
        'SEASON_OPPONENT': 4,
        'SEASON_TEAM': 4,
        'SEASON_YEAR': 4,
        'TEAM_ID': 4,
    }
    embedding_features = features_params.feature_name_to_machine_learning_dataset_params['embedding_features'].feature_names
    embeddings_dims = [embedding_feature_dims.get(f, 4) for f in embedding_features]

    embeddings = EmbeddingsLayerParams(
        input_tensor_name='embedding_features',
        output_tensor_name=[f'embedded_{i}' for i in range(len(embeddings_dims))],
        embeddings_dims=embeddings_dims,
        combine_params_kwargs={'dim': -1, 'combine_func_str': 'concat'},
    )

    # 2. Combine tabular + lagged features, then fuse with categorical embeddings
    tabular_features = CombineTensorsParams(
        input_tensor_name=['tabular', 'lagged'],
        output_tensor_name='all_tabular',
        combine_func_str='concat',
    )
    combined_tabular_embeddings = CombineTensorsParams(
        input_tensor_name=embeddings.output_tensor_name + [tabular_features.output_tensor_name],
        output_tensor_name='combined_tabular_embeddings',
        combine_func_str='concat',
        dim=-1,
    )

    # 3. Positional encodings (HOME, START_POSITION_CALCULATED)
    num_positional_features = len(
        features_params.feature_name_to_machine_learning_dataset_params['positional_encoding_features'].feature_names
    )
    positional_encodings = EmbeddingsLayerParams(
        input_tensor_name='positional_encoding_features',
        output_tensor_name=[f'embedded_positional_encoding_{i}' for i in range(num_positional_features)],
        embeddings_dims=[1, 11],
        combine_params_kwargs={'dim': -1, 'combine_func_str': 'sum'},
        max_norm=1.0,
    )
    with_positional_encoding = CombineTensorsParams(
        input_tensor_name=[combined_tabular_embeddings.output_tensor_name] + positional_encodings.output_tensor_name,
        output_tensor_name='with_positional_encoding',
        combine_func_str='concat',
        dim=-1,
    )

    # 4. Linear projection into transformer space
    inputs_encoder_params = DenseLayersParams(
        input_tensor_name=with_positional_encoding.output_tensor_name,
        output_tensor_name='transformer_inputs',
        num_nodes_per_layer=[d_model],
    )

    # 5. Team-level contextual attention (within vs between teams)
    transformer_encoder_params_1 = SplitAttentionCombineWrapperParams(
        input_tensor_name={'x': inputs_encoder_params.output_tensor_name,
                           'mask': 'real_player_mask'},
        output_tensor_name='transformer_output_1',
        d_model=d_model,
        n_heads=n_heads,
        ffdim=d_model * 4,
        dropout_rate=0.30,
        norm_at_beginning=True,
        n_bench_players=features_params.num_bench_players,
        num_layers=1,
        activation=ActivationFunction.GELU,
    )

    # 6. Global multi-head self-attention across all 22 active players
    transformer_encoder_params = TransformerEncoderPytorchParams(
        input_tensor_name={'x': transformer_encoder_params_1.output_tensor_name,
                           'mask': 'real_player_mask'},
        output_tensor_name='transformer_output',
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * 4,
        dropout=0.30,
        norm_first=True,
        num_layers=2,
        activation=ActivationFunction.GELU,
    )

    # 7. Optional residual / skip connection
    if use_skip_connection:
        post_transformer_features = CombineTensorsParams(
            input_tensor_name=[transformer_encoder_params.output_tensor_name,
                               inputs_encoder_params.output_tensor_name],
            output_tensor_name='post_transformer_combined',
            combine_func_str='concat',
            dim=-1,
        )
        head_input_tensor_name = post_transformer_features.output_tensor_name
    else:
        post_transformer_features = None
        head_input_tensor_name = transformer_encoder_params.output_tensor_name

    # 8. Output dense head: predicts the 11 target stats
    output_dense = DenseLayersParams(
        input_tensor_name=head_input_tensor_name,
        output_tensor_name='outputs_pred',
        num_nodes_per_layer=[64, num_outputs],
        use_batchnorm=True,
        activation=ActivationFunction.GELU,
        dropout_rate=0.25,
    )

    # 9. Compute Yahoo Fantasy Points from predicted stats
    calculate_fantasy_points_params = CalculateFantasyPointsParams(
        input_tensor_name='outputs_pred',
        output_tensor_name='fantasy_points',
        site=Sites.YAHOO,
        stat_names=output_feature_names,
    )

    # 10. Reshape to daily batches for DFS lineup optimization
    metadata_feature_names = features_params.feature_name_to_machine_learning_dataset_params[METADATA].feature_names
    num_players_per_game = (5 + features_params.num_bench_players) * 2
    date_ind = metadata_feature_names.index('DATE')

    def make_daily_reshape(output_name: str, sample_source: str, ind_in_sample: Optional[int] = None) -> ReshapeToSamplePerDayParams:
        input_tensors = [sample_source, METADATA] if sample_source != METADATA else [METADATA, METADATA]
        kwargs = dict(
            input_tensor_name=input_tensors,
            output_tensor_name=output_name,
            sample_per_game_tensor_name=sample_source,
            metadata_name=METADATA,
            num_players_per_game=num_players_per_game,
            date_ind_in_metadata=date_ind,
            num_days_per_batch=num_days_per_batch,
        )
        if ind_in_sample is not None:
            kwargs['ind_in_sample_per_game_tensor'] = ind_in_sample
        return ReshapeToSamplePerDayParams(**kwargs)

    fp_pred_reshape = make_daily_reshape('fantasy_points_pred_sample_per_day', calculate_fantasy_points_params.output_tensor_name)
    fp_true_reshape = make_daily_reshape('fantasy_points_true_sample_per_day', METADATA, metadata_feature_names.index('YAHOO_POINTS'))
    salary_reshape = make_daily_reshape('salary_sample_per_day', METADATA, metadata_feature_names.index('YAHOO_SALARY'))
    position_reshape = make_daily_reshape('position_sample_per_day', METADATA, metadata_feature_names.index('YAHOO_POSITION'))
    date_reshape = make_daily_reshape('date_sample_per_day', METADATA, date_ind)
    keep_mask_reshape = make_daily_reshape('keep_mask_sample_per_day', 'keep_mask')

    true_fp_and_salary_combined = CombineTensorsParams(
        input_tensor_name=[fp_true_reshape.output_tensor_name, salary_reshape.output_tensor_name],
        output_tensor_name='fantasy_points_salary_true_sample_per_day_combined',
        dim=-1,
        combine_func_str='stack',
    )

    optimize_teams_params = OptimizeTeamParams(
        input_tensor_name=[
            fp_pred_reshape.output_tensor_name,
            fp_true_reshape.output_tensor_name,
            salary_reshape.output_tensor_name,
            position_reshape.output_tensor_name,
            date_reshape.output_tensor_name,
        ],
        output_tensor_name='optimize_teams_results',
        site=Sites.YAHOO,
        num_teams_in_set=1,
    )

    # Assemble architecture layer list in execution order
    layer_params_list = [
        embeddings, tabular_features, combined_tabular_embeddings,
        positional_encodings, with_positional_encoding, inputs_encoder_params,
        transformer_encoder_params_1, transformer_encoder_params,
    ]
    if post_transformer_features is not None:
        layer_params_list.append(post_transformer_features)
    layer_params_list += [
        output_dense, calculate_fantasy_points_params,
        fp_pred_reshape, fp_true_reshape, salary_reshape, position_reshape,
        date_reshape, keep_mask_reshape, true_fp_and_salary_combined,
        optimize_teams_params,
    ]

    return ArchitectureParamsFantasyBasketballPredictions(
        layer_params_list=layer_params_list,
        output_tensor_names=['outputs_pred'],
    )


def create_training_params(
    features_params: FeaturesParamsFantasyBasketballPredictions,
    num_days_per_batch: int,
) -> TrainingParamsFantasyBasketballPredictions:
    model_id = get_next_model_id(PROJECT_NAME)
    true_label_dataset = features_params.feature_name_to_machine_learning_dataset_params['true_label']
    true_label_names = true_label_dataset.feature_names
    true_label_is_scaled = isinstance(true_label_dataset, MachineLearningDatasetTabularScalingParams)

    loss_params = MSELossParams(
        name_for_logs='mse',
        column_names=true_label_names,
        tensor_names_sample_weights='sample_weights',
        tensor_names_pred='outputs_pred',
        tensor_names_true='true_label',
        tensor_names_keep_mask='keep_mask',
    )

    optimize_teams_loss_params = OptimizeTeamsLossParams(
        tensor_names_true='fantasy_points_true_sample_per_day',
        tensor_names_pred='optimize_teams_results',
        weight=0.0,
        name_for_logs='optimize_teams_loss',
    )

    callbacks_params = (
        ModelCheckpointParams(),
        EarlyStoppingParams(patience=4),
        ReduceLearningRateOnPlateauParams(patience=2),
        LogModelResultsParams(),
        CollectTruePredictedOutputsParamsFantasyBasketballPredictions(
            column_names={
                'true_label': [f'{c}_true' for c in true_label_names],
                'outputs_pred': [f'{c}_pred' for c in true_label_names],
                METADATA: features_params.feature_name_to_machine_learning_dataset_params[METADATA].feature_names,
                'keep_mask': ['NOT_IGNORE'],
            },
            scaler_dir=os.path.join(models_dir(PROJECT_NAME), str(model_id), 'scalers') if true_label_is_scaled else None,
        ),
    )

    metrics_params = (
        MAEMetricParams(
            tensor_names_pred='outputs_pred',
            tensor_names_true='true_label',
            tensor_names_keep_mask='keep_mask',
            reduction_dim=(0, 1),
            name_for_logs='mae_metrics',
            column_names=true_label_names,
        ),
        optimize_teams_loss_params,
    )

    return TrainingParamsFantasyBasketballPredictions(
        batch_size=None,
        num_days_per_batch=num_days_per_batch,
        learning_rate=0.001,
        weight_decay=1e-4,
        num_epochs=None,
        training_dtype=torch.float32,
        dataloader_num_workers=0,
        gradient_clip_val=1.0,
        loss_params=(loss_params,),
        callbacks_params=callbacks_params,
        metrics_params=metrics_params,
    )


def create_params(use_skip_connection: bool = False, d_model: int = 128) -> MachineLearningParamsFantasyBasketballPredictions:
    num_days_per_batch = 4
    features_params = create_features_params()
    architecture_params = create_architecture_params(
        features_params,
        num_days_per_batch=num_days_per_batch,
        use_skip_connection=use_skip_connection,
        d_model=d_model,
    )
    training_params = create_training_params(features_params, num_days_per_batch=num_days_per_batch)
    return MachineLearningParamsFantasyBasketballPredictions(
        features_params=features_params,
        architecture_params=architecture_params,
        training_params=training_params,
    )


def test_params(params):
    """Verify serialization round-trips (to_dict -> init_from_dict and save -> load)."""
    # 1. to_dict -> init_from_dict
    params_dict = params.to_dict()
    params2 = MachineLearningParamsFantasyBasketballPredictions.init_from_dict(params_dict)
    compare_dataclasses(params, params2)

    # 2. save_to_json -> load_params
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        test_path = tmp.name
    try:
        params.save_to_json(path=test_path)
        params3 = MachineLearningParamsFantasyBasketballPredictions.load_params(test_path)
        compare_dataclasses(params, params3)
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def clean_empty_model_dirs(project_name: str = PROJECT_NAME):
    """Remove model directories that do not contain a trained model file."""
    project_models_dir = models_dir(project_name)
    if not os.path.isdir(project_models_dir):
        return
    for d in os.listdir(project_models_dir):
        curr_model_dir = os.path.join(project_models_dir, d)
        if not os.path.isdir(curr_model_dir):
            continue
        has_pt = os.path.isfile(os.path.join(curr_model_dir, 'model.pt'))
        has_ckpt = any(f.endswith('.ckpt') for f in os.listdir(curr_model_dir))
        if not (has_pt or has_ckpt):
            shutil.rmtree(curr_model_dir, ignore_errors=True)


def main(
    use_lightning: bool = False,
    params: Optional[MachineLearningParamsFantasyBasketballPredictions] = None,
    clean_dirs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if clean_dirs:
        clean_empty_model_dirs(PROJECT_NAME)

    define_logger(
        format='%(asctime)s [%(levelname)s] - %(message)s',
        level=logging.INFO,
        filename=None,
        print_to_console=True,
        filemode=None,
    )

    if params is None:
        params = create_params()
    test_params(params)

    run_model = RunModelFantasyBasketballPredictions(
        params=params, description=None, use_lightning=use_lightning
    )
    eval_results, predict_results = run_model.run_full_training_session()

    output_feather = os.path.join(
        models_dir(PROJECT_NAME), str(run_model.model_id), 'predict_results.feather'
    )
    predict_results.to_feather(output_feather)

    return eval_results, predict_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fantasy Basketball Predictions Model")
    parser.add_argument(
        "--use_lightning",
        action="store_true",
        help="Use PyTorch Lightning training loop instead of legacy manual PyTorch loop",
    )
    parser.add_argument(
        "--fast-dev-run",
        "--fast_dev_run",
        action="store_true",
        dest="fast_dev_run",
        help="Run 1 train and 1 val batch to quickly verify code before full training",
    )
    parser.add_argument(
        "--auto-lr-find",
        "--auto_lr_find",
        action="store_true",
        dest="auto_lr_find",
        help="Run Lightning Tuner learning rate finder before training",
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="Enable Stochastic Weight Averaging callback",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorchProfiler during training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for full reproducibility (default: 42)",
    )
    parser.add_argument(
        "--track-grad-norm",
        "--track_grad_norm",
        type=float,
        default=None,
        dest="track_grad_norm",
        help="Track and log gradient norm (e.g. 2 for L2 norm)",
    )
    parser.add_argument(
        "--skip-connection",
        "--skip_connection",
        action="store_true",
        dest="skip_connection",
        help="Enable the pre-transformer skip connection to the output dense head",
    )
    parser.add_argument(
        "--d_model",
        "--d-model",
        type=int,
        default=128,
        dest="d_model",
        help="Transformer embedding dimension (default: 128)",
    )
    args = parser.parse_args()

    use_lightning = args.use_lightning
    use_skip = getattr(args, "skip_connection", False)
    d_model = getattr(args, "d_model", 128)
    params = create_params(use_skip_connection=use_skip, d_model=d_model)
    if args.fast_dev_run:
        params.training_params.fast_dev_run = True
    if args.auto_lr_find:
        params.training_params.auto_lr_find = True
    if args.swa:
        params.training_params.use_swa = True
    if args.profile:
        params.training_params.run_profiler = True
    if args.seed is not None:
        params.training_params.seed = args.seed
    if args.track_grad_norm is not None:
        params.training_params.track_grad_norm = args.track_grad_norm

    main(use_lightning=use_lightning, params=params)
