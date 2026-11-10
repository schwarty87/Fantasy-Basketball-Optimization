from constants import NUM_TEAMS, DFS_MAX_TEAM_SALARY, DFS_POSITIONS
from dataclasses import dataclass
from enums import Sites
from machine_learning.enums import DataSplitType
from machine_learning.machine_learning_datasets import MachineLearningDatasetCombined
from machine_learning.registries import params_registry
from typing import List, Optional, Dict, Any
from machine_learning.params.abstract_params import LayerParams
from machine_learning.constants import METADATA


@params_registry.register
@dataclass(kw_only=True)
class CalculateFantasyPointsParams(LayerParams):
    """Parameters for the layer that converts the model predictions to fantasy points."""
    # A list of the raw stat names of the input tensor. The order is important
    stat_names: List[str]

    # the fantasy site to use
    site: Sites = Sites.YAHOO

    # optional standard scaler parameters to unscale predictions back to raw physical stats
    scaler_mean: Optional[List[float]] = None
    scaler_scale: Optional[List[float]] = None

    def update_after_loading_data(self, updated_params: Dict[str, Any]):
        if 'scaler_mean' in updated_params:
            self.scaler_mean = updated_params['scaler_mean']
        if 'scaler_scale' in updated_params:
            self.scaler_scale = updated_params['scaler_scale']


def update_calculate_fantasy_points_params(params: Any, dataset_combined: Any = None, data_loaders: Any = None) -> None:
    """Populates scaler_mean and scaler_scale on CalculateFantasyPointsParams from the true_label scaler."""
    arch_params = getattr(params, 'architecture_params', None)
    if arch_params is None or not hasattr(arch_params, 'layer_params_list'):
        return

    cfp_layers = [p for p in arch_params.layer_params_list if isinstance(p, CalculateFantasyPointsParams)]
    if not cfp_layers:
        return

    scaler = None
    if dataset_combined is not None and hasattr(dataset_combined, 'machine_learning_datasets_stacked'):
        ds = getattr(dataset_combined.machine_learning_datasets_stacked, 'datasets', {}).get('true_label')
        scaler = getattr(ds, 'scaler', None)
    elif data_loaders:
        try:
            ds = MachineLearningDatasetCombined.get_single_dataset_from_data_loaders(
                data_loaders=data_loaders,
                data_split_type=DataSplitType.TRAIN,
                feature_type='true_label',
            )
            scaler = getattr(ds, 'scaler', None)
        except Exception:
            pass

    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        for p in cfp_layers:
            if hasattr(scaler, 'feature_names_in_'):
                col_to_mean = dict(zip(scaler.feature_names_in_, scaler.mean_))
                col_to_scale = dict(zip(scaler.feature_names_in_, scaler.scale_))
                mean_list = [float(col_to_mean.get(s, 0.0)) for s in p.stat_names]
                scale_list = [float(col_to_scale.get(s, 1.0)) for s in p.stat_names]
            else:
                mean_list = [float(m) for m in scaler.mean_]
                scale_list = [float(s) for s in scaler.scale_]
            p.update_after_loading_data({'scaler_mean': mean_list, 'scaler_scale': scale_list})


@params_registry.register
@dataclass(kw_only=True)
class ReshapeToSamplePerDayParams(LayerParams):
    sample_per_game_tensor_name: str
    ind_in_sample_per_game_tensor: int | None = None
    metadata_name: str = METADATA
    date_ind_in_metadata: int = 0
    max_num_games_per_day: int = NUM_TEAMS // 2
    num_players_per_game: int = 22
    num_days_per_batch: int = 1


@params_registry.register
@dataclass(kw_only=True)
class OptimizeTeamParams(LayerParams):
    """Parameters for the differentiable optimization layer to find the best fantasy team."""
    site: Sites = Sites.YAHOO
    max_salary: int | None = None
    num_players: int | None = None
    num_teams_in_set: int = 1
    max_shared_players_in_teams_pairs: int | None = None
    avoid_single_player_in_all_sets: bool = False
    run_in_train: bool = False
    verbose: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.max_salary is None:
            self.max_salary = DFS_MAX_TEAM_SALARY[self.site]
        if self.num_players is None:
            self.num_players = len(DFS_POSITIONS[self.site])
        if self.num_teams_in_set == 1:
            self.max_shared_players_in_teams_pairs = None
            self.avoid_single_player_in_all_sets = False