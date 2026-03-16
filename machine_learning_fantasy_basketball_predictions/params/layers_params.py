from constants import NUM_TEAMS, DFS_MAX_TEAM_SALARY, DFS_POSITIONS
from dataclasses import dataclass
from enums import Sites
from machine_learning.registries import params_registry
from typing import List
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