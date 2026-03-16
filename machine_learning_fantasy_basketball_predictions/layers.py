from functools import partial
from joblib import Parallel, delayed
import numpy as np
import torch
import torch.nn.functional as F
from constants import SCORING, Sites, NUM_TO_POSITION
from typing import Tuple, Dict, Any, Callable
from find_best_team_linear_programming import (FindBestSetOfTeamsLinearProgrammingYahoo,
                                               FindBestSetOfTeamsLinearProgrammingYahooScipy)
from machine_learning.registries import layers_registry
from machine_learning.layers.abstract_layer import AbstractLayer, LayersInputOutputShape
from machine_learning_fantasy_basketball_predictions.params.layers_params import (CalculateFantasyPointsParams,
  OptimizeTeamParams, ReshapeToSamplePerDayParams)


@layers_registry.register
class CalculateFantasyPoints(AbstractLayer):
    """A non-learnable layer that calculates DFS points given the raw stats."""

    def __init__(self, params: CalculateFantasyPointsParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)

        self.scoring_vector = torch.zeros(len(params.stat_names))

        # if we already calculated the value, return the value
        stat_names_lower = [n.lower() for n in params.stat_names]
        if f'{params.site}_points'.lower() in stat_names_lower:
            self.scoring_vector[stat_names_lower.index(f'{params.site}_points')] = 1.
        else:
            scoring = SCORING[params.site].copy()

            # updating the scoring values per stat if we have different stats for calculating the points value
            if 'PTS' in scoring and 'PTS' not in params.stat_names:
                if all([s in params.stat_names for s in ('3PM', '2PM', 'FTM')]):
                    scoring['3PM'] = scoring['PTS'] * 3
                    scoring['2PM'] = scoring['PTS'] * 2
                    scoring['FTM'] = scoring['PTS']
                    del scoring['PTS']
                else:
                    raise ValueError('Missing points column from stat_names!')

            stat_to_idx = {name: i for i, name in enumerate(params.stat_names)}
            for stat, weight in scoring.items():
                if stat in params.stat_names:
                    self.scoring_vector[stat_to_idx[stat]] = weight
                else:
                    raise ValueError(f"We're missing the stat {stat} for calculating fantasy points!")

        self.register_buffer('weights', self.scoring_vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, ..., num_stats]
        fantasy_points = torch.einsum('...s, s -> ...', x, self.weights)  # [batch_size, ...]
        return fantasy_points

    def output_shape(self) -> LayersInputOutputShape:
        return self.input_shape[:-1]


@layers_registry.register
class ReshapeToSamplePerDay(AbstractLayer):
    """Reshape the model outputs, so that each sample contains all games from a single day"""

    def __init__(self, params: ReshapeToSamplePerDayParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)
        self._sample_per_game_tensor_name = params.sample_per_game_tensor_name
        self._ind_in_sample_per_game_tensor = params.ind_in_sample_per_game_tensor
        self._metadata_name = params.metadata_name
        self._date_ind_in_metadata = params.date_ind_in_metadata
        self._num_players_per_game = params.num_players_per_game
        self._max_num_games_per_day = params.max_num_games_per_day
        self._num_day_per_batch = params.num_days_per_batch

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # extract data
        sample_per_game = x[self._sample_per_game_tensor_name]
        if self._ind_in_sample_per_game_tensor is not None:
            sample_per_game = sample_per_game[..., self._ind_in_sample_per_game_tensor]
        dates = x[self._metadata_name][..., 0, self._date_ind_in_metadata]

        sample_per_day = torch.zeros(self._num_day_per_batch, self._max_num_games_per_day, self._num_players_per_game,
                                     dtype=sample_per_game.dtype, device=sample_per_game.device)

        unique_dates, group_indices = torch.unique(dates, return_inverse=True, sorted=False)
        one_hot = F.one_hot(group_indices, num_classes=self._num_day_per_batch)
        item_idx = (torch.cumsum(one_hot, dim=0) - 1) * one_hot
        item_idx = item_idx.sum(dim=1)
        if sample_per_game.ndim == 3 and sample_per_game.shape[-1] == 1:
            sample_per_game = sample_per_game[..., 0]
        sample_per_day[group_indices, item_idx] = sample_per_game
        return sample_per_day

    def output_shape(self) -> LayersInputOutputShape:
        return self._max_num_games_per_day, self._num_players_per_game


OptimizeTeamFn_true_results = dict()

class OptimizeTeamFn(torch.autograd.Function):
    """
    Custom autograd function to handle the non-differentiable team optimization step.
    The forward pass uses PuLP to find the optimal team, while the backward pass
    provides a custom gradient to guide the model's predictions.
    """

    @staticmethod
    def _run_single_day(points: np.ndarray, salaries: np.ndarray, positions: np.ndarray,
                        optimizer: FindBestSetOfTeamsLinearProgrammingYahoo, cache_key: float | None = None
                        ) -> np.ndarray:
        """
        Run the optimizer for a single day
        :param points: A 1d float array of points (the value we want to optimize)
        :param salaries: A 1d float array of salaries, the same length as points
        :param positions: A 1d str array of positions, the same length as points
        :param optimizer: The optimizer to use
        :param cache_key: the key to use for caching
        :return: A 1d bool array, the same length as points, where any player who appeared in any of the optimized
                 teams is set to True
        """
        if cache_key is not None and cache_key in OptimizeTeamFn_true_results.keys():
            return OptimizeTeamFn_true_results[cache_key]

        optimizer.run_inner(points=points, salaries=salaries, positions=positions)
        results_set = optimizer.get_results_boolean_array()
        results_set = results_set.any(axis=0)

        if cache_key is not None:
            OptimizeTeamFn_true_results[cache_key] = results_set

        return results_set

    @staticmethod
    def forward(ctx: Any, y_pred: torch.Tensor, y_true: torch.Tensor, salaries: torch.Tensor,
                positions: torch.Tensor, optimizer: FindBestSetOfTeamsLinearProgrammingYahoo,
                num_to_position: Callable[[np.ndarray], np.ndarray], dates: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the team optimization.

        Args:
            ctx: Context object for saving information for the backward pass.
            y_pred: Predicted fantasy points for each player.
            y_true: True fantasy points for each player.
            salaries: Salary of each player.
            positions: Position of each player.
            optimizer: An instance of a linear programming optimizer for team selection.
            num_to_position: A function to convert position numbers to string representations.
            dates: Dates of each row

        Returns:
            A boolean tensor representing the optimal team based on true points.
        """
        # Detach tensors from the computation graph and convert to numpy
        y_pred_np = y_pred.cpu().detach().numpy()
        y_true_np = y_true.cpu().detach().numpy()
        salaries_np = salaries.cpu().detach().numpy()
        positions_np = positions.cpu().detach().numpy()
        dates = dates.cpu().detach().numpy()[:, 0]

        # # Get masks for valid players (positive salary and valid position)
        # salary_mask = (salaries_np > 0)
        # positions_mask = np.isin(positions_np, list(NUM_TO_POSITION.keys()))
        # mask = salary_mask & positions_mask

        # # Initialize results arrays
        # results_pred_all = np.zeros_like(y_pred_np, dtype=bool)
        # results_true_all = np.zeros_like(y_true_np, dtype=bool)

        # Iterate over each day in the batch
        optimize_team_funcs_pred = list()
        optimize_team_funcs_true = list()
        # for day_num, day_mask in enumerate(mask):
        for day_num, date_day in enumerate(dates):
            # if not np.any(day_mask):
            #     continue

            # Filter for active players for the current day
            points_pred = y_pred_np[day_num] #[day_mask]
            points_true = y_true_np[day_num] #[day_mask]
            salaries_day = salaries_np[day_num] #[day_mask]
            positions_day = positions_np[day_num] #[day_mask]
            # date_day = dates[day_num]

            # Find optimal team based on predicted points
            optimize_team_funcs_pred.append(partial(OptimizeTeamFn._run_single_day, points=points_pred,
                                                    salaries=salaries_day, positions=num_to_position(positions_day),
                                                    optimizer=optimizer
                                                    )
                                            )

            # Find optimal team based on true points
            optimize_team_funcs_true.append(partial(OptimizeTeamFn._run_single_day, points=points_true,
                                                    salaries=salaries_day, positions=num_to_position(positions_day),
                                                    optimizer=optimizer, cache_key=date_day
                                                    )
                                            )

        results = list()
        all_funcs = optimize_team_funcs_pred + optimize_team_funcs_true
        for i, f in enumerate(all_funcs):
            results.append(f())

        # results = [f() for f in optimize_team_funcs_pred + optimize_team_funcs_true]
        # # results = Parallel(n_jobs=-1)([delayed(f)() for f in optimize_team_funcs_pred + optimize_team_funcs_true])
        results_pred = results[:len(optimize_team_funcs_pred)]
        results_true = results[len(optimize_team_funcs_pred):]

        results_pred_all = np.vstack(results_pred).astype(float)
        results_true_all = np.vstack(results_true).astype(float)

        has_team_pred = np.any(results_pred_all, axis=1)
        remove = np.logical_not(has_team_pred)
        results_true_all[remove] = 0

        # Save tensors for the backward pass
        ctx.save_for_backward(torch.from_numpy(results_pred_all).to(y_pred.device, dtype=y_pred.dtype),
                              torch.from_numpy(results_true_all).to(y_pred.device, dtype=y_pred.dtype),
                              )

        # Return the optimal team based on true points
        return torch.from_numpy(results_pred_all).float().to(y_pred.device)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None, None, None]:
        """
        Performs the backward pass of the team optimization using an SPO+ inspired approach.

        Args:
            ctx: Context object for retrieving saved information.
            grad_output: The gradient from the subsequent layer.

        Returns:
            A tuple of gradients for each input of the forward pass.
        """
        # Retrieve saved tensors
        predicted_team, true_team = ctx.saved_tensors

        # Calculate the SPO+ gradient: 2 * (predicted_team - true_team)
        spo_grad = - 0.0001 * (predicted_team.float() - true_team.float())

        # We don't want to punish the model for giving a player a high score, just missing ones who should've had a high
        # score
        spo_grad[true_team == 0] = 0

        return spo_grad * grad_output, None, None, None, None, None, None


@layers_registry.register
class OptimizeTeam(AbstractLayer):
    """Finds the optimal fantasy team, given the predicted/true stats, salaries and positions."""

    def __init__(self, params: OptimizeTeamParams, input_shape: Tuple[int, ...]):
        super().__init__(params=params, input_shape=input_shape)
        if params.site == Sites.YAHOO:
            linear_programming_class = FindBestSetOfTeamsLinearProgrammingYahooScipy
        else:
            raise ValueError(f"We haven't implemented FindBestSetOfTeamsLinearProgramming{params.site} yet!")
        self.optimizer = linear_programming_class(num_teams_in_set=params.num_teams_in_set,
                                                  max_shared_players_in_teams_pairs=params.max_shared_players_in_teams_pairs,
                                                  avoid_single_player_in_all_sets=params.avoid_single_player_in_all_sets,
                                                  verbose=params.verbose,
                                                  position_col=None,
                                                  salary_col=None,
                                                  points_col=None,
                                                  player_col=None,
                                                  gap_rel=0.01
                                                  )
        self._num_to_position_vectorized = np.vectorize(NUM_TO_POSITION.__getitem__)
        self._position_nums = list(NUM_TO_POSITION.keys())
        self._num_teams_in_set = params.num_teams_in_set

    def _num_to_position(self, arr: np.ndarray) -> np.ndarray:
        mask = np.isin(arr, self._position_nums)
        return_arr = np.full(arr.shape, fill_value='', dtype=object)
        if np.any(mask):
            return_arr[mask] = self._num_to_position_vectorized(arr[mask])
        return return_arr

    def forward(self, x: Dict[str, torch.tensor]) -> torch.Tensor:
        # x_np = {k: v.cpu().detach().numpy() for k, v in x.items()}

        points_pred = x['fantasy_points_pred_sample_per_day'].view(x['fantasy_points_pred_sample_per_day'].shape[0], -1)
        points_true = x['fantasy_points_true_sample_per_day'].view(x['fantasy_points_true_sample_per_day'].shape[0], -1)
        salaries = x['salary_sample_per_day'].view(x['salary_sample_per_day'].shape[0], -1)
        positions = x['position_sample_per_day'].view(x['position_sample_per_day'].shape[0], -1)
        date = x['date_sample_per_day'].view(x['date_sample_per_day'].shape[0], -1)

        # Apply the custom autograd function to handle the non-differentiable optimization
        optimal_team = OptimizeTeamFn.apply(points_pred, points_true, salaries, positions, self.optimizer,
                                            self._num_to_position, date)

        return optimal_team.view_as(x['fantasy_points_pred_sample_per_day'])

    def output_shape(self) -> LayersInputOutputShape:
        return self.input_shape[0]
