# https://github.com/mattbrondum/Fantasy-Football-Optimization/blob/master/NBA/Optimization/optimize.py
# https://zwlevonian.medium.com/integer-linear-programming-with-pulp-optimizing-a-draftkings-nfl-lineup-5e7524dd42d3
# https://stackoverflow.com/questions/74156804/python-pulp-optimization-limiting-total-number-of-one-variable-chosen/74157623#74157623
# https://stackoverflow.com/questions/66326293/python-pulp-positional-constraints-for-a-lineup-generator/66327972#66327972

import bisect
import copy
import datetime
import logging
import heapq
import pulp
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Union, Dict, List, Set, Tuple
from pulp.apis.core import LpSolver
from constants import DFS_POSITIONS, DFS_MAX_TEAM_SALARY, NUM_TO_POSITION
from enums import Sites

DEFAULT_LP_SOLVER = getattr(pulp, 'COIN_CMD', getattr(pulp, 'PULP_CBC_CMD', None))

NUM_PLAYERS_PER_POSITION = {Sites.YAHOO: {'PG': {'min_amount': 1, 'max_amount': 3},
                                          'SG': {'min_amount': 1, 'max_amount': 3},
                                          'SF': {'min_amount': 1, 'max_amount': 3},
                                          'PF': {'min_amount': 1, 'max_amount': 3},
                                          'C': {'min_amount': 1, 'max_amount': 2}
                                          }
                            }

ADDITIONAL_POSITIONS = {Sites.YAHOO: {'G': {'relevant_positions': ['SG', 'PG'],
                                            'min_amount': 3
                                            },
                                      'F': {'relevant_positions': ['SF', 'PF'],
                                            'min_amount': 3
                                            }
                                      }
                        }


class FindBestSetOfTeamsLinearProgramming(ABC):
    """
    Use linear programming to find the best set of teams. The idea is to define a binary variable for each
     player/team_num, and define constraints so that it finds the optimal variables given the constraints.
     We use the base positions here (PG/SG/SF/PF/C) and not G/F/UTIL, which are only filled afterward. Each position has
     constraints on the min/max amount allowed in each team. For example, for Yahoo we could define a constraint of at
     least 1 PG, and at most 3 PG's (in the case where the G and UTIl are filled by players whose base position is PG).
    """

    def __init__(self, position_col: str, salary_col: str, points_col: str, num_teams_in_set: int = 1,
                 max_shared_players_in_teams_pairs: int | None = None, avoid_single_player_in_all_sets: bool = True,
                 player_col: str = 'PLAYER_ID', verbose: bool = True, gap_rel: float | None = None):
        """
        :param position_col: The name of the column to use for positions data
        :param salary_col: The name of the column to use for the salaries data
        :param points_col: The name of the column to use for the points data
        :param num_teams_in_set: The number of teams in the set
        :param max_shared_players_in_teams_pairs: No more than this amount of players can be shared between each pair
                                                  of teams
        :param avoid_single_player_in_all_sets: If True, a single player can't appear in all teams
        :param player_col: The name of the column to use for the players data (usually either PLAYER_ID or NAME)
        :param verbose: Whether to log info to console or not
        :param gap_rel: Stop optimizing if within this percent of the optimum
        """
        self._num_teams_in_set = num_teams_in_set
        self._max_shared_players_in_teams_pairs = max_shared_players_in_teams_pairs if num_teams_in_set > 1 else None
        self._avoid_single_player_in_all_sets = avoid_single_player_in_all_sets if num_teams_in_set > 2 else False

        self._position_col = position_col
        self._salary_col = salary_col
        self._player_col = player_col
        self._points_col = points_col

        self._players: np.ndarray = np.array([])
        self._player_id_to_points: Dict[int, Union[int, float]] = dict()
        self._player_id_to_salary: Dict[int, Union[int, float]] = dict()
        self._player_id_to_position: Dict[int, str] = dict()
        self._player_teamnum_set: Set = set()

        self._verbose = verbose

        self._positions = DFS_POSITIONS[self._site]
        self._additional_positions = ADDITIONAL_POSITIONS[self._site]
        self._max_salary_per_team = DFS_MAX_TEAM_SALARY[self._site]
        self._num_players_per_position = NUM_PLAYERS_PER_POSITION[self._site]
        self._num_players_per_team = len(self._positions)

        self._player_id_to_index = None
        self._gap_rel = gap_rel

    def positions(self):
        return self._positions

    @staticmethod
    def create_player_team_sets(players: Union[np.ndarray, List[Union[str, int]]], num_teams_in_set: int = 1) -> Set:
        """
        Create a set of tuples with the relevant information for each player/team in each tuple
        :param players: An array with the names/ids of the players pool that we can choose from to fill the teams
        :param num_teams_in_set: The number of teams in the final set
        :return: The set of tuples with information for each binary variable
        """
        player_team_sets = set()
        for team_num in range(num_teams_in_set):
            for player in players:
                player_team_sets.add((player, team_num))
        return player_team_sets

    def _create_player_variables(self):
        """
        Create the pulp binary variables where each one represents a player/team.
        If self._max_shared_players_in_teams_pairs was given, we'll create binary variables for each
        player/pair-of-teams, so that we could add more constraints on them
        """
        self._player_variables = pulp.LpVariable.dicts("player_variables", self._player_teamnum_set, cat=pulp.LpBinary)

        if self._max_shared_players_in_teams_pairs is not None:
            self._shared_players = \
                pulp.LpVariable.dicts("shared_players", [(i, j, player) for i in range(self._num_teams_in_set)
                                                        for j in range(i + 1, self._num_teams_in_set)
                                                        for player in self._players],
                                      cat="Binary")

    def _define_problem(self):
        """ Define the pulp problem """
        self.prob = pulp.LpProblem('find_best_dfs_lineups_set', pulp.LpMaximize)

    def _define_objective_function(self):
        """ Define the function to maximize """
        self.prob += pulp.lpSum(self._player_id_to_points[player] * self._player_variables[player, team_num] for
                                player, team_num in self._player_teamnum_set)

    def _define_constraints(self, locked_dfs_list: List[pd.DataFrame]):
        """ Define all the constraints """
        raise NotImplementedError

    def _solve(self, solver: LpSolver | None = None):
        """
        Solve the linear programming problem. This should only be called after all the variables have been
        initialized and the constraints defined
        :param solver: the solver to use when solving the linear programming problem
        """
        if solver is None:
            solver = DEFAULT_LP_SOLVER(msg=False, timeLimit=60)
        if self._verbose:
            logging.info('solving optimal set of teams using linear programming...')
        self.prob.solve(solver=solver)
        if self._verbose:
            if self.prob.status == pulp.LpStatusOptimal:
                logging.info('found optimal linear programming solution!')
            else:
                logging.info("couldn't find the optimal linear programming solution!")

    def _filter(self, points: np.ndarray, salaries: np.ndarray, positions: np.ndarray, player_ids: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positive_salary = salaries > 0
        in_positions = np.isin(positions, self._positions)
        points_is_finite = np.isfinite(points)
        mask = positive_salary & in_positions & points_is_finite

        if not np.any(mask):
            return points[mask], salaries[mask], positions[mask], player_ids[mask], np.array([])

        # remove players who have multiple players with higher points, lower salary and same position, efficiently
        df = pd.DataFrame({
            'points': points[mask],
            'salary': salaries[mask],
            'position': positions[mask],
            'player_ids': player_ids[mask],
        })
        df.sort_values(by=['position', 'salary', 'points'], ascending=[True, True, False], inplace=True)
        keep_indices = []
        domination_counts = []
        for pos, group in df.groupby('position', sort=False):
            n = self._num_teams_in_set * NUM_PLAYERS_PER_POSITION[self._site][pos]['max_amount']

            group_indices = group.index.values
            group_points = group['points'].values
            top_scores = []
            seen_scores = []
            for idx, pts in zip(group_indices, group_points):
                # How many scores seen so far are strictly greater than current pts?
                # seen_scores is sorted ascending. bisect_right finds the insertion point after equal values.
                # All values to the right are strictly greater.
                dom_count = len(seen_scores) - bisect.bisect_right(seen_scores, pts)
                # Add current score to history (maintain sorted order)
                bisect.insort(seen_scores, pts)
                if len(top_scores) >= n and top_scores[0] > pts:
                    continue
                keep_indices.append(idx)
                domination_counts.append(dom_count)
                if len(top_scores) < n:
                    heapq.heappush(top_scores, pts)
                elif pts > top_scores[0]:
                    heapq.heapreplace(top_scores, pts)
        df_filtered = df.loc[keep_indices]

        return (df_filtered['points'].values,
                df_filtered['salary'].values,
                df_filtered['position'].values,
                df_filtered['player_ids'].values,
                np.array(domination_counts)
                )

    def run_inner(self, points: np.ndarray, salaries: np.ndarray, positions: np.ndarray,
                  player_ids: np.ndarray | None = None, locked_dfs_list: List[pd.DataFrame] | None = None):
        if player_ids is None:
            player_ids = np.arange(len(points))
        self._player_id_to_index = {pid: i for i, pid in enumerate(player_ids)}
        points, salaries, positions, self._players, dom_counts = self._filter(points, salaries, positions, player_ids)

        self._player_id_to_points = \
            {player_id: player_points for player_id, player_points in zip(self._players, points)}
        self._player_id_to_salary = \
            {player_id: salary for player_id, salary in zip(self._players, salaries)}
        self._player_id_to_position = \
            {player_id: position for player_id, position in zip(self._players, positions)}

        self._player_teamnum_set = \
            self.create_player_team_sets(players=self._players, num_teams_in_set=self._num_teams_in_set)

        self._create_player_variables()
        self._define_problem()
        self._define_objective_function()
        self._define_constraints(locked_dfs_list)

        solver = DEFAULT_LP_SOLVER(msg=False, timeLimit=60, gapRel=self._gap_rel)#, threads=1)
        self._solve(solver=solver)

    def run(self, df: pd.DataFrame, locked_dfs: Dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
        """
        This is the main method that runs the entire pipeline
        :param df: A df with all the information needed to solve the problem. Should have columns that correspond to
                   self._position_col/self._salary_col/self._player_col/self._points_col
        :param locked_dfs: A dict of competition_id: df with locked players. The length of the dict should be
                           self._num_teams_in_set. Each df should have a column with the name found in self._player_col,
                           self._salary_col, self._position_col, and "Pos" (the position in the actual team)
        :return: The best set of players given all the constraints
        """
        if locked_dfs is not None:
            if len(locked_dfs) != self._num_teams_in_set:
                raise ValueError('the length of locked_dfs has to equal self._num_teams_in_set!')

            # Setting the prediction value for locked dfs as a very small number, so they won't be chosen aside from
            # where they're forced to be
            assign_kwargs = {self._points_col: min(-1, df[self._points_col].min() - 1)}
            locked_dfs = {contest_id: curr_df.assign(**assign_kwargs) for contest_id, curr_df in locked_dfs.items()}

            contest_ids = list()
            locked_dfs_list = list()
            for contest_id, locked_df in locked_dfs.items():
                contest_ids.append(contest_id)
                locked_dfs_list.append(locked_df)

            if all([curr_df.empty for curr_df in locked_dfs_list]):
                for c in locked_dfs_list[0].columns:
                    if c not in df.columns:
                        df[c] = None
            else:
                df = pd.concat([df] + locked_dfs_list, axis=0).drop_duplicates(subset=['PLAYER_ID'], keep='last')
        else:
            contest_ids = range(self._num_teams_in_set)
            locked_dfs_list = [pd.DataFrame(columns=['Pos', self._player_col]) for _ in range(self._num_teams_in_set)]
            df = df.copy()

        df[self._position_col] = df[self._position_col].replace(NUM_TO_POSITION)

        player_ids = df[self._player_col].values
        positions = df[self._position_col].values
        salaries = df[self._salary_col].values
        points = df[self._points_col].values

        self.run_inner(points, salaries, positions, player_ids, locked_dfs_list)

        results_df = self.create_results_df(locked_dfs_list)
        results_df['contest_id'] = contest_ids
        return results_df

    @abstractmethod
    def create_results_df(self, locked_dfs_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Create a df with the results from the linear programming solution
        :param locked_dfs_list: A list of dfs, where each df has the players/positions that must be filled
        :return: The df with the results from the linear programming problem
        """
        raise NotImplementedError

    def get_results_boolean_array(self) -> np.ndarray:
        results = np.zeros((self._num_teams_in_set, len(self._player_id_to_index)), dtype=int)

        if len(self._players) > 0:
            player_variables_are_int = isinstance(list(self._player_variables.values())[0], np.integer)
            for team_num in range(self._num_teams_in_set):
                for player_num in self._players:
                    if player_variables_are_int:
                        if self._player_variables[player_num, team_num]:
                            results[team_num, self._player_id_to_index[player_num]] = 1
                    elif self._player_variables[player_num, team_num].varValue:
                        results[team_num, self._player_id_to_index[player_num]] = 1
        return results

class FindBestSetOfTeamsLinearProgrammingYahoo(FindBestSetOfTeamsLinearProgramming):
    """
    Use the pulp package to use linear programming to find the optimal set of DFS teams for Yahoo
    """
    def __init__(self, position_col: str | None = 'YAHOO_POSITION', salary_col: str | None = 'YAHOO_SALARY',
                 points_col: str | None = 'YAHOO_POINTS', num_teams_in_set: int = 1,
                 max_shared_players_in_teams_pairs: int | None = None, avoid_single_player_in_all_sets: bool = True,
                 player_col: str | None = 'PLAYER_ID', verbose: bool = True, gap_rel: float | None = None):
        self._site = Sites.YAHOO

        super().__init__(position_col=position_col, salary_col=salary_col, points_col=points_col,
                         num_teams_in_set=num_teams_in_set,
                         avoid_single_player_in_all_sets=avoid_single_player_in_all_sets,
                         max_shared_players_in_teams_pairs=max_shared_players_in_teams_pairs,
                         player_col=player_col, verbose=verbose, gap_rel=gap_rel)

    def _get_current_num_players_per_position_dict(self, locked_df: pd.DataFrame | None = None
                                                   ) -> Dict[str, Dict[str, int]]:
        """
        update the values from self._num_players_per_position with the locked players df.
         For example, if we have a player who was locked in the UTIl position, we'll reduce the max_amount of each of
         the positions by 1 (except for the locked players position) since none of the other positions could fill the
         UTIl spot anymore
        :param locked_df: A df with the locked players/positions. Should have a self._player_col column, and a "Pos"
                          column
        :return: The updated num_players_per_position dict
        """
        num_players_per_position = copy.deepcopy(self._num_players_per_position)
        if locked_df is None:
            return num_players_per_position
        if 'UTIL' in locked_df['Pos'].values:
            util_player_id = locked_df.loc[locked_df['Pos'] == 'UTIL', self._player_col].values[0]
            util_player_pos = self._player_id_to_position[util_player_id]
            for k in num_players_per_position.keys():
                if k != util_player_pos:
                    num_players_per_position[k]['max_amount'] -= 1
        for pos in ['G', 'F']:
            if pos in locked_df['Pos'].values:
                util_player_id = locked_df.loc[locked_df['Pos'] == pos, self._player_col].values[0]
                util_player_pos = self._player_id_to_position[util_player_id]
                change_pos = f'P{pos}' if util_player_pos == f'S{pos}' else f'S{pos}'
                num_players_per_position[change_pos]['max_amount'] -= 1
        return num_players_per_position

    def _define_constraints(self, locked_dfs_list: List[pd.DataFrame]):
        for teamnum in range(self._num_teams_in_set):
            curr_player_sets = \
                {(player, teamnum) for player, teamnum2 in self._player_teamnum_set if teamnum2 == teamnum}
            locked_df = locked_dfs_list[teamnum] if locked_dfs_list else None

            # Number of players constraint
            self.prob += pulp.lpSum(self._player_variables[player, teamnum] for player, _ in curr_player_sets) == \
                         self._num_players_per_team

            # Total salary constraint
            self.prob += pulp.lpSum(self._player_id_to_salary[player] * self._player_variables[player, teamnum] for
                                    player, _ in curr_player_sets) <= self._max_salary_per_team

            # # Prevent a single player from playing multiple positions in each game...
            # for player in self._players:
            #     self.prob += pulp.lpSum(self._player_variables[player, teamnum] for player2, _ in curr_player_sets if
            #                             player2 == player) <= 1

            # Constraining number of players for each position
            num_players_per_position = self._get_current_num_players_per_position_dict(locked_df)
            for pos, min_max_num in num_players_per_position.items():
                curr_sum = pulp.lpSum(self._player_variables[player, teamnum] for player, _ in curr_player_sets if
                                      self._player_id_to_position[player] == pos)
                if min_max_num.get('min_amount') is not None:
                    self.prob += curr_sum >= min_max_num['min_amount']
                if min_max_num.get('max_amount') is not None:
                    self.prob += curr_sum <= min_max_num['max_amount']
            if self._additional_positions is not None:
                for p, vals in self._additional_positions.items():
                    self.prob += \
                        pulp.lpSum(self._player_variables[player, teamnum] for player, _ in curr_player_sets if
                                   self._player_id_to_position[player] in vals['relevant_positions']) >= \
                        vals['min_amount']

            # Forcing locked players to be chosen
            if locked_df is not None:
                for player_id in locked_df[self._player_col].values:
                    self.prob += self._player_variables[player_id, teamnum] >= 1

        if self._avoid_single_player_in_all_sets and self._num_teams_in_set > 1:
            # Prevent a single player from appearing in all sets
            for player in self._players:
                self.prob += pulp.lpSum(self._player_variables[player, teamnum] for player2, teamnum in
                                        self._player_teamnum_set if player2 == player) <= (self._num_teams_in_set - 1)

        # limiting the max amount of shared players between pairs of teams.
        # We'll do it by constraining the sum of the _max_shared_players_in_teams_pairs variables for each pair of
        # teams, and making sure each of the variables is larger than the sum (minus 1) of player_variables for a given
        # player/pair-of-teams
        if self._max_shared_players_in_teams_pairs is not None:
            for i in range(self._num_teams_in_set):
                for j in range(i + 1, self._num_teams_in_set):
                    self.prob += \
                        pulp.lpSum(self._shared_players[(i, j, player)] for player in self._players) <= \
                        self._max_shared_players_in_teams_pairs
                    for player in self._players:
                        self.prob += \
                            self._shared_players[(i, j, player)] >= \
                            self._player_variables[player, i] + self._player_variables[player, j] - 1

    def create_results_df(self, locked_dfs_list: List[pd.DataFrame] | None = None) -> pd.DataFrame:
        selected_df = pd.DataFrame(index=range(self._num_teams_in_set), columns=self._positions)

        # filling locked players
        if locked_dfs_list is not None:
            for teamnum, locked_df in enumerate(locked_dfs_list):
                for i, locked_player in locked_df.iterrows():
                    selected_df.loc[teamnum, locked_player['Pos']] = locked_player[self._player_col]

        # filling unlocked players
        for player, teamnum in self._player_teamnum_set:
            if self._player_variables[player, teamnum].varValue == 0:
                continue
            if (locked_dfs_list is not None) and (player in locked_dfs_list[teamnum][self._player_col].values):
                continue
            pos = self._player_id_to_position[player]
            if np.isnan(selected_df.loc[teamnum, pos]):
                selected_df.loc[teamnum, pos] = player
            else:
                new_pos = pos[1] if pos != 'C' and np.isnan(selected_df.loc[teamnum, pos[1]]) else 'UTIL'
                selected_df.loc[teamnum, new_pos] = player

        if self._num_teams_in_set > 1:
            num_shared_per_teams_pair = dict()
            for team1 in range(self._num_teams_in_set):
                for team2 in range(team1 + 1, self._num_teams_in_set):
                    num_shared_per_teams_pair[(team1, team2)] = \
                        len(set(selected_df.loc[team1].values) & set(selected_df.loc[team2].values))
            if self._verbose:
                logging.info(f'num_shared_per_teams_pair: {num_shared_per_teams_pair}')

        if self._verbose:
            logging.info(f'\nnum times each player is used:\n{pd.Series(selected_df.values.flatten()).value_counts()}')
        selected_df['salary'] = selected_df[self._positions].map(lambda x: self._player_id_to_salary[x]).sum(axis=1)
        selected_df['points'] = selected_df[self._positions].map(lambda x: self._player_id_to_points[x]).sum(axis=1)
        if self._verbose:
            logging.info(f'\n{selected_df.to_string()}')

        return selected_df


class FindBestSetOfTeamsLinearProgrammingYahooScipy(FindBestSetOfTeamsLinearProgrammingYahoo):

    def run_inner(self, points: np.ndarray, salaries: np.ndarray, positions: np.ndarray,
                  player_ids: np.ndarray | None = None, locked_dfs_list: List[pd.DataFrame] | None = None):
        """
        Highly optimized implementation using Scipy HiGHS solver with Dense Matrices.
        """
        self._player_variables = {}

        # Fallback to PuLP for complex pairwise overlap constraints
        if self._max_shared_players_in_teams_pairs is not None:
            return super().run_inner(points, salaries, positions, player_ids, locked_dfs_list)

        if player_ids is None:
            player_ids = np.arange(len(points))

        self._player_id_to_index = {pid: i for i, pid in enumerate(player_ids)}

        # 1. Filter out dominated or irrelevant players
        points, salaries, positions, self._players, dom_counts = self._filter(points, salaries, positions, player_ids)

        # --- STATE INITIALIZATION ---
        self._player_id_to_points = {pid: pts for pid, pts in zip(self._players, points)}
        self._player_id_to_salary = {pid: sal for pid, sal in zip(self._players, salaries)}
        self._player_id_to_position = {pid: pos for pid, pos in zip(self._players, positions)}

        n_players = len(self._players)
        num_teams = self._num_teams_in_set
        n_vars = n_players * num_teams

        if n_players < self._num_players_per_team:
            return

        # --- 2. Build Constraint Matrix for a SINGLE team (A_single) ---
        # We use dense lists/arrays which are faster for small N (< 1000)
        constraints_list = []
        lb_list = []
        ub_list = []

        # Size
        constraints_list.append(np.ones(n_players))
        lb_list.append(self._num_players_per_team)
        ub_list.append(self._num_players_per_team)

        # Salary
        constraints_list.append(salaries)
        lb_list.append(0)
        ub_list.append(self._max_salary_per_team)

        # Positions
        for pos, limits in self._num_players_per_position.items():
            row = (positions == pos).astype(float)
            constraints_list.append(row)
            lb_list.append(limits.get('min_amount', 0))
            ub_list.append(limits.get('max_amount', n_players))

        if self._additional_positions:
            for pos, vals in self._additional_positions.items():
                row = np.isin(positions, vals['relevant_positions']).astype(float)
                constraints_list.append(row)
                lb_list.append(vals.get('min_amount', 0))
                ub_list.append(n_players)

        A_single = np.vstack(constraints_list)  # Shape: (n_constraints, n_players)
        b_l_single = np.array(lb_list)
        b_u_single = np.array(ub_list)

        # --- 3. Vectorized Construction of Multi-Team Matrices ---

        # A_teams: Block diagonal repetition of A_single
        # Use Kronecker product: I_T (kron) A_single
        # [ A 0 0 ]
        # [ 0 A 0 ]
        # [ 0 0 A ]
        A_teams = np.kron(np.eye(num_teams), A_single)
        b_l_teams = np.tile(b_l_single, num_teams)
        b_u_teams = np.tile(b_u_single, num_teams)

        A_final = A_teams
        b_l_final = b_l_teams
        b_u_final = b_u_teams

        # --- 4. Diversity Constraints ---
        # "Avoid single player in all sets": Sum(x_p) <= T-1
        # Matrix: [ I_N | I_N | ... | I_N ] (Horizontal tile of identities)
        if self._avoid_single_player_in_all_sets and num_teams > 1:
            A_diversity = np.tile(np.eye(n_players), (1, num_teams))
            b_l_diversity = np.zeros(n_players)
            b_u_diversity = np.full(n_players, num_teams - 1)

            # # The only players who could possibly be chosen for all teams are the ones who aren't "dominated"
            # # (lower/equal salary and higher points) by 3 or more players from their position
            # not_dominated = dom_counts < 3
            # A_diversity = A_diversity[not_dominated]
            # b_l_diversity = b_l_diversity[not_dominated]
            # b_u_diversity = b_u_diversity[not_dominated]

            A_final = np.vstack([A_final, A_diversity])
            b_l_final = np.concatenate([b_l_final, b_l_diversity])
            b_u_final = np.concatenate([b_u_final, b_u_diversity])

        # --- 5. Symmetry Breaking Constraints ---
        # Score(Team t) >= Score(Team t+1)
        if num_teams > 1:
            # Construct comparison matrix M for scores:
            # [ 1 -1  0 ]
            # [ 0  1 -1 ]
            M = np.eye(num_teams - 1, num_teams, k=0) - np.eye(num_teams - 1, num_teams, k=1)
            # We want M @ (Points_per_team) >= 0
            # Points_per_team = [P @ x_0, P @ x_1, ...]
            # Constraint row: P (kron) row_of_M

            # Efficient construction:
            # A_sym = M (kron) P
            # Shape: (num_teams-1, n_vars)
            A_sym = np.kron(M, points)
            b_l_sym = np.zeros(num_teams - 1)
            b_u_sym = np.full(num_teams - 1, np.inf)

            A_final = np.vstack([A_final, A_sym])
            b_l_final = np.concatenate([b_l_final, b_l_sym])
            b_u_final = np.concatenate([b_u_final, b_u_sym])

        # --- 6. Bounds and Locked Players ---
        c = np.tile(-points, num_teams)
        lb_vars = np.zeros(n_vars)
        ub_vars = np.ones(n_vars)

        if locked_dfs_list:
            for t_idx, locked_df in enumerate(locked_dfs_list):
                if not locked_df.empty:
                    locked_ids = set(locked_df[self._player_col].values)
                    # Vectorized search for indices might be faster for large lists,
                    # but simple loop is fine for N=50
                    for i, pid in enumerate(self._players):
                        if pid in locked_ids:
                            lb_vars[i + (t_idx * n_players)] = 1

        # --- 7. Solve ---
        solver_options = {'presolve': True}
        if self._gap_rel is not None:
            solver_options['mip_rel_gap'] = self._gap_rel
        else:
            solver_options['mip_rel_gap'] = 0.01

        res = milp(c=c,
                   constraints=LinearConstraint(A_final, b_l_final, b_u_final),
                   integrality=1,
                   bounds=Bounds(lb_vars, ub_vars),
                   options=solver_options)

        # --- 8. Results ---
        if res.success:
            selected = np.round(res.x).astype(int).reshape((num_teams, n_players))
            for t_idx in range(num_teams):
                for i, pid in enumerate(self._players):
                    self._player_variables[pid, t_idx] = selected[t_idx, i]
        else:
            for t_idx in range(num_teams):
                for pid in self._players:
                    self._player_variables[pid, t_idx] = 0


if __name__ == '__main__':
    try:
        df = pd.read_feather('C:/Users/schwa/PycharmProjects/Fantasy-Basketball-Predictions/features/features.feather')
    except:
        df = pd.read_feather('C:/Users/schwa/Documents/python_projects/Fantasy-Basketball-Predictions/features/features.feather')
    df = df.loc[df['DATE'] == datetime.date(2023, 12, 11)]
    df = df.loc[df['YAHOO_SALARY'].notnull()]

    num_teams_in_set = 1
    avoid_single_player_in_all_sets = True
    max_shared_players_in_teams_pairs = None
    find_best_set = FindBestSetOfTeamsLinearProgrammingYahoo(num_teams_in_set=num_teams_in_set,
                                                             avoid_single_player_in_all_sets=avoid_single_player_in_all_sets,
                                                             max_shared_players_in_teams_pairs=max_shared_players_in_teams_pairs,
                                                             gap_rel=0.01)

    find_best_set_scipy = FindBestSetOfTeamsLinearProgrammingYahooScipy(num_teams_in_set=num_teams_in_set,
                                                                        avoid_single_player_in_all_sets=avoid_single_player_in_all_sets,
                                                                        max_shared_players_in_teams_pairs=max_shared_players_in_teams_pairs,
                                                                        gap_rel=0.01)
    import time
    t = [0, 0]
    for _ in range(10):
        start = time.perf_counter()
        results_df = find_best_set.run(df=df)
        t[0] += time.perf_counter() - start
        start = time.perf_counter()
        results_df_scipy = find_best_set_scipy.run(df=df)
        t[1] += time.perf_counter() - start
    print(t)
    # results_df = find_best_set.create_results_df(locked_dfs_list=None)
    # results_df_scipy = find_best_set_scipy.create_results_df(locked_dfs_list=None)
    print(results_df.to_string())
    print(results_df_scipy.to_string())
    # print(results_df.nunique(axis=1))

