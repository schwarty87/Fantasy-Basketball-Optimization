import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from constants import DFS_POSITIONS_YAHOO, MAX_SALARY_YAHOO
from utils import load_and_cache_players_list
from typing import List, Set, Optional, Tuple

RANDOM_SEED = 42
ELIGIBILITY = {'PG': [1], 'SG': [2], 'SF': [3], 'PF': [4], 'C': [5], 'G': [1, 2], 'F': [3, 4], 'UTIL': [1, 2, 3, 4, 5]}


class CHANNELS:
    ids = 0
    salary = 1
    points = 2


class FindBestTeamGA:
    def __init__(self, n_iterations=1000, n_teams_in_df=1000, keep_top_n=250, mutate_pct=0.2,
                 points_col='YAHOO_POINTS', definitely_in: Optional[pd.DataFrame] = None,
                 at_least_one_min_player: bool = False):
        self.n_iterations = n_iterations
        self.n_teams_in_df = n_teams_in_df
        self.keep_top_n = keep_top_n
        self.mutate_pct = mutate_pct
        self.points_col = points_col
        self.at_least_one_min_player = at_least_one_min_player

        self.use_cols = ['PLAYER_ID', 'YAHOO_SALARY', points_col]

        self.random_generator = np.random.default_rng(RANDOM_SEED)

        self.max_scores = list()

        self.positions = DFS_POSITIONS_YAHOO
        self.n_positions = len(self.positions)
        self.max_salary = MAX_SALARY_YAHOO

        self.df: pd.DataFrame = pd.DataFrame()
        self.eligibility = {}
        self.id_to_points = np.vectorize({}.__getitem__)
        self.id_to_salary = np.vectorize({}.__getitem__)
        self.res = pd.DataFrame()
        self.best_set = pd.DataFrame()

        self.player_id_to_name = dict()

        self.definitely_in = definitely_in

    def _single_iteration(self):
        self._selection()
        self._crossover()
        self._mutation()
        self._add_new_teams_to_old()
        self._fill_definitely_in()
        self._drop_illegal_rows_from_players()
        self.max_scores.append(np.max(self.players[:, :, CHANNELS.points].sum(axis=1)))

    def _add_new_teams_to_old(self):
        self.players = np.vstack([self.players, self.add_players])

    def _fill_definitely_in(self):
        if self.definitely_in is None or self.definitely_in.empty:
            return
        for i, curr_definitely_in in self.definitely_in.iterrows():
            fill_ind = self.positions.index(curr_definitely_in['Pos'])
            self.players[:, fill_ind, CHANNELS.ids] = curr_definitely_in['PLAYER_ID']
            self.players[:, fill_ind, CHANNELS.salary] = curr_definitely_in['Salary']
            self.players[:, fill_ind, CHANNELS.points] = 0

    def _mutation(self):
        mask = self.random_generator.uniform(size=self.add_players.shape[:-1]) < self.mutate_pct
        mask = np.dstack([mask] * 3)

        players_for_mutate = np.empty(shape=(self.n_teams_in_df, self.n_positions))
        for i, pos in enumerate(self.positions):
            players_for_mutate[:, i] = self.random_generator.choice(self.eligibility[pos][:, CHANNELS.ids],
                                                                    size=(self.n_teams_in_df,),
                                                                    replace=True)
        players_for_mutate = np.stack([players_for_mutate, self.id_to_salary(players_for_mutate),
                                       self.id_to_points(players_for_mutate)],
                                      axis=-1)

        self.add_players = np.where(mask, players_for_mutate, self.add_players)

    def _crossover(self):
        half_len_positions = int(self.n_positions / 2)
        rand_inds = self.random_generator.choice(range(self.players.shape[0]), self.n_teams_in_df)
        self.add_players = self.players[rand_inds]
        n_add_players = self.add_players.shape[0]
        rand_cols = self.random_generator.choice(np.arange(self.n_positions), half_len_positions, replace=False)
        n_add_players_half = int(n_add_players / 2)
        self.add_players[:n_add_players_half, rand_cols, :], self.add_players[-n_add_players_half:, rand_cols, :] = \
            self.add_players[-n_add_players_half:, rand_cols, :], self.add_players[:n_add_players_half, rand_cols, :]

    def _selection(self):
        points_sum = self.players[:, :, CHANNELS.points].sum(axis=1)
        sort_inds = np.argsort(points_sum)[::-1]
        keep_inds = sort_inds[:self.keep_top_n]
        self.players = self.players[keep_inds]

    def _drop_illegal_rows_from_players(self):
        _, u_inds = np.unique(self.players[:, :, CHANNELS.ids], return_index=True, axis=0)
        self.players = self.players[u_inds]
        n_unique_per_row = np.count_nonzero(np.diff(np.sort(self.players[:, :, CHANNELS.ids])), axis=1) + 1
        total_salary_per_team = np.sum(self.players[:, :, CHANNELS.salary], axis=1)
        keep_inds = (total_salary_per_team <= self.max_salary) & (n_unique_per_row == self.n_positions)
        if self.at_least_one_min_player:
            keep_inds &= np.any(self.players[:, :, CHANNELS.salary] == 10, axis=1)
        self.players = self.players[keep_inds]

    def _init_teams(self):
        self.players = np.empty(shape=(self.n_teams_in_df * 2, self.n_positions))
        for i, pos in enumerate(self.positions):
            try:
                self.players[:, i] = self.random_generator.choice(self.eligibility[pos][:, CHANNELS.ids],
                                                                  size=(self.n_teams_in_df * 2,), replace=True)
            except ValueError as er:
                print('$$$$$$$ FAILED WHEN GETTING ELIGIBLE PLAYERS!! $$$$$$$')
                print(f'position: {pos}')
                players_per_position = {p: el.shape[0] for p, el in self.eligibility.items()}
                print(f'eligible players per position:\n{players_per_position}')
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                print(er)
                raise ValueError(er)
        self.players = np.dstack([self.players, self.id_to_salary(self.players), self.id_to_points(self.players)])
        self._drop_illegal_rows_from_players()
        self.players = self.players[:self.n_teams_in_df]

    def summarize(self, convert_ids_to_names: bool = True, player_id_to_name=None, print_result: bool = False,
                  true_points_col: Optional[str] = None):
        players = self.players[:, :, CHANNELS.ids].astype(int)
        if convert_ids_to_names:
            if player_id_to_name is None:
                players_list = load_and_cache_players_list(print_usage=False)
                self.player_id_to_name = players_list.set_index('PLAYER_ID')['NAME'].to_dict()

        self.res = pd.DataFrame(players, columns=self.positions)
        self.res['points'] = np.sum(self.players[:, :, CHANNELS.points], axis=1)
        self.res['salary'] = np.sum(self.players[:, :, CHANNELS.salary], axis=1)
        if true_points_col is not None:
            id_to_true_points = self.df[['PLAYER_ID', true_points_col]].set_index('PLAYER_ID').iloc[:, 0].to_dict()
            id_to_true_points = np.vectorize(id_to_true_points.__getitem__)
            true_points = id_to_true_points(players)
            self.res['true_points'] = true_points.sum(axis=1)
        self.res.sort_values('points', inplace=True, ascending=False)
        self.res.reset_index(drop=True, inplace=True)

        # dropping duplicate rows
        keep_inds = pd.DataFrame(np.sort(self.res[self.positions].values, axis=1),
                                 index=self.res.index).drop_duplicates().index
        self.res = self.res.loc[keep_inds].reset_index(drop=True)

        if print_result:
            print_res = self.res.copy()
            print_res[DFS_POSITIONS_YAHOO] = print_res[DFS_POSITIONS_YAHOO].replace(self.player_id_to_name)
            print(print_res.iloc[:100].to_string(index=False))

        return self.res

    def call(self, df: pd.DataFrame, progress_bar: bool = True, positions: Optional[List[str]] = None,
             max_salary: Optional[int] = None, definitely_in: Optional[pd.DataFrame] = None):
        """
        run the genetic algorithm to find optimal teams given the constraints
        :param df: df with the players' data, including IDs, salaries, positions
        :param progress_bar: whether to use a progress bar when running
        :param positions: which positions to use. This is useful if we want to update a team that was submitted already
                          and some of its players were locked already.
                          If None, use all positions
        :param max_salary: the max allowed salary. This is useful if we want to update a team that was submitted already
                           and some of its players were locked already.
                           If None, use the default MAX_SALARY
        :param definitely_in: pd.DataFrame with ['PLAYER_ID', 'Pos', 'Salary'] columns. These players have to be
               in the lineup in the given positions. This is useful when we want to update a team in the middle of the
               night after some players have been locked into the lineup already
        """
        if positions is not None:
            self.positions = positions
            self.n_positions = len(self.positions)
        if max_salary is not None:
            self.max_salary = max_salary
        if definitely_in is not None:
            self.definitely_in = definitely_in
        if self.at_least_one_min_player:
            self.min_salary = df['YAHOO_SALARY'].min()

        self.df = df
        self.eligibility = {p: self.df.query(f'YAHOO_POSITION.isin({ELIGIBILITY[p]})')[self.use_cols].values for p in
                            self.positions}
        id_to_salary = self.df[['PLAYER_ID', 'YAHOO_SALARY']].set_index('PLAYER_ID').iloc[:, 0].to_dict()
        self.id_to_salary = np.vectorize(id_to_salary.__getitem__)
        id_to_points = self.df[['PLAYER_ID', self.points_col]].set_index('PLAYER_ID').iloc[:, 0].to_dict()
        self.id_to_points = np.vectorize(id_to_points.__getitem__)

        self._init_teams()
        iter_vals = range(self.n_iterations)
        if progress_bar:
            iter_vals = tqdm(iter_vals, desc='running GA to find the optimal team')
        for iteration in iter_vals:
            self._single_iteration()

            # the values should be monotonically increasing since we always keep the best from the previous iteration
            if iteration > 10 and self.max_scores[-1] == self.max_scores[-10]:
                break

    @staticmethod
    def _find_best_set_of_teams(team_sets: List[List[Set]], team_points: List[np.ndarray],
                                max_allowed_joint_players: int,
                                curr_group: Optional[Tuple[List[Set[int]], float, List[int]]] = None,
                                best_group: Optional[Tuple[List[int], float, List[int]]] = None,
                                avoid_single_player_in_all_sets: bool = True) -> Tuple[List[Set[int]], float, List[int]]:
        """
        Given a list of lists of sets, and a value that corresponds to each set, find the optimal group that has the
        highest combined value, while staying within the constraints
        :param team_sets: A list of list of sets, where each set has player ids for a team. This is the result of
                          multiple runs of FindBestTeamGA -> summarize
        :param team_points: a list of points, one for each set in team_sets
        :param max_allowed_joint_players: Max size of the intersection between two teams
        :param curr_group: the current group that we're evaluating
        :param best_group: The best group so far
        :param avoid_single_player_in_all_sets: If True, don't allow a single player to appear in all teams
        :return: The optimal group. Index  0 has the sets, index 1 has the total points, and index 2 has the indexes of
                 the sets
        """
        if curr_group is None:
            curr_group = ([], 0., [])
        curr_group_len = len(curr_group[0])
        if curr_group_len == len(team_sets):
            is_best = best_group is None or curr_group[1] > best_group[1]
            if avoid_single_player_in_all_sets:
                is_best = len(set.intersection(*curr_group[0])) == 0
            if is_best:
                best_group = (curr_group[0], curr_group[1], curr_group[2])
            return best_group
        for i, (curr_team_set, curr_team_points) in enumerate(zip(team_sets[curr_group_len], team_points[curr_group_len])):
            next_group = (curr_group[0] + [curr_team_set], curr_group[1] + curr_team_points, curr_group[2] + [i])
            if best_group is not None:
                max_possible_points = next_group[1] + sum([tp[0] for tp in team_points[curr_group_len + 1:]])
                if max_possible_points <= best_group[1]:
                    break
            if curr_group_len > 0:
                if any([len(curr_team_set.intersection(in_team)) > max_allowed_joint_players for in_team in curr_group[0]]):
                    continue
            best_group = FindBestTeamGA._find_best_set_of_teams(team_sets=team_sets,
                                                                team_points=team_points,
                                                                curr_group=next_group,
                                                                best_group=best_group,
                                                                avoid_single_player_in_all_sets=avoid_single_player_in_all_sets,
                                                                max_allowed_joint_players=max_allowed_joint_players)
        return best_group

    @staticmethod
    def find_best_set(ga_results: List[pd.DataFrame], max_allowed_joint_players: int = 3,
                      avoid_single_player_in_all_sets: bool = True) -> pd.DataFrame:
        """

        :param ga_results: A list of results from running the genetic algorithm (self.res, after running self.call()
                          then self.summarize()). The length of the output wil be the same as the length of this list
        :param max_allowed_joint_players: Max size of the intersection between two teams
        :param avoid_single_player_in_all_sets: If True, don't allow a single player to appear in all teams
        :return: The teams to use, with player ID's (not names)
        """
        team_sets = [[set(curr_row) for _, curr_row in
                      curr_ga_results.sort_values('points', ascending=False)[DFS_POSITIONS_YAHOO].iterrows()] for
                     curr_ga_results in ga_results]

        best_set_of_teams_ids, best_set_of_teams_pts, best_set_of_teams_inds = \
            FindBestTeamGA._find_best_set_of_teams(team_sets=team_sets,
                                                   team_points=[curr_ga_results['points'].values for
                                                                 curr_ga_results in ga_results],
                                                   avoid_single_player_in_all_sets=avoid_single_player_in_all_sets,
                                                   max_allowed_joint_players=max_allowed_joint_players)

        best_set_of_teams = [curr_ga_results.iloc[i] for curr_ga_results, i in zip(ga_results, best_set_of_teams_inds)]
        best_set_of_teams = pd.concat(best_set_of_teams, axis=1).T.reset_index(drop=True)

        return best_set_of_teams


if __name__ == '__main__':
    import time
    import sys
    t = [0, 0, 0]
    df = pd.read_csv('C:/Users/schwa/Documents/python_projects/Fantasy-Basketball-Predictions/tmp/df_tmp.csv')
    fbtga = FindBestTeamGA(points_col='YAHOO_POINTS_pred', n_iterations=10000)
    start = time.perf_counter()
    fbtga.call(df=df)
    t[0] = time.perf_counter() - start
    start = time.perf_counter()
    res = fbtga.summarize(true_points_col='YAHOO_POINTS_true', convert_ids_to_names=False)
    t[1] = time.perf_counter() - start
    start = time.perf_counter()
    fbtga.find_best_set([res, res, res])
    t[2] = time.perf_counter() - start
    print(t)
    sys.exit(0)


    # import datetime
    # from train_nn import TrainNeuralNetwork
    # from utils import add_missing_points_columns_to_predictions
    params_update_dict = {'continue_model_name': 'extra_weight_on_ast', 'training': {'run_extra_callbacks': False}}
    # tnn = TrainNeuralNetwork(name='extra_weight_on_ast', continue_model_name='extra_weight_on_ast', update_params=params_update_dict)
    # predictions = tnn.predict(save_predictions=False)
    # predictions = predictions[0]
    # predictions = predictions.loc[predictions['DATE'] == datetime.date(2023, 3, 1)]
    # add_missing_points_columns_to_predictions(predictions_df=predictions)
    #
    # fbtga = FindBestTeamGA(points_col='YAHOO_POINTS_pred')
    # fbtga.call(df=predictions)
    # fbtga.summarize(convert_ids_to_names=False, print_result=False)
    # ga_teams = fbtga.res  # the results from the genetic algorithm (the best teams, sorted by predicted points)

    # fbtga.find_best_set()

    FindBestTeamGA.find_best_set(ga_results=[ga_teams, ga_teams, ga_teams])



    aaa=1
