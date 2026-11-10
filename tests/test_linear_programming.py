import unittest
import numpy as np
import pandas as pd
import pulp

from constants import Sites
from find_best_team_linear_programming import (
    DEFAULT_LP_SOLVER,
    FindBestSetOfTeamsLinearProgrammingYahoo,
    FindBestSetOfTeamsLinearProgrammingYahooScipy,
    _get_default_lp_solver,
)


def _generate_synthetic_dfs_pool(n_players: int = 60, seed: int = 42) -> pd.DataFrame:
    """Helper to generate a realistic DFS pool with balanced positions."""
    rng = np.random.default_rng(seed)
    base_positions = ['PG', 'SG', 'SF', 'PF', 'C']
    positions = rng.choice(base_positions, size=n_players)
    salaries = rng.integers(10, 50, size=n_players)
    # Projected points roughly correlated with salary + uniform noise
    points = salaries * 0.85 + rng.uniform(2.0, 15.0, size=n_players)

    return pd.DataFrame({
        'PLAYER_ID': np.arange(1001, 1001 + n_players),
        'YAHOO_POSITION': positions,
        'YAHOO_SALARY': salaries,
        'YAHOO_POINTS': points,
    })


class TestLpSolverResolution(unittest.TestCase):
    """Tests that default LP solver resolves to an executable solver on the system."""

    def test_solver_resolution_not_none(self):
        solver_cls = _get_default_lp_solver()
        self.assertIsNotNone(solver_cls, "Expected an available solver class to be found.")

    def test_default_solver_is_available(self):
        solver = DEFAULT_LP_SOLVER(msg=False)
        self.assertTrue(bool(solver.available()), f"Solver {solver} reported unavailable on this system.")

    def test_solver_executes_simple_milp(self):
        """Verify solver can solve a simple problem without raising PulpSolverError."""
        prob = pulp.LpProblem("test_problem", pulp.LpMaximize)
        x = pulp.LpVariable("x", lowBound=0, upBound=10, cat=pulp.LpInteger)
        y = pulp.LpVariable("y", lowBound=0, upBound=10, cat=pulp.LpInteger)

        prob += x + 2 * y
        prob += x + y <= 8

        solver = DEFAULT_LP_SOLVER(msg=False)
        prob.solve(solver=solver)

        self.assertEqual(prob.status, pulp.LpStatusOptimal)
        self.assertEqual(x.varValue, 0)
        self.assertEqual(y.varValue, 8)
        self.assertEqual(pulp.value(prob.objective), 16)


class TestYahooLinearProgrammingSingleTeam(unittest.TestCase):
    """Tests for Yahoo single-team optimization."""

    def setUp(self):
        self.df = _generate_synthetic_dfs_pool(n_players=70, seed=123)
        self.solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=1,
            verbose=False,
        )

    def test_optimal_lineup_meets_all_constraints(self):
        team = self.solver.run(df=self.df)

        # 1. Output shape & status
        self.assertEqual(len(team), 1)
        self.assertTrue(self.solver.last_result.success)
        self.assertEqual(self.solver.prob.status, pulp.LpStatusOptimal)

        # 2. Salary Cap ($200)
        total_salary = team['salary'].iloc[0]
        self.assertLessEqual(total_salary, 200)
        self.assertGreater(total_salary, 0)

        # 3. 8 distinct players across roster slots
        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']
        chosen_ids = team[slots].iloc[0].values
        self.assertEqual(len(set(chosen_ids)), 8, "Expected 8 unique players across slots.")

        # 4. Check positional slot eligibility
        id_to_pos = self.df.set_index('PLAYER_ID')['YAHOO_POSITION'].to_dict()
        self.assertEqual(id_to_pos[team['PG'].iloc[0]], 'PG')
        self.assertEqual(id_to_pos[team['SG'].iloc[0]], 'SG')
        self.assertIn(id_to_pos[team['G'].iloc[0]], ['PG', 'SG'])
        self.assertEqual(id_to_pos[team['SF'].iloc[0]], 'SF')
        self.assertEqual(id_to_pos[team['PF'].iloc[0]], 'PF')
        self.assertIn(id_to_pos[team['F'].iloc[0]], ['SF', 'PF'])
        self.assertEqual(id_to_pos[team['C'].iloc[0]], 'C')
        self.assertIn(id_to_pos[team['UTIL'].iloc[0]], ['PG', 'SG', 'SF', 'PF', 'C'])

        # 5. Total points positive and maximized
        self.assertGreater(team['points'].iloc[0], 0)


class TestYahooLinearProgrammingLockedPlayers(unittest.TestCase):
    """Tests for Yahoo lineup optimization with locked players."""

    def test_locked_player_forced_into_lineup(self):
        df = _generate_synthetic_dfs_pool(n_players=60, seed=99)

        # Pick an intentionally bad player (low points, high salary)
        locked_player_id = 9999
        bad_player = pd.DataFrame([{
            'PLAYER_ID': locked_player_id,
            'YAHOO_POSITION': 'C',
            'YAHOO_SALARY': 35,
            'YAHOO_POINTS': 1.0,  # Terribly suboptimal
        }])
        df = pd.concat([df, bad_player], ignore_index=True)

        locked_df = pd.DataFrame([{
            'PLAYER_ID': locked_player_id,
            'Pos': 'C',
            'YAHOO_POSITION': 'C',
            'YAHOO_SALARY': 35,
        }])

        solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=1,
            verbose=False,
        )

        team = solver.run(df=df, locked_dfs={0: locked_df})

        # The locked player must appear in the team at the locked slot 'C'
        self.assertEqual(team['C'].iloc[0], locked_player_id)
        # Roster must still be valid and within salary cap
        self.assertLessEqual(team['salary'].iloc[0], 200)
        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']
        self.assertEqual(len(set(team[slots].iloc[0].values)), 8)


class TestYahooLinearProgrammingMultiTeam(unittest.TestCase):
    """Tests for Yahoo multi-team optimization with overlap constraints."""

    def test_multi_team_diversity_and_overlap(self):
        df = _generate_synthetic_dfs_pool(n_players=80, seed=777)

        max_shared = 4
        solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=2,
            max_shared_players_in_teams_pairs=max_shared,
            avoid_single_player_in_all_sets=False,
            verbose=False,
        )

        teams = solver.run(df=df)

        self.assertEqual(len(teams), 2)
        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']

        team0_players = set(teams[slots].iloc[0].values)
        team1_players = set(teams[slots].iloc[1].values)

        shared = team0_players & team1_players
        self.assertLessEqual(
            len(shared),
            max_shared,
            f"Expected at most {max_shared} shared players, but got {len(shared)}: {shared}"
        )

        for i in range(2):
            self.assertLessEqual(teams['salary'].iloc[i], 200)
            self.assertEqual(len(set(teams[slots].iloc[i].values)), 8)


class TestYahooLinearProgrammingScipy(unittest.TestCase):
    """Tests for the Scipy HiGHS solver variant."""

    def test_scipy_optimal_single_team(self):
        df = _generate_synthetic_dfs_pool(n_players=60, seed=42)

        solver = FindBestSetOfTeamsLinearProgrammingYahooScipy(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=1,
            verbose=False,
        )

        team = solver.run(df=df)

        self.assertEqual(len(team), 1)
        self.assertLessEqual(team['salary'].iloc[0], 200)
        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']
        self.assertEqual(len(set(team[slots].iloc[0].values)), 8)
        self.assertGreater(team['points'].iloc[0], 0)


class TestYahooLinearProgrammingLateSwapAndResidualMilp(unittest.TestCase):
    """Tests for late swap residual MILP, slot locking, and entry_id output."""

    def test_late_swap_partial_slot_locks(self):
        """Verify partial locks in UTIL and C correctly reduce residual cap and fill remaining slots."""
        df = _generate_synthetic_dfs_pool(n_players=60, seed=101)

        # Early game locked players: Jokic in C ($55, 60 pts) and Brunson in UTIL ($40, 45 pts)
        locked_c_id = 9001
        locked_util_id = 9002
        locked_df = pd.DataFrame([
            {'PLAYER_ID': locked_c_id, 'Pos': 'C', 'YAHOO_POSITION': 'C', 'YAHOO_SALARY': 55.0, 'YAHOO_POINTS': 60.0},
            {'PLAYER_ID': locked_util_id, 'Pos': 'UTIL', 'YAHOO_POSITION': 'PG', 'YAHOO_SALARY': 40.0, 'YAHOO_POINTS': 45.0},
        ])

        solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=1,
            verbose=False,
        )

        team = solver.run(df=df, locked_dfs={777: locked_df})

        self.assertEqual(len(team), 1)
        self.assertEqual(team['C'].iloc[0], locked_c_id)
        self.assertEqual(team['UTIL'].iloc[0], locked_util_id)

        # Total salary must be <= 200 (residual cap 200 - 95 = 105 for remaining 6 slots)
        self.assertLessEqual(team['salary'].iloc[0], 200.0)
        self.assertGreater(team['points'].iloc[0], 105.0)  # At least 60 + 45

        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']
        chosen = team[slots].iloc[0].values
        self.assertEqual(len(set(chosen)), 8)
        self.assertIn('entry_id', team.columns)
        self.assertIn('contest_id', team.columns)
        self.assertEqual(team['entry_id'].iloc[0], 777)
        self.assertEqual(team['contest_id'].iloc[0], 777)

    def test_fully_locked_entry_returns_gracefully(self):
        """Verify an entry with all 8 slots locked returns immediately without MILP failure."""
        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']
        locked_records = []
        for i, s in enumerate(slots):
            locked_records.append({
                'PLAYER_ID': 8000 + i,
                'Pos': s,
                'YAHOO_POSITION': s if s in ['PG', 'SG', 'SF', 'PF', 'C'] else 'PG',
                'YAHOO_SALARY': 22.0,
                'YAHOO_POINTS': 30.0,
            })
        locked_df = pd.DataFrame(locked_records)

        df = _generate_synthetic_dfs_pool(n_players=40, seed=55)
        solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=1,
            verbose=False,
        )

        team = solver.run(df=df, locked_dfs={999: locked_df})
        self.assertEqual(len(team), 1)
        self.assertTrue(solver.last_result.success)
        self.assertEqual(team['entry_id'].iloc[0], 999)
        self.assertEqual(team['salary'].iloc[0], 22.0 * 8)
        self.assertEqual(team['points'].iloc[0], 30.0 * 8)

    def test_three_teams_avoid_single_player_in_all_sets(self):
        """Verify avoid_single_player_in_all_sets prevents any single player from appearing in all 3 sets."""
        df = _generate_synthetic_dfs_pool(n_players=80, seed=888)

        # Make one superstar exceptionally good so every team would naturally choose him
        superstar_id = 99999
        superstar_row = pd.DataFrame([{
            'PLAYER_ID': superstar_id,
            'YAHOO_POSITION': 'PG',
            'YAHOO_SALARY': 30.0,
            'YAHOO_POINTS': 100.0,  # Dominant
        }])
        df = pd.concat([df, superstar_row], ignore_index=True)

        solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            position_col='YAHOO_POSITION',
            salary_col='YAHOO_SALARY',
            points_col='YAHOO_POINTS',
            player_col='PLAYER_ID',
            num_teams_in_set=3,
            avoid_single_player_in_all_sets=True,
            max_shared_players_in_teams_pairs=5,
            verbose=False,
        )

        teams = solver.run(df=df)
        self.assertEqual(len(teams), 3)

        slots = ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']
        all_chosen = [set(teams[slots].iloc[t].values) for t in range(3)]
        in_all_three = all_chosen[0] & all_chosen[1] & all_chosen[2]
        self.assertEqual(len(in_all_three), 0, f"Expected no player in all 3 teams, but got {in_all_three}")


if __name__ == '__main__':
    unittest.main()
