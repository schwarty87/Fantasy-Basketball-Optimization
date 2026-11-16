import unittest
import datetime
import numpy as np
import pandas as pd

from machine_learning_fantasy_basketball_predictions.params.params import FeaturesParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import (
    MachineLearningDatasetCombinedFantasyBasketballPredictions,
)


class TestPreGridFeaturesAndLags(unittest.TestCase):
    def test_features_params_declares_source_dependencies(self):
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'tabular': unittest.mock.MagicMock(
                    feature_names=['PTS_mean_ratio']
                ),
                'lagged': unittest.mock.MagicMock(
                    feature_names=['MIN_t-1', 'STARTER_t-1', 'AST_t-2']
                ),
            }
        )
        cols = set(params.all_feature_names_list)
        # Should include base columns needed to compute derived features
        self.assertIn('PTS', cols)
        self.assertIn('PTS_mean5', cols)
        self.assertIn('MIN', cols)
        self.assertIn('STARTER', cols)
        self.assertIn('AST', cols)
        self.assertIn('EJECTED', cols)
        self.assertIn('LEFT_MIDGAME', cols)

    def test_chronological_lags_and_starter_fallback(self):
        # Player 1 plays 3 games chronologically
        df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1), datetime.date(2023, 1, 3), datetime.date(2023, 1, 5)],
            'GAME_ID': [1, 2, 3],
            'TEAM_ID': [10, 10, 10],
            'PLAYER_ID': [101, 101, 101],
            'MIN': [25.0, 32.0, 28.0],
            'STARTER': [True, True, False],
            'START_POSITION_CALCULATED': ['PG', 'PG', 'BN1'],
        })
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'lagged': unittest.mock.MagicMock(feature_names=['MIN_t-1', 'MIN_t-2', 'STARTER_t-1']),
            }
        )
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df, params)

        self.assertIn('MIN_t-1', res.columns)
        self.assertIn('MIN_t-2', res.columns)
        self.assertIn('STARTER_t-1', res.columns)

        # Game 2: lag 1 from Game 1 (MIN=25.0, STARTER=1.0)
        self.assertEqual(res.loc[res['GAME_ID'] == 2, 'MIN_t-1'].iloc[0], 25.0)
        self.assertEqual(res.loc[res['GAME_ID'] == 2, 'STARTER_t-1'].iloc[0], 1.0)

        # Game 3: lag 1 from Game 2 (MIN=32.0), lag 2 from Game 1 (MIN=25.0)
        self.assertEqual(res.loc[res['GAME_ID'] == 3, 'MIN_t-1'].iloc[0], 32.0)
        self.assertEqual(res.loc[res['GAME_ID'] == 3, 'MIN_t-2'].iloc[0], 25.0)

    def test_mean_ratio_calculation(self):
        # Strictly causal: _mean_ratio at game t uses player's game t-1 stat / PTS_mean5 (debut -> 1.0)
        df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)] * 4 + [datetime.date(2023, 1, 3)] * 4,
            'GAME_ID': [1] * 4 + [2] * 4,
            'TEAM_ID': [1] * 8,
            'PLAYER_ID': [1, 2, 3, 4] * 2,
            'PTS': [20.0, 30.0, 0.0, 15.0, 99.0, 1.0, 0.0, 2.0],
            'PTS_mean5': [10.0, 10.0, 10.0, 10.0, 10.0, 2.0, 0.0, 0.0],
        })
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'tabular': unittest.mock.MagicMock(feature_names=['PTS_mean_ratio']),
            }
        )
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df, params)

        self.assertIn('PTS_mean_ratio', res.columns)
        # Game 1 debuts default cleanly to 1.0
        g1 = res[res['GAME_ID'] == 1]
        self.assertTrue((g1['PTS_mean_ratio'] == 1.0).all())
        # Game 2 uses game 1 PTS (normal 20/10=2.0, clamped >10, 0/0->1.0, x/0->10.0)
        g2 = res[res['GAME_ID'] == 2]
        self.assertEqual(g2.loc[g2['PLAYER_ID'] == 1, 'PTS_mean_ratio'].iloc[0], 2.0)
        self.assertEqual(g2.loc[g2['PLAYER_ID'] == 2, 'PTS_mean_ratio'].iloc[0], 10.0)
        self.assertEqual(g2.loc[g2['PLAYER_ID'] == 3, 'PTS_mean_ratio'].iloc[0], 1.0)
        self.assertEqual(g2.loc[g2['PLAYER_ID'] == 4, 'PTS_mean_ratio'].iloc[0], 10.0)

        # Changing game 2's own PTS must have zero effect on game 2's PTS_mean_ratio
        df_mut = df.copy()
        df_mut.loc[df_mut['GAME_ID'] == 2, 'PTS'] = [5.0, 5.0, 50.0, 0.0]
        res_mut = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df_mut, params)
        pd.testing.assert_series_equal(res['PTS_mean_ratio'], res_mut['PTS_mean_ratio'])

    def test_court_position_uses_starter_position_without_changing_site_position_rounded(self):
        """Verify COURT_POSITION uses START_POSITION on starters while SITE_POSITION_ROUNDED stays unchanged."""
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'positional_encoding_features': unittest.mock.MagicMock(feature_names=['HOME', 'COURT_POSITION']),
            }
        )
        self.assertTrue({'SITE_POSITION', 'START_POSITION', 'STARTER'}.issubset(set(params.all_feature_names_list)))

        raw_df = pd.DataFrame({
            'STARTER': [True, False, True],
            'START_POSITION': ['C', '', None],
            'SITE_POSITION': [2.2, 4.1, 1.8],  # rounded: [2 (SG), 4 (PF), 2 (SG)]
            'YAHOO_POSITION': ['SG', 'PF', 'SG'],
        })
        dummy_ds = unittest.mock.MagicMock(spec=MachineLearningDatasetCombinedFantasyBasketballPredictions)
        dummy_ds.params = params
        dummy_ds._add_player_history_columns = (
            MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns
        )
        with unittest.mock.patch(
            'machine_learning.machine_learning_datasets.MachineLearningDatasetCombined._load_features',
            return_value=raw_df.copy(),
        ):
            out = MachineLearningDatasetCombinedFantasyBasketballPredictions._load_features(dummy_ds)

        # SITE_POSITION_ROUNDED stays strictly rounded(SITE_POSITION)
        self.assertEqual(list(out['SITE_POSITION_ROUNDED']), [2, 4, 2])
        # Starter with valid START_POSITION='C' gets 5; bench and missing START_POSITION fall back to SITE_POSITION_ROUNDED
        self.assertEqual(list(out['COURT_POSITION']), [5, 4, 2])

    def test_role_promotion_uses_strict_chronological_lag_and_preserves_role_change(self):
        """Verify spot-start promotion after bench game pulls t-1 from the bench game and preserves STARTER_t-1 == 0.0."""
        df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1), datetime.date(2023, 1, 3), datetime.date(2023, 1, 5)],
            'GAME_ID': [1, 2, 3],
            'TEAM_ID': [10, 10, 10],
            'PLAYER_ID': [101, 101, 101],
            'PTS': [10.0, 28.0, 15.0],
            'STARTER': [True, False, True],
            'START_POSITION_CALCULATED': ['PG', 'BN1', 'PG'],
        })
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'lagged': unittest.mock.MagicMock(feature_names=['PTS_t-1', 'PTS_t-2', 'STARTER_t-1']),
            }
        )
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df, params)

        game3 = res.loc[res['GAME_ID'] == 3].iloc[0]
        self.assertEqual(game3['PTS_t-1'], 28.0)
        self.assertEqual(game3['PTS_t-2'], 10.0)
        self.assertEqual(game3['STARTER_t-1'], 0.0)

    def test_add_player_history_columns_preserves_grid_order_when_columns_exist(self):
        """Verify _add_player_history_columns is a no-op that never re-sorts an already-computed grid DataFrame."""
        df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)],
            'GAME_ID': [1, 1],
            'TEAM_ID': [20, 10],
            'PLAYER_ID': [500, 0],  # Starter (500) before dummy bench slot (0)
            'PTS': [20.0, 0.0],
            'PTS_t-1': [18.0, 0.0],
        })
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'lagged': unittest.mock.MagicMock(feature_names=['PTS_t-1']),
            }
        )
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df, params)
        self.assertEqual(list(res['PLAYER_ID']), [500, 0])

    def test_debut_lag_fallback_ignores_bench_order_when_spc_excluded_from_inputs(self):
        """Verify debut lag fallback groups only by STARTER when START_POSITION_CALCULATED is omitted from inputs."""
        base_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)] * 3,
            'GAME_ID': [1, 1, 1],
            'TEAM_ID': [10, 10, 10],
            'PLAYER_ID': [101, 102, 103],
            'MIN': [24.0, 6.0, 12.0],
            'STARTER': [False, False, False],
            'START_POSITION_CALCULATED': ['BN1', 'BN9', 'BN1'],
        })
        df_bn9 = base_df.copy()
        df_bn9.loc[df_bn9['PLAYER_ID'] == 103, 'START_POSITION_CALCULATED'] = 'BN9'

        params_no_spc = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'lagged': unittest.mock.MagicMock(feature_names=['MIN_t-1']),
                'positional_encoding_features': unittest.mock.MagicMock(feature_names=['HOME', 'SITE_POSITION_ROUNDED']),
            }
        )
        res1 = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(base_df.copy(), params_no_spc)
        res2 = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df_bn9.copy(), params_no_spc)
        self.assertEqual(
            res1.loc[res1['PLAYER_ID'] == 103, 'MIN_t-1'].iloc[0],
            res2.loc[res2['PLAYER_ID'] == 103, 'MIN_t-1'].iloc[0],
        )

        params_with_spc = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'lagged': unittest.mock.MagicMock(feature_names=['MIN_t-1']),
                'positional_encoding_features': unittest.mock.MagicMock(feature_names=['HOME', 'START_POSITION_CALCULATED']),
            }
        )
        res_with_1 = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(base_df.copy(), params_with_spc)
        res_with_2 = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df_bn9.copy(), params_with_spc)
        self.assertNotEqual(
            res_with_1.loc[res_with_1['PLAYER_ID'] == 103, 'MIN_t-1'].iloc[0],
            res_with_2.loc[res_with_2['PLAYER_ID'] == 103, 'MIN_t-1'].iloc[0],
        )

    def test_sort_features_30_player_grid_preserves_away_then_home_order(self):
        """Verify 30-player grid keeps [HOME=0 PG..BN10, HOME=1 PG..BN10] even when Home TEAM_ID < Away TEAM_ID."""
        positions = ['PG', 'SG', 'SF', 'PF', 'C'] + [f'BN{i}' for i in range(1, 11)]
        rows = []
        # Home team (TEAM_ID=10 < Away TEAM_ID=20)
        for idx, pos in enumerate(positions[:8]):
            rows.append({
                'DATE': datetime.date(2023, 1, 1),
                'SEASON_YEAR': 2023,
                'GAME_ID': 100,
                'HOME': 1.0,
                'TEAM_ID': 10,
                'TEAM': 'HOM',
                'PLAYER_ID': 100 + idx,
                'START_POSITION_CALCULATED': pos,
            })
        for idx, pos in enumerate(positions[:9]):
            rows.append({
                'DATE': datetime.date(2023, 1, 1),
                'SEASON_YEAR': 2023,
                'GAME_ID': 100,
                'HOME': 0.0,
                'TEAM_ID': 20,
                'TEAM': 'AWY',
                'PLAYER_ID': 200 + idx,
                'START_POSITION_CALCULATED': pos,
            })
        df = pd.DataFrame(rows)
        dummy_ds = unittest.mock.MagicMock()
        dummy_ds.params.sample_per_game = True
        dummy_ds.params.num_bench_players = 10
        dummy_ds._get_start_positions_list = lambda: MachineLearningDatasetCombinedFantasyBasketballPredictions._get_start_positions_list(dummy_ds)
        sorted_df = MachineLearningDatasetCombinedFantasyBasketballPredictions._sort_features(dummy_ds, df)

        self.assertEqual(len(sorted_df), 30)
        self.assertTrue((sorted_df.iloc[:15]['HOME'] == 0.0).all())
        self.assertTrue((sorted_df.iloc[15:]['HOME'] == 1.0).all())
        self.assertEqual(list(sorted_df.iloc[:15]['START_POSITION_CALCULATED']), positions)
        self.assertEqual(list(sorted_df.iloc[15:]['START_POSITION_CALCULATED']), positions)


if __name__ == '__main__':
    unittest.main()
