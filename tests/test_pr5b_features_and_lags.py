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
        df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)] * 4,
            'GAME_ID': [1] * 4,
            'TEAM_ID': [1] * 4,
            'PLAYER_ID': [1, 2, 3, 4],
            'PTS': [20.0, 30.0, 0.0, 15.0],
            'PTS_mean5': [10.0, 2.0, 0.0, 0.0],  # normal (2.0), clamped (>10), 0/0 (1.0), x/0 (10.0)
        })
        params = FeaturesParamsFantasyBasketballPredictions(
            feature_name_to_machine_learning_dataset_params={
                'tabular': unittest.mock.MagicMock(feature_names=['PTS_mean_ratio']),
            }
        )
        res = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(df, params)

        self.assertIn('PTS_mean_ratio', res.columns)
        self.assertEqual(res.loc[res['PLAYER_ID'] == 1, 'PTS_mean_ratio'].iloc[0], 2.0)
        self.assertEqual(res.loc[res['PLAYER_ID'] == 2, 'PTS_mean_ratio'].iloc[0], 10.0)  # clamped
        self.assertEqual(res.loc[res['PLAYER_ID'] == 3, 'PTS_mean_ratio'].iloc[0], 1.0)   # 0 / 0 -> 1.0
        self.assertEqual(res.loc[res['PLAYER_ID'] == 4, 'PTS_mean_ratio'].iloc[0], 10.0)  # x / 0 -> 10.0

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


if __name__ == '__main__':
    unittest.main()
