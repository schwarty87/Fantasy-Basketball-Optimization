import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import datetime
import tempfile
import shutil

from utils import sharpe
from machine_learning_fantasy_basketball_predictions.run_model import RunModelFantasyBasketballPredictions
from machine_learning.enums import DataSplitType


class TestSharpeRatio(unittest.TestCase):
    def test_sharpe_known_values(self):
        pnl = [0.8, -1.0, 0.8, 0.8, -1.0, 0.8]
        expected = float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(252.0))
        self.assertAlmostEqual(sharpe(pnl), expected, places=5)
        self.assertAlmostEqual(sharpe(pd.Series(pnl)), expected, places=5)
        self.assertAlmostEqual(sharpe(np.array(pnl)), expected, places=5)

    def test_sharpe_single_element_or_empty(self):
        self.assertEqual(sharpe([]), 0.0)
        self.assertEqual(sharpe([1.0]), 0.0)
        self.assertEqual(sharpe(pd.Series([], dtype=float)), 0.0)
        self.assertEqual(sharpe(pd.Series([5.0])), 0.0)

    def test_sharpe_zero_variance(self):
        self.assertEqual(sharpe([1.0, 1.0, 1.0]), 0.0)
        self.assertEqual(sharpe([-0.5, -0.5, -0.5]), 0.0)

    def test_sharpe_with_nans(self):
        pnl_with_nan = [0.8, np.nan, -1.0, 0.8, np.nan, 0.8]
        clean_pnl = [0.8, -1.0, 0.8, 0.8]
        expected = float(np.mean(clean_pnl) / np.std(clean_pnl, ddof=1) * np.sqrt(252.0))
        self.assertAlmostEqual(sharpe(pnl_with_nan), expected, places=5)
        self.assertAlmostEqual(sharpe(pd.Series(pnl_with_nan)), expected, places=5)

    def test_sharpe_all_nans(self):
        self.assertEqual(sharpe([np.nan, np.nan]), 0.0)
        self.assertEqual(sharpe(pd.Series([np.nan, np.nan])), 0.0)


class TestBacktestEvaluatorRawPoints(unittest.TestCase):
    def test_process_predictions_df_merges_raw_points(self):
        predictions_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1), datetime.date(2023, 1, 1)],
            'PLAYER_ID': [101, 102],
            'YAHOO_POINTS_pred': [35.0, 42.0],
            'YAHOO_POINTS_true': [30.0, 36.0],  # shrunk per-48
            'YAHOO_SALARY': [25.0, 30.0],
            'YAHOO_POSITION': [1, 2],
        })
        contests_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)],
            'contestId': [999],
            'entry_fee': [10.0],
            'score': [250.0],
        })
        mock_player_stats = pd.DataFrame({
            'DATE': ['2023-01-01', '2023-01-01'],
            'PLAYER_ID': [101, 102],
            'YAHOO_POINTS': [33.0, 39.5],  # raw un-shrunk OT points
        })

        with patch('machine_learning_fantasy_basketball_predictions.run_model.StatsLoaderPlayers') as mock_loader:
            mock_loader.return_value.load_saved_data.return_value = mock_player_stats
            processed = RunModelFantasyBasketballPredictions._process_predictions_df(
                predictions_df=predictions_df,
                contests_df=contests_df,
            )

        self.assertIn('YAHOO_POINTS_raw', processed.columns)
        self.assertEqual(processed.loc[processed['PLAYER_ID'] == 101, 'YAHOO_POINTS_raw'].iloc[0], 33.0)
        self.assertEqual(processed.loc[processed['PLAYER_ID'] == 102, 'YAHOO_POINTS_raw'].iloc[0], 39.5)

    def test_process_predictions_df_retains_existing_raw_points(self):
        predictions_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)],
            'PLAYER_ID': [101],
            'YAHOO_POINTS_pred': [35.0],
            'YAHOO_POINTS_true': [30.0],
            'YAHOO_POINTS_raw': [34.0],
            'YAHOO_SALARY': [25.0],
            'YAHOO_POSITION': [1],
        })
        contests_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)],
            'contestId': [999],
            'entry_fee': [10.0],
            'score': [250.0],
        })

        with patch('machine_learning_fantasy_basketball_predictions.run_model.StatsLoaderPlayers') as mock_loader:
            processed = RunModelFantasyBasketballPredictions._process_predictions_df(
                predictions_df=predictions_df,
                contests_df=contests_df,
            )
            mock_loader.assert_not_called()

        self.assertEqual(processed['YAHOO_POINTS_raw'].iloc[0], 34.0)

    def test_add_extra_metrics_scores_lineup_with_raw_points(self):
        # Build slate with 60 players so day_df.shape[0] >= 50
        dates = [datetime.date(2023, 1, 1)] * 60
        player_ids = list(range(1, 61))
        # 8 players who will be selected by LP (highest pred within cap)
        # Positions: 1=PG, 2=SG, 3=SF, 4=PF, 5=C, etc.
        positions = [1, 2, 3, 4, 5, 1, 3, 5] + [1] * 52
        salaries = [25.0] * 8 + [10.0] * 52  # 8 * 25 = 200 (exact cap)
        preds = [40.0] * 8 + [5.0] * 52
        true_shrunk = [35.0] * 8 + [5.0] * 52  # 8 * 35 = 280
        raw_ot = [45.0] * 8 + [5.0] * 52        # 8 * 45 = 360

        predict_df = pd.DataFrame({
            'DATE': dates,
            'PLAYER_ID': player_ids,
            'YAHOO_POSITION': positions,
            'YAHOO_SALARY': salaries,
            'YAHOO_POINTS_pred': preds,
            'YAHOO_POINTS_true': true_shrunk,
            'YAHOO_POINTS_raw': raw_ot,
            'split_type': ['test'] * 60,
        })

        contests_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)] * 10,
            'contestId': [1001] * 10,
            'entry_fee': [5.0] * 10,
            # Human entries scored between 300 and 340 (beats 280, loses to 360!)
            'score': [310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0, 350.0, 355.0],
        })

        temp_dir = tempfile.mkdtemp()
        try:
            mock_params = MagicMock()
            mock_params.continue_model_id = None
            with patch('machine_learning.run_model.models_dir', return_value=temp_dir), \
                 patch('machine_learning.run_model.get_next_model_id', return_value=9999), \
                 patch('machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballDataModule'), \
                 patch('machine_learning_fantasy_basketball_predictions.run_model.FantasyBasketballLightningModule'):
                runner = RunModelFantasyBasketballPredictions(params=mock_params, use_lightning=True)
            eval_df = pd.DataFrame(index=['TEST'], data={'loss': [0.5]})

            with patch.object(runner, '_load_contests_df', return_value=contests_df):
                with patch.object(runner, '_process_predictions_df', return_value=predict_df):
                    result = runner._add_extra_metrics_to_eval_results(eval_df, predict_df)

            # If raw_ot was used (360), predicted_score is 360, beating all human scores (310-355 -> percentile 1.0)
            # If true_shrunk was used (280), predicted_score would be 280, losing to all human scores (percentile 0.0)
            self.assertEqual(result.loc['TEST', 'mean_score_paid'], 360.0)
            self.assertEqual(result.loc['TEST', 'mean_percentile_paid'], 1.0)
            self.assertIn('sharpe_5050_paid', result.columns)
            self.assertIn('sharpe_top10_paid', result.columns)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
