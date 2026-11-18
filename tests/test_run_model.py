import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import datetime
import tempfile
import shutil

from compare_models import (
    average_prediction_dfs,
    compare_model_groups,
    compute_mde,
    evaluate_model_group,
    paired_bootstrap_test,
)
from utils import sharpe
from machine_learning_fantasy_basketball_predictions.run_model import (
    RunModelFantasyBasketballPredictions,
    add_blended_quantile_point_columns,
)
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

    def test_process_predictions_df_fails_fast_when_stats_loader_fails_or_missing_column(self):
        """Verify _process_predictions_df propagates errors and raises ValueError if YAHOO_POINTS_raw cannot be populated."""
        predictions_df = pd.DataFrame({
            'DATE': [datetime.date(2023, 1, 1)],
            'PLAYER_ID': [101],
            'YAHOO_POINTS_pred': [35.0],
            'YAHOO_POINTS_true': [30.0],
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
            mock_loader.return_value.load_saved_data.side_effect = RuntimeError("disk error")
            with self.assertRaises(RuntimeError):
                RunModelFantasyBasketballPredictions._process_predictions_df(
                    predictions_df=predictions_df,
                    contests_df=contests_df,
                )

        with patch('machine_learning_fantasy_basketball_predictions.run_model.StatsLoaderPlayers') as mock_loader:
            # Missing 'YAHOO_POINTS' column in player_boxscores
            mock_loader.return_value.load_saved_data.return_value = pd.DataFrame({'DATE': ['2023-01-01'], 'PLAYER_ID': [101]})
            with self.assertRaises(ValueError):
                RunModelFantasyBasketballPredictions._process_predictions_df(
                    predictions_df=predictions_df,
                    contests_df=contests_df,
                )

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
                result = runner._add_extra_metrics_to_eval_results(eval_df, predict_df)

            # If raw_ot was used (360), predicted_score is 360, beating all human scores (310-355 -> percentile 1.0)
            # If true_shrunk was used (280), predicted_score would be 280, losing to all human scores (percentile 0.0)
            self.assertEqual(result.loc['TEST', 'mean_score_paid'], 360.0)
            self.assertEqual(result.loc['TEST', 'mean_percentile_paid'], 1.0)
            self.assertIn('sharpe_5050_paid', result.columns)
            self.assertIn('sharpe_top10_paid', result.columns)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_midgame_exit_starter_retained_in_lp_while_dummy_slots_dropped(self):
        """Verify NOT_IGNORE==0 midgame-exit starter remains eligible in backtest LP while -999 dummy slots are dropped."""
        d1 = datetime.date(2023, 1, 1)
        # Player 1 is a $30 starter projected at 55 FP who exited mid-game (NOT_IGNORE=0, YAHOO_POINTS_raw=6.0)
        # Players 2..8 are 7 other $24 starters projected at 40 FP with YAHOO_POINTS_raw=40.0
        # Players 9..60 are $10 bench players projected at 5 FP with YAHOO_POINTS_raw=5.0
        # Player 61 is a dummy slot with MIN_true=-999 (must be dropped even though pred=99.0)
        predict_df = pd.DataFrame({
            'DATE': [d1] * 61,
            'PLAYER_ID': list(range(1, 62)),
            'YAHOO_POSITION': [1, 2, 3, 4, 5, 1, 3, 5] + [1] * 52 + [1],
            'YAHOO_SALARY': [30.0] + [24.0] * 7 + [10.0] * 52 + [10.0],
            'YAHOO_POINTS_pred': [55.0] + [40.0] * 7 + [5.0] * 52 + [99.0],
            'YAHOO_POINTS_true': [6.0] + [40.0] * 7 + [5.0] * 52 + [-999.0],
            'MIN_true': [8.0] + [34.0] * 7 + [12.0] * 52 + [-999.0],
            'YAHOO_POINTS_raw': [6.0] + [40.0] * 7 + [5.0] * 52 + [0.0],
            'NOT_IGNORE': [0] + [1] * 59 + [0],
            'split_type': ['test'] * 61,
        })
        contests_df = pd.DataFrame({
            'DATE': [d1] * 5,
            'contestId': [1001] * 5,
            'entry_fee': [5.0] * 5,
            'score': [300.0] * 5,
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
                result = runner._add_extra_metrics_to_eval_results(eval_df, predict_df)

            # Player 1 (6.0) + 7 * 40.0 (280.0) = 286.0
            self.assertAlmostEqual(result.loc['TEST', 'mean_score_paid'], 286.0, places=5)
            self.assertAlmostEqual(result.loc['TEST', 'mean_lineup_score_all_dates'], 286.0, places=5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mean_lineup_score_all_dates_includes_non_contest_dates(self):
        """Verify mean_lineup_score_all_dates averages over D1 and D2 while contest metrics use only D1."""
        d1 = datetime.date(2023, 1, 1)
        d2 = datetime.date(2023, 1, 2)

        def _make_day(d: datetime.date, starter_raw: float) -> pd.DataFrame:
            return pd.DataFrame({
                'DATE': [d] * 60,
                'PLAYER_ID': list(range(1, 61)),
                'YAHOO_POSITION': [1, 2, 3, 4, 5, 1, 3, 5] + [1] * 52,
                'YAHOO_SALARY': [25.0] * 8 + [10.0] * 52,
                'YAHOO_POINTS_pred': [40.0] * 8 + [5.0] * 52,
                'YAHOO_POINTS_true': [starter_raw] * 8 + [5.0] * 52,
                'YAHOO_POINTS_raw': [starter_raw] * 8 + [5.0] * 52,
                'split_type': ['test'] * 60,
            })

        # D1 lineup scores 8 * 45 = 360.0; D2 lineup scores 8 * 30 = 240.0
        predict_df = pd.concat([_make_day(d1, 45.0), _make_day(d2, 30.0)], ignore_index=True)
        contests_df = pd.DataFrame({
            'DATE': [d1] * 5,
            'contestId': [1001] * 5,
            'entry_fee': [5.0] * 5,
            'score': [300.0] * 5,
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
                result = runner._add_extra_metrics_to_eval_results(eval_df, predict_df)

            self.assertAlmostEqual(result.loc['TEST', 'mean_lineup_score_all_dates'], 300.0, places=5)
            self.assertEqual(result.loc['TEST', 'n_lineup_dates'], 2)
            self.assertAlmostEqual(result.loc['TEST', 'mean_score_paid'], 360.0, places=5)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_per_contest_top10_pct_profit_9x_accuracy_and_dropped_salaried(self):
        """Verify per-contest top10_pct, 9x payout, salary-tier biases, min_mae_sal, and dropped salaried count."""
        d1 = datetime.date(2023, 1, 1)
        # 61 real players on D1:
        # - 1..8 selected starters (sal=25, MIN_pred=35, MIN_true=33, raw_FP=45, pred_FP=50) -> bias=+5.0 (per48_true[0]=30 != raw_FP)
        # - 9..30 sal=12 low-tier (MIN_pred=15, MIN_true=17, raw_FP=10, pred_FP=8) -> bias=-2.0
        # - 31..60 sal=35 high-tier (MIN_pred=32, MIN_true=30, raw_FP=40, pred_FP=37) -> bias=-3.0
        # - 61 sal=12 NOT_IGNORE=0 midgame-exit (MIN_pred=20, MIN_true=4, raw_FP=6, pred_FP=5) -> excluded from fp_bias/rmse
        player_ids = list(range(1, 62))
        positions = [1, 2, 3, 4, 5, 1, 3, 5] + [1] * 53
        salaries = [25.0] * 8 + [12.0] * 22 + [35.0] * 30 + [12.0]
        preds_fp = [50.0] * 8 + [8.0] * 22 + [37.0] * 30 + [5.0]  # lineup picks 1..8 (8*25=200), avg pred=50 -> bias=+5 on selected
        raw_fp = [45.0] * 8 + [10.0] * 22 + [40.0] * 30 + [6.0]
        per48_true_fp = [30.0] + [45.0] * 7 + [10.0] * 22 + [40.0] * 30 + [4.0]
        min_pred = [35.0] * 8 + [15.0] * 22 + [32.0] * 30 + [20.0]
        min_true = [33.0] * 8 + [17.0] * 22 + [30.0] * 30 + [4.0]

        predict_df = pd.DataFrame({
            'DATE': [d1] * 61,
            'PLAYER_ID': player_ids,
            'YAHOO_POSITION': positions,
            'YAHOO_SALARY': salaries,
            'YAHOO_POINTS_pred': preds_fp,
            'YAHOO_POINTS_true': per48_true_fp,
            'MIN_pred': min_pred,
            'MIN_true': min_true,
            'NOT_IGNORE': [1] * 60 + [0],
            'split_type': ['TEST'] * 61,
        })
        # Player stats has players 1..61 PLUS 1 dropped active $20 player (id=999, MIN=25, IS_OUT=False)
        # AND 1 inactive OUT $25 player (id=998, MIN=0, IS_OUT=True -> must NOT be counted as dropped)
        # Active >= $15 count: 8 (sal=25) + 30 (sal=35) + 1 (id=999, sal=20) = 39
        player_stats_df = pd.DataFrame({
            'DATE': [d1] * 63,
            'PLAYER_ID': player_ids + [999, 998],
            'YAHOO_POINTS': raw_fp + [20.0, 0.0],
            'YAHOO_SALARY': salaries + [20.0, 25.0],
            'YAHOO_POSITION': positions + [1, 1],
            'MIN': min_true + [25.0, 0.0],
            'IS_OUT': [False] * 61 + [False, True],
        })
        # Two paid contests on D1 (lineup score = 8 * 45 = 360.0):
        # - contest 2001: beats all 10 entries -> percentile=1.0 (Top-10% WIN: +8.0)
        # - contest 2002: beats 7/10 entries -> percentile=0.7 (Top-10% LOSS: -1.0, 50/50 WIN: +0.8)
        # Daily avg percentile = 0.85 (< 0.90), so per-contest top10_pct MUST be 0.5 (not 0.0)!
        contests_df = pd.DataFrame({
            'DATE': [d1] * 20,
            'contestId': [2001] * 10 + [2002] * 10,
            'entry_fee': [5.0] * 20,
            'score': [300.0] * 10 + [300.0] * 7 + [380.0] * 3,
        })

        eval_res, details = RunModelFantasyBasketballPredictions.evaluate_predictions_df(
            predict_results=predict_df,
            contests_df=contests_df,
            player_stats_df=player_stats_df,
            return_details=True,
        )

        self.assertAlmostEqual(eval_res.loc['TEST', 'top10_pct_paid'], 0.5, places=5)
        self.assertAlmostEqual(eval_res.loc['TEST', 'mean_profit_top10_9x_paid'], 3.5, places=5)  # (8 - 1) / 2
        self.assertEqual(eval_res.loc['TEST', 'portfolio_value_top10_9x_paid'], [3.5])
        self.assertEqual(eval_res.loc['TEST', 'n_contests_paid'], 2)
        self.assertEqual(eval_res.loc['TEST', 'n_contest_dates_paid'], 1)
        self.assertEqual(eval_res.loc['TEST', 'n_salaried_players'], 39)
        self.assertEqual(eval_res.loc['TEST', 'n_dropped_salaried_players'], 1)

        # Tier biases & MIN MAE
        self.assertAlmostEqual(eval_res.loc['TEST', 'fp_bias_sal_le_15'], -2.0, places=5)
        self.assertAlmostEqual(eval_res.loc['TEST', 'fp_bias_sal_16_30'], 5.0, places=5)
        self.assertAlmostEqual(eval_res.loc['TEST', 'fp_bias_sal_gt_30'], -3.0, places=5)
        self.assertAlmostEqual(eval_res.loc['TEST', 'min_mae_sal'], 2.0, places=5)
        expected_rmse = float(np.sqrt((8 * 25 + 22 * 4 + 30 * 9) / 60))
        self.assertAlmostEqual(eval_res.loc['TEST', 'fp_rmse_raw'], expected_rmse, places=5)
        self.assertEqual(len(details['percentile_per_contest']), 2)

    def test_compare_models_average_prediction_dfs_bootstrap_and_mde(self):
        """Verify physical seed prediction averaging, alignment checks, paired bootstrap, and MDE."""
        d1 = datetime.date(2024, 1, 1)
        df1 = pd.DataFrame({
            'DATE': [d1, d1],
            'PLAYER_ID': [10, 20],
            'split_type': ['VALIDATION', 'VALIDATION'],
            'YAHOO_POINTS_pred': [30.0, 20.0],
            'MIN_pred': [32.0, 22.0],
            'YAHOO_POINTS_true': [35.0, 18.0],
        })
        # Reverse row order to test deterministic sorting alignment
        df2 = pd.DataFrame({
            'DATE': [d1, d1],
            'PLAYER_ID': [20, 10],
            'split_type': ['VALIDATION', 'VALIDATION'],
            'YAHOO_POINTS_pred': [24.0, 34.0],
            'MIN_pred': [26.0, 36.0],
            'YAHOO_POINTS_true': [18.0, 35.0],
        })
        avg_df = average_prediction_dfs([df1, df2])
        self.assertEqual(list(avg_df['PLAYER_ID']), [10, 20])
        self.assertAlmostEqual(float(avg_df.loc[0, 'YAHOO_POINTS_pred']), 32.0, places=5)
        self.assertAlmostEqual(float(avg_df.loc[1, 'YAHOO_POINTS_pred']), 22.0, places=5)
        self.assertAlmostEqual(float(avg_df.loc[0, 'MIN_pred']), 34.0, places=5)
        self.assertEqual(list(avg_df['YAHOO_POINTS_true']), [35.0, 18.0])

        df_bad = df2.copy()
        df_bad.loc[0, 'PLAYER_ID'] = 999
        with self.assertRaises(ValueError):
            average_prediction_dfs([df1, df_bad])

        a = [10.0] * 30
        b = [12.0 if i % 2 == 0 else 11.0 for i in range(30)]
        boot = paired_bootstrap_test(a, b, n_resamples=2000, seed=7)
        self.assertEqual(boot['n'], 30)
        self.assertAlmostEqual(boot['mean_diff'], 1.5, places=5)
        self.assertLess(boot['p_value'], 0.01)
        self.assertAlmostEqual(boot['mde_80'], compute_mde(boot['se']), places=5)

    def test_quantile_coverage_metrics_and_cash_gpp_routing(self):
        """Verify quantile calibration metrics (including NOT_IGNORE=0 early exits), Cash/GPP routing, and alpha validation."""
        d1 = datetime.date(2023, 1, 1)
        with self.assertRaises(ValueError):
            add_blended_quantile_point_columns(pd.DataFrame({'YAHOO_POINTS_pred': [10.0]}), alpha_cash=1.5)

        # Build 60 players across two valid DFS archetypes:
        # - Lineup A (players 1..8, sal=25): Safe high-floor PG=1 (mean=41, q20=40, q80=43) + starters 2..8 (mean=40, q20=39, q80=42); true raw=46 each (368 total)
        # - Lineup B (player 9 replacing player 1 at PG, sal=25): Boom/bust ceiling -> mean=39, q20=10, q80=75; true raw=55 (377 total)
        # Point baseline (alpha=0): PG=1 (41.0) > PG=9 (39.0) -> deterministically picks Lineup A (368, beats 8/10 -> Top10 LOSS)
        # Cash blend (alpha_cash=0.35): PG=1 (40.65) > PG=9 (28.85) -> picks Lineup A (368)
        # GPP blend (alpha_gpp=0.85): PG=9 (69.6) > PG=1 (42.7) -> picks Lineup B (377)
        # Note player 10 is sal=10 (sal <= 12) AND NOT_IGNORE=0 (early exit) with y_raw=4 < q20=6 < q80=14: included in q20_coverage!
        n = 60
        player_ids = list(range(1, n + 1))
        positions = [1, 2, 3, 4, 5, 1, 3, 5] + [1] * (n - 8)
        salaries = [25.0] * 9 + [10.0] + [10.0] * (n - 10)
        mean_pred = [41.0] + [40.0] * 7 + [39.0, 10.0] + [5.0] * (n - 10)
        q20_pred = [40.0] + [39.0] * 7 + [10.0, 6.0] + [3.0] * (n - 10)
        q80_pred = [43.0] + [42.0] * 7 + [75.0, 14.0] + [8.0] * (n - 10)
        raw_fp = [46.0] * 8 + [55.0, 4.0] + [5.0] * (n - 10)
        not_ignore = [1] * 9 + [0] + [1] * (n - 10)

        predict_df = pd.DataFrame({
            'DATE': [d1] * n,
            'PLAYER_ID': player_ids,
            'YAHOO_POSITION': positions,
            'YAHOO_SALARY': salaries,
            'YAHOO_POINTS_pred': mean_pred,
            'YAHOO_POINTS_q20_pred': q20_pred,
            'YAHOO_POINTS_q80_pred': q80_pred,
            'YAHOO_POINTS_true': raw_fp,
            'MIN_true': [30.0] * 9 + [4.0] + [10.0] * (n - 10),
            'NOT_IGNORE': not_ignore,
            'split_type': ['VALIDATION'] * n,
        })
        player_stats_df = pd.DataFrame({
            'DATE': [d1] * n,
            'PLAYER_ID': player_ids,
            'YAHOO_POINTS': raw_fp,
            'YAHOO_SALARY': salaries,
            'YAHOO_POSITION': positions,
            'MIN': [30.0] * 9 + [4.0] + [10.0] * (n - 10),
            'IS_OUT': [False] * n,
        })
        # Contest 3001 thresholds: scores in [350, 369, 370] -> Lineup A (368) beats 8/10 (0.80: 50/50 WIN, Top10 LOSS)
        # while Lineup B (377) beats 10/10 (1.00: Top10 WIN!)
        contests_df = pd.DataFrame({
            'DATE': [d1] * 10,
            'contestId': [3001] * 10,
            'entry_fee': [5.0] * 10,
            'score': [350.0] * 8 + [370.0] * 2,
        })

        eval_res, details = RunModelFantasyBasketballPredictions.evaluate_predictions_df(
            predict_results=predict_df,
            contests_df=contests_df,
            player_stats_df=player_stats_df,
            return_details=True,
            alpha_cash=0.35,
            alpha_gpp=0.85,
        )

        # Quantile coverage includes player 10 (NOT_IGNORE=0 early exit where y=4 <= q20=6)
        expected_q20_cov = (0 + 0 + 1 + 0)  # players 1..9 > q20; player 10 <= q20; players 11..60 (y=5 > q20=3) -> 1/60
        expected_q80_cov = 51 / 60.0  # players 9..60 (52 - 1 wait: player 9 raw=55 <= q80=75, 10: 4<=14, 11..60: 5<=8 -> 52/60)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'q20_coverage'], 1.0 / 60.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'q80_coverage'], 52.0 / 60.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'q20_coverage_sal_le_12'], 1.0 / 51.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'q80_coverage_sal_le_12'], 1.0, places=5)
        self.assertEqual(eval_res.loc['VALIDATION', 'quantile_crossing_rate'], 0.0)

        # Point baseline (players 1..8 tied with 9 at mean=40, tie-breaks to 1..8 -> 368.0) vs Cash (368.0) vs GPP (377.0)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'mean_lineup_score_cash_all_dates'], 368.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'mean_lineup_score_gpp_all_dates'], 377.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'mean_percentile_cash_paid'], 0.8, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'mean_percentile_gpp_paid'], 1.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'mean_profit_5050_paid'], 0.8, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'top10_pct_paid'], 1.0, places=5)
        self.assertAlmostEqual(eval_res.loc['VALIDATION', 'mean_profit_top10_9x_paid'], 8.0, places=5)
        self.assertIn('predicted_score_gpp', details['all_dates_scores'].columns)

        # alpha_gpp=0.0 collapses GPP lineup back to point predictions (score 368.0 -> percentile 0.8 -> Top10 LOSS)
        with patch('compare_models.load_predict_results', side_effect=[predict_df, predict_df]):
            grp = evaluate_model_group(
                [1, 2],
                split='VALIDATION',
                include_ensemble=True,
                contests_df=contests_df,
                player_stats_df=player_stats_df,
                alpha_cash=0.0,
                alpha_gpp=0.0,
            )
        self.assertAlmostEqual(grp['ensemble_eval'].loc['VALIDATION', 'top10_pct_paid'], 0.0, places=5)
        self.assertAlmostEqual(grp['ensemble_eval'].loc['VALIDATION', 'mean_lineup_score_gpp_all_dates'], 368.0, places=5)

    def test_quantile_coverage_uses_per48_true_not_raw_ot(self):
        """Verify q20/q80 coverage uses per-48 YAHOO_POINTS_true rather than raw OT points when both exist."""
        d1 = datetime.date(2023, 1, 1)
        predict_df = pd.DataFrame({
            'DATE': [d1] * 60,
            'PLAYER_ID': list(range(1, 61)),
            'YAHOO_POSITION': [1, 2, 3, 4, 5, 1, 3, 5] + [1] * 52,
            'YAHOO_SALARY': [25.0] * 8 + [10.0] * 52,
            'YAHOO_POINTS_pred': [40.0] * 8 + [10.0] * 52,
            'YAHOO_POINTS_q20_pred': [30.0] * 60,
            'YAHOO_POINTS_q80_pred': [50.0] * 60,
            # per-48 YAHOO_POINTS_true <= q80 (100% <= q80, 0% <= q20), while raw OT > q80 (0% <= q80)
            'YAHOO_POINTS_true': [42.0] * 60,
            'YAHOO_POINTS_raw': [65.0] * 60,
            'NOT_IGNORE': [1] * 60,
            'split_type': ['VALIDATION'] * 60,
        })
        eval_res = RunModelFantasyBasketballPredictions._calculate_prediction_accuracy_metrics(
            eval_results=pd.DataFrame(index=['VALIDATION']),
            predict_results=predict_df,
        )
        self.assertEqual(eval_res.loc['VALIDATION', 'q20_coverage'], 0.0)
        self.assertEqual(eval_res.loc['VALIDATION', 'q80_coverage'], 1.0)

    def test_compare_model_groups_pairs_contest_percentiles_by_date(self):
        """Verify compare_model_groups aggregates contest percentiles by DATE before paired bootstrapping."""
        temp_dir = tempfile.mkdtemp()
        try:
            dir_a = os.path.join(temp_dir, 'model_a')
            dir_b = os.path.join(temp_dir, 'model_b')
            os.makedirs(dir_a, exist_ok=True)
            os.makedirs(dir_b, exist_ok=True)

            dates = [datetime.date(2024, 1, d) for d in range(1, 7)]
            n_per_day = 60
            positions = [1, 2, 3, 4, 5, 1, 3, 5] * 2 + [1] * (n_per_day - 16)
            salaries = [25.0] * 16 + [10.0] * (n_per_day - 16)
            preds_a = [40.0] * 8 + [20.0] * 8 + [5.0] * (n_per_day - 16)
            preds_b = [20.0] * 8 + [40.0] * 8 + [5.0] * (n_per_day - 16)

            rows_a, rows_b, ps_rows = [], [], []
            for day_idx, d in enumerate(dates):
                if day_idx == 0:
                    raw_pts = [31.25] * 8 + [43.75] * 8 + [5.0] * (n_per_day - 16)  # A=250, B=350
                else:
                    raw_pts = [43.75] * 8 + [31.25] * 8 + [5.0] * (n_per_day - 16)  # A=350, B=250
                base = pd.DataFrame({
                    'DATE': [d] * n_per_day,
                    'PLAYER_ID': list(range(1, n_per_day + 1)),
                    'YAHOO_POSITION': positions,
                    'YAHOO_SALARY': salaries,
                    'YAHOO_POINTS_true': raw_pts,
                    'NOT_IGNORE': [1] * n_per_day,
                    'split_type': ['VALIDATION'] * n_per_day,
                })
                df_a = base.copy()
                df_a['YAHOO_POINTS_pred'] = preds_a
                rows_a.append(df_a)
                df_b = base.copy()
                df_b['YAHOO_POINTS_pred'] = preds_b
                rows_b.append(df_b)
                ps_rows.append(pd.DataFrame({
                    'DATE': [d] * n_per_day,
                    'PLAYER_ID': list(range(1, n_per_day + 1)),
                    'YAHOO_POINTS': raw_pts,
                    'YAHOO_SALARY': salaries,
                    'YAHOO_POSITION': positions,
                    'MIN': [30.0] * n_per_day,
                    'IS_OUT': [False] * n_per_day,
                }))

            pd.concat(rows_a, ignore_index=True).to_feather(os.path.join(dir_a, 'predict_results.feather'))
            pd.concat(rows_b, ignore_index=True).to_feather(os.path.join(dir_b, 'predict_results.feather'))
            player_stats_df = pd.concat(ps_rows, ignore_index=True)

            # Date 0 has 10 contests (A beats 0.1, B beats 0.9); Dates 1..5 have 1 contest each (A beats 0.6, B beats 0.4).
            # Date-level pairing gives n=6 dates and mean_diff = (0.8 + 5*(-0.2))/6 < 0 (contest-level would give n=15, > 0).
            contest_rows = []
            for cid in range(1000, 1010):
                for score in [200.0] + [300.0] * 8 + [400.0]:
                    contest_rows.append({'DATE': dates[0], 'contestId': cid, 'entry_fee': 5.0, 'score': score})
            for idx, d in enumerate(dates[1:], start=1):
                cid = 2000 + idx
                for score in [200.0] * 4 + [280.0] * 2 + [400.0] * 4:
                    contest_rows.append({'DATE': d, 'contestId': cid, 'entry_fee': 5.0, 'score': score})
            contests_df = pd.DataFrame(contest_rows)

            with patch.object(RunModelFantasyBasketballPredictions, '_load_contests_df', return_value=contests_df), \
                 patch('compare_models.StatsLoaderPlayers') as mock_loader:
                mock_loader.return_value.load_saved_data.return_value = player_stats_df
                comp = compare_model_groups([dir_a], [dir_b], split='VALIDATION', n_resamples=500, seed=42)

            self.assertIn('daily_percentile_all', comp['bootstrap_tests'])
            self.assertNotIn('contest_percentile_all', comp['bootstrap_tests'])
            boot = comp['bootstrap_tests']['daily_percentile_all']
            self.assertEqual(boot['n'], 6)
            self.assertLess(boot['mean_diff'], 0.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
