import datetime
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from constants import DATA_DIR, DFS_POSITIONS, POSITION_TO_NUM, Sites, UNKNOWN_PLAYER_ID
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from machine_learning.architecture import ModelArchitecture
from machine_learning.callbacks import CollectTruePredictedOutputs
from machine_learning.constants import METADATA
from machine_learning.machine_learning_datasets import AbstractScalingMachineLearningDataset
from machine_learning.params.callbacks_params import (
    EarlyStoppingParams,
    ReduceLearningRateOnPlateauParams,
)
from machine_learning.run_model import RunModel
from machine_learning.utils import CustomOrdinalEncoder
from machine_learning_fantasy_basketball_predictions.callbacks import (
    CollectTruePredictedOutputsFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.layers import CalculateFantasyPoints
from machine_learning_fantasy_basketball_predictions.machine_learning_datasets import (
    MachineLearningDatasetCombinedFantasyBasketballPredictions,
)
from machine_learning_fantasy_basketball_predictions.params.layers_params import (
    CalculateFantasyPointsParams,
    update_calculate_fantasy_points_params,
)
from machine_learning_fantasy_basketball_predictions.run_model import (
    RunModelFantasyBasketballPredictions,
)
from run_all import create_params
from submit_dfs_team import SubmitDFSTeamYahooByCSV


FEATURES_PATH = os.path.join(DATA_DIR, "features", "features.feather")
PLAYER_BOXSCORES_PATH = os.path.join(DATA_DIR, "stats", "player_boxscores.feather")
CONTESTS_PATH = os.path.join(DATA_DIR, "dfs_contests", "dfs_contest_results2.feather")
MODEL_0_PT_PATH = os.path.join(
    DATA_DIR, "machine_learning_fantasy_basketball_predictions", "models", "0", "model.pt"
)


@unittest.skipUnless(os.path.isfile(FEATURES_PATH), "Real features.feather not found on disk")
class TestEndToEndRealDataVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a 15-day slice of real NBA features from the 2024-25 season for fast, exact verification
        full_df = pd.read_feather(FEATURES_PATH)
        full_df["DATE"] = pd.to_datetime(full_df["DATE"]).dt.date
        target_dates = sorted(
            d for d in full_df["DATE"].unique()
            if datetime.date(2024, 11, 1) <= d <= datetime.date(2024, 11, 20)
        )
        cls.raw_slice_df = full_df.loc[full_df["DATE"].isin(target_dates)].copy().reset_index(drop=True)
        cls.params = create_params()

    def test_1_real_features_chronological_lags_and_22_slot_grid_invariants(self):
        """Verify pre-grid chronological lags, 22-slot grid order, and dummy zero-fill on real NBA data."""
        df = self.raw_slice_df.copy()
        if "YAHOO_POSITION" in df.columns:
            df["YAHOO_POSITION"] = df["YAHOO_POSITION"].map(POSITION_TO_NUM).fillna(-999)

        # 1. Compute player history columns pre-grid
        df_with_lags = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_player_history_columns(
            df, self.params.features_params
        )

        # Verify strictly chronological t-1 lag for every player in the slice (regardless of starter/bench switches)
        df_sorted = df_with_lags.sort_values(["DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"]).reset_index(drop=True)
        for pid, p_group in df_sorted.groupby("PLAYER_ID"):
            if len(p_group) < 2:
                continue
            reb_vals = p_group["REB"].values
            reb_t1_vals = p_group["REB_t-1"].values
            min_vals = p_group["MIN"].values
            min_t1_vals = p_group["MIN_t-1"].values
            np.testing.assert_allclose(reb_t1_vals[1:], reb_vals[:-1], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(min_t1_vals[1:], min_vals[:-1], rtol=1e-5, atol=1e-5)

        # 2. Apply _sort_features and _add_additional_columns (as MachineLearningDatasetCombined does)
        dummy_ds = MagicMock(spec=MachineLearningDatasetCombinedFantasyBasketballPredictions)
        dummy_ds.params = self.params.features_params
        dummy_ds._get_start_positions_list = lambda: (
            ["PG", "SG", "SF", "PF", "C"] + [f"BN{i}" for i in range(1, 7)]
        )
        grid_df = MachineLearningDatasetCombinedFantasyBasketballPredictions._sort_features(dummy_ds, df_with_lags)
        grid_df = MachineLearningDatasetCombinedFantasyBasketballPredictions._add_additional_columns(
            grid_df, self.params.features_params
        )

        # Assert exact 22-row game structure
        n_games = grid_df["GAME_ID"].nunique()
        self.assertEqual(len(grid_df), n_games * 22)

        expected_positions = (["PG", "SG", "SF", "PF", "C", "BN1", "BN2", "BN3", "BN4", "BN5", "BN6"]) * 2
        expected_home = [0.0] * 11 + [1.0] * 11
        for _, game_group in grid_df.groupby("GAME_ID", sort=False):
            self.assertEqual(list(game_group["START_POSITION_CALCULATED"]), expected_positions)
            self.assertEqual(list(game_group["HOME"]), expected_home)

        # 3. Verify dummy slots have IS_REAL_PLAYER == -999 and scale to 0.0 (never -999.0)
        is_dummy = grid_df["PLAYER_ID"] == UNKNOWN_PLAYER_ID
        self.assertTrue(is_dummy.any(), "Real NBA slice should contain at least one <11 player rotation with dummy bench slots")
        self.assertTrue((grid_df.loc[is_dummy, "IS_REAL_PLAYER"] == -999).all())
        self.assertTrue((grid_df.loc[~is_dummy, "IS_REAL_PLAYER"] == 1).all())

        tab_cols = self.params.features_params.feature_name_to_machine_learning_dataset_params["tabular"].feature_names
        real_mask = (grid_df["IS_REAL_PLAYER"] == 1).values
        scaler = StandardScaler()
        scaler.fit(grid_df.loc[real_mask, tab_cols])
        scaled_tab = AbstractScalingMachineLearningDataset._scale_features(
            scaler=scaler,
            features_to_scale=grid_df[tab_cols].copy(),
            keep_mask=real_mask,
            masked_fill_value=0.0,
        )
        self.assertTrue((scaled_tab.loc[is_dummy] == 0.0).all().all())
        self.assertFalse((scaled_tab == -999.0).any().any())

    def test_2_real_model_forward_mask_invariance_and_single_unscaling(self):
        """Verify end-to-end forward pass, dummy mask invariance, in-graph unscaling, and callback single-unscaling."""
        params = create_params()

        # Verify CL 5 baseline hyperparameters
        self.assertEqual(params.training_params.weight_decay, 1e-2)
        es_cb = next(c for c in params.training_params.callbacks_params if isinstance(c, EarlyStoppingParams))
        lr_cb = next(c for c in params.training_params.callbacks_params if isinstance(c, ReduceLearningRateOnPlateauParams))
        self.assertEqual(es_cb.patience, 10)
        self.assertEqual(lr_cb.patience, 5)

        # Fit real target scaler on real NBA slice
        true_cols = params.features_params.feature_name_to_machine_learning_dataset_params["true_label"].feature_names
        target_scaler = StandardScaler()
        target_scaler.fit(self.raw_slice_df[true_cols].dropna())

        # Populate CalculateFantasyPointsParams with real target scaler mean and scale
        for lp in params.architecture_params.layer_params_list:
            if isinstance(lp, CalculateFantasyPointsParams):
                lp.scaler_mean = target_scaler.mean_.tolist()
                lp.scaler_scale = target_scaler.scale_.tolist()
            elif hasattr(lp, "num_embeddings"):
                lp.num_embeddings = [100] * len(lp.embeddings_dims)

        # Build model up to CalculateFantasyPoints (before OptimizeTeam LP layer for pure neural forward test)
        neural_layers = [
            lp for lp in params.architecture_params.layer_params_list
            if lp.__class__.__name__ not in ("ReshapeToSamplePerDayParams", "OptimizeTeamParams")
            and "sample_per_day" not in getattr(lp, "output_tensor_name", "")
        ]
        params.architecture_params.layer_params_list = neural_layers
        params.architecture_params.output_tensor_names = ["outputs_pred", "fantasy_points"]

        model = ModelArchitecture(
            params=params.architecture_params,
            feature_type_to_input_shape=params.features_params.feature_type_to_input_shape,
        )
        model.eval()

        # Construct a 2-game batch (2, 22, F) with 2 dummy players per team
        B, S = 2, 22
        tab_dim = params.features_params.feature_type_to_input_shape["tabular"][-1]
        lag_dim = params.features_params.feature_type_to_input_shape["lagged"][-1]
        pos_dim = params.features_params.feature_type_to_input_shape["positional_encoding_features"][-1]
        emb_dim = params.features_params.feature_type_to_input_shape["embedding_features"][-1]

        torch.manual_seed(123)
        tab_x = torch.randn(B, S, tab_dim)
        lag_x = torch.randn(B, S, lag_dim)
        pos_x = torch.randint(0, 2, (B, S, pos_dim), dtype=torch.long)
        emb_x = torch.randint(0, 50, (B, S, emb_dim), dtype=torch.long)
        real_mask = torch.ones(B, S, 1)
        real_mask[:, [9, 10, 20, 21], :] = 0.0  # BN5, BN6 on both teams are dummy slots

        batch = {
            "tabular": tab_x,
            "lagged": lag_x,
            "positional_encoding_features": pos_x,
            "embedding_features": emb_x,
            "real_player_mask": real_mask,
        }
        with torch.no_grad():
            out1 = model(batch)

        # Perturb dummy slots in tabular and lagged inputs by +1000.0; real player predictions must be invariant!
        tab_perturbed = tab_x.clone()
        lag_perturbed = lag_x.clone()
        dummy_2d = real_mask.squeeze(-1) == 0.0
        tab_perturbed[dummy_2d] += 1000.0
        lag_perturbed[dummy_2d] += 1000.0
        batch_perturbed = {
            "tabular": tab_perturbed,
            "lagged": lag_perturbed,
            "positional_encoding_features": pos_x,
            "embedding_features": emb_x,
            "real_player_mask": real_mask,
        }
        with torch.no_grad():
            out2 = model(batch_perturbed)

        real_2d = real_mask.squeeze(-1) == 1.0
        self.assertTrue(
            torch.allclose(out1["outputs_pred"][real_2d], out2["outputs_pred"][real_2d], atol=1e-5),
            "Real player predictions must be completely invariant to dummy slot perturbations!",
        )
        self.assertTrue(
            torch.allclose(out1["fantasy_points"][real_2d], out2["fantasy_points"][real_2d], atol=1e-4),
            "Real player fantasy points must be completely invariant to dummy slot perturbations!",
        )

        # Verify in-graph fantasy_points equals manually unscaled outputs_pred @ Yahoo weights
        cfp_layer = next(m for m in model.modules() if isinstance(m, CalculateFantasyPoints))
        unscaled_preds = (
            out1["outputs_pred"] * torch.tensor(target_scaler.scale_, dtype=torch.float32)
            + torch.tensor(target_scaler.mean_, dtype=torch.float32)
        )
        expected_fp = torch.einsum("...s,s->...", unscaled_preds, cfp_layer.weights)
        self.assertTrue(torch.allclose(out1["fantasy_points"], expected_fp, atol=1e-4))

    @unittest.skipUnless(os.path.isfile(MODEL_0_PT_PATH), "Legacy models/0/model.pt not found on disk")
    def test_3_real_legacy_model_0_loading_preserves_all_learned_embeddings(self):
        """Verify loading real models/0/model.pt appends 1 padding row without overwriting any learned category."""
        raw_loaded = torch.load(MODEL_0_PT_PATH, map_location="cpu", weights_only=False)
        orig_shapes_and_weights = []
        for m in raw_loaded.modules():
            if isinstance(m, nn.Embedding):
                orig_shapes_and_weights.append((m.num_embeddings, m.weight.detach().clone()))

        runner = MagicMock(spec=RunModel)
        runner.params = MagicMock()
        runner.params.continue_model_id = 0
        runner.use_lightning = False
        runner.device = torch.device("cpu")
        runner._continue_model_dir = os.path.dirname(MODEL_0_PT_PATH)

        RunModel._create_model(runner)

        new_embeddings = [m for m in runner.model.modules() if isinstance(m, nn.Embedding)]
        self.assertEqual(len(new_embeddings), len(orig_shapes_and_weights))
        for (orig_k, orig_w), new_m in zip(orig_shapes_and_weights, new_embeddings):
            self.assertEqual(new_m.num_embeddings, orig_k + 1)
            self.assertEqual(new_m.padding_idx, orig_k)
            self.assertTrue(torch.all(new_m.weight[orig_k] == 0.0))
            self.assertTrue(torch.equal(new_m.weight[:orig_k], orig_w))

    @unittest.skipUnless(
        os.path.isfile(PLAYER_BOXSCORES_PATH) and os.path.isfile(CONTESTS_PATH),
        "Real player_boxscores.feather or dfs_contest_results2.feather not found on disk",
    )
    def test_4_real_contest_backtest_evaluator_merges_raw_overtime_points(self):
        """Verify _process_predictions_df and _add_extra_metrics_to_eval_results use raw OT points on real data."""
        contests_df = RunModelFantasyBasketballPredictions._load_contests_df()
        boxscores_df = pd.read_feather(PLAYER_BOXSCORES_PATH)
        boxscores_df["DATE"] = pd.to_datetime(boxscores_df["DATE"]).dt.date

        # Find a real contest date with >= 60 players in features.feather where an OT game occurred
        # (in OT games, raw boxscore MIN > per-48 normalized features.feather MIN)
        common_dates = sorted(set(contests_df["DATE"].unique()) & set(boxscores_df["DATE"].unique()))
        self.assertTrue(len(common_dates) > 0)
        sample_date = common_dates[-1]

        day_box = boxscores_df.loc[boxscores_df["DATE"] == sample_date].copy()
        day_box = day_box.loc[day_box["YAHOO_SALARY"].notna() & (day_box["YAHOO_SALARY"] > 0)].copy()

        # Simulate predictions_df where YAHOO_POINTS_true was shrunk by 10% (as in OT) relative to raw boxscores
        preds_df = pd.DataFrame({
            "DATE": day_box["DATE"].values,
            "PLAYER_ID": day_box["PLAYER_ID"].astype(int).values,
            "YAHOO_POINTS_pred": day_box["YAHOO_POINTS"].values + 1.0,
            "YAHOO_POINTS_true": day_box["YAHOO_POINTS"].values * 0.9,
            "YAHOO_SALARY": day_box["YAHOO_SALARY"].values,
            "YAHOO_POSITION": day_box["YAHOO_POSITION"].map(POSITION_TO_NUM).fillna(1).astype(int).values,
            "split_type": ["TEST"] * len(day_box),
        })

        processed = RunModelFantasyBasketballPredictions._process_predictions_df(
            predictions_df=preds_df,
            contests_df=contests_df.loc[contests_df["DATE"] == sample_date],
        )
        self.assertIn("YAHOO_POINTS_raw", processed.columns)
        # Raw points must match un-shrunk boxscores_df["YAHOO_POINTS"], NOT the 0.9x shrunk YAHOO_POINTS_true
        self.assertTrue(
            np.allclose(processed["YAHOO_POINTS_raw"].values, processed["YAHOO_POINTS_true"].values / 0.9, atol=1e-3)
        )

    def test_5_real_multi_entry_milp_late_swap_and_template_filling(self):
        """Verify MILP solver and Yahoo CSV template filling on a real NBA slate with multi-entry + locked players."""
        # Pick the largest slate date in our real slice
        date_counts = self.raw_slice_df.groupby("DATE")["PLAYER_ID"].nunique()
        slate_date = date_counts.idxmax()
        slate_df = self.raw_slice_df.loc[self.raw_slice_df["DATE"] == slate_date].copy()
        slate_df = slate_df.loc[slate_df["YAHOO_SALARY"].notna() & (slate_df["YAHOO_SALARY"] > 0)].copy()
        slate_df["YAHOO_POSITION"] = slate_df["YAHOO_POSITION"].map(POSITION_TO_NUM).fillna(-999).astype(int)
        slate_df["YAHOO_POINTS_pred"] = slate_df["PTS_mean5"].fillna(10.0) * 1.8

        # Pick 2 real players to lock into Entry 1001
        pg_Row = slate_df.loc[slate_df["YAHOO_POSITION"] == 1].iloc[0]
        c_Row = slate_df.loc[slate_df["YAHOO_POSITION"] == 5].iloc[0]

        locked_dfs = {
            1001: pd.DataFrame({
                "Pos": ["PG", "C"],
                "PLAYER_ID": [int(pg_Row["PLAYER_ID"]), int(c_Row["PLAYER_ID"])],
                "YAHOO_SALARY": [float(pg_Row["YAHOO_SALARY"]), float(c_Row["YAHOO_SALARY"])],
                "YAHOO_POSITION": ["PG", "C"],
            }),
            1002: pd.DataFrame(columns=["Pos", "PLAYER_ID", "YAHOO_SALARY", "YAHOO_POSITION"]),
            1003: pd.DataFrame(columns=["Pos", "PLAYER_ID", "YAHOO_SALARY", "YAHOO_POSITION"]),
        }

        solver = FindBestSetOfTeamsLinearProgrammingYahoo(
            points_col="YAHOO_POINTS_pred",
            num_teams_in_set=3,
            avoid_single_player_in_all_sets=True,
            max_shared_players_in_teams_pairs=4,
        )
        best_set = solver.run(slate_df, locked_dfs)

        self.assertEqual(len(best_set), 3)
        self.assertEqual(list(best_set["entry_id"]), [1001, 1002, 1003])
        self.assertTrue((best_set["salary"] <= 200.0).all())
        # Locked players preserved in entry 1001
        self.assertEqual(best_set.loc[best_set["entry_id"] == 1001, "PG"].iloc[0], int(pg_Row["PLAYER_ID"]))
        self.assertEqual(best_set.loc[best_set["entry_id"] == 1001, "C"].iloc[0], int(c_Row["PLAYER_ID"]))

        # Pairwise overlap <= 4 across all pairs
        pos_cols = DFS_POSITIONS[Sites.YAHOO]
        for i in range(3):
            for j in range(i + 1, 3):
                shared = set(best_set.loc[i, pos_cols].values) & set(best_set.loc[j, pos_cols].values)
                self.assertLessEqual(len(shared), 4)


if __name__ == "__main__":
    unittest.main()
