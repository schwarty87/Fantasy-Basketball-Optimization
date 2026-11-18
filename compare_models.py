import argparse
import os
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from data_collectors.stats_loaders import StatsLoaderPlayers
from machine_learning.utils import models_dir
from machine_learning_fantasy_basketball_predictions.constants import PROJECT_NAME
from machine_learning_fantasy_basketball_predictions.run_model import RunModelFantasyBasketballPredictions
from utils import convert_date_col_to_date

Z_TWO_SIDED_005_POWER_80 = 1.959963984540054 + 0.8416212335729143  # 2.801585


def resolve_model_dir(model_ref: str | int, project_name: str = PROJECT_NAME) -> str:
    """Resolve a model ID or directory path to a validated model directory."""
    ref_str = str(model_ref)
    if os.path.isdir(ref_str):
        return ref_str
    candidate = os.path.join(models_dir(project_name), ref_str)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"Could not find model directory for {model_ref!r} at {candidate}")


def filter_predictions_split(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Return a copy of predictions filtered to the target split (e.g. VALIDATION or TEST)."""
    out = RunModelFantasyBasketballPredictions._normalize_predictions_df(df)
    if split is None or 'split_type' not in out.columns:
        return out
    target = split.strip().upper()
    valid = {target, target.lower()}
    return out.loc[out['split_type'].astype(str).isin(valid)].copy().reset_index(drop=True)


def load_predict_results(
    model_ref: str | int,
    split: str | None = None,
    project_name: str = PROJECT_NAME,
) -> pd.DataFrame:
    """Load saved predict_results.feather for a model ID or path."""
    mdir = resolve_model_dir(model_ref, project_name=project_name)
    feather_path = os.path.join(mdir, 'predict_results.feather')
    if not os.path.isfile(feather_path):
        raise FileNotFoundError(f"Missing predict_results.feather in {mdir}")
    df = pd.read_feather(feather_path)
    return filter_predictions_split(df, split=split)


def average_prediction_dfs(dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average physical *_pred columns across seed prediction DataFrames aligned on DATE and PLAYER_ID."""
    if not dfs:
        raise ValueError("average_prediction_dfs requires at least one prediction DataFrame")
    if len(dfs) == 1:
        return dfs[0].copy()

    sort_keys = [k for k in ('DATE', 'PLAYER_ID', 'split_type') if k in dfs[0].columns]
    prepared: List[pd.DataFrame] = []
    ref_keys: pd.DataFrame | None = None

    for idx, df in enumerate(dfs):
        curr = df.copy()
        if 'DATE' in curr.columns:
            curr['DATE'] = convert_date_col_to_date(curr['DATE'])
        if 'PLAYER_ID' in curr.columns:
            curr['PLAYER_ID'] = curr['PLAYER_ID'].astype(int)
        if sort_keys:
            curr = curr.sort_values(sort_keys, kind='mergesort').reset_index(drop=True)
        else:
            curr = curr.reset_index(drop=True)

        if ref_keys is None:
            ref_keys = curr[sort_keys] if sort_keys else pd.DataFrame(index=curr.index)
        else:
            if len(curr) != len(ref_keys):
                raise ValueError(f"Prediction DataFrame {idx} row count {len(curr)} != {len(ref_keys)}")
            if sort_keys and not curr[sort_keys].equals(ref_keys):
                raise ValueError(f"Prediction DataFrame {idx} does not align with reference on keys {sort_keys}")
        prepared.append(curr)

    out = prepared[0].copy()
    pred_cols = [
        c for c in out.columns
        if c.endswith('_pred') and pd.api.types.is_numeric_dtype(out[c])
    ]
    for col in pred_cols:
        if all(col in p.columns for p in prepared):
            stacked = np.stack([p[col].to_numpy(dtype=np.float64) for p in prepared], axis=0)
            out[col] = np.mean(stacked, axis=0).astype(out[col].dtype if np.issubdtype(out[col].dtype, np.floating) else np.float64)

    return out


def compute_mde(se: float, factor: float = Z_TWO_SIDED_005_POWER_80) -> float:
    """Minimum Detectable Effect at 80% power, two-sided alpha=0.05."""
    if not np.isfinite(se):
        return float('nan')
    return float(factor * se)


def paired_bootstrap_test(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Run a paired two-sided bootstrap significance test for (b - a)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Paired arrays must have the same shape, got {a_arr.shape} vs {b_arr.shape}")

    finite = np.isfinite(a_arr) & np.isfinite(b_arr)
    diffs = b_arr[finite] - a_arr[finite]
    n = int(diffs.size)
    if n == 0:
        return {
            'n': 0,
            'mean_a': float('nan'),
            'mean_b': float('nan'),
            'mean_diff': float('nan'),
            'se': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'p_value': float('nan'),
            'mde_80': float('nan'),
        }

    mean_a = float(np.mean(a_arr[finite]))
    mean_b = float(np.mean(b_arr[finite]))
    mean_diff = float(np.mean(diffs))
    if n < 2:
        return {
            'n': n,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'se': float('nan'),
            'ci_low': mean_diff,
            'ci_high': mean_diff,
            'p_value': 1.0,
            'mde_80': float('nan'),
        }

    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    if se == 0.0:
        p_val = 1.0 if mean_diff == 0.0 else 0.0
        return {
            'n': n,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'mean_diff': mean_diff,
            'se': 0.0,
            'ci_low': mean_diff,
            'ci_high': mean_diff,
            'p_value': p_val,
            'mde_80': 0.0,
        }

    rng = np.random.default_rng(seed=seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[boot_indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    centered_boot = boot_means - mean_diff
    p_value = float(np.mean(np.abs(centered_boot) >= abs(mean_diff)))

    return {
        'n': n,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'mean_diff': mean_diff,
        'se': se,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': p_value,
        'mde_80': compute_mde(se),
    }


def _aggregate_daily_contest_series(ppc: pd.DataFrame, paid_only: bool) -> pd.DataFrame:
    base_cols = ['percentile', 'profit_5050', 'profit_top10_9x']
    if ppc.empty:
        return pd.DataFrame(columns=['DATE'] + base_cols)
    subset = ppc.loc[ppc['entry_fee'].notnull() & (ppc['entry_fee'] > 0)] if paid_only else ppc
    if subset.empty:
        return pd.DataFrame(columns=['DATE'] + base_cols)
    agg_cols = base_cols + [c for c in ('percentile_cash', 'percentile_gpp') if c in subset.columns]
    return (
        subset.groupby('DATE', as_index=False)[agg_cols]
        .mean()
    )


def _pair_and_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_col: str,
    val_col: str,
    n_resamples: int,
    seed: int,
) -> Dict[str, float]:
    if df_a.empty or df_b.empty:
        return paired_bootstrap_test([], [], n_resamples=n_resamples, seed=seed)
    merged = df_a[[key_col, val_col]].merge(
        df_b[[key_col, val_col]],
        on=key_col,
        how='inner',
        suffixes=('_a', '_b'),
    )
    return paired_bootstrap_test(
        merged[f'{val_col}_a'].to_numpy(dtype=np.float64),
        merged[f'{val_col}_b'].to_numpy(dtype=np.float64),
        n_resamples=n_resamples,
        seed=seed,
    )


def evaluate_model_group(
    model_refs: Sequence[str | int],
    split: str = 'VALIDATION',
    include_ensemble: bool = False,
    contests_df: pd.DataFrame | None = None,
    player_stats_df: pd.DataFrame | None = None,
    project_name: str = PROJECT_NAME,
    alpha_cash: float = 0.35,
    alpha_gpp: float = 0.85,
) -> Dict[str, Any]:
    """Re-evaluate saved model predictions via the canonical DFS evaluator and aggregate across seeds."""
    split_upper = split.strip().upper()
    if contests_df is None:
        contests_df = RunModelFantasyBasketballPredictions._load_contests_df()
    if player_stats_df is None:
        player_stats_df = StatsLoaderPlayers().load_saved_data()

    dfs: List[pd.DataFrame] = []
    seed_evals: List[pd.DataFrame] = []
    seed_details: List[Dict[str, pd.DataFrame]] = []

    for ref in model_refs:
        df = load_predict_results(ref, split=split_upper, project_name=project_name)
        dfs.append(df)
        ev, det = RunModelFantasyBasketballPredictions.evaluate_predictions_df(
            predict_results=df,
            contests_df=contests_df,
            player_stats_df=player_stats_df,
            return_details=True,
            alpha_cash=alpha_cash,
            alpha_gpp=alpha_gpp,
        )
        seed_evals.append(ev)
        seed_details.append(det)

    scalar_rows = []
    for ref, ev in zip(model_refs, seed_evals):
        row = {}
        if split_upper in ev.index:
            for col, val in ev.loc[split_upper].items():
                if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool):
                    row[col] = float(val)
        row['model_ref'] = str(ref)
        scalar_rows.append(row)
    summary_df = pd.DataFrame(scalar_rows)

    ensemble_eval = None
    ensemble_details = None
    if include_ensemble and len(dfs) > 1:
        ens_df = average_prediction_dfs(dfs)
        ensemble_eval, ensemble_details = RunModelFantasyBasketballPredictions.evaluate_predictions_df(
            predict_results=ens_df,
            contests_df=contests_df,
            player_stats_df=player_stats_df,
            return_details=True,
            alpha_cash=alpha_cash,
            alpha_gpp=alpha_gpp,
        )

    return {
        'model_refs': [str(r) for r in model_refs],
        'split': split_upper,
        'seed_evals': seed_evals,
        'seed_details': seed_details,
        'summary_df': summary_df,
        'ensemble_eval': ensemble_eval,
        'ensemble_details': ensemble_details,
    }


def _average_seed_details(seed_details: Sequence[Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """Average per-date / per-contest detail outcomes across multiple seeds of an arm."""
    if len(seed_details) == 1:
        return seed_details[0]

    score_cols = ['predicted_score'] + [
        c for c in ('predicted_score_cash', 'predicted_score_gpp')
        if all(c in d['all_dates_scores'].columns for d in seed_details)
    ]
    all_scores = pd.concat(
        [d['all_dates_scores'][['DATE'] + score_cols] for d in seed_details],
        ignore_index=True,
    )
    mean_scores = all_scores.groupby('DATE', as_index=False)[score_cols].mean()

    opt_ppc_cols = [
        c for c in ('percentile_cash', 'percentile_gpp', 'predicted_score_cash', 'predicted_score_gpp')
        if all(c in d['percentile_per_contest'].columns for d in seed_details)
    ]
    ppc_mean_cols = ['percentile', 'profit_5050', 'profit_top10_9x'] + opt_ppc_cols
    ppc_cols = ['contestId', 'DATE', 'entry_fee'] + ppc_mean_cols
    ppc_concat = pd.concat(
        [d['percentile_per_contest'][ppc_cols] for d in seed_details],
        ignore_index=True,
    )
    ppc_agg: Dict[str, str] = {'entry_fee': 'first'}
    ppc_agg.update({c: 'mean' for c in ppc_mean_cols})
    mean_ppc = ppc_concat.groupby(['contestId', 'DATE'], as_index=False).agg(ppc_agg)
    return {'all_dates_scores': mean_scores, 'percentile_per_contest': mean_ppc}


def _select_bootstrap_col(df: pd.DataFrame, preferred_col: str, fallback_col: str) -> pd.DataFrame:
    col = preferred_col if preferred_col in df.columns else fallback_col
    return df.rename(columns={col: '_boot_val'}) if col != '_boot_val' else df


def compare_model_groups(
    models_a: Sequence[str | int],
    models_b: Sequence[str | int],
    split: str = 'VALIDATION',
    include_ensemble: bool = False,
    n_resamples: int = 5000,
    seed: int = 42,
    project_name: str = PROJECT_NAME,
    alpha_cash: float = 0.35,
    alpha_gpp: float = 0.85,
) -> Dict[str, Any]:
    """Evaluate and statistically compare two sets of models (e.g. A vs. B)."""
    contests_df = RunModelFantasyBasketballPredictions._load_contests_df()
    player_stats_df = StatsLoaderPlayers().load_saved_data()

    res_a = evaluate_model_group(
        models_a, split=split, include_ensemble=include_ensemble,
        contests_df=contests_df, player_stats_df=player_stats_df, project_name=project_name,
        alpha_cash=alpha_cash, alpha_gpp=alpha_gpp,
    )
    res_b = evaluate_model_group(
        models_b, split=split, include_ensemble=include_ensemble,
        contests_df=contests_df, player_stats_df=player_stats_df, project_name=project_name,
        alpha_cash=alpha_cash, alpha_gpp=alpha_gpp,
    )

    det_a = (
        res_a['ensemble_details']
        if include_ensemble and res_a['ensemble_details'] is not None
        else _average_seed_details(res_a['seed_details'])
    )
    det_b = (
        res_b['ensemble_details']
        if include_ensemble and res_b['ensemble_details'] is not None
        else _average_seed_details(res_b['seed_details'])
    )

    ppc_a = det_a['percentile_per_contest']
    ppc_b = det_b['percentile_per_contest']

    daily_all_a = _aggregate_daily_contest_series(ppc_a, paid_only=False)
    daily_all_b = _aggregate_daily_contest_series(ppc_b, paid_only=False)
    daily_paid_a = _aggregate_daily_contest_series(ppc_a, paid_only=True)
    daily_paid_b = _aggregate_daily_contest_series(ppc_b, paid_only=True)

    tests = {
        'daily_lineup_score': _pair_and_bootstrap(
            det_a['all_dates_scores'], det_b['all_dates_scores'], 'DATE', 'predicted_score', n_resamples, seed
        ),
        'daily_percentile_all': _pair_and_bootstrap(
            daily_all_a, daily_all_b, 'DATE', 'percentile', n_resamples, seed
        ),
        'daily_percentile_paid': _pair_and_bootstrap(
            daily_paid_a, daily_paid_b, 'DATE', 'percentile', n_resamples, seed
        ),
        'daily_profit_5050_all': _pair_and_bootstrap(
            daily_all_a, daily_all_b, 'DATE', 'profit_5050', n_resamples, seed
        ),
        'daily_profit_5050_paid': _pair_and_bootstrap(
            daily_paid_a, daily_paid_b, 'DATE', 'profit_5050', n_resamples, seed
        ),
        'daily_profit_top10_9x_all': _pair_and_bootstrap(
            daily_all_a, daily_all_b, 'DATE', 'profit_top10_9x', n_resamples, seed
        ),
        'daily_profit_top10_9x_paid': _pair_and_bootstrap(
            daily_paid_a, daily_paid_b, 'DATE', 'profit_top10_9x', n_resamples, seed
        ),
    }

    if 'predicted_score_cash' in det_a['all_dates_scores'].columns or 'predicted_score_cash' in det_b['all_dates_scores'].columns:
        tests['daily_lineup_score_cash'] = _pair_and_bootstrap(
            _select_bootstrap_col(det_a['all_dates_scores'], 'predicted_score_cash', 'predicted_score'),
            _select_bootstrap_col(det_b['all_dates_scores'], 'predicted_score_cash', 'predicted_score'),
            'DATE', '_boot_val', n_resamples, seed,
        )
    if 'predicted_score_gpp' in det_a['all_dates_scores'].columns or 'predicted_score_gpp' in det_b['all_dates_scores'].columns:
        tests['daily_lineup_score_gpp'] = _pair_and_bootstrap(
            _select_bootstrap_col(det_a['all_dates_scores'], 'predicted_score_gpp', 'predicted_score'),
            _select_bootstrap_col(det_b['all_dates_scores'], 'predicted_score_gpp', 'predicted_score'),
            'DATE', '_boot_val', n_resamples, seed,
        )
    if 'percentile_cash' in daily_all_a.columns or 'percentile_cash' in daily_all_b.columns:
        tests['daily_percentile_cash_all'] = _pair_and_bootstrap(
            _select_bootstrap_col(daily_all_a, 'percentile_cash', 'percentile'),
            _select_bootstrap_col(daily_all_b, 'percentile_cash', 'percentile'),
            'DATE', '_boot_val', n_resamples, seed,
        )
        tests['daily_percentile_cash_paid'] = _pair_and_bootstrap(
            _select_bootstrap_col(daily_paid_a, 'percentile_cash', 'percentile'),
            _select_bootstrap_col(daily_paid_b, 'percentile_cash', 'percentile'),
            'DATE', '_boot_val', n_resamples, seed,
        )
    if 'percentile_gpp' in daily_all_a.columns or 'percentile_gpp' in daily_all_b.columns:
        tests['daily_percentile_gpp_all'] = _pair_and_bootstrap(
            _select_bootstrap_col(daily_all_a, 'percentile_gpp', 'percentile'),
            _select_bootstrap_col(daily_all_b, 'percentile_gpp', 'percentile'),
            'DATE', '_boot_val', n_resamples, seed,
        )
        tests['daily_percentile_gpp_paid'] = _pair_and_bootstrap(
            _select_bootstrap_col(daily_paid_a, 'percentile_gpp', 'percentile'),
            _select_bootstrap_col(daily_paid_b, 'percentile_gpp', 'percentile'),
            'DATE', '_boot_val', n_resamples, seed,
        )

    return {'group_a': res_a, 'group_b': res_b, 'bootstrap_tests': tests}


def _format_comparison_report(comparison: Dict[str, Any]) -> str:
    lines: List[str] = []
    ga = comparison['group_a']
    gb = comparison['group_b']
    split = ga['split']
    lines.append(f"=== Model Comparison ({split}) ===")
    lines.append(f"Group A: {ga['model_refs']} | Group B: {gb['model_refs']}")

    key_metrics = [
        'pearson_corr', 'spearman_corr', 'daily_spearman_corr',
        'fp_rmse_raw', 'fp_bias_raw', 'fp_bias_sal_le_15', 'fp_bias_sal_16_30', 'fp_bias_sal_gt_30',
        'q20_coverage', 'q80_coverage', 'q20_coverage_sal_le_12', 'q80_coverage_sal_le_12', 'quantile_crossing_rate',
        'min_mae_sal', 'n_salaried_players', 'n_dropped_salaried_players',
        'mean_lineup_score_all_dates', 'mean_lineup_score_cash_all_dates', 'mean_lineup_score_gpp_all_dates',
        'mean_percentile_all', 'mean_percentile_cash_all', 'mean_percentile_gpp_all',
        'top10_pct_all', 'mean_profit_5050_all', 'sharpe_5050_all',
        'mean_profit_top10_9x_all', 'sharpe_top10_9x_all',
        'mean_percentile_paid', 'mean_percentile_cash_paid', 'mean_percentile_gpp_paid',
        'top10_pct_paid', 'mean_profit_5050_paid', 'sharpe_5050_paid',
        'mean_profit_top10_9x_paid', 'sharpe_top10_9x_paid',
    ]

    has_ens_a = ga.get('ensemble_eval') is not None and split in ga['ensemble_eval'].index
    has_ens_b = gb.get('ensemble_eval') is not None and split in gb['ensemble_eval'].index

    lines.append("\n--- Group Summary (Mean across seeds; +/- SE if >=2 seeds) ---")
    header = f"{'Metric':<32} | {'Group A':>18} | {'Group B':>18}"
    if has_ens_a:
        header += f" | {'Ens A':>10}"
    if has_ens_b:
        header += f" | {'Ens B':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    df_a, df_b = ga['summary_df'], gb['summary_df']

    def _fmt_col(df: pd.DataFrame, col: str) -> str:
        if col not in df.columns or df[col].dropna().empty:
            return "—"
        vals = df[col].dropna().to_numpy(dtype=np.float64)
        mean = float(np.mean(vals))
        if len(vals) >= 2:
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            return f"{mean:.4f} ± {se:.4f}"
        return f"{mean:.4f}"

    def _fmt_ens(ens_eval: pd.DataFrame | None, col: str) -> str:
        if ens_eval is None or split not in ens_eval.index or col not in ens_eval.columns:
            return "—"
        val = ens_eval.loc[split, col]
        return f"{float(val):.4f}" if pd.notna(val) else "—"

    for m in key_metrics:
        row_str = f"{m:<32} | {_fmt_col(df_a, m):>18} | {_fmt_col(df_b, m):>18}"
        if has_ens_a:
            row_str += f" | {_fmt_ens(ga['ensemble_eval'], m):>10}"
        if has_ens_b:
            row_str += f" | {_fmt_ens(gb['ensemble_eval'], m):>10}"
        lines.append(row_str)

    lines.append("\n--- Paired Bootstrap (B - A) ---")
    lines.append(f"{'Series':<28} | {'N':>6} | {'Mean A':>8} | {'Mean B':>8} | {'Diff':>8} | {'SE':>7} | {'p':>7} | {'MDE80':>7}")
    lines.append("-" * 96)
    for name, stats in comparison['bootstrap_tests'].items():
        lines.append(
            f"{name:<28} | {stats['n']:>6d} | {stats['mean_a']:>8.4f} | {stats['mean_b']:>8.4f} | "
            f"{stats['mean_diff']:>+8.4f} | {stats['se']:>7.4f} | {stats['p_value']:>7.4f} | {stats['mde_80']:>7.4f}"
        )

    return "\n".join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare Fantasy Basketball Models using canonical DFS metrics and paired bootstrap.")
    parser.add_argument('--models-a', nargs='+', required=True, help="Baseline model IDs or directories")
    parser.add_argument('--models-b', nargs='+', required=True, help="Candidate model IDs or directories")
    parser.add_argument('--split', choices=['VALIDATION', 'TEST'], default='VALIDATION', help="Data split to evaluate (default: VALIDATION)")
    parser.add_argument('--ensemble', action='store_true', help="Also average physical predictions within each multi-seed group")
    parser.add_argument('--bootstrap-resamples', type=int, default=5000, help="Bootstrap resample count (default: 5000)")
    parser.add_argument('--seed', type=int, default=42, help="Bootstrap RNG seed (default: 42)")
    parser.add_argument('--alpha-cash', type=float, default=0.35, help="Blend weight on q20 for Cash contests (default: 0.35)")
    parser.add_argument('--alpha-gpp', type=float, default=0.85, help="Blend weight on q80 for GPP contests (default: 0.85)")
    args = parser.parse_args()

    result = compare_model_groups(
        models_a=args.models_a,
        models_b=args.models_b,
        split=args.split,
        include_ensemble=args.ensemble,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
        alpha_cash=args.alpha_cash,
        alpha_gpp=args.alpha_gpp,
    )
    print(_format_comparison_report(result))
