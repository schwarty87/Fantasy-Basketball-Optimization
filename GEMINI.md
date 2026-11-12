# Fantasy Basketball Optimization - Assistant Guidelines & Architecture

## Core Pipeline Architecture

### Daily Model Retraining (Intentional Design)
- In `run_full_pipeline.py`, the pipeline is intentionally designed to train a fresh model every morning via `run_once_per_day()`.
- **Purpose**: Incorporate the latest game results and apply extra weights on players who have scheduled games today so the model specializes on active players.
- **Do NOT "optimize away" daily retraining**: Retraining from scratch each morning when `_todays_model_already_trained()` is false is the expected design behavior, not an issue or oversight.
- **Future Investigation**: There is an open investigation to explore fine-tuning from the previous day's checkpoint rather than retraining from scratch, but full retraining remains the default until empirical tests are completed.
- **Model Validation & Fallback Safety**:
  - `_is_valid_trained_model_dir()` validates that a model directory contains `params.json`, weights (`model.pt` or `*.ckpt`), target scalers (`scalers/true_label.pkl`), and encoders (`categorical_encoder/embedding_features.pkl`).
  - If training today's model fails or produces an incomplete directory, the failed temporary directory is deleted (never deleting base models like `models/0`), and the pipeline safely falls back to the latest valid `model_YYYY_MM_DD` (or base model `0`) while reloading its schema-matched `params.json` with `last_month_in_train = True`.
  - Redundant retraining inside `run_full_pipeline()` prior to game tip-off is omitted, preserving `run_once_per_day()` as the single retraining source of truth.

### Embeddings & Unknown Category Zero-Padding
- `CustomOrdinalEncoder` with `handle_unknown='use_encoded_value'` assigns unseen categories the integer index `len(categories)`.
- `EmbeddingsLayer` allocates `num_unique_values = len(categories) + 1` embedding rows and configures `padding_idx = len(categories)`. Unseen players or teams automatically map to frozen `0.0` vectors with zero gradients, preventing index errors.

### Feature NaN Filtering & Undrafted Rookies
- In `CalcFeatures._drop_nans()`, `YEARS_SINCE_DRAFT` is intentionally excluded from the required non-null check columns so that undrafted rookies (with `NaN` draft years) are retained in the feature set.

### Shared Stats & Feature Pipeline (`update_stats.py` & `run_full_pipeline.py`)
- **Single Source of Truth**: All stats data fetching (season schedules, team boxscores, player boxscores, and calculated features) is standardized in `update_all_stats()` in `update_stats.py`.
- Both the standalone CLI (`python update_stats.py`) and `run_full_pipeline.py` (`run_once_per_day()` and `run_full_pipeline()`) call `update_all_stats()` to ensure zero logic drift.
- **Upcoming Games Flag**:
  - `add_upcoming_games=False`: Used during 8:00 AM daily model retraining so models train only on completed games.
  - `add_upcoming_games=True`: Used right before game times in `run_full_pipeline()` to incorporate today's rosters, salaries, and matchups for prediction.
- **Season Schedule Caching & Resilience**:
  - Season schedules are static full-season files cached in `data/season_schedules/{season}.feather` with a 24-hour TTL.
  - When `stats.nba.com` times out or fails, `scrape_season_schedules()` automatically falls back to the existing cached schedule file to prevent pipeline crashes. `update_stats.py` also supports `--no-schedules` to explicitly skip schedule queries.
  - **Live Postponement & Time Shift Detection**: To detect game postponements (`PPD`) or tip-off time shifts throughout the day without re-downloading the entire 1,267-game season schedule, `update_todays_schedule_from_scoreboard()` queries NBA's lightweight daily `scoreboardv3` endpoint (~5-10 games). It automatically patches `postponedStatus`, `gameStatusText`, and `GAME_DATE_TIME` in the cached season schedule file both during the morning run and right before lineup submission. `load_schedule(drop_ppd=True)` filters out all games marked with `postponedStatus == 'Y'` or `PPD` / `Postponed`.

### Browser Profile & DFS Lineup Submission
- **Dedicated Automated Profile**: To prevent `DevToolsActivePort` / file-lock collisions when the user has desktop Chrome running, Selenium uses a dedicated automated profile at `submitted_teams/browser_profile/` (with fallback to default Chrome user data if missing).
- **Interactive Login**: Run `python submit_dfs_team.py --login` to open a visible Chrome window, log in to Yahoo once with "keep me signed in", and save cookies directly to `submitted_teams/browser_profile/`.
- **Permissions & Headless**: Selenium is configured with automation-suppression flags (`--disable-blink-features=AutomationControlled`, notification suppressions) and dedicated download directories to ensure it runs headlessly without permission prompts or popups.
- **Submitted Teams Storage**: Generated Yahoo CSV templates and submission logs are stored in `submitted_teams/` in the project root. Daily CSV files and `browser_profile/` are ignored by Git.
- **Private League Contest Filtering**: In `_get_template_df()` in `submit_dfs_team.py`, private league contests (e.g. `'DFS league 2026-27'`, `'dfs league 2025-26'`) are automatically filtered out of the downloaded template CSV by shifting subsequent contest rows up. This preserves the player pool on the right while preventing automated submissions from altering private league lineups.

### Prediction Merging & Salary Robustness (`run_full_pipeline.py`)
- **Metadata Column Collisions**: Model predictions include metadata columns (`YAHOO_POSITION`, `YAHOO_SALARY`). In `_get_todays_predictions()`, overlapping columns (`GAME_ID`, `TEAM_ID`, `YAHOO_POSITION`, `YAHOO_SALARY`) are stripped from `predictions` before merging with `stats_loaders_players.df` to prevent pandas `_x` / `_y` duplicate column suffixes.
- **Authoritative DFS Salaries & Positions**: Missing or placeholder (`-999.0` / `<= 0`) salaries and unmapped positions in `predictions` are authoritatively resolved using the downloaded Yahoo `teams_template`. Positions are mapped to integer codes (`POSITION_TO_NUM`) and unmapped fallback positions are handled gracefully.

### In-Graph Fantasy Points Unscaling, Gating & Single-Date Inference (`layers.py` & `run_full_pipeline.py`)
- **In-Graph Unscaling (`CalculateFantasyPoints`)**: Standardized model predictions (`outputs_pred`) are unscaled back to physical units using non-persistent buffers `scaler_mean` and `scaler_scale` before taking the dot product with Yahoo scoring weights. This ensures fantasy point calculations and LP lineup decisions operate on true physical stat estimates rather than arbitrary standard deviations.
- **Dynamic Daily Reshaping (`ReshapeToSamplePerDay`)**: Day batching dynamically sizes `num_days = max(num_days_per_batch, len(unique_dates))` and `max_games = max(max_num_games_per_day, max_games_in_batch)` to eliminate out-of-bounds index errors on varying slate sizes.
- **Prediction Gating & Solver Efficiency (`OptimizeTeam`)**: During standard model training, `OptimizeTeam` returns zeros without invoking LP solvers unless `run_in_train=True`. In evaluation or prediction mode (`requires_grad=False`), true-team LP solving is skipped, preventing solver overhead and zero-stat caching.
- **Single-Date Inference (`create_date_filtered_data_loader`)**: Rather than running full forward passes across entire seasons in `_get_todays_predictions()`, `create_date_filtered_data_loader([run_date])` constructs a targeted DataLoader filtered via epoch day mapping for fast ~10ms slate inference.
- **DST-Safe Scheduling (`_next_8am_et`)**: Pipeline scheduling rigorously localizes next 8:00 AM intervals via `NY_TZ.localize(datetime.datetime.combine(target_date, datetime.time(8, 0)))` to prevent daylight saving hour shifts, and cleanly terminates when `run_immediately=True` to prevent infinite busy-wait loops.

### Residual MILP Late-Swap & Multi-Entry Keying (`find_best_team_linear_programming.py` & `submit_dfs_team.py`)
- **Exact (Player, Slot, Team) Formulation**: `FindBestSetOfTeamsLinearProgrammingYahoo.run` implements an exact mixed-integer linear program using `scipy.optimize.milp` and sparse `csc_matrix`. Open slots $O_t$ and residual cap $C_t = 200.0 - \text{locked\_salary}_t$ are modeled directly without post-hoc heuristic slot filling.
- **Multi-Entry `Entry ID` Keying**: `SubmitDFSTeamYahooByCSV` keys lineup sets by `Entry ID` rather than `Contest ID`. Multiple lineups submitted into the same contest (e.g. multi-entry GPPs) preserve independent row identity without colliding or overwriting each other in the Yahoo upload template.
- **Cash vs. GPP Strategy Routing**: In `run_full_pipeline()`, cash contests (50/50, Double Up, Head-to-Head) are solved independently to maximize expected median/floor without forced diversity, whereas tournament (GPP) entries are solved jointly with pairwise overlap constraints (`max_shared_players_in_teams_pairs = 4`) and single-player set avoidance (`avoid_single_player_in_all_sets = True` when $\ge 3$).

### Masked Input Scalers & Transformer Attention Masking (`machine_learning/` & `run_all.py`)
- **Real-Player Indicator (`IS_REAL_PLAYER`)**: Identifies actual participating NBA players (`PLAYER_ID != UNKNOWN_PLAYER_ID`) to isolate them from synthetic bench-padding slots (6 per team, 12 per game). `NOT_IGNORE` strictly combines `IS_REAL_PLAYER & ~(EJECTED | LEFT_MIDGAME)`.
- **Masked Scaler Fitting & Zero Fill (`masked_fill_value`)**: `tabular` and `lagged` feature datasets configure `relevant_keep_mask_dataset_name='real_player_mask'` and `masked_fill_value=0.0`. `StandardScaler` calculates means and standard deviations strictly on active players, and fills dummy padding slots with `0.0` (normalized mean) rather than arbitrary unscaled `-999.0` sentinels.
- **Contextual Key-Padding Masking (`_build_safe_key_padding_mask`)**: Plumbs `real_player_mask` through `SplitAttentionCombineWrapper`, `SplitAttentionCombine`, and `TransformerEncoderPytorch`. Within-team attention masks dummy teammates, between-team attention masks dummy opponents (`between1` uses `m2`, `between2` uses `m1`), and global attention ignores dummy slots, with automatic row unmasking to prevent softmax `NaN`s on fully-padded batches.
- **Principled Regularization & Decoupled Weight Decay**: To prevent post-convergence overfitting once clean inputs and attention are unmasked:
  - `TrainingParams.weight_decay` (`1e-4` default) partitions model parameters into decay (2D weight tensors in linear & attention projections) and no-decay (1D biases, LayerNorm parameters, and categorical embedding tables) in `AdamW`.
  - Tighter early stopping (`patience=4`) and proactive plateau scheduling (`ReduceLearningRateOnPlateau(patience=2)`) terminate training before idiosyncratic training-set memorization occurs.
  - Increased dropout on the final dense prediction head (`0.25`) and transformer attention layers (`0.30`).

### Contest Backtest Evaluator & Sharpe Ratio (`machine_learning_fantasy_basketball_predictions/run_model.py` & `utils.py`)
- **Raw Overtime Points for Contest Backtesting**: Neural network training targets remain strictly per-48 normalized to prevent gradient descent from overfitting stochastic overtime noise. However, historical Yahoo contest leaderboards score human contestants on raw regulation + overtime boxscores. `_process_predictions_df` inner-merges raw `YAHOO_POINTS` as `YAHOO_POINTS_raw` from `player_boxscores.feather`, and `_add_extra_metrics_to_eval_results` scores LP-selected lineups on `YAHOO_POINTS_raw` to eliminate artificial 10-25% overtime penalties during contest backtesting.
- **Daily PnL Sharpe Calculation**: `utils.sharpe` computes annualized Sharpe directly on daily PnL vectors (`profit_5050`, `profit_top10`) using sample standard deviation (`ddof=1`), safely returning `0.0` on zero variance or $N < 2$, eliminating division-by-zero artifacts on cumulative account balance series. Both `sharpe_5050` and `sharpe_top10` are recorded in `eval_results` and logged to TensorBoard under `DFS_Contests/`.

### Pre-Grid Player History Features & Rolling Form (`machine_learning_fantasy_basketball_predictions/`)
- **Pre-Grid Chronological Lags (`_add_player_history_columns`)**: History features (`*_t-1` through `*_t-3`, `_mean_ratio`) are computed at the end of `_load_features()` across all completed games sorted by `['DATE', 'GAME_ID', 'TEAM_ID', 'PLAYER_ID']` before `_sort_features` truncates bench players or inserts dummy rows, avoiding dropped-game gaps or dummy-row leakage. Lags are strictly chronological per `PLAYER_ID`, using prior-lag carry-forward and per-role means as debut fallbacks.
- **Omission of Salary from Neural Net Inputs**: External DFS salaries (`YAHOO_SALARY`) are intentionally excluded from neural network feature inputs. Imputing 71k+ missing historical salaries fabricates data, and feeding betting/pricing tags to the model anchors predictions to Yahoo's algorithm, destroying the model's ability to find mispriced value. Salaries belong exclusively in the linear programming solver's budget constraint.
- **Preserved `BY_HISTOGRAM` Sample Weighting**: Retains domain-informed sample weighting on tail events (rare 50-point breakouts and duds) with mild 20-35% variance bounds (`max_class_weight=1.0`), stabilizing gradients while preserving tournament upside sensitivity.

## Data Consistency & Validation Guidelines

### Feather & PyArrow Type Strictness (Pandas 2.2+ / PyArrow 19+)
- Feather files strictly validate column types upon writing. Never mix booleans and numbers:
  - `player_boxscores.feather`:
    - `W_L`: strictly `bool`
    - `HOME`: strictly `bool`
    - `GAME_ID`, `TEAM_ID`, `PLAYER_ID`: strictly `int64`
    - Boolean indicator columns (`LEFT_MIDGAME`, `EJECTED`, `IS_OUT`, `DOUBLE_DOUBLE`, `TRIPLE_DOUBLE`, `CONFIRMED`, `IS_QUESTIONABLE`): strictly non-null `bool` (fill NaNs with `False`).
  - `team_boxscores.feather`:
    - `W_L`: strictly `int8`
    - `HOME`: strictly `int8`
    - `GAME_ID`, `TEAM_ID`: strictly `int64`

### Season 2023 Game Count (NBA Cup / In-Season Tournament)
- In `assert_valid_df_all()` in `utils.py`, completed seasons are expected to have 1230 games.
- Season 2023 is allowed to have either 1230 or 1231 games because team boxscores include the NBA In-Season Tournament Championship game (`62300001` on 2023-12-09).

### Player Name Normalization
- All suffixes in `constants.FIX_PLAYER_NAMES` must include punctuation (e.g., `'Terrence Shannon Jr.'`, `'Michael Porter Jr.'`) to maintain 1-to-1 matching with historical boxscores and avoid `assert_valid_df_players` name collisions.
