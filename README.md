# Daily Fantasy Basketball AI Optimizer

This project is an end-to-end system for daily fantasy basketball (Yahoo DFS). Every day, it pulls the latest NBA box scores, injury reports, starting lineups, and player salaries, trains a PyTorch model to predict player fantasy points, finds the best valid lineup using integer linear programming, and automatically submits (and updates) the teams before tipoff.

## How It Works

### 1. Predicting Player Stats (`machine_learning/` & `machine_learning_fantasy_basketball_predictions/`)
A player's stats depend heavily on who else is on the court. For example, when a star sits out, their usage and rebounds get redistributed among their teammates, and facing a fast-paced or weak interior defense changes everyone's ceiling.

* **Game-Level Attention (`SplitAttentionCombine`)**: Instead of predicting each player in isolation, `BatchGamesSampler` (`samplers.py`) groups all players in a game into a `(2, 21, d)` tensor (Team A vs. Team B, padded to 21 roster slots). In `machine_learning/layers/layers.py`, `SplitAttentionCombine` runs multi-head self-attention within each team's roster and cross-attention against the opposing team's roster, so the model can learn teammate usage shifts and matchup effects directly.
* **Features & Learned Embeddings**: Combines rolling multi-window box-score stats (`calc_features.py`) with learned embeddings for `PLAYER_ID`, `TEAM_ID`, `START_POSITION_CALCULATED` (inferred starting role vs. bench), and `HOME`.
* **Multi-Task Training (`run_all.py`)**: Rather than only predicting `FANTASY_POINTS`, the model jointly predicts all 11 underlying box-score categories (`PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `FGM/FGA`, `FTM/FTA`, `FG3M/FG3A`, `DOUBLE_DOUBLE`, `TRIPLE_DOUBLE`). This gives the shared `PLAYER_ID` embeddings much richer supervision about what kind of player each person is.
* **Loss Functions (`layers.py`, `corn.py`)**: Alongside standard regression losses, the repo includes losses tailored to lineup selection:
  * `PairwiseRankingLoss` & `BradleyTerryLoss`: Pairwise ranking losses that put extra weight on salary upsets (when a cheaper player outscores a more expensive one) and same-salary comparisons.
  * `CORN` (`corn.py`): Conditional ordinal regression across 50 score buckets to model the full distribution of outcomes rather than just the mean.
  * `OptimizeTeamFn`: A decision-focused loss (SPO+) that backpropagates through the linear programming lineup solver so prediction errors are weighted by whether they actually lead to picking a worse lineup.
* **Daily Retraining**: Every morning, the model retrains on the latest games with exponential time decay (`lambda = 0.001`) and 2x sample weight on players who have a game scheduled that night.

### 2. Finding the Optimal Lineup (`find_best_team_linear_programming.py`)
Given the predicted fantasy points, we need to pick an 8-player Yahoo lineup (`PG`, `SG`, `G`, `SF`, `PF`, `F`, `C`, `UTIL`) under a 200-unit salary cap.

* **Pareto Pruning**: Before calling the solver, we drop any player who is strictly dominated by another player with the same position eligibility (equal or higher salary and equal or lower projected points). This shrinks the candidate pool by ~80%.
* **Integer Linear Programming (MILP)**: We formulate the roster rules and multi-position eligibility using binary decision variables and Kronecker-product constraint matrices, and solve for the globally optimal lineup using SciPy's `milp` (HiGHS branch-and-bound).
* **Multi-Entry Diversity**: When entering multiple contests at once, `FindBestSetOfTeamsLinearProgrammingYahoo` iteratively adds exclusion constraints to generate a diverse set of top lineups—capping how many players any two lineups can share (`max_shared_players_in_teams_pairs = 4`) and making sure no single player appears in every lineup when entering 3+ contests.

### 3. Automated Pre-Tipoff Updates & Submission (`run_full_pipeline.py`, `submit_dfs_team.py`)
NBA games start at different times throughout the evening, and late injury scratches or starting lineups often come out 15–20 minutes before a game's tipoff, after earlier games have already started.

* **Updating Before Every Game Window**: `run_full_pipeline_continuously()` wakes up 10 minutes before each distinct tipoff time on that night's schedule, scrapes the latest injury and starter news (`RotoWire`, `BasketballMonster`, and the `Underdog` NBA feed), and re-runs the model predictions.
* **Respecting Locked Players**: `SubmitDFSTeamYahooByCSV` uses headless Selenium to download Yahoo's live bulk-edit CSV template, checks which players in each contest have already locked because their game started (`get_locked_dfs()`), and passes them as fixed constraints into the MILP solver to re-optimize the remaining open slots.
* **Smart Slot Ordering (`_sort_players_by_time`)**: Before uploading the CSV back to Yahoo, players in earlier games are placed into the most specific slots (`PG`, `SG`, `SF`, `PF`, `C`), while the flexible slots (`G`, `F`, `UTIL`) are saved for players in the latest games. That way, if a late-night starter gets scratched, the solver has as much position flexibility as possible to swap players around.

## Main Entry Points
* **`run_all.py`**: Entry point for configuring model architectures, features, and training runs. Model, layer, dataset, and callback configurations are defined as dataclasses inheriting from `AbstractParams` (`machine_learning/params/abstract_params.py`) and saved as JSON files.
* **`run_full_pipeline.py`**: Runs the full daily loop—updates historical stats and retrains the model at 8:00 AM ET, then wakes up before each game window to pull live injury/starter updates, re-optimize unlocked lineup slots, and submit to Yahoo.
* **`update_stats.py`**: Standalone script to fetch schedules, team/player box scores, and calculated features.
* **`submit_dfs_team.py`**: Handles downloading the Yahoo CSV template, ordering roster slots by game time, and uploading lineups via Selenium (run `python submit_dfs_team.py --login` once to save your login session to `submitted_teams/browser_profile/`).
