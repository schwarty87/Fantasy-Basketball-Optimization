# Fantasy Basketball Optimization - Assistant Guidelines & Architecture

## Core Pipeline Architecture

### Daily Model Retraining (Intentional Design)
- In `run_full_pipeline.py`, the pipeline is intentionally designed to train a fresh model every morning via `run_once_per_day()`.
- **Purpose**: Incorporate the latest game results and apply extra weights on players who have scheduled games today so the model specializes on active players.
- **Do NOT "optimize away" daily retraining**: Retraining from scratch each morning when `_todays_model_already_trained()` is false is the expected design behavior, not an issue or oversight.
- **Future Investigation**: There is an open investigation to explore fine-tuning from the previous day's checkpoint rather than retraining from scratch, but full retraining remains the default until empirical tests are completed.

### Shared Stats & Feature Pipeline (`update_stats.py` & `run_full_pipeline.py`)
- **Single Source of Truth**: All stats data fetching (season schedules, team boxscores, player boxscores, and calculated features) is standardized in `update_all_stats()` in `update_stats.py`.
- Both the standalone CLI (`python update_stats.py`) and `run_full_pipeline.py` (`run_once_per_day()` and `run_full_pipeline()`) call `update_all_stats()` to ensure zero logic drift.
- **Upcoming Games Flag**:
  - `add_upcoming_games=False`: Used during 8:00 AM daily model retraining so models train only on completed games.
  - `add_upcoming_games=True`: Used right before game times in `run_full_pipeline()` to incorporate today's rosters, salaries, and matchups for prediction.

### Browser Profile & DFS Lineup Submission
- **Dedicated Automated Profile**: To prevent `DevToolsActivePort` / file-lock collisions when the user has desktop Chrome running, Selenium uses a dedicated automated profile at `submitted_teams/browser_profile/` (with fallback to default Chrome user data if missing).
- **Interactive Login**: Run `python submit_dfs_team.py --login` to open a visible Chrome window, log in to Yahoo once with "keep me signed in", and save cookies directly to `submitted_teams/browser_profile/`.
- **Permissions & Headless**: Selenium is configured with automation-suppression flags (`--disable-blink-features=AutomationControlled`, notification suppressions) and dedicated download directories to ensure it runs headlessly without permission prompts or popups.
- **Submitted Teams Storage**: Generated Yahoo CSV templates and submission logs are stored in `submitted_teams/` in the project root. Daily CSV files and `browser_profile/` are ignored by Git.

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
