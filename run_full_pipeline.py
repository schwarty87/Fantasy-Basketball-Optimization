import datetime
import logging
import os
import shutil
import time
from typing import Any
import pandas as pd
import numpy as np
from datetime import timedelta
from data_collectors.stats_loaders import StatsLoaderPlayers, StatsLoaderTeams
from calc_features import CalcFeatures
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions, get_current_time, define_logger, load_schedule, \
    get_current_date, load_and_cache_players_list, scrape_season_schedules, get_next_game_time, update_dir_name, DATA_DIR, \
    get_current_season, add_game_times_to_df, update_todays_schedule_from_scoreboard
from submit_dfs_team import SubmitDFSTeamYahooByCSV
from update_stats import update_all_stats
from constants import NY_TZ, DFS_POSITIONS, BROWSER, Sites, POSITION_TO_NUM, get_browser_profile_path
from enums import Browsers
from machine_learning_fantasy_basketball_predictions.params.params import \
    MachineLearningParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.run_model import RunModelFantasyBasketballPredictions
from machine_learning.enums import DataSplitType


class RunFullPipeline:
    def __init__(self, load_params_from_model_id: int = 0, season: int | None = None,
                 run_date: datetime.date | str = get_current_date(),
                 headless: bool = True, minutes_before_game_to_run: int = 10,
                 params_update_dict: dict | None = None, run_immediately: bool = False,
                 initialize_daily_job: bool = True, force_retrain: bool = False):
        """
        Run everything we need to continuously update stats, train new models, predict/create teams and submit them
        continuously
        :param load_params_from_model_id: Use the params from this model to train the new model
        :param season: The season that we're running
        :param run_date: the date to run (should always be today's date unless debugging)
        :param headless: whether to run selenium in headless mode or not
        :param minutes_before_game_to_run: the amount of time we want to run the pipeline before each game
        :param run_immediately: if True, don't wait for the game times to run. Should only be true when debugging
        :param params_update_dict: data to update params with
        :param initialize_daily_job: if True, initialize today's schedule and daily run via _reset()
        :param force_retrain: if True, retrain today's model even if already trained
        """
        define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO,
                      filename=os.path.join(DATA_DIR, 'logs', f"logs_{get_current_date()}.txt"), print_to_console=True, filemode='a')

        self.season = season or get_current_season()
        self.headless = headless
        self.minutes_before_game_to_run = minutes_before_game_to_run
        self._run_immediately = run_immediately
        self.force_retrain = force_retrain

        models_base = os.path.join(DATA_DIR, 'machine_learning_fantasy_basketball_predictions', 'models')
        if not os.path.isdir(os.path.join(models_base, str(load_params_from_model_id))):
            logging.warning(f"Model ID {load_params_from_model_id} directory not found in {models_base}. Falling back to model 0.")
            load_params_from_model_id = 0
        self._load_params_from_model_id = load_params_from_model_id

        self.dates_to_game_times = dict()   #self._get_game_times_by_day_dict()

        self._model_params_update_dict = params_update_dict.copy() if params_update_dict is not None else dict()
        if 'training' not in self._model_params_update_dict:
            self._model_params_update_dict['training'] = dict()
        self._model_params_update_dict['training']['run_extra_callbacks'] = False

        self.stats_loaders_players = StatsLoaderPlayers(season=self.season)
        self.stats_loaders_teams = StatsLoaderTeams(season=self.season)
        self.calc_features = CalcFeatures()

        self._run_date = self._resolve_run_date(run_date)

        define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO,
                      filename=os.path.join(DATA_DIR, 'logs', f"logs_{self._run_date}.txt"), print_to_console=True, filemode='a')
        self._exclude_contest_ids = set()

        self.train_nn = None
        if initialize_daily_job:
            self._reset(run_date)

        self.submit_teams = None
        # self.google_drive_driver = None    # driver for reading data from Google Drive

    @staticmethod
    def _resolve_run_date(run_date: datetime.date | str) -> datetime.date:
        """
        Parses run_date into a datetime.date object.
        Supports datetime.date instances, the string 'today', or 'YYYY-MM-DD' strings.
        """
        if isinstance(run_date, datetime.date):
            return run_date
        if isinstance(run_date, str):
            if run_date == 'today':
                return get_current_date()
            try:
                return datetime.date.fromisoformat(run_date)
            except ValueError:
                pass
        logging.error(f'Invalid run_date: {run_date!r}. Expected datetime.date, "today", or "YYYY-MM-DD". Defaulting to today.')
        return get_current_date()

    def _reset(self, run_date: datetime.date | str):
        """
        Should be called at the beginning of each run, and every time we start a new day. Calculate all the relevant
        info for the current day. We get the new model name that we want to run, and (re)initialize the
        TrainNeuralNetwork object. In addition, we'll calculate the times that we want to run the predictions/team
        creation, and reinitialize the exclude_contests list. Finally, we'll run run_once_per_day to collect the
        previous day's results.
        :param run_date: The date that we're running now
        """
        self._run_date = self._resolve_run_date(run_date)

        define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO,
                      filename=os.path.join(DATA_DIR, 'logs', f"logs_{self._run_date}.txt"), print_to_console=True, filemode='a')

        today_8am = datetime.datetime(year=self._run_date.year, month=self._run_date.month, day=self._run_date.day,
                                      hour=8, minute=0, second=0)
        today_8am = today_8am.astimezone(NY_TZ)
        self._exclude_contest_ids = set()

        if not self._run_immediately:
            self._wait_until_certain_time(wait_until=today_8am)
        self.run_once_per_day()

    @staticmethod
    def _wait_until_certain_time(wait_until: datetime.datetime):
        current_time = get_current_time()
        if wait_until.tzinfo is None and current_time.tzinfo is not None:
            wait_until = wait_until.replace(tzinfo=current_time.tzinfo)
        elif wait_until.tzinfo is not None and current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=wait_until.tzinfo)
        wait_seconds = (wait_until - current_time).total_seconds()
        if wait_seconds > 0:
            logging.info(f'sleeping until {wait_until}')
            time.sleep(wait_seconds)

    def _todays_model_already_trained(self, date: datetime.date | str | None = None) -> bool:
        target_date = date or self._run_date
        todays_model_dir = os.path.join(
            DATA_DIR, 'machine_learning_fantasy_basketball_predictions', 'models',
            f'model_{str(target_date).replace("-", "_")}'
        )
        if not os.path.isdir(todays_model_dir):
            return False
        return os.path.isfile(os.path.join(todays_model_dir, 'model.pt')) or any(
            f.endswith('.ckpt') for f in os.listdir(todays_model_dir)
        )

    def _has_games_today(self):
        return self._run_date in self.dates_to_game_times.keys()

    @staticmethod
    def _add_game_times_to_df(df: pd.DataFrame, schedule: pd.DataFrame | None = None) -> pd.DataFrame:
        return add_game_times_to_df(df, schedule=schedule)

    @staticmethod
    def _parse_position(val: Any) -> int | float:
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            first_p = val.replace('/', ',').split(',')[0].strip()
            return POSITION_TO_NUM.get(first_p, POSITION_TO_NUM.get(val, np.nan))
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val
        return np.nan

    def _get_todays_predictions(self, teams_template: pd.DataFrame | None = None) -> pd.DataFrame:
        metrics_params = self.train_nn._change_metrics_params_tuple_to_per_sample(self.train_nn.params.training_params.metrics_params)
        loss_params = self.train_nn._change_metrics_params_tuple_to_per_sample(self.train_nn.params.training_params.loss_params)
        predictions = self.train_nn.predict(data_split_type=DataSplitType.TRAIN, metrics_params=metrics_params,
                                            loss_params=loss_params)
        predictions.rename(columns=lambda x: x.replace('CollectTruePredictedOutputsParamsFantasyBasketballPredictions_', ''),
                           inplace=True)
        predictions = predictions.loc[predictions['DATE'] == self._run_date].copy()

        # Drop overlapping columns in predictions before merging from stats_loaders_players to prevent _x/_y collisions
        for col in ['GAME_ID', 'TEAM_ID', 'YAHOO_POSITION', 'YAHOO_SALARY']:
            if col in predictions.columns:
                predictions.drop(columns=[col], inplace=True)

        # Merge GAME_ID, TEAM_ID (and YAHOO_POSITION, YAHOO_SALARY if available) from stats_loaders_players
        merge_cols = ['PLAYER_ID', 'DATE']
        for col in ['GAME_ID', 'TEAM_ID', 'YAHOO_POSITION', 'YAHOO_SALARY']:
            if col in self.stats_loaders_players.df.columns:
                merge_cols.append(col)

        players_df_today = self.stats_loaders_players.df[merge_cols].drop_duplicates(subset=['PLAYER_ID', 'DATE'])
        predictions = predictions.merge(players_df_today, on=['DATE', 'PLAYER_ID'], how='left')

        # Ensure YAHOO_SALARY and YAHOO_POSITION exist in predictions
        if 'YAHOO_SALARY' not in predictions.columns:
            predictions['YAHOO_SALARY'] = np.nan
        if 'YAHOO_POSITION' not in predictions.columns:
            predictions['YAHOO_POSITION'] = np.nan

        # Treat placeholder/negative values (e.g. IGNORE=-999) as missing
        predictions.loc[predictions['YAHOO_SALARY'] <= 0, 'YAHOO_SALARY'] = np.nan

        logging.info(f'number of players after running predictions: {predictions.shape[0]}')

        if teams_template is not None:
            player_name_to_id = self.stats_loaders_players.get_player_name_to_id()
            player_id_to_name = {player_id: player_name for player_name, player_id in player_name_to_id.items()}
            prediction_names = predictions['PLAYER_ID'].map(player_id_to_name)
            in_template = prediction_names.isin(teams_template['Name'].values)
            predictions = predictions.loc[in_template].reset_index(drop=True)

            # filling fantasy salaries and positions using template as authoritative source
            name_to_salary = dict(zip(teams_template['Name'], teams_template['Salary']))
            id_to_salary = {
                pid: name_to_salary.get(player_id_to_name.get(pid))
                for pid in predictions['PLAYER_ID'].values
                if player_id_to_name.get(pid) in name_to_salary
            }
            mapped_salaries = predictions['PLAYER_ID'].map(id_to_salary)
            predictions['YAHOO_SALARY'] = mapped_salaries.fillna(predictions['YAHOO_SALARY']).astype(float)

            name_to_position = dict(zip(teams_template['Name'], teams_template['Position']))
            id_to_position = {
                pid: name_to_position.get(player_id_to_name.get(pid))
                for pid in predictions['PLAYER_ID'].values
                if player_id_to_name.get(pid) in name_to_position
            }
            mapped_positions = predictions['PLAYER_ID'].map(id_to_position)
            predictions['YAHOO_POSITION'] = (
                mapped_positions.fillna(predictions['YAHOO_POSITION'])
                .apply(self._parse_position)
                .fillna(1)
                .astype(int)
            )

            logging.info(f'number of players after filtering with players in template: {predictions.shape[0]}')

        add_missing_points_columns_to_predictions(predictions_df=predictions)
        predictions = self._add_game_times_to_df(predictions)

        # dropping players whose game time already passed
        cutoff_time = get_next_game_time(template=teams_template)
        if pd.notna(cutoff_time):
            predictions = predictions.loc[predictions['GAME_DATE_TIME'] >= cutoff_time]

        logging.info(f'number of players after filtering with cutoff time: {predictions.shape[0]}')

        return predictions

    def run_once_per_day(self):
        scrape_season_schedules(self.season, save_dir=update_dir_name('season_schedules'))
        update_todays_schedule_from_scoreboard(date=self._run_date, season=self.season)

        self.dates_to_game_times = self._get_game_times_by_day_dict()

        # getting times when we want to run the predictions/team creation etc.
        self._todays_game_times = self.dates_to_game_times.get(self._run_date)
        if self._run_date in self.dates_to_game_times.keys():
            self._run_times = [t - timedelta(minutes=self.minutes_before_game_to_run) for t in self._todays_game_times]
        else:
            self._run_times = None

        self.players_list = load_and_cache_players_list(season=self.season)
        update_all_stats(
            season=self.season,
            add_upcoming_games=False,
            stats_loader_players=self.stats_loaders_players,
            stats_loader_teams=self.stats_loaders_teams,
            calc_features=self.calc_features,
            update_schedules=False,
            players_list=self.players_list,
        )

        # Daily Model Retraining Architecture:
        # Every morning the pipeline runs once to collect the previous day's results and
        # train a brand new model incorporating the latest data (and extra weights on today's active players).
        # TODO: Future investigation: evaluate fine-tuning an existing checkpoint vs. full daily retrain.
        params_path = os.path.join(DATA_DIR, f'machine_learning_fantasy_basketball_predictions/models/{self._load_params_from_model_id}/params.json')
        params = MachineLearningParamsFantasyBasketballPredictions.load_params(params_path)
        params.training_params.fast_dev_run = False
        params.features_params.last_month_in_train = True
        todays_model_name = 'model_' + str(self._run_date).replace('-', '_')
        already_trained = self._todays_model_already_trained()

        if (not already_trained or self.force_retrain) and self._has_games_today():
            logging.info(f"Training fresh model for today ({todays_model_name})...")
            self.train_nn = RunModelFantasyBasketballPredictions(params=params, description=None)
            self.train_nn.train()

            # change the model name from the model id to today's date
            model_dir = self.train_nn._model_dir
            target_dir = os.path.join(os.path.dirname(model_dir), todays_model_name)
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            os.rename(model_dir, target_dir)
        elif already_trained:
            logging.info(
                f"Today's model '{todays_model_name}' has already been trained. "
                f"Skipping retraining (pass --force-retrain to train from scratch)."
            )
        elif not self._has_games_today():
            logging.info(f"No games scheduled for today ({self._run_date}). Skipping model training.")

        params.continue_model_id = todays_model_name
        self.train_nn = RunModelFantasyBasketballPredictions(params=params, description=None)

    @staticmethod
    def _browser_profile_path(browser: Browsers = BROWSER) -> str | None:
        return get_browser_profile_path(browser)

    def run_full_pipeline(self):
        profile_path = self._browser_profile_path()
        if 'TO_YEAR' in self.players_list.columns:
            nbacom_names = self.players_list.query(f'TO_YEAR >= {self.season}')['NAME'].unique()
        else:
            nbacom_names = self.players_list['NAME'].unique()
        submit_teams = SubmitDFSTeamYahooByCSV(headless=self.headless, profile_path=profile_path,
                                               nbacom_names=nbacom_names)
        contest_ids = submit_teams.contest_ids
        if len(contest_ids) == 0:
            return

        update_todays_schedule_from_scoreboard(date=self._run_date, season=self.season)

        update_all_stats(
            season=self.season,
            add_upcoming_games=True,
            stats_loader_players=self.stats_loaders_players,
            stats_loader_teams=self.stats_loaders_teams,
            calc_features=self.calc_features,
            update_schedules=False,
            players_list=self.players_list,
        )

        self.train_nn.create_data_loaders()
        self.train_nn.update_categorical_params()
        if not self._todays_model_already_trained():
            self.train_nn.train()
        predictions = self._get_todays_predictions(teams_template=submit_teams.template)

        locked_dfs = submit_teams.get_locked_dfs()

        player_name_to_id = self.stats_loaders_players.get_player_name_to_id()
        player_id_to_name = {player_id: player_name for player_name, player_id in player_name_to_id.items()}
        locked_dfs = \
            {contest_id: locked_df.assign(PLAYER_ID=locked_df['NAME'].replace(player_name_to_id))
             for contest_id, locked_df in locked_dfs.items()}

        locked_df_shapes = {contest_id: df.shape[0] for contest_id, df in locked_dfs.items()}
        logging.info(f'number of locked players per contest:{locked_df_shapes}')
        logging.info(f'predictions shape before callingFindBestTeamGA :\n{predictions.shape[0]}')

        # all_ga_results = dict()
        # for contest_id, locked_df in locked_dfs.items():
        #     find_best_teams = FindBestTeamGA(points_col='YAHOO_POINTS_pred', at_least_one_min_player=False,
        #                                      n_teams_in_df=2500, keep_top_n=300, mutate_pct=0.3)
        #     find_best_teams.call(df=predictions, definitely_in=locked_df)
        #     all_ga_results[contest_id] = find_best_teams.summarize(player_id_to_name=player_id_to_name)
        #
        # if len(all_ga_results) == 1:
        #     best_set2 = list(all_ga_results.values())[0].iloc[[0]].assign(contest_id=contest_ids[0])
        # else:
        #      # TODO move constants to params
        #     best_set2 = \
        #          FindBestTeamGA.find_best_set(ga_results=[all_ga_results[contest_id] for contest_id in contest_ids],
        #                                       max_allowed_joint_players=4,
        #                                       avoid_single_player_in_all_sets=len(contest_ids) >= 3)
        #     best_set2['contest_id'] = contest_ids
        # best_set2[DFS_POSITIONS[Sites.YAHOO]] = best_set2[DFS_POSITIONS[Sites.YAHOO]].replace(player_id_to_name)
        # best_set =best_set2

        # finding best set using linear programming
        # locked_dfs = {k: v for i, (k, v) in enumerate(locked_dfs.items()) if i == 0}
        num_teams_in_set = len(locked_dfs)
        find_best_team_lp = FindBestSetOfTeamsLinearProgrammingYahoo(points_col='YAHOO_POINTS_pred',
                                                                     num_teams_in_set=num_teams_in_set,
                                                                     avoid_single_player_in_all_sets=num_teams_in_set >= 3,
                                                                     max_shared_players_in_teams_pairs=4)
        best_set = find_best_team_lp.run(predictions, locked_dfs)
        position_cols = DFS_POSITIONS[Sites.YAHOO]
        best_set[position_cols] = best_set[position_cols].apply(lambda col: col.map(player_id_to_name))

        # submitting the best set to yahoo
        submit_teams.submit_best_set(best_set)

    def run_full_pipeline_continuously(self):
        """ run the full pipeline in a loop, each time we'll update the game. If there are no games today we'll wait """

        while True:
            # loop for each day

            # check if we got to the next day
            current_date = get_current_date()
            if current_date != self._run_date:
                self._reset(current_date)

            # if there are no remaining games today, wait until the next day
            if not self._has_games_today() or self._todays_game_times is None or \
                    get_current_time() > max(self._todays_game_times):
                tomorrow_8am = (get_current_time() + timedelta(days=1)).replace(hour=8, minute=0, second=0,
                                                                                microsecond=0)
                if not self._run_immediately:
                    self._wait_until_certain_time(tomorrow_8am)
                continue

            # run pipeline a few minutes before each game
            for game_time in self._todays_game_times:
                if not self._run_immediately:
                    self._wait_until_certain_time(game_time - timedelta(minutes=self.minutes_before_game_to_run))

                    # check if we're a reasonable amount of time before the game
                    min_acceptable_time = game_time - timedelta(minutes=self.minutes_before_game_to_run)
                    max_acceptable_time = game_time - timedelta(minutes=1)
                    if not min_acceptable_time < get_current_time() < max_acceptable_time:
                        continue

                try:
                    self.run_full_pipeline()
                except Exception as e:
                    logging.exception(f'Error executing run_full_pipeline before game at {game_time}: {e}')
                    continue

    @staticmethod
    def _get_game_times_by_day_dict() -> dict:
        schedule = load_schedule(drop_ppd=True)
        game_times_by_day = schedule.groupby('DATE')['GAME_DATE_TIME'].unique().to_dict()
        return game_times_by_day


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="NBA Fantasy Pipeline")
    parser.add_argument('--train-only', action='store_true',
                        help="Only run morning training job (run_once_per_day) and exit")
    parser.add_argument('--model-id', type=int, default=0,
                        help="Model ID to load params from (default: 0)")
    parser.add_argument('--run-immediately', action='store_true',
                        help="Do not wait for 8:00 AM or game times")
    parser.add_argument('--date', type=str, default='today',
                        help="Run date (default: today)")
    parser.add_argument('--force-retrain', action='store_true',
                        help="Force retraining of today's model even if it was already trained")
    parser.add_argument('--run-once', action='store_true',
                        help="Run the pipeline once immediately to generate predictions, optimize lineups, and submit to Yahoo, then exit")
    parser.add_argument('--no-headless', action='store_true',
                        help="Run the browser in visible mode (default: headless)")
    args = parser.parse_args()

    params_update_dict = {'continue_model_name': 'extra_weight_on_ast'}
    ropd = RunFullPipeline(
        run_date=args.date,
        headless=not args.no_headless,
        params_update_dict=params_update_dict,
        run_immediately=args.run_immediately or args.train_only or args.run_once,
        minutes_before_game_to_run=7,
        load_params_from_model_id=args.model_id,
        initialize_daily_job=not args.train_only,
        force_retrain=args.force_retrain,
    )
    if args.train_only:
        ropd.run_once_per_day()
        logging.info("Morning model training completed successfully!")
    elif args.run_once:
        ropd.run_full_pipeline()
        logging.info("End-to-end pipeline execution and lineup submission completed successfully!")
    else:
        ropd.run_full_pipeline_continuously()