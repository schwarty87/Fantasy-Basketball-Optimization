import datetime
import logging
import os
import time
import pandas as pd
from datetime import timedelta
from typing import Optional, Dict, Union
from data_collectors.stats_loaders import StatsLoaderPlayers, StatsLoaderTeams
from calc_features import CalcFeatures
from find_best_team_linear_programming import FindBestSetOfTeamsLinearProgrammingYahoo
from utils import add_missing_points_columns_to_predictions, get_current_time, define_logger, load_schedule, \
    get_current_date, load_and_cache_players_list, scrape_season_schedules, get_next_game_time, update_dir_name, DATA_DIR
from submit_dfs_team import SubmitDFSTeamYahooByCSV
from constants import NY_TZ, DFS_POSITIONS, BROWSER, Sites, POSITION_TO_NUM
from enums import Browsers
from machine_learning_fantasy_basketball_predictions.params.params import \
    MachineLearningParamsFantasyBasketballPredictions
from machine_learning_fantasy_basketball_predictions.run_model import RunModelFantasyBasketballPredictions
from machine_learning.enums import DataSplitType


class RunFullPipeline:
    def __init__(self, load_params_from_model_id: int, season: int = 2025,
                 run_date: Union[datetime.date, str] = get_current_date(),
                 headless: bool = True, minutes_before_game_to_run: int = 10,
                 params_update_dict: Optional[Dict] = None, run_immediately: bool = False):
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
        """
        define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO,
                      filename=os.path.join(DATA_DIR, 'logs', f"logs_{get_current_date()}.txt"), print_to_console=True, filemode='a')

        self.season = season
        self.headless = headless
        self.minutes_before_game_to_run = minutes_before_game_to_run
        self._run_immediately = run_immediately
        self._load_params_from_model_id = load_params_from_model_id

        self.dates_to_game_times = dict()   #self._get_game_times_by_day_dict()

        self._model_params_update_dict = params_update_dict
        if 'training' not in self._model_params_update_dict.keys():
            self._model_params_update_dict['training'] = dict()
        self._model_params_update_dict['training']['run_extra_callbacks'] = False

        self.stats_loaders_players = StatsLoaderPlayers(season=season)
        self.stats_loaders_teams = StatsLoaderTeams(season=season)
        self.calc_features = CalcFeatures()

        self.train_nn = None
        self._reset(run_date)

        self.submit_teams = None
        # self.google_drive_driver = None    # driver for reading data from Google Drive

    def _reset(self, run_date: Union[datetime.date, str]):
        """
        Should be called at the beginning of each run, and every time we start a new day. Calculate all the relevant
        info for the current day. We get the new model name that we want to run, and (re)initialize the
        TrainNeuralNetwork object. In addition, we'll calculate the times that we want to run the predictions/team
        creation, and reinitialize the exclude_contests list. Finally, we'll run run_once_per_day to collect the
        previous day's results.
        :param run_date: The date that we're running now
        """
        if isinstance(run_date, datetime.date):
            self._run_date = run_date
        elif isinstance(run_date, str) and run_date == 'today':
            self._run_date = get_current_date()
        else:
            logging.error('run_date has to be either a datetime.date or "today"!!!')

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
        wait_seconds = (wait_until - get_current_time()).total_seconds()
        if wait_seconds > 0:
            logging.info(f'sleeping until {wait_until}')
            time.sleep(wait_seconds)

    @staticmethod
    def _todays_model_already_trained() -> bool:
        return os.path.isfile(os.path.join(DATA_DIR, 'machine_learning_fantasy_basketball_predictions', 'models',
                                           'model_' + str(get_current_date()).replace('-', '_'), 'model.pt'))

    def _has_games_today(self):
        return self._run_date in self.dates_to_game_times.keys()

    @staticmethod
    def _add_game_times_to_df(df: pd.DataFrame) -> pd.DataFrame:
        schedules_dir = update_dir_name('season_schedules')
        schedule_names = sorted(os.listdir(schedules_dir))
        schedules = pd.concat([pd.read_feather(os.path.join(schedules_dir, s)) for s in schedule_names], axis=0)
        df = df.merge(schedules, how='left', left_on=['GAME_ID', 'TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID_HOME'])
        df = df.merge(schedules, how='left', left_on=['GAME_ID', 'TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID_AWAY'])
        df['GAME_DATE_TIME'] = df['GAME_DATE_TIME_x'].fillna(df['GAME_DATE_TIME_y'])

        drop_cols = [c + suff for c in schedules for suff in ('_x', '_y') if c + suff in df]
        df.drop(columns=drop_cols, inplace=True)

        assert not df['GAME_DATE_TIME'].isnull().any(), 'missing game times!!!!'

        return df

    def _get_todays_predictions(self, teams_template: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        metrics_params = self.train_nn._change_metrics_params_tuple_to_per_sample(self.train_nn.params.training_params.metrics_params)
        loss_params = self.train_nn._change_metrics_params_tuple_to_per_sample(self.train_nn.params.training_params.loss_params)
        predictions = self.train_nn.predict(data_split_type=DataSplitType.TRAIN, metrics_params=metrics_params,
                                            loss_params=loss_params)
        predictions.rename(columns=lambda x: x.replace('CollectTruePredictedOutputsParamsFantasyBasketballPredictions_', ''),
                           inplace=True)
        predictions = predictions.loc[predictions['DATE'] == self._run_date]

        # adding game id
        predictions = predictions.merge(self.stats_loaders_players.df[['PLAYER_ID', 'DATE', 'GAME_ID', 'TEAM_ID',
                                                                       'YAHOO_POSITION', 'YAHOO_SALARY']],
                                        on=['DATE', 'PLAYER_ID'], how='left')

        logging.info(f'number of players after running predictions: {predictions.shape[0]}')

        if teams_template is not None:
            player_name_to_id = self.stats_loaders_players.get_player_name_to_id()
            player_id_to_name = {player_id: player_name for player_name, player_id in player_name_to_id.items()}
            prediction_names = predictions['PLAYER_ID'].replace(player_id_to_name)
            # prediction_names2 = prediction_names.replace(FIX_PLAYER_NAMES_NBA_TO_YAHOO)
            # in_template = prediction_names2.isin(teams_template['Name'].values)
            in_template = prediction_names.isin(teams_template['Name'].values)
            predictions = predictions.loc[in_template].reset_index(drop=True)

            # filling fantasy salaries and positions
            if predictions['YAHOO_SALARY'].isnull().any():
                id_to_salary = \
                    {int(id): teams_template.loc[teams_template['Name'] == player_id_to_name[id], 'Salary'].values[0]
                     for id in predictions['PLAYER_ID'].values}
                predictions['YAHOO_SALARY'] = (
                    predictions['YAHOO_SALARY'].fillna(predictions['PLAYER_ID'].replace(id_to_salary)))
                predictions['YAHOO_SALARY'] = predictions['YAHOO_SALARY'].astype(float)
            if predictions['YAHOO_POSITION'].isnull().any():
                id_to_position = \
                    {int(id): teams_template.loc[teams_template['Name'] == player_id_to_name[id], 'Position'].values[0]
                     for id in predictions['PLAYER_ID'].values}
                predictions['YAHOO_POSITION'] = (
                    predictions['YAHOO_POSITION'].fillna(predictions['PLAYER_ID'].replace(id_to_position)))
                predictions['YAHOO_POSITION'] = predictions['YAHOO_POSITION'].map(POSITION_TO_NUM)
                predictions['YAHOO_POSITION'] = predictions['YAHOO_POSITION'].astype(int)

            logging.info(f'number of players after filtering with players in template: {predictions.shape[0]}')

        add_missing_points_columns_to_predictions(predictions_df=predictions)
        predictions = self._add_game_times_to_df(predictions)

        # dropping players whose game time already passed
        # cutoff_time = get_current_time() + timedelta(minutes=1 )
        cutoff_time = get_next_game_time(template=teams_template)
        predictions = predictions.loc[predictions['GAME_DATE_TIME'] >= cutoff_time]

        logging.info(f'number of players after filtering with cutoff time: {predictions.shape[0]}')

        return predictions

    def run_once_per_day(self):
        scrape_season_schedules(self.season, save_dir=update_dir_name('season_schedules'))

        self.dates_to_game_times = self._get_game_times_by_day_dict()

        # getting times when we want to run the predictions/team creation etc.
        self._todays_game_times = self.dates_to_game_times.get(self._run_date)
        if self._run_date in self.dates_to_game_times.keys():
            self._run_times = [t - timedelta(minutes=self.minutes_before_game_to_run) for t in self._todays_game_times]
        else:
            self._run_times = None

        self.players_list = load_and_cache_players_list(season=self.season)
        self.stats_loaders_players.call(players_list=self.players_list, add_upcoming_games=False)
        self.stats_loaders_teams.call()

        self.calc_features.call(players_list=self.players_list, add_upcoming_games=False)

        # training/loading model
        params_path = os.path.join(DATA_DIR, f'machine_learning_fantasy_basketball_predictions/models/{self._load_params_from_model_id}/params.json')
        params = MachineLearningParamsFantasyBasketballPredictions.load_params(params_path)
        # params.features_params.extra_weight_on_certain_date = self._run_date
        params.features_params.last_month_in_train = True
        todays_model_name = 'model_' + str(get_current_date()).replace('-', '_')
        if not self._todays_model_already_trained() and self._has_games_today():
            self.train_nn = RunModelFantasyBasketballPredictions(params=params, description=None)
            self.train_nn.train()

            # change the model name from the model id to today's date
            model_dir = self.train_nn._model_dir
            os.rename(model_dir,
                      os.path.join(os.path.dirname(model_dir), todays_model_name)
                      )

            import sys
            sys.exit(0)
        params.continue_model_id = todays_model_name
        self.train_nn = RunModelFantasyBasketballPredictions(params=params, description=None)

    @staticmethod
    def _browser_profile_path() -> Optional[str]:
        if BROWSER == Browsers.CHROME:
            if os.path.isdir('/home/danieli/.config/google-chrome'):
                return '/home/danieli/.config/google-chrome'
            if os.path.isdir("C:/Users/schwa/AppData/Local/Google/Chrome/User Data"):
                return "C:/Users/schwa/AppData/Local/Google/Chrome/User Data"
            return None
        elif BROWSER == Browsers.FIREFOX:
            if os.path.isdir('/home/danieli/.mozilla/firefox'):
                return '/home/danieli/.mozilla/firefox/f0lpo4uh.default-release'
            else:
                return 'C:/Users/schwa/AppData/Roaming/Mozilla/Firefox/Profiles/irl9n0gn.default-release'
        raise ValueError('no browser profile path found!')

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

        self.stats_loaders_players.call(players_list=self.players_list)
        self.stats_loaders_teams.call()
        self.calc_features.call(players_list=self.players_list)

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

                # try:
                self.run_full_pipeline()
                # except Exception as e:
                #     logging.warning(f'There was an error when running run_full_pipeline {e}!!!\ncontinuing...')
                #     traceback.print_exc()
                #     import sys
                #     sys.exit(0)
                #     continue

    @staticmethod
    def _get_game_times_by_day_dict() -> Dict:
        schedule = load_schedule(drop_ppd=True)
        game_times_by_day = schedule.groupby('DATE')['GAME_DATE_TIME'].unique().to_dict()
        return game_times_by_day


if __name__ == '__main__':
    run_immediately = False
    headless = True
    params_update_dict = {'continue_model_name': 'extra_weight_on_ast'}
    run_date = 'today'
    ropd = RunFullPipeline(run_date=run_date, headless=headless, params_update_dict=params_update_dict,
                           run_immediately=run_immediately, minutes_before_game_to_run=7, load_params_from_model_id=121)
    ropd.run_full_pipeline_continuously()