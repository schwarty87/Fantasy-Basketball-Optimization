import os
import time
import datetime
import glob
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union, List, Tuple
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import SessionNotCreatedException, StaleElementReferenceException, TimeoutException
from constants import DFS_POSITIONS, NY_TZ, BROWSER, Sites, get_browser_profile_path
from utils import get_current_time, get_current_date, get_next_game_time, FixPlayerNames, retry, remove_accents
from enums import Browsers

if BROWSER == Browsers.CHROME:
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
elif BROWSER == Browsers.FIREFOX:
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
else:
    raise ValueError('BROWSER has to be one of the values in the Browsers enum!')

UPCOMING_CONTESTS_URL = 'https://sports.yahoo.com/dailyfantasy/contests/upcoming'
LIVE_CONTESTS_URL = 'https://sports.yahoo.com/dailyfantasy/contests/live'
EDIT_LINEUP_URL = 'https://sports.yahoo.com/dailyfantasy/contest/csv/edit'

ELIGIBILITY = {'PG': ['PG'],
               'SG': ['SG'],
               'SF': ['SF'],
               'PF': ['PF'],
               'C': ['C'],
               'G': ['PG', 'SG'],
               'F': ['SF', 'PF'],
               'UTIL': ['PG', 'SG', 'SF', 'PF', 'C']
               }


class SubmitDFSTeamYahooByCSV:
    def __init__(self, headless: bool = True,
                 profile_path: Optional[str] = None,
                 nbacom_names: Optional[Union[List[str], np.ndarray]] = None):
        # import sys
        # sys.exit(0)
        
        self._site = Sites.YAHOO
        self._positions = DFS_POSITIONS[self._site]

        self.headless = headless
        self.profile_path = profile_path or get_browser_profile_path(BROWSER)
        self.nbacom_names = nbacom_names

        self.template = self._get_template_df()

        if self.template is None:
            self.contest_ids = list()
            return

        logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        has_contest_inds = self.template['Contest ID'].apply(lambda x: isinstance(x, int))
        template_str = f'template after downloading:\n{self.template.loc[has_contest_inds, ["Contest ID"] + self._positions].to_string()}'
        template_str = remove_accents(template_str)
        logging.info(template_str)
        logging.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        has_id = self.template['ID'].str.len() > 1
        self.id_to_game_time = \
            self.template.loc[has_id, ['ID + Name', 'Time']].set_index('ID + Name').iloc[:, 0].to_dict()
        self.id_to_salary =  \
            self.template.loc[has_id, ['ID + Name', 'Salary']].set_index('ID + Name').astype(int).iloc[:, 0].to_dict()
        self.id_to_name =  \
            self.template.loc[has_id, ['ID + Name', 'Name']].set_index('ID + Name').iloc[:, 0].to_dict()
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}

        self.contest_ids = [contest_id for contest_id in self.template['Contest ID'].unique() if
                            isinstance(contest_id, int)]

    @staticmethod
    def _submitted_teams_dir() -> str:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'submitted_teams')
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _download_dir() -> str:
        d = os.path.join(SubmitDFSTeamYahooByCSV._submitted_teams_dir(), 'downloads')
        os.makedirs(d, exist_ok=True)
        return d

    def _create_browser_options(self) -> Options:
        download_dir = self._download_dir()
        if BROWSER == Browsers.CHROME:
            chrome_options = Options()
            if self.profile_path is not None:
                chrome_options.add_argument(f"--user-data-dir={self.profile_path}")
            if self.headless:
                chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--profile-directory=Default')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-infobars')
            chrome_options.add_argument('--disable-notifications')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("prefs", {
                "download.default_directory": download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
                "profile.default_content_setting_values.notifications": 2,
                "credentials_enable_service": False,
                "profile.password_manager_enabled": False
            })
            chrome_options.page_load_strategy = 'eager'
            return chrome_options
        elif BROWSER == Browsers.FIREFOX:
            firefox_options = Options()
            if self.profile_path is not None:
                firefox_options.add_argument("-profile")
                firefox_options.add_argument(self.profile_path)
            if self.headless:
                firefox_options.add_argument('--headless')
            firefox_options.set_preference("browser.download.folderList", 2)
            firefox_options.set_preference("browser.download.dir", download_dir)
            firefox_options.set_preference("browser.download.useDownloadDir", True)
            firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk",
                                           "text/csv,application/csv,text/comma-separated-values,application/vnd.ms-excel")
            firefox_options.set_preference("dom.webnotifications.enabled", False)
            firefox_options.set_preference("permissions.default.desktop-notification", 2)
            firefox_options.page_load_strategy = 'eager'
            return firefox_options
        else:
            raise ValueError('BROWSER has to be one of the values in the Browsers enum!')

    @retry(exceptions=SessionNotCreatedException, max_retries=3, delay_seconds=1)
    def _start_driver(self):
        webdriver_browser = getattr(webdriver, BROWSER.name.title())
        self.driver = webdriver_browser(service=Service(), options=self._create_browser_options())

        logging.info('finished starting driver')

    def _find_element_by_two_markers(self, by1: By, value1: str, attribute_name: str, attribute_value: str,
                                     max_tries: int = 3):
        for i in range(max_tries):
            elements = [b for b in self.driver.find_elements(by=by1, value=value1) if
                        getattr(b, attribute_name).lower() == attribute_value.lower()]
            if len(elements) > 0:
                return elements[0]
            time.sleep(0.5)
        raise ValueError(f"couldn't find element {by1}: {value1}, {attribute_name}: {attribute_value}!")

    def _click_btn(self, by: By, value: str, sleep: Optional[float] = None):
        WebDriverWait(self.driver, 20).until(EC.element_to_be_clickable((by, value)))
        if sleep:
            time.sleep(sleep)
        btn = self.driver.find_element(by, value)
        btn.click()

    @staticmethod
    @retry(exceptions=StaleElementReferenceException, max_retries=3, delay_seconds=1)
    def _is_on_my_contests_page(driver: webdriver) -> bool:
        """ check if we're on the page to edit the lineup """
        found_buttons = driver.find_elements(By.LINK_TEXT, "Edit Entries")
        return len(found_buttons) == 1

    def _navigate_to_submit_csv_page(self) -> bool:
        self.driver.get(EDIT_LINEUP_URL)
        if 'login.yahoo.com' in self.driver.current_url.lower():
            logging.error("Yahoo authentication required: browser was redirected to login page. "
                          "Please ensure you are logged into Yahoo in your active browser profile.")
            return False
        try:
            WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.LINK_TEXT, "Download a .csv template")))
            return True
        except TimeoutException:
            try:
                WebDriverWait(self.driver, 5).until(EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT,
                    "Oops! We could not find any contest entries to edit. Head over to the Lobby or hit Create to enter"
                    " some contests.")))
                logging.warning("we're not in any competitions!")
                return False
            except TimeoutException:
                if 'login.yahoo.com' in self.driver.current_url.lower():
                    logging.error("Yahoo authentication required: redirected to login during navigation.")
                    return False
                raise

    @staticmethod
    @retry(exceptions=(FileNotFoundError, pd.errors.EmptyDataError, PermissionError), max_retries=10, delay_seconds=2)
    def _load_csv(filename: str, only_recent_downloads: Optional[Union[int, float]] = 300) -> Tuple[pd.DataFrame, str]:
        # Search both dedicated downloads directory and user Downloads folder
        search_dirs = [
            SubmitDFSTeamYahooByCSV._download_dir(),
            os.path.join(os.path.expanduser("~"), 'Downloads')
        ]
        template_paths = []
        for d in search_dirs:
            if os.path.isdir(d):
                template_paths.extend(glob.glob(os.path.join(d, f'{filename}*.csv')))

        # Exclude temporary partially-downloaded files
        template_paths = [p for p in template_paths if not p.endswith(('.part', '.crdownload', '.tmp'))]

        if len(template_paths) == 0:
            raise FileNotFoundError(f'there are no files with the name {filename}!')
        creation_times = [os.path.getctime(file_path) for file_path in template_paths]
        if only_recent_downloads is not None:
            max_time = datetime.datetime.fromtimestamp(np.max(creation_times))
            if (get_current_time(tz=None) - max_time).total_seconds() > only_recent_downloads:
                raise FileNotFoundError(f'there are no recent files with the name {filename}!')
        path_of_latest_template = template_paths[np.argmax(creation_times)]
        if os.path.getsize(path_of_latest_template) == 0:
            raise pd.errors.EmptyDataError(f"file {path_of_latest_template} is still being written")
        logging.info(f'loading downloaded template from {path_of_latest_template}...')
        return pd.read_csv(path_of_latest_template), path_of_latest_template

    @staticmethod
    def load_downloaded_csv(filename: str = 'Yahoo_DF_contest_lineups_bulk_update_template',
                            nbacom_names: Optional[Union[List[str], np.ndarray]] = None,
                            only_recent_downloads: Optional[Union[int, float]] = 300,
                            delete_file: bool = True) -> pd.DataFrame:
        """
        Load the template file that was downloaded from the site
        :param filename: The name of the file to look for
        :param nbacom_names: The names we want to use to correct the names in the file
        :param only_recent_downloads: The file must have been downloaded at most this many seconds ago
        :param delete_file: Whether to delete the file after reading it or not
        :return: The template file
        """
        template, path_of_latest_template = \
            SubmitDFSTeamYahooByCSV._load_csv(filename=filename, only_recent_downloads=only_recent_downloads)

        template['Name'] = template['First Name'] + ' ' + template['Last Name']
        template['Name'] = FixPlayerNames(compare_names=nbacom_names).fix_names(template['Name'])

        template['Contest ID'] = template['Contest ID'].astype(object)
        template['Entry ID'] = template['Entry ID'].astype(object)
        has_contest_id = template['Contest ID'].astype(str).str.strip().str.len() > 1
        template.loc[has_contest_id, 'Contest ID'] = template.loc[has_contest_id, 'Contest ID'].astype(int)
        template.loc[has_contest_id, 'Entry ID'] = template.loc[has_contest_id, 'Entry ID'].astype(int)

        # converting times to datetimes
        todays_date = get_current_date()
        template['Time'] = template['Time'].astype(object)
        has_time = template['Time'].astype(str).str.contains(':')
        date_strings = str(todays_date) + ' ' + template.loc[has_time, 'Time'].str.replace(r' E[S|D]T', '', regex=True)
        template.loc[has_time, 'Time'] = pd.to_datetime(date_strings, format='%Y-%m-%d %I:%M%p').dt.tz_localize(NY_TZ)
        template.loc[~has_time, 'Time'] = template.loc[has_time, 'Time'].max()
        if delete_file and os.path.isfile(path_of_latest_template):
            os.remove(path_of_latest_template)

        return template

    def _get_template_df(self) -> Optional[pd.DataFrame]:
        try:
            self._start_driver()
            on_submit_csv_page = self._navigate_to_submit_csv_page()
            if not on_submit_csv_page:
                return None
            self._click_btn(by=By.LINK_TEXT, value='Download a .csv template')
            template = self.load_downloaded_csv(nbacom_names=self.nbacom_names)
            logging.info('finished downloading and loading template df')
        except Exception as e:
            if os.path.isfile(self._template_path()):
                template = pd.read_csv(self._template_path())
            else:
                raise ValueError(f"we weren't able to load the template\n{e}")
        finally:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()

        # remove the row with the dfs league
        dfs_league_row = template['Contest Title'].str.contains('NBA Free 6-Team 155-Round League Round')
        if dfs_league_row.any():
            cols = template.loc[:, :' '].columns
            i = dfs_league_row[dfs_league_row].index[0]
            template.loc[i: template.index[-2], cols] = template.loc[i + 1: template.index[-1], cols].values

        # # remove the row with the dfs league
        # dfs_league_row = template['Contest Title'].str.contains('dfs league 2025-26')
        # if dfs_league_row.any():
        #     cols = template.loc[:, :' '].columns
        #     i = dfs_league_row[dfs_league_row].index[0]
        #     template.loc[i: template.index[-2], cols] = template.loc[i + 1: template.index[-1], cols].values

        return template

    @staticmethod
    def _id_to_value(dict: Dict, id: str):
        if id in dict.keys():
            return dict[id]
        values = [v for k, v in dict.items() if id.split(' - ')[0] in k]
        if len(values) == 1:
            return values[0]
        raise KeyError(f"couldn't find {id} in dict!")

    def get_locked_dfs(self) -> Dict[int, pd.DataFrame]:
        """get a df with the locked players for each contest"""
        locked_dfs = dict()
        next_game_time = get_next_game_time(template=self.template)
        for _, curr_row in self.template.iterrows():
            locked_players = list()
            contest_id = curr_row['Contest ID']
            if not isinstance(contest_id, int):
                continue
            contest_id = int(contest_id)
            for p in self._positions:
                if len(curr_row[p]) <= 1:
                    continue
                curr_id = curr_row[p]
                player_game_time = self._id_to_value(dict=self.id_to_game_time, id=curr_id)
                # if get_current_time() + datetime.timedelta(minutes=1) > game_time:
                if next_game_time > player_game_time:
                    name = self._id_to_value(dict=self.id_to_name, id=curr_id)
                    locked_players.append({'Pos': p,
                                           'YAHOO_SALARY': self._id_to_value(self.id_to_salary, curr_id),
                                           'NAME': name,
                                           'YAHOO_POSITION': self.template.loc[self.template['Name'] == name, 'Position'].values[0]
                                           })
            if len(locked_players) > 0:
                locked_dfs[contest_id] = pd.DataFrame(locked_players)
            else:
                locked_dfs[contest_id] = pd.DataFrame(columns=['Pos', 'NAME', 'YAHOO_SALARY', 'YAHOO_POSITION'])
        return locked_dfs

    @staticmethod
    def _template_path() -> str:
        return os.path.join(SubmitDFSTeamYahooByCSV._submitted_teams_dir(),
                            str(get_current_date()).replace('-', '_') + '.csv')

    def _sort_players_by_time(self, teams_df: pd.DataFrame):
        """Sorting each team so the players who play in the earlier games will be in the more restrictive positions"""
        if 'contest_id' in teams_df.columns:
            teams_df = teams_df.set_index('contest_id')

        teams_df_sorted = teams_df.copy()
        eligibility = {p: self.template.loc[self.template['Position'].isin(ELIGIBILITY[p]), 'ID + Name'].values
                       for p in self._positions}

        next_game_time = get_next_game_time(template=self.template)
        for contest_id, team_row in teams_df.iterrows():
            row_players = set([pp for pp in team_row[self._positions].values if
                               self._id_to_value(self.id_to_game_time, pp) >= next_game_time])
            for p in self._positions:
                is_locked = self._id_to_value(self.id_to_game_time, team_row[p]) < next_game_time
                if is_locked:
                    teams_df_sorted.loc[contest_id, p] = team_row[p]
                else:
                    options = sorted(set(eligibility[p]) & set(row_players))
                    options_game_times = [self._id_to_value(self.id_to_game_time, option) for option in options]
                    earliest_game_player = options[np.argmin(options_game_times)]
                    teams_df_sorted.loc[contest_id, p] = earliest_game_player
                    row_players.remove(earliest_game_player)
        return teams_df_sorted.reset_index()

    def _assert_template_csv_finished_saving(self):
        # making sure that the csv finished saving before moving on
        template_for_comparison = \
            self.template.iloc[:len(self.contest_ids)][['Contest ID', 'Entry ID'] + self._positions]
        template_for_comparison[['Contest ID', 'Entry ID']] = \
            template_for_comparison[['Contest ID', 'Entry ID']].astype(int)
        for i in range(5):
            template2 = pd.read_csv(self._template_path())
            template2 = template2.iloc[:len(self.contest_ids)]
            template2[['Contest ID', 'Entry ID']] = template2[['Contest ID', 'Entry ID']].astype(int)
            if template_for_comparison.equals(template2[['Contest ID', 'Entry ID'] + self._positions]):
                break
            elif i < 4:
                time.sleep(0.5)
            else:
                raise ValueError('having trouble saving template csv!')

    def _fill_template_with_predicted_teams(self, best_set_with_ids: pd.DataFrame):
        for _, curr_row in best_set_with_ids.iterrows():
            contest_id = curr_row['contest_id']
            ind_in_template = self.template.loc[self.template['Contest ID'] == contest_id].index[0]
            self.template.loc[ind_in_template, self._positions] = curr_row[self._positions].values

    def submit_best_set(self, best_set: pd.DataFrame):
        """
        Submit the given set of teams to yahoo by updating the template and uploading a csv
        :param best_set: a dataframe with the best set. The columns should be (all positions), salary, contest_id
                         The value in each position should be the player name (not player id))
        """
        best_set_with_ids = best_set.copy()
        best_set_with_ids[self._positions] = best_set_with_ids[self._positions].replace(self.name_to_id)

        # sorting the players by time
        best_set_with_ids = self._sort_players_by_time(teams_df=best_set_with_ids)

        all_converted = best_set_with_ids[self._positions].isin(self.template['ID + Name'].values).all(axis=None)
        if not all_converted:
            unconverted_inds = ~best_set_with_ids[self._positions].isin(self.template['ID + Name'].values)
            uncoverted_names = best_set_with_ids[self._positions].values[unconverted_inds]
            logging.error(f'not all names were converted to ids!\n{uncoverted_names}')
            raise ValueError(f'not all names were converted to ids!\n{uncoverted_names}')

        logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        before_converting_str = f'best set before converting names to yahoo ids:\n{best_set.to_string()}'
        before_converting_str = remove_accents(before_converting_str)
        logging.info(before_converting_str)
        after_converting_str = f'best set after converting names to yahoo ids:\n{best_set_with_ids.to_string()}'
        after_converting_str = remove_accents(after_converting_str)
        logging.info(after_converting_str)
        logging.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

        # filling the template with the predicted teams
        template_before = self.template.iloc[:len(self.contest_ids)][self._positions].copy()
        self._fill_template_with_predicted_teams(best_set_with_ids)

        self.template.to_csv(self._template_path(), index=False)
        self._assert_template_csv_finished_saving()

        # self._open_edit_entries_window()
        self._start_driver()
        submitted_team = False
        try:
            on_submit_csv_page = self._navigate_to_submit_csv_page()
            if not on_submit_csv_page:
                return

            # File upload: Try universal input[type='file'] first, then fallback to accessible_name
            file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
            if file_inputs:
                choose_file_button = file_inputs[0]
            else:
                attribute_value = 'Choose file' if BROWSER == Browsers.CHROME else 'browse… no file selected.'
                choose_file_button = self._find_element_by_two_markers(by1=By.TAG_NAME, value1='input',
                                                                       attribute_name='accessible_name',
                                                                       attribute_value=attribute_value)
            choose_file_button.send_keys(self._template_path())
            detected_txt = f"We have detected {len(self.contest_ids)} lineup"
            if len(self.contest_ids) > 1:
                detected_txt += "s"
            WebDriverWait(self.driver, 10).until(
                lambda driver: self._find_element_by_two_markers(By.TAG_NAME, 'span', 'text', detected_txt))
            upload_and_edit_button = self._find_element_by_two_markers(by1=By.TAG_NAME, value1='button',
                                                                       attribute_name='text',
                                                                       attribute_value='Upload and edit entries')

            upload_and_edit_button.click()
            entry_str = 'entries' if len(self.contest_ids) > 1 else 'entry'
            success_text = f"Contest {entry_str} edited successfully!"
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda d: d.find_elements(By.PARTIAL_LINK_TEXT, success_text) or
                              d.find_elements(By.XPATH, f"//*[contains(text(), '{success_text}') or contains(text(), 'edited successfully')]")
                )
                submitted_team = True
            except TimeoutException:
                logging.error("Success confirmation message element was not detected within 15s timeout.")
                submitted_team = False
        except Exception as e:
            logging.exception('there was an exception in submit_best_set:')
            submitted_team = False
        finally:
            self.driver.quit()

        if submitted_team:
            logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            logging.info(f'finished submitting best sets to yahoo via csv!\n{best_set.to_string()}')
            logging.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            has_contest_inds = self.template['Contest ID'].apply(lambda x: isinstance(x, int))
            template_str = f'template after submitting:\n{self.template.loc[has_contest_inds, ["Contest ID"] + self._positions].to_string()}'
            template_str = remove_accents(template_str)
            logging.info(template_str)
            logging.info('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    @classmethod
    def interactive_login(cls, browser: Browsers = BROWSER, profile_path: Optional[str] = None):
        """
        Open a visible browser session for the user to log into Yahoo once.
        Cookies and session tokens will be saved in the dedicated profile directory so that
        all subsequent headless runs stay logged in.
        """
        resolved_path = profile_path or get_browser_profile_path(browser)
        print("=" * 60)
        print("Yahoo DFS Interactive Login Helper")
        print("=" * 60)
        print(f"Browser: {browser.name}")
        print(f"Profile directory: {resolved_path}")
        print("\nOpening browser window...")
        print("Please log into Yahoo and make sure to check 'Stay signed in' / 'Keep me signed in'.")
        print("Navigate until you reach your Yahoo account or contest page, then return here.")

        instance = cls.__new__(cls)
        instance.headless = False
        instance.profile_path = resolved_path
        instance._site = Sites.YAHOO
        instance._positions = DFS_POSITIONS[instance._site]
        instance._start_driver()
        try:
            instance.driver.get(EDIT_LINEUP_URL)
            input("\n--> Press ENTER in this console when you have finished signing in... <--\n")
            current_url = instance.driver.current_url
            if 'login.yahoo.com' not in current_url.lower():
                print("SUCCESS: Successfully signed into Yahoo! Profile saved.")
            else:
                print("WARNING: Current URL still indicates a login page. Please verify credentials.")
        finally:
            instance.driver.quit()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Yahoo DFS Lineup Submission and Authentication")
    parser.add_argument('--login', action='store_true',
                        help="Open visible browser to log into Yahoo and save session in dedicated profile")
    parser.add_argument('--headless', action='store_true', default=False,
                        help="Run browser in headless mode")
    args = parser.parse_args()

    if args.login:
        SubmitDFSTeamYahooByCSV.interactive_login()
    else:
        SubmitDFSTeamYahooByCSV(headless=args.headless)