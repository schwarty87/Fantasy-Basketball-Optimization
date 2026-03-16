import os
import traceback

import pandas as pd
import datetime
from typing import Tuple
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, WebDriverException
from selenium.webdriver.common.by import By
from tqdm import tqdm
from constants import FIX_TEAM_NAMES

team_names = [n for n in FIX_TEAM_NAMES.keys() if ' ' in n]


def init_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless=new')
    chrome_options.add_argument('--no-sandbox')
    return webdriver.Chrome(service=Service(), options=chrome_options)


def full_loop(df_path: str, gids: Tuple[Tuple[int, int], ...]):
    gid = -1
    while True:
        try:
            all_gids = sum([list(range(ids[0], ids[1] + 1)) for ids in gids], [])

            if os.path.isfile(df_path):
                df = pd.read_csv(tmp_df_path)
                start_gid = max(gid, df['ESPN_GAME_ID'].max() + 1)
                all_gids = [gid for gid in all_gids if gid >= start_gid]
                possibly_left_midgame = df.apply(tuple, axis=1).to_list()
            else:
                possibly_left_midgame = list()

            driver = init_driver()
            for i, gid in enumerate(tqdm(all_gids)):
                curr_url = f'https://www.espn.com/nba/recap/_/gameId/{gid}'

                driver.get(curr_url)

                no_story_available = driver.find_elements(By.TAG_NAME, 'h5')
                if len(no_story_available) > 0 and no_story_available[0].text == 'No Story Available':
                    continue

                page_not_found = driver.find_elements(By.TAG_NAME, 'h1')
                if len(page_not_found) > 0 and page_not_found[0].text == 'Page not found.':
                    continue

                # game date
                game_time = driver.find_element(By.CLASS_NAME, 'ContentList').find_element(By.TAG_NAME, 'span').text
                if len(game_time) == 0:
                    continue
                game_time = datetime.datetime.strptime(game_time, '%I:%M %p, %B %d, %Y')
                if game_time.hour <= 10:
                    curr_date = (game_time - datetime.timedelta(days=1)).date()
                else:
                    curr_date = game_time.date()

                teams = [elem.get_attribute('textContent') for elem in driver.find_elements(By.CSS_SELECTOR, '.HUcap.mpjVY')]
                teams = [t for t in teams if t]
                if len(teams) != 2:
                    teams = [t.text for t in driver.find_elements(By.TAG_NAME, 'h2')]
                    teams = [t for t in teams if t]
                if len(teams) != 2:
                    raise ValueError('length of teams isnt 2!')
                teams = [FIX_TEAM_NAMES.get(t, t) for t in teams]


                # story = driver.find_element(By.XPATH, '//*[@id="fittPageContainer"]/div[2]/div/div[5]/div/div[1]/section/div/div[2]')
                # story = driver.find_element(By.XPATH, '//*[@id="fittPageContainer"]/div[2]/div/div/div/div[5]/div/div[1]/section/div/div[2]')
                story = driver.find_element(By.CSS_SELECTOR, '.Story__Body.t__body')
                for paragraph in story.find_elements(By.TAG_NAME, 'p'):
                    curr_text = paragraph.text

                    if any([s in curr_text for s in ('left', 'exited', 'sprained', 'injured', 'before leaving', 'suffered',
                                                     'despite losing', 'left the game', 'left in the', 'spraining',
                                                     "didn’t return", "did not return", "never returned", "went down",
                                                     "left after playing", "sat out the rest of the game",
                                                     "helped off the court", 'lost forward', 'lost guard', 'lost center')]) and \
                            'left wing' not in curr_text and 'seconds left' not in curr_text and 'place of injured' not in curr_text:
                        for link_element in paragraph.find_elements(By.TAG_NAME, 'a'):
                            curr_name = link_element.text
                            if curr_name not in team_names and curr_name.lower() != curr_name:
                                possibly_left_midgame.append((link_element.text, curr_date, curr_text, gid, teams[0], teams[1]))
                                # print(i)

                        pd.DataFrame(possibly_left_midgame, columns=['NAME', 'DATE', 'TEXT', 'ESPN_GAME_ID', 'TEAM1', 'TEAM2']).to_csv(df_path, index=False)

                if gid % 10 == 0:
                    driver.quit()
                    driver = init_driver()

            break
        except (WebDriverException, NoSuchElementException):
            traceback.print_exc()
            driver.quit()
            continue

    # except Exception as e:
    #     traceback.print_exc()
    #     # import sys
    #     # sys.exit(0)
    #     driver.quit()
    #     raise e


def main(df_path: str, gids: Tuple[Tuple[int, int], ...]):
    full_loop(df_path, gids)

    # while True:
    #     try:
    #         full_loop(df_path, gids)
    #         break
    #     except (WebDriverException, NoSuchElementException):
    #         continue


if __name__ == '__main__':
    gids = (#(400578293, 400579508),
            # (400827888, 400829117),
            # (400899375, 400900608),
            # (400974437, 400975976),
            # (401070213, 401071903),
            # (401160623, 401161615), (401224686, 401224790),
            # (401266805, 401283400), (401307309, 401307891),
            # (401358773, 401361049),
            # (401468016, 401469385),
            # (401584689, 401585828),
            (401704627, 401705766),)  # from espn

    tmp_df_path = 'C:/users/schwa/Desktop/possibly_left_midgame.csv'
    main(df_path=tmp_df_path, gids=gids)