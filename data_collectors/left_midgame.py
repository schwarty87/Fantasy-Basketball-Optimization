import json
import os
import datetime
import logging
import ollama
import pandas as pd
import numpy as np
import typing
import re
import requests
import utils
from nba_api.stats import endpoints
from pydantic import BaseModel, Field


LEFT_MIDGAME_DF_COLUMNS = ['NAME', 'TEAM', 'DATE', 'LEFT_MIDGAME']
IRRELEVANT_SHORT_DESCRIPTIONS = ['halftime', 'rebound', 
                                 'end of game', 'foul', 'end of 4th', 'missed 3pt', 
                                 '+2 points', '+1 point', '+3 points', 'missed fg', 
                                 'jump ball', 'end of 3rd', 'turnover', 'challenge', 
                                 'missed ft', 'violation', 'end of 1st'
                                ]
IRRELEVANT_DESCRIPTION_PATTERNS = r"STEAL \(\d+ STL\)|BLOCK \(\d+ BLK\)"
EMPTY_DF = pd.DataFrame(columns=LEFT_MIDGAME_DF_COLUMNS)
INJURY_KEYWORDS = r"(" \
                  r"leaves? (the )?game|will not return|questionable to return|" \
                  r"headed to the locker|heads? to the locker|helped to the|carried to the|" \
                  r"assisted to the|medical staff|training staff|trainer|" \
                  r"shaken up|hard fall|slow to get up|" \
                  r"sprain|strain|tweak|soreness|tightness|spasm|fracture|torn|contusion" \
                  r")"
ARTICLE_INJURY_KEYWORDS = [
    r"leaves? (the )?game", r"will not return", r"left with", 
    r"headed to the locker", r"heads? to the locker", 
    r"helped to the locker", r"assisted to the locker",
    r"carried off", r"suffered a", r"sustained a", 
    r"exited (in|during|with)", r"did not return", 
    r"diagnosed with", r"x-rays were"
]
ARTICLE_INJURY_REGEX = re.compile(r"|".join(ARTICLE_INJURY_KEYWORDS), re.IGNORECASE)


class DiscoveredInjury(BaseModel):
    player_name: str = Field(
        description="The exact full name of the NBA player mentioned."
    )
    analysis_justification: str = Field(
        description="Reason step-by-step why this player suffered a new, mid-game injury based ONLY on explicit text actions (e.g., walking off, limping, heading to the locker room)."
    )
    injury_description: str = Field(
        description="The specific body part or injury condition sustained DURING the game play."
    )
    is_pre_game_inactive: bool = Field(
        description="True if the text indicates the player was already out, scratched, recovering from prior surgery, or inactive before tip-off. False if the injury occurred mid-game."
    )
    did_return: bool | None = Field(
        default=None,
        description="True if the text states they checked back in, False if ruled out, None if unspecified."
    )

class InjuryExtractionContainer(BaseModel):
    injuries: typing.List[DiscoveredInjury]


def _extract_injuries_cpu_optimized(full_corpus: str) -> typing.List[typing.Dict]:
    system_instruction = (
        "You are a strict data extraction engine optimized for fantasy basketball intelligence. "
        "Your task is to parse sports recaps and identify players who sustained a NEW injury DURING the game. "
        "CRITICAL RULES:\n"
        "1. Filter out pre-game inactives: If a player did not play because they were 'recovering from surgery', 'inactive', 'scratched', 'without', or 'injury management', mark is_pre_game_inactive as True.\n"
        "2. Avoid narrative confusion: Do not mistake scoring runs, comebacks, or standard aggressive play descriptions as injuries.\n"
        "3. Literal verification: The text must explicitly describe the injury event occurring during play."
    )

    prompt = f"Analyze this game recap string and extract mid-game injuries:\n\n{full_corpus}"

    try:
        response = ollama.chat(
            model='llama3.2:3b',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': prompt}
            ],
            format=InjuryExtractionContainer.model_json_schema(),
            options={
                'temperature': 0.0 # Force deterministic output, minimizing hallucinations
            }
        )
        
        parsed_json = json.loads(response['message']['content'])
        true_mid_game_injuries = [
            injury for injury in parsed_json.get('injuries', [])
            if not injury['is_pre_game_inactive'] and not injury['did_return']
        ]
        if true_mid_game_injuries:
            aaa=1
        return true_mid_game_injuries

    except Exception as e:
        print(f"Error: {e}")
        return []


class LeftMidgame:
    def __init__(self, nbacom_players_list: typing.List[str] | pd.Series | np.ndarray | pd.DataFrame | None = None,
                 player_id_to_name: typing.Dict[int, str] | None = None,
                 player_name_to_id: typing.Dict[str, int] | None = None):
        if isinstance(nbacom_players_list, pd.DataFrame):
            nbacom_players_list = nbacom_players_list['NAME'].values
        self.player_id_to_name = player_id_to_name or {}
        self.player_name_to_id = player_name_to_id or {}
        self.fix_player_names = utils.FixPlayerNames(compare_names=nbacom_players_list)
        self.left_midgame_players = set()
        
        self.save_dir = os.path.join(utils.update_dir_name('cached_data'), 'left_midgame')
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

    @staticmethod
    def _get_espn_game_ids_for_date(date: datetime.date) -> typing.List[int]:
        date_str = date.strftime('%Y%m%d')
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        data = response.json()
        return [event['id'] for event in data['events']]

    @staticmethod
    def _check_play_by_play_data(json_data: typing.Dict) -> typing.List[typing.Dict]:
        plays = json_data.get('plays', [])
        midgame_exits = []
        for play in plays:
            short_description = play.get('shortDescription')
            if short_description in IRRELEVANT_SHORT_DESCRIPTIONS:
                continue

            text = play.get('text', '')
            
            if re.search(INJURY_KEYWORDS, text, re.IGNORECASE):
                clock = play.get('clock', {}).get('displayValue', '00:00')
                period = play.get('period', {}).get('number', 1)
                
                # Extract the actual players tagged in this live event
                for participant in play.get('participants', []):
                    player_name = participant.get('athlete', {}).get('displayName')
                    
                    if player_name:
                        midgame_exits.append({
                            "player": player_name,
                            "period": f"Q{period}",
                            "clock": clock,
                            "live_log": text
                        })

        return midgame_exits

    @staticmethod
    def _extract_midgame_exits_from_recap(json_data: typing.Dict) -> typing.List[typing.Dict]:
        """
        Scans the journalist's game recap article to verify if a specific
        player suffered a mid-game injury that the play-by-play might have missed.
        """
        article_data = json_data.get('article', {})
        headline = article_data.get('headline', '')
        description = article_data.get('description', '')
        story = article_data.get('story', '')
        
        full_text = f"{headline}\n{description}\n{story}"
        
        return _extract_injuries_cpu_optimized(full_text)

    def _check_espn_midgame_injuries(self, game_id: int) -> typing.List[typing.Dict]:
        # Call ESPN's CDN game endpoint
        url = f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print("Failed to fetch data from ESPN.")
            return []

        data = response.json()
        
        # play_by_play_midgame_exits = self._check_play_by_play_data(data)
        article_exits = self._extract_midgame_exits_from_recap(data)

        return article_exits

    def collect_data(self, date: datetime.date = utils.get_current_date()) -> typing.Set[str]:
        """
        Scrapes/Loads players who left the game mid-game for a specific date using play-by-play.
        Returns a set of player names who left mid-game on that date.
        """
        self.left_midgame_players = set()
        game_ids = utils.date_to_game_ids(date)
        if len(game_ids) == 0:
            return self.left_midgame_players
        
        espn_gids = self._get_espn_game_ids_for_date(date)
        if not espn_gids:
            return self.left_midgame_players

        for espn_gid in espn_gids:
            injury_events = self._check_espn_midgame_injuries(espn_gid)
            if injury_events:
                self.left_midgame_players.add(injury_events)

        return self.left_midgame_players





if __name__ == '__main__':
    from utils import load_and_cache_players_list
    players = load_and_cache_players_list(season=2025)
    player_id_to_name = dict(zip(players['PLAYER_ID'], players['NAME']))
    player_name_to_id = dict(zip(players['NAME'], players['PLAYER_ID']))
    
    lm = LeftMidgame(
        nbacom_players_list=players['NAME'].to_list(),
        player_id_to_name=player_id_to_name,
        player_name_to_id=player_name_to_id
    )
    
    # Try a recent date
    test_date = datetime.date(2025, 2, 5) # Example date
    print(f"Finding players who left mid-game on {test_date}...")
    left_players = lm.collect_data(test_date)
    print("Found players:", left_players)

