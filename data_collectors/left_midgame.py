import os
import datetime
import logging
import re
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

import utils
from utils import (
    FixPlayerNames,
    retry,
    get_current_date,
    update_dir_name,
)

LEFT_MIDGAME_DF_COLUMNS = ['NAME', 'TEAM', 'DATE', 'LEFT_MIDGAME']
EMPTY_DF = pd.DataFrame(columns=LEFT_MIDGAME_DF_COLUMNS)

CLOCK_OR_COURT_PATTERN = re.compile(
    r'(\b(loose|air|jump|game|dead|live)\s+ball\b|\b\d+:\d+\s+left\b|\b\d+\s+seconds?\s+left\b|\bleft\s+(wing|corner|side|baseline|alone|open)\b)',
    re.IGNORECASE,
)
PREGAME_INACTIVE_PATTERN = re.compile(
    r'(\bin place of\b|\bstarted for\b|\bwithout injured\b|\bwithout\s+the\s+injured\b|'
    r'\bmissed his\b|\bmissed their\b|\bmissed the last\b|\brecovering from\b|'
    r'\bsidelined\b|\bhas not played\b|\bout with\b|\bwere unavailable\b|\bwas unavailable\b|'
    r'\bcrowd\b|\bfans\b)',
    re.IGNORECASE,
)
RETURNED_PATTERN = re.compile(
    r'(\breturned to the game\b|\bwas able to return\b|\blater returned\b|\bchecked back in\b|'
    r'\bcame back (in|into the game)\b|\bcleared to return\b|\breturned in the\b|'
    r'\b(but\s+)?returned\s+(and\s+played|to\s+the|before\s+the)\b)',
    re.IGNORECASE,
)
SECOND_HALF_OUT_PATTERN = re.compile(
    r'\b(without\s+.*?\s+for\s+the\s+second\s+half|ruled\s+out\s+(at\s+halftime|for\s+the\s+(second\s+half|rest|remainder))|sat\s+out\s+the\s+second\s+half)\b',
    re.IGNORECASE,
)
MIDGAME_EXIT_PATTERN = re.compile(
    r'('
    r'\b(left|leaving)\s+(the\s+)?(game|court|floor)\b|'
    r'\b(left|leaving)\s+with\b|'
    r'\bexited\s+(the\s+)?game\b|'
    r'\bexited\s+with\b|'
    r'\bexited\s+in\s+the\b|'
    r'\bdeparted\s+(the\s+)?game\b|'
    r'\bdeparted\s+with\b|'
    r'\bsat\s+out\s+the\s+(second\s+half|rest|remainder)\b|'
    r'\bruled\s+out\b.*?\b(for\s+the\s+(game|rest|remainder)|at\s+halftime|in\s+the\s+(second\s+half|third|fourth|3rd|4th))\b|'
    r'\b(did\s+not|didn\'t|will\s+not|never)\s+return\b|'
    r'\b(headed|heading|limped|helped|assisted|carried|taken|went|escorted|needed\s+help\s+.*?)\s+(to|into)\s+the\s+locker\s+room\b|'
    r'\b(helped|assisted|carried|limped)\s+off\s+the\s+court\b|'
    r'\binjured\s+his\s+\w+\s+in\s+the\s+(first|second|third|fourth|1st|2nd|3rd|4th)\b|'
    r'\bsuffered\s+a\s+.*?\s+in\s+the\s+(first|second|third|fourth|1st|2nd|3rd|4th)\s+(quarter|half)\b|'
    r'\b(was\s+)?(ejected|tossed)\b|'
    r'\b(ejection|disqualified)\b|'
    r'\bwent\s+out\s+with\b|'
    r'\b(didn\'t\s+play|did\s+not\s+play|sat\s+out|missed|without\s+.*?\s+for)\s+the\s+second\s+half\b|'
    r'\b(lost|losing)\s+.*?\s+to\s+(a\s+)?.*?(injury|strain|sprain|contusion|soreness)\b|'
    r'\b(lost|losing)\s+.*?\s+in\s+the\s+(second\s+half|third|fourth|1st|2nd|3rd|4th)\b|'
    r'\bbefore\s+(leaving|straining|spraining|injuring|hurting|heading)\b'
    r')',
    re.IGNORECASE,
)
BYSTANDER_PATTERN = re.compile(
    r'\b(foul\s+on|colliding\s+with|collision\s+with|into|against|foot\s+of\s+teammate)\s+([A-Z][a-z]+(\s+[A-Z][a-z]+)?)\b',
    re.IGNORECASE,
)
NON_PLAYER_SUBJECT_PATTERN = re.compile(
    r'\b(udoka|coach|head\s+coach|assistant\s+coach|referee|official|crew\s+chief|umpire|trainer|team|fans?|crowd|players?)\b',
    re.IGNORECASE,
)


def get_player_surname(full_name: str) -> str:
    """Extracts the true surname of a player, stripping generational suffixes."""
    tokens = full_name.split()
    suffixes = {'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'v'}
    while len(tokens) > 1 and tokens[-1].lower() in suffixes:
        tokens.pop()
    return tokens[-1] if tokens else full_name


def clause_belongs_to_player(
    clauses: list[str],
    player_name: str,
    surname: str,
    other_players: list[str] | None = None,
) -> tuple[bool, bool]:
    """
    Determines whether any clause containing an exit verb belongs to the specified player
    by tracking active grammatical subjects across compound clauses and preventing
    ejections of coaches, officials, or other non-player entities from being falsely
    attributed to a player.

    Returns (has_exit, is_ejection).
    """
    other_players = other_players or []

    # 1. If player is explicitly mentioned in an inactive clause, disqualify immediately
    for c in clauses:
        c_strip = c.strip()
        has_player = (player_name.lower() in c_strip.lower()) or (
            len(surname) >= 4 and re.search(rf'\b{re.escape(surname.lower())}\b', c_strip.lower())
        )
        if has_player and PREGAME_INACTIVE_PATTERN.search(c_strip):
            return False, False

    # 2. Track discourse subject across clauses
    active_subject_is_player = False
    for c in clauses:
        c_strip = c.strip()
        if not c_strip:
            continue

        has_player = (player_name.lower() in c_strip.lower()) or (
            len(surname) >= 4 and re.search(rf'\b{re.escape(surname.lower())}\b', c_strip.lower())
        )

        has_other_player = any(
            (op.lower() in c_strip.lower()) or (
                len(get_player_surname(op)) >= 4 and re.search(rf'\b{re.escape(get_player_surname(op).lower())}\b', c_strip.lower())
            )
            for op in other_players if op.lower() != player_name.lower()
        )
        other_subject_match = re.search(
            r'\b([A-Z][a-z]+)\s+(was\s+(tossed|ejected|hit|assessed)|received\b)',
            c_strip,
        )
        has_other_named_entity = (
            other_subject_match is not None
            and other_subject_match.group(1).lower() not in (player_name.lower(), surname.lower())
        )
        has_other_subject = bool(NON_PLAYER_SUBJECT_PATTERN.search(c_strip)) or has_other_player or has_other_named_entity

        if has_player:
            active_subject_is_player = True
        elif has_other_subject:
            active_subject_is_player = False

        if active_subject_is_player:
            if MIDGAME_EXIT_PATTERN.search(c_strip):
                is_ej = bool(re.search(r'\b(ejected|ejection|disqualified|tossed)\b', c_strip, re.IGNORECASE))
                return True, is_ej

    return False, False


class LeftMidgame:
    """
    Collects and identifies NBA players who exited mid-game due to injury, illness,
    or other physical reasons, ensuring they are not conflated with standard rotation benches.
    """

    def __init__(
        self,
        nbacom_players_list: list[str] | pd.Series | np.ndarray | pd.DataFrame | None = None,
        player_id_to_name: dict[int, str] | None = None,
        player_name_to_id: dict[str, int] | None = None,
    ):
        if isinstance(nbacom_players_list, pd.DataFrame):
            nbacom_players_list = nbacom_players_list['NAME'].values
        self.player_id_to_name = player_id_to_name or {}
        self.player_name_to_id = player_name_to_id or {}
        self.fix_player_names = FixPlayerNames(compare_names=nbacom_players_list)
        self.left_midgame_players: set[str] = set()
        self.df = EMPTY_DF
        self.save_dir = update_dir_name('left_midgame')

    @staticmethod
    @retry(max_retries=3, delay_seconds=1, return_on_fail=[])
    def _get_espn_game_ids_for_date(date: datetime.date) -> list[int]:
        """Fetch all ESPN game event IDs for a specific calendar date."""
        date_str = date.strftime('%Y%m%d')
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []
        data = response.json()
        return [int(event['id']) for event in data.get('events', []) if 'id' in event]

    @staticmethod
    def _check_play_by_play_data(json_data: dict) -> list[dict]:
        """Scans play-by-play logs for explicit ejection or injury exit events."""
        plays = json_data.get('plays', [])
        exits = []
        for play in plays:
            text = play.get('text', '')
            if not text:
                continue
            is_ejection = bool(re.search(r'\b(ejection|ejected)\b', text, re.IGNORECASE))
            is_injury = bool(re.search(MIDGAME_EXIT_PATTERN, text))
            if is_ejection or is_injury:
                for participant in play.get('participants', []):
                    player_name = participant.get('athlete', {}).get('displayName')
                    if player_name:
                        exits.append({
                            'player': player_name,
                            'live_log': text,
                            'is_ejection': is_ejection,
                        })
        return exits

    def _extract_midgame_exits_from_html(
        self,
        html_text: str,
        roster_players: dict[str, str | int] | None = None,
    ) -> list[dict]:
        """
        Extracts players who exited mid-game from recap article HTML by linking
        the player's identity to active exit clauses while excluding pre-game inactives,
        court positions, and players who returned to play.
        """
        if not html_text:
            return []

        roster_players = roster_players or {}
        paragraphs = re.split(r'</p>|\n\s*\n', html_text)

        # Pass 1: Identify players who returned to action
        returned_players = set()
        second_half_out_players = set()
        for p in paragraphs:
            p_text = BeautifulSoup(p, 'html.parser').get_text()
            if not p_text.strip():
                continue

            p_clean = CLOCK_OR_COURT_PATTERN.sub('', p_text)
            p_players = set()
            for a in BeautifulSoup(p, 'html.parser').find_all('a', href=re.compile(r'/player/_/id/')):
                raw = a.get_text().strip()
                if raw:
                    p_players.add(raw)
            for r_name in roster_players:
                last = get_player_surname(r_name)
                if (r_name.lower() in p_clean.lower()) or (
                    len(last) >= 4 and re.search(rf'\b{re.escape(last.lower())}\b', p_clean.lower())
                ):
                    p_players.add(r_name)

            if SECOND_HALF_OUT_PATTERN.search(p_text):
                for p_name in p_players:
                    last = get_player_surname(p_name)
                    if p_name.lower() in p_text.lower() or (
                        len(last) >= 4 and re.search(rf'\b{re.escape(last.lower())}\b', p_text.lower())
                    ):
                        second_half_out_players.add(p_name)

            if RETURNED_PATTERN.search(p_text):
                if len(p_players) == 1:
                    returned_players.update(p_players)
                else:
                    cleaned_p = re.sub(r',\s*who\s+[^,]+,', ' ', p_text)
                    sentences = re.split(r'\. |\.\n', cleaned_p)
                    for s in sentences:
                        clauses = re.split(r';| but | after | while | in which |, then | and ', s)
                        for c in clauses:
                            if RETURNED_PATTERN.search(c):
                                for p_name in p_players:
                                    last = get_player_surname(p_name)
                                    if p_name.lower() in c.lower() or (
                                        len(last) >= 4 and re.search(rf'\b{re.escape(last.lower())}\b', c.lower())
                                    ):
                                        returned_players.add(p_name)

        # Remove players who were ruled out for the second half from returned_players
        returned_players = returned_players - second_half_out_players

        # Pass 2: Extract mid-game departures
        exits = []
        seen = set()

        for p in paragraphs:
            soup = BeautifulSoup(p, 'html.parser')
            p_text = soup.get_text()
            if not p_text.strip():
                continue

            p_clean = CLOCK_OR_COURT_PATTERN.sub('', p_text)
            mentioned_players = set()
            for a in soup.find_all('a', href=re.compile(r'/player/_/id/')):
                raw = a.get_text().strip()
                if raw:
                    mentioned_players.add(raw)

            for r_name in roster_players:
                last = get_player_surname(r_name)
                if (r_name.lower() in p_clean.lower()) or (
                    len(last) >= 4 and re.search(rf'\b{re.escape(last.lower())}\b', p_clean.lower())
                ):
                    mentioned_players.add(r_name)

            cleaned_p = re.sub(r',\s*who\s+[^,]+,', ' ', p_text)
            sentences = re.split(r'\. |\.\n', cleaned_p)

            for s in sentences:
                s_clean = CLOCK_OR_COURT_PATTERN.sub('', s)

                # Find all players in this specific sentence
                players_in_s = []
                for raw_name in mentioned_players:
                    last_name = get_player_surname(raw_name)
                    if (raw_name.lower() in s_clean.lower()) or (
                        len(last_name) >= 4 and re.search(rf'\b{re.escape(last_name.lower())}\b', s_clean.lower())
                    ):
                        players_in_s.append(raw_name)

                if not players_in_s:
                    continue

                clauses = re.split(r';| but | after | while | in which |, then | and ', s_clean)

                for raw_name in players_in_s:
                    if raw_name in seen or raw_name in returned_players:
                        continue
                    fixed_names = self.fix_player_names.fix_names([raw_name])
                    fixed_name = str(fixed_names[0]) if len(fixed_names) > 0 else raw_name
                    if fixed_name in seen or fixed_name in returned_players:
                        continue
                    last_name = get_player_surname(raw_name)

                    # Check if raw_name is a bystander in s
                    is_bystander = False
                    for b_match in BYSTANDER_PATTERN.finditer(s):
                        target = b_match.group(2).lower()
                        if raw_name.lower() in target or (len(last_name) >= 4 and last_name.lower() in target):
                            is_bystander = True
                            break
                    if is_bystander:
                        continue

                    has_exit, is_ejection = clause_belongs_to_player(clauses, raw_name, last_name, other_players=players_in_s)
                    if has_exit:
                        exits.append({
                            'NAME': fixed_name,
                            'RAW_NAME': raw_name,
                            'REASON': s.strip(),
                            'IS_EJECTION': is_ejection,
                        })
                        seen.add(raw_name)
                        seen.add(fixed_name)

        return exits

    def _extract_midgame_exits_from_recap(self, json_data: dict) -> list[dict]:
        """Extracts players who exited mid-game from the ESPN article JSON."""
        article_data = json_data.get('article', {})
        story_html = article_data.get('story', '')
        if not story_html:
            headline = article_data.get('headline', '')
            description = article_data.get('description', '')
            story_html = f"<p>{headline}</p><p>{description}</p>"

        roster_players = {}
        for team in json_data.get('boxscore', {}).get('players', []):
            for stat in team.get('statistics', []):
                for ath in stat.get('athletes', []):
                    name = ath.get('athlete', {}).get('displayName', '')
                    dnp = ath.get('didNotPlay', False)
                    if name and not dnp:
                        roster_players[name] = ath.get('athlete', {}).get('id')

        return self._extract_midgame_exits_from_html(story_html, roster_players=roster_players)

    @retry(max_retries=3, delay_seconds=1, return_on_fail={})
    def _fetch_espn_game_summary(self, game_id: int) -> dict:
        url = f"https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return {}
        return response.json()

    def collect_data(
        self,
        date: datetime.date = get_current_date(),
        players_played: set[str] | list[str] | None = None,
    ) -> set[str]:
        """
        Scrapes or loads players who left mid-game on a given date.
        Returns a set of normalized player names.
        """
        self.left_midgame_players = set()
        save_path = os.path.join(self.save_dir, f"{str(date).replace('-', '_')}.feather")

        # Load from disk if previously cached and not today
        if os.path.isfile(save_path) and date != get_current_date():
            try:
                cached_df = pd.read_feather(save_path)
                if not cached_df.empty and 'NAME' in cached_df.columns:
                    left_mask = cached_df['LEFT_MIDGAME'] if 'LEFT_MIDGAME' in cached_df.columns else True
                    names = set(cached_df.loc[left_mask, 'NAME'].dropna().unique())
                    self.left_midgame_players = names
                    self.df = cached_df
                    return self.left_midgame_players
            except Exception as e:
                logging.warning(f"Failed to read cached left_midgame file {save_path}: {e}")

        espn_gids = self._get_espn_game_ids_for_date(date)
        if not espn_gids:
            self.df = EMPTY_DF
            return self.left_midgame_players

        exit_records = []
        for espn_gid in espn_gids:
            summary = self._fetch_espn_game_summary(espn_gid)
            if not summary:
                continue

            recap_exits = self._extract_midgame_exits_from_recap(summary)
            for item in recap_exits:
                name = item['NAME']
                exit_records.append({
                    'NAME': name,
                    'DATE': date,
                    'LEFT_MIDGAME': True,
                    'IS_EJECTION': item.get('IS_EJECTION', False),
                    'REASON': item.get('REASON', ''),
                })
                self.left_midgame_players.add(name)

        # Cross-reference with players who actually played if provided
        if players_played is not None:
            played_set = set(players_played)
            self.left_midgame_players = self.left_midgame_players & played_set
            exit_records = [r for r in exit_records if r['NAME'] in self.left_midgame_players]

        if exit_records:
            self.df = pd.DataFrame(exit_records)
            if 'TEAM' not in self.df.columns:
                self.df['TEAM'] = ''
            self.df = self.df[['NAME', 'TEAM', 'DATE', 'LEFT_MIDGAME']].drop_duplicates(subset=['NAME', 'DATE'])
        else:
            self.df = EMPTY_DF.copy()

        # Save cache
        try:
            self.df.to_feather(save_path)
        except Exception as e:
            logging.warning(f"Failed to save left_midgame feather file {save_path}: {e}")

        return self.left_midgame_players

    def collect_df(self, date: datetime.date = get_current_date()) -> pd.DataFrame:
        self.collect_data(date=date)
        return self.df


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

    test_date = datetime.date(2024, 11, 1)
    print(f"Finding players who left mid-game on {test_date}...")
    left_players = lm.collect_data(test_date)
    print("Found players:", left_players)
