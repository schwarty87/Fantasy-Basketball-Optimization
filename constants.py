import datetime
import os
from typing import Union
import pytz

from machine_learning.constants import DATA_DIR
from enums import Browsers, Sites

IGNORE_VAL = -999
UNKNOWN_PLAYER_ID = 0
NY_TZ = pytz.timezone('America/New_York')

START_POSITIONS = ['PG', 'SG', 'SF', 'PF', 'C']
ROTOGURU_PASSWORDS = ['S9436562151', 'S2671592161']
ROTOWIRE_SITE_NAME_TO_ID = {Sites.FANDUEL: 2, Sites.DRAFTKINGS: 1, Sites.YAHOO: 3}
YAHOO_CATEGORIES = ['PTS', 'REB', 'AST', 'STL', 'BLK', '3PM',  'FGM', 'FGA', 'FTA', 'FTM', 'TOV']
DFS_POSITIONS = {Sites.YAHOO: ['PG', 'SG', 'G', 'SF', 'PF', 'F', 'C', 'UTIL']}
DFS_MAX_TEAM_SALARY = {Sites.YAHOO: 200}
POSITION_TO_NUM = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
NUM_TO_POSITION = {v: k for k, v in POSITION_TO_NUM.items()}
# CONTEST_IDS_FILE_ID_IN_GOOGLE_DRIVE = '1_ww9GHan5D6l5w_dP6es28ciucuaPK3f'


SECONDS_PER_DAY = 60 * 60 * 24


class URLS:
    TRANSACTIONS_NBACOM = 'https://stats.nba.com/js/data/playermovement/NBA_Player_Movement.json'
    BOXSCORES_NBACOM = 'https://stats.nba.com/stats/{player_or_team}gamelogs?DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00&Location=&MeasureType={measure_type}&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=Totals&Period=0&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType={season_type}&ShotClockRange=&VsConference=&VsDivision='
    BOXSCORES_NBACOM2 = 'https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo=&Direction=DESC&ISTRound=&LeagueID=00&PlayerOrTeam={player_or_team}&Season={season_str}&SeasonType={season_type}&Sorter=DATE'
    BOXSCORES_NBACOM_TEAM = 'https://stats.nba.com/stats/teamgamelogs?DateFrom=&DateTo=&GameSegment=&ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType={measure_type}&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=Totals&Period=0&PlusMinus=N&Rank=N&Season={season}&SeasonSegment=&SeasonType={season_type}&ShotClockRange=&VsConference=&VsDivision='
    GAMES_SCOREBOARD_NBACOM = 'https://stats.nba.com/stats/scoreboardv3?GameDate={date}&LeagueID=00'
    SINGLE_BOXSCORES_NBACOM = 'https://stats.nba.com/stats/boxscoretraditionalv3?EndPeriod=1&EndRange=0&GameID={gid}&RangeType=0&StartPeriod=1&StartRange=0'
    PLAYERS_LIST_NBACOM = 'https://stats.nba.com/stats/playerindex?College=&Country=&DraftPick=&DraftRound=&DraftYear=&Height=&Historical=1&LeagueID=00&Season={season_str}&SeasonType=Regular%20Season&TeamID=0&Weight='
    PLAYERS_LIST_NBACOM2 = 'https://stats.nba.com/stats/leaguedashplayerbiostats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=PerGame&Period=0&PlayerExperience=&PlayerPosition=&Season={season_str}&SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='
    PLAYERS_LIST_NBACOM3 = 'https://stats.nba.com/stats/playerindex?College=&Country=&DraftPick=&DraftRound=&DraftYear=&Height=&Historical=1&LeagueID=00&Season={season_str}&SeasonType=Regular&TeamID=0&Weight='
    LINEUPS_ROTOGRINDERS = 'https://rotogrinders.com/lineups/nba?date={date}&site={site}'
    INJURIES_ROTOGRINDERS = 'https://rotogrinders.com/grids/nba-dfs-injury-report-the-situation-room-impact-ratings-analysis-2703084'
    SEASON_SCHEDULE = 'https://stats.nba.com/stats/scheduleleaguev2?Season={season_str}&LeagueID=00'
    SEASON_SCHEDULE2 = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json'
    NBACOM_GAMES_LINEUPS = 'https://stats.nba.com/js/data/leaders/00_daily_lineups_{datestr}.json'
    ROTOWIRE_INJURIES = 'https://www.rotowire.com/basketball/tables/injury-report.php?team=ALL&pos=ALL'
    BASKETBALL_MONSTER_LINEUPS = "https://basketballmonster.com/nbalineups.aspx"
    EJECTIONS_SPOTRAC = 'https://www.spotrac.com/nba/fines-suspensions/fines/ejection/'

    @staticmethod
    def _season_int_to_str(season: Union[int, str]):
        if isinstance(season, int):
            return str(season) + '-' + str(season + 1)[-2:]
        return season

    def boxscores_nbacom(self, player_or_team: str, measure_type: str, season: Union[int, str],
                         season_type: str = 'Regular Season'):
        return self.BOXSCORES_NBACOM.format(player_or_team=player_or_team, measure_type=measure_type,
                                            season=self._season_int_to_str(season), season_type=season_type)

    def boxscores_nbacom2(self, player_or_team: str, season: Union[int, str],
                         season_type: str = 'Regular Season'):
        return self.BOXSCORES_NBACOM2.format(player_or_team=player_or_team, season_str=self._season_int_to_str(season),
                                             season_type=season_type)

    def boxscores_nbacom_team(self, measure_type: str, season: Union[int, str],
                              season_type: str = 'Regular Season') -> str:
        if season_type == 'Regular Season':
            season_type = 'Regular%20Season'
        return self.BOXSCORES_NBACOM_TEAM.format(measure_type=measure_type,
                                                 season=self._season_int_to_str(season),
                                                 season_type=season_type)

    def games_scoreboard_nbacom(self, date: str):
        return self.GAMES_SCOREBOARD_NBACOM.format(date=date)

    def single_boxscores_nbacom(self, gid: int | str):
        return self.SINGLE_BOXSCORES_NBACOM.format(gid=gid)

    def players_list_nbacom(self, season: Union[int, str]):
        return self.PLAYERS_LIST_NBACOM.format(season_str=self._season_int_to_str(season))

    def players_list_nbacom2(self, season: Union[int, str]):
        return self.PLAYERS_LIST_NBACOM2.format(season_str=self._season_int_to_str(season))

    def players_list_nbacom3(self, season: Union[int, str]):
        return self.PLAYERS_LIST_NBACOM3.format(season_str=self._season_int_to_str(season))

    def lineups_rotogrinders(self, date: Union[datetime.date, str], site: Sites):
        if isinstance(date, datetime.date):
            date = str(date)
        return self.LINEUPS_ROTOGRINDERS.format(date=date, site=site.name.lower())

    def season_schedule(self, season: Union[int, str]):
        return self.SEASON_SCHEDULE.format(season_str=self._season_int_to_str(season))
        # return self.SEASON_SCHEDULE2

    def games_lineups_nbacom(self, datestr: str):
        return self.NBACOM_GAMES_LINEUPS.format(datestr=datestr.replace('-', '').replace('/', ''))


TEAM_ID_TO_NAME = {1610612742: 'DAL', 1610612745: 'HOU', 1610612747: 'LAL', 1610612740: 'NOP', 1610612753: 'ORL',
                   1610612759: 'SAS', 1610612737: 'ATL', 1610612751: 'BKN', 1610612738: 'BOS', 1610612766: 'CHA',
                   1610612741: 'CHI', 1610612743: 'DEN', 1610612765: 'DET', 1610612744: 'GSW', 1610612754: 'IND',
                   1610612763: 'MEM', 1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN', 1610612752: 'NYK',
                   1610612760: 'OKC', 1610612755: 'PHI', 1610612756: 'PHX', 1610612757: 'POR', 1610612758: 'SAC',
                   1610612761: 'TOR', 1610612762: 'UTA', 1610612764: 'WAS', 1610612739: 'CLE', 1610612746: 'LAC'}
TEAM_IDS = list(TEAM_ID_TO_NAME.keys())
NUM_TEAMS = len(TEAM_IDS)

FIX_PLAYER_NAMES = {
    'MJ Walker': 'M.J. Walker',
    'Cameron Reynolds': 'Cam Reynolds',
    'Nene Hilario': 'Nene',
    'PJ Dozier': 'P.J. Dozier',
    'Enes Kanter': 'Enes Freedom',
    "Louis Williams":  "Lou Williams",
    "C.J. McCollum": "CJ McCollum",
    "Timothe Luwawu": "Timothe Luwawu-Cabarrot",
    "Harry Giles": "Harry Giles III",
    "Marcus Morris": "Marcus Morris Sr.",
    "Robert Williams": "Robert Williams III",
    "Kelly Oubre": "Kelly Oubre Jr.",
    "Danuel House": "Danuel House Jr.",
    "Derrick Jones": "Derrick Jones Jr.",
    "Marvin Bagley": "Marvin Bagley III",
    "James Ennis": "James Ennis III",
    "Otto Porter": "Otto Porter Jr.",
    "Michael Porter": "Michael Porter Jr.",
    "Kevin Knox": "Kevin Knox II",
    "Larry Nance": "Larry Nance Jr.",
    "Juan Hernangomez": "Juancho Hernangomez",
    "Sviatoslav Mykhailiuk": "Svi Mykhailiuk",
    "Wes Matthews": "Wesley Matthews",
    "DeAndre Bembry": "DeAndre' Bembry",
    "J.J. Redick": "JJ Redick",
    "Mohamed Bamba": "Mo Bamba",
    "Ishmael Smith": "Ish Smith",
    "Guillermo Hernangomez": "Willy Hernangomez",
    "Lonnie Walker": "Lonnie Walker IV",
    "Nazareth Mitrou-Long": "Naz Mitrou-Long",
    "B.J. Johnson": "BJ Johnson",
    "Frank Mason": "Frank Mason III",
    "TJ Leaf": "T.J. Leaf",
    "Charlie Brown": "Charlie Brown Jr.",
    "Matthew Mooney": "Matt Mooney",
    "Melvin Frazier": "Melvin Frazier Jr.",
    "Jacob Evans III": "Jacob Evans",
    "J.R. Smith": "JR Smith",
    "Michael Frazier": "Michael Frazier II",
    "C.J. Miles": "CJ Miles",
    "Jose Barea": "J.J. Barea",
    "Wade Baldwin": "Wade Baldwin IV",
    "D.J. Stephens": "DJ Stephens",
    "Mitch Creek": "Mitchell Creek",
    "Walter Lemon Jr.": "Walt Lemon Jr.",
    "R.J. Hunter": "RJ Hunter",
    "Matt Williams": "Matt Williams Jr.",
    "Vince Hunter": "Vincent Hunter",
    "Johnny O'Bryant": "Johnny O'Bryant III",
    "A.J. Hammons": "AJ Hammons",
    "James Webb": "James Webb III",
    "Trey McKinney Jones": "Trey McKinney-Jones",
    "Andrew White": "Andrew White III",
    "K.J. McDaniels": "KJ McDaniels",
    "Joseph Young": "Joe Young",
    "Mike Young": "Michael Young",
    "Tim Hardaway": "Tim Hardaway Jr.",
    'PJ Tucker': 'P.J. Tucker',
    "PJ Washington": "P.J. Washington",
    "P.J. Washington Jr.": "P.J. Washington",
    "Troy Brown": "Troy Brown Jr.",
    "Gary Trent": "Gary Trent Jr.",
    "KJ Martin": "Kenyon Martin Jr.",
    "Kenyon Martin": "Kenyon Martin Jr.",
    "Kevin Porter": "Kevin Porter Jr.",
    "Jaren Jackson": "Jaren Jackson Jr.",
    "Didi Louzada Silva": "Didi Louzada",
    "Gary Payton": "Gary Payton II",
    "Ishmail Wainright": "Ish Wainright",
    "Wendell Carter": "Wendell Carter Jr.",
    "Trey Murphy": "Trey Murphy III",
    "Vernon Carey": "Vernon Carey Jr.",
    "Robert Woodard": "Robert Woodard II",
    "R.J. Nembhard": "RJ Nembhard Jr.",
    "Nicolas Claxton": "Nic Claxton",
    "McKinley Wright": "McKinley Wright IV",
    "Kira Lewis": "Kira Lewis Jr.",
    "Greg Brown": "Greg Brown III",
    "Dennis Smith": "Dennis Smith Jr.",
    "David Duke": "David Duke Jr.",
    "Duane Washington": "Duane Washington Jr.",
    "Xavier Tillman Sr.": "Xavier Tillman",
    "Josh Primo": "Joshua Primo",
    "Chaundee Brown": "Chaundee Brown Jr.",
    "Derrick Walton": "Derrick Walton Jr.",
    "RJ Nembhard": "RJ Nembhard Jr.",
    "Lindy Waters": "Lindy Waters III",
    "Scotty Pippen": "Scotty Pippen Jr.",
    "Vince Williams": "Vince Williams Jr.",
    "Vincent Williams": "Vince Williams Jr.",
    "Vincent Williams Jr.": "Vince Williams Jr.",
    "Kenneth Lofton": "Kenneth Lofton Jr.",
    "Herb Jones": "Herbert Jones",
    "OG Anunoby": "O.G. Anunoby",
    "Jabari Smith": "Jabari Smith Jr.",
    "Jabari Smith Jr": "Jabari Smith Jr.",
    "Lu Dort": "Luguentz Dort",
    "Wendell Moore": "Wendell Moore Jr.",
    "Cam Johnson": "Cameron Johnson",
    "AJ Green": "A.J. Green",
    "Brandon Boston Jr.": "Brandon Boston",
    "Ron Harper": "Ron Harper Jr.",
    "Michael Foster": "Michael Foster Jr.",
    "Mark Willams": "Mark Williams",
    "Jeff Dowtin": "Jeff Dowtin Jr.",
    "JD Davidson": "Jermareo Davidson",
    "John Butler": "John Butler Jr.",
    "TyTy Washington": "TyTy Washington Jr.",
    "Moe Wagner": "Moritz Wagner",
    "Mo Wagner": "Moritz Wagner",
    "TJ Warren": "T.J. Warren",
    "Devonte Graham": "Devonte' Graham",
    "Shaq Harrison": "Shaquille Harrison",
    "Brian Bowen": "Brian Bowen II",
    "D.J. Stewart": "DJ Stewart",
    "Tre Scott": "Trevon Scott",
    "GG Jackson II": "GG Jackson",
    "Gregory Jackson": "GG Jackson",
    "Gregory Jackson II": "GG Jackson",
    "G.G. Jackson": "GG Jackson",
    "Jaime Jaquez": "Jaime Jaquez Jr.",
    "Nick Smith": "Nick Smith Jr.",
    "Patrick Baldwin": "Patrick Baldwin Jr.",
    "Reggie Bullock Jr.": "Reggie Bullock",
    "Andre Jackson": "Andre Jackson Jr.",
    "Nate Williams": "Jeenathan Williams",
    "Jermaine Samuels": "Jermaine Samuels Jr.",
    "Craig Porter": "Craig Porter Jr.",
    "RJ Hampton": "R.J. Hampton",
    "Dereck Lively": "Dereck Lively II",
    "AJ Lawson": "A.J. Lawson",
    "Nah'Shon Hyland": "Bones Hyland",
    'DJ Carton': 'D.J. Carton',
    'Miles Mcbride': 'Miles McBride',
    'EJ Liddell': 'E.J. Liddell',
    'Ricky Council': 'Ricky Council IV',
    'DeAndre Hunter': "De'Andre Hunter",
    "Cameron Reddish": "Cam Reddish",
    "Jae'sean Tate": "Jae'Sean Tate",
    'Matthew Hurt': 'Matt Hurt',
    'Onurulp Bitim': 'Onuralp Bitim',
    'DeAndre Ayton': 'Deandre Ayton',
    "Cameron Thomas": "Cam Thomas",
    'Marjon Beauchamp': 'MarJon Beauchamp',
    'Bryce Mcgowens': 'Bryce McGowens',
    'DJ Wilson': 'D.J. Wilson',
    'Dom Barlow': 'Dominick Barlow',
    'J.T. Thor': 'JT Thor',
    'Troy Brown Jr': 'Troy Brown Jr.',
    'Moe Harkless': 'Maurice Harkless',
    'Ron Holland II': 'Ronald Holland II',
    'Bub Carrington': 'Carlton Carrington',
    'Jakob Poltl': 'Jakob Poeltl',
    'Ron Holland': 'Ronald Holland II',
    'Yongxi Cui': 'Cui Yongxi',
    'Kevin McCullar': 'Kevin McCullar Jr.',
    'Terrence Shannon': 'Terrence Shannon Jr',
    'Alex Sarr': 'Alexandre Sarr',
    'Tristan Da Silva': 'Tristan da Silva',
    'DaRon Holmes': 'DaRon Holmes II',
    'Kenneth Simpson': 'KJ Simpson',
    'R.J. Barrett': 'RJ Barrett',
    'Trey Jemison': 'Trey Jemison III',
    'Boo Buie': 'Boo Buie III',
    'Robert Dillingham': 'Rob Dillingham',
    'Robert DIllingham': 'Rob Dillingham',
    'David Jones': 'David Jones Garcia',
    'Tolu Smith III': 'Tolu Smith',
    'Jimmy Butler III': 'Jimmy Butler',
    'Keion Brooks': 'Keion Brooks Jr.',
    'Cam Payne': 'Cameron Payne',
    'Walter Clayton': 'Walter Clayton Jr.',
    "Nae'qwan Tomlin": "Nae'Qwan Tomlin",
    'Nigel Hayes': 'Nigel Hayes-Davis',
    'David Jones-Garcia': 'David Jones Garcia',
    'Eli Ndiaye': 'Eli John Ndiaye',
    'Nikola Djurisic': 'Nikola Durisic',
    'EJ Harkless': 'Elijah Harkless',
    'Hansen Yang': 'Yang Hansen',
    'L.J. Cryer': 'LJ Cryer',
    "Eli N'Diaye": 'Eli John Ndiaye',
    'Bobby Portis Jr.': 'Bobby Portis'
}

FIX_TEAM_NAMES = {'NOR': 'NOP',
                  'NO': 'NOP',
                  'PHO': 'PHX',
                  'GS': 'GSW',
                  'NY': 'NYK',
                  'SA': 'SAS',
                  'SAN': 'SAS',
                  'BRO': 'BKN',
                  'CHR': 'CHA',
                  'Atlanta Hawks': 'ATL',
                  'Boston Celtics': 'BOS',
                  'Brooklyn Nets': 'BKN',
                  'Charlotte Hornets': 'CHA',
                  'Chicago Bulls': 'CHI',
                  'Cleveland Cavaliers': 'CLE',
                  'Cleveland Cavs': 'CLE',
                  'Dallas Mavericks': 'DAL',
                  'Denver Nuggets': 'DEN',
                  'Detroit Pistons': 'DET',
                  'Golden State Warriors': 'GSW',
                  'Houston Rockets': 'HOU',
                  'Indiana Pacers': 'IND',
                  'Los Angeles Lakers': 'LAL',
                  'Los Angeles Clippers': 'LAC',
                  'Memphis Grizzlies': 'MEM',
                  'Miami Heat': 'MIA',
                  'Milwaukee Bucks': 'MIL',
                  'Minnesota Timberwolves': 'MIN',
                  'New Orleans Pelicans': 'NOP',
                  'New York Knicks': 'NYK',
                  'Oklahoma City Thunder': 'OKC',
                  'Orlando Magic': 'ORL',
                  'Philadelphia 76ers': 'PHI',
                  'Phoenix Suns': 'PHX',
                  'Portland Trail Blazers': 'POR',
                  'Sacramento Kings': 'SAC',
                  'San Antonio Spurs': 'SAS',
                  'Toronto Raptors': 'TOR',
                  'Utah Jazz': 'UTA',
                  'Washington Wizards': 'WAS',
                  'Atlanta': 'ATL',
                  'Brooklyn': 'BKN',
                  'Boston': 'BOS',
                  'Charlotte': 'CHA',
                  'Chicago': 'CHI',
                  'Cleveland': 'CLE',
                  'Dallas': 'DAL',
                  'Denver': 'DEN',
                  'Detroit': 'DET',
                  'Golden State': 'GSW',
                  'Golden St.': 'GSW',
                  'Houston': 'HOU',
                  'Indiana': 'IND',
                  'L.A. Clippers': 'LAC',
                  'L.A. Lakers': 'LAL',
                  'Memphis': 'MEM',
                  'Miami': 'MIA',
                  'Milwaukee': 'MIL',
                  'Minnesota': 'MIN',
                  'New Orleans': 'NOP',
                  'New York': 'NYK',
                  'Oklahoma City': 'OKC',
                  'Orlando': 'ORL',
                  'Philadelphia': 'PHI',
                  'Phoenix': 'PHX',
                  'Portland': 'POR',
                  'Sacramento': 'SAC',
                  'San Antonio': 'SAS',
                  'Toronto': 'TOR',
                  'Utah': 'UTA',
                  'Washington': 'WAS',
                  'Hawks': 'ATL',
                  'Celtics': 'BOS',
                  'Nets': 'BKN',
                  'Hornets': 'CHA',
                  'Bulls': 'CHI',
                  'Cavaliers': 'CLE',
                  'Cavs': 'CLE',
                  'Mavericks': 'DAL',
                  'Nuggets': 'DEN',
                  'Pistons': 'DET',
                  'Warriors': 'GSW',
                  'Rockets': 'HOU',
                  'Pacers': 'IND',
                  'Lakers': 'LAL',
                  'Clippers': 'LAC',
                  'Grizzlies': 'MEM',
                  'Heat': 'MIA',
                  'Bucks': 'MIL',
                  'Timberwolves': 'MIN',
                  'Wolves': 'MIN',
                  'Pelicans': 'NOP',
                  'Knicks': 'NYK',
                  'Thunder': 'OKC',
                  'Magic': 'ORL',
                  '76ers': 'PHI',
                  'Suns': 'PHX',
                  'Trail Blazers': 'POR',
                  'Blazers': 'POR',
                  'Kings': 'SAC',
                  'Spurs': 'SAS',
                  'Raptors': 'TOR',
                  'Jazz': 'UTA',
                  'Wizards': 'WAS',
                  }

TEAM_NAME_TO_ID = {'ATL': 1610612737,
                   'BKN': 1610612751,
                   'BRO': 1610612751,
                   'BOS': 1610612738,
                   'CHA': 1610612766,
                   'CHR': 1610612766,
                   'CHI': 1610612741,
                   'CLE': 1610612739,
                   'DAL': 1610612742,
                   'DEN': 1610612743,
                   'DET': 1610612765,
                   'GSW': 1610612744,
                   'GS': 1610612744,
                   'HOU': 1610612745,
                   'IND': 1610612754,
                   'LAC': 1610612746,
                   'LAL': 1610612747,
                   'MEM': 1610612763,
                   'MIA': 1610612748,
                   'MIL': 1610612749,
                   'MIN': 1610612750,
                   'NOP': 1610612740,
                   'NOR': 1610612740,
                   'NYK': 1610612752,
                   'NY': 1610612752,
                   'OKC': 1610612760,
                   'ORL': 1610612753,
                   'PHI': 1610612755,
                   'PHX': 1610612756,
                   'PHO': 1610612756,
                   'POR': 1610612757,
                   'SAC': 1610612758,
                   'SAS': 1610612759,
                   'SAN': 1610612759,
                   'TOR': 1610612761,
                   'UTA': 1610612762,
                   'WAS': 1610612764
                   }

SCORING = {Sites.FANDUEL: {'PTS': 1, 'REB': 1.2, 'AST': 1.5, 'STL': 3, 'BLK': 3, 'TOV': -1},
           Sites.DRAFTKINGS: {'PTS': 1, '3PM': 0.5, 'REB': 1.25, 'AST': 1.5, 'STL': 2, 'BLK': 2, 'TOV': -0.5,
                          'DOUBLE_DOUBLE': 1.5, 'TRIPLE_DOUBLE': 3},
           Sites.YAHOO: {'PTS': 1, 'REB': 1.2, 'AST': 1.5, 'STL': 3, 'BLK': 3, 'TOV': -1}
           }

SUPERSTARS = ['LeBron James', 'Kevin Durant', 'Anthony Davis', 'Tim Duncan', 'Kobe Bryant', 'Jrue Holiday',
              'Julius Randle', 'Nikola Jokic', 'Joel Embiid', 'Giannis Antetokounmpo', 'Luka Doncic', 'Trae Young',
              'James Harden', 'Karl-Anthony Towns', 'Damian Lillard', 'Kawhi Leonard', 'Paul George', 'Jayson Tatum',
              'Stephen Curry', 'Tyrese Haliburton', 'LaMelo Ball', 'Anthony Edwards', 'Kyrie Irving', 'Devin Booker',
              'Domantas Sabonis', 'Ja Morant', 'Bradley Beal', 'Jimmy Butler', 'Chris Paul', 'Bam Adebayo',
              'Shai Gilgeous-Alexander', 'Karl-Anthony Towns', 'Tyrese Maxey', 'Donovan Mitchell', 'Victor Wembanyama',
              'Chet Holmgren', 'Zion Williamson', 'DeMar DeRozan', "De'Aron Fox", 'Jaylen Brown', 'Brandon Ingram',
              'Paolo Banchero', 'Zach LaVine', 'Chris Bosh', 'Kristaps Porzingis', 'Julius Randle', 'Russell Westbrook',
              'Klay Thompson', 'Cade Cunningham', 'LaMelo Ball', 'Paskal Siakam', 'Nikola Vucevic', 'Jalen Brunson',
              'Jaren Jackson Jr.', 'Tyrese Haliburton', 'Bam Adebayo', 'Mikal Bridges', 'Alperen Sengun', 'Myles Turner']


BROWSER = Browsers.FIREFOX