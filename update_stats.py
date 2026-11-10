import argparse
import logging
import pandas as pd
from utils import (
    define_logger,
    get_current_date,
    get_current_season,
    load_and_cache_players_list,
    scrape_season_schedules,
    update_dir_name,
    update_todays_schedule_from_scoreboard,
)
from data_collectors.stats_loaders import StatsLoaderTeams, StatsLoaderPlayers
from calc_features import CalcFeatures


def update_all_stats(
    season: int | None = None,
    with_features: bool = True,
    add_upcoming_games: bool = False,
    team_only: bool = False,
    player_only: bool = False,
    stats_loader_players: StatsLoaderPlayers | None = None,
    stats_loader_teams: StatsLoaderTeams | None = None,
    calc_features: CalcFeatures | None = None,
    update_schedules: bool = True,
    players_list: pd.DataFrame | None = None,
) -> tuple[StatsLoaderPlayers | None, StatsLoaderTeams | None, CalcFeatures | None]:
    """
    Run data collectors to bring season schedules, team boxscores, player boxscores, and calculated features up to date.
    Shared between standalone CLI (update_stats.py) and the daily pipeline in run_full_pipeline.py.
    """
    define_logger(format='%(asctime)s [%(levelname)s] - %(message)s', level=logging.INFO, print_to_console=True)
    target_season = season or get_current_season()
    logging.info(f'Starting stats update for NBA season {target_season}...')

    # 1. Season schedules
    if update_schedules and not player_only:
        logging.info(f'Updating season schedules for season {target_season}...')
        scrape_season_schedules(target_season, save_dir=update_dir_name('season_schedules'))
        update_todays_schedule_from_scoreboard(date=get_current_date(), season=target_season)

    if not team_only and players_list is None:
        logging.info(f'Loading active players list for season {target_season}...')
        players_list = load_and_cache_players_list(season=target_season)

    # 2. Team stats
    slt = stats_loader_teams
    if not player_only:
        logging.info(f'Updating team boxscores for season {target_season}...')
        if slt is None:
            slt = StatsLoaderTeams(season=target_season)
        slt.call()
        logging.info('Successfully updated team boxscores!')

    # 3. Player stats
    slp = stats_loader_players
    if not team_only:
        logging.info(f'Updating player boxscores for season {target_season}...')
        if slp is None:
            slp = StatsLoaderPlayers(season=target_season)
        slp.call(players_list=players_list, add_upcoming_games=add_upcoming_games)
        logging.info('Successfully updated player boxscores!')

    # 4. Features calculation
    cf = calc_features
    if with_features and not team_only:
        logging.info(f'Calculating model features for season {target_season}...')
        if cf is None:
            cf = CalcFeatures()
        cf.call(players_list=players_list, add_upcoming_games=add_upcoming_games)
        logging.info('Successfully updated features.feather!')

    logging.info('All stats and features updated successfully!')
    return slp, slt, cf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NBA Stats & Feature Data Updater")
    parser.add_argument('--season', type=int, default=None,
                        help="Target NBA season year (e.g. 2026 for 2026-27). Defaults to current season.")
    parser.add_argument('--no-features', action='store_true',
                        help="Skip feature calculation after updating boxscores.")
    parser.add_argument('--no-schedules', action='store_true',
                        help="Skip updating season schedules.")
    parser.add_argument('--add-upcoming', action='store_true',
                        help="Include today's upcoming games in updates.")
    parser.add_argument('--team-only', action='store_true',
                        help="Only update team boxscores.")
    parser.add_argument('--player-only', action='store_true',
                        help="Only update player boxscores.")
    args = parser.parse_args()

    update_all_stats(
        season=args.season,
        with_features=not args.no_features,
        add_upcoming_games=args.add_upcoming,
        team_only=args.team_only,
        player_only=args.player_only,
        update_schedules=not args.no_schedules,
    )
