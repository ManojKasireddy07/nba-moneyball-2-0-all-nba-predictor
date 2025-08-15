"""
Ingestion script for pulling NBA season data and All‑NBA selections from
Basketball‑Reference.  This module uses the community maintained
``basketball_reference_web_scraper`` library to download per‑season player
totals and advanced statistics.  It also includes a helper for scraping
historical All‑NBA teams directly from Basketball‑Reference.

Usage (via Makefile):

    python src/data/ingest_bref.py --start 2000 --end 2025 --raw-dir data/raw

On first run this will create ``data/raw`` if it doesn’t exist and
download two CSV files per season (basic totals and advanced stats)
alongside a single ``all_nba.csv`` summarizing award winners.

Note: Downloading data from Basketball‑Reference requires an internet
connection.  Respect their rate limits and fair‑use policy – you may
wish to insert ``time.sleep`` calls between requests for long loops.
"""

from __future__ import annotations

import argparse
import enum
import json
import os
import sys
from typing import Dict, List

import pandas as pd


def safe_enum_to_value(value):
    """Convert enumeration values (from basketball_reference_web_scraper) to strings.

    The scraper returns instances of Enum for teams, positions, etc.  To
    persist them to CSV we convert them to their ``name`` attribute.  Non‑enum
    values are returned unchanged.
    """
    if isinstance(value, enum.Enum):
        return value.name
    return value


def fetch_season_totals(season_end_year: int) -> pd.DataFrame:
    """Fetch basic per‑season totals for all players.

    Parameters
    ----------
    season_end_year : int
        The year in which the season ended (e.g. ``2023`` for the 2022‑23 season).

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per player and totals stats.
    """
    try:
        from basketball_reference_web_scraper import client  # type: ignore
    except ImportError:
        raise ImportError(
            "basketball_reference_web_scraper is required to fetch data. "
            "Install it via `pip install basketball_reference_web_scraper`."
        )

    records: List[Dict] = client.players_season_totals(season_end_year=season_end_year)
    df = pd.DataFrame(records)
    # Convert enum values to strings
    for col in df.columns:
        df[col] = df[col].apply(safe_enum_to_value)
    df["season_end_year"] = season_end_year
    return df


def fetch_season_advanced(season_end_year: int) -> pd.DataFrame:
    """Fetch advanced per‑season totals for all players.

    Parameters
    ----------
    season_end_year : int
        The year in which the season ended.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per player and advanced stats.
    """
    try:
        from basketball_reference_web_scraper import client  # type: ignore
    except ImportError:
        raise ImportError(
            "basketball_reference_web_scraper is required to fetch data. "
            "Install it via `pip install basketball_reference_web_scraper`."
        )

    records: List[Dict] = client.players_advanced_season_totals(
        season_end_year=season_end_year
    )
    df = pd.DataFrame(records)
    for col in df.columns:
        df[col] = df[col].apply(safe_enum_to_value)
    df["season_end_year"] = season_end_year
    return df


def scrape_all_nba() -> pd.DataFrame:
    """Scrape historical All‑NBA selections from Basketball‑Reference.

    The page https://www.basketball-reference.com/awards/all_league.html contains
    a table where each row corresponds to a season and lists the players on
    the First, Second and Third All‑NBA teams.  This function parses the
    tables into a tidy DataFrame with columns:

    ``season_end_year``, ``player``, ``team_rank`` (1, 2 or 3)
    """
    url = "https://www.basketball-reference.com/awards/all_league.html"
    # Pandas automatically uses lxml/html5lib under the hood.  A network
    # connection is required – if access fails, an HTTPError will be raised.
    tables = pd.read_html(url, header=0)
    # The first table is typically the All‑NBA teams; skip summary tables that
    # follow.  We locate the table with columns like 'Season', 'First Team', etc.
    all_league_table = None
    for tbl in tables:
        cols = [c.lower() for c in tbl.columns]
        if "first team" in cols:
            all_league_table = tbl
            break
    if all_league_table is None:
        raise RuntimeError("Could not locate All‑NBA table on awards page.")
    # Melt the table into long format: each team column becomes one row
    long_df = all_league_table.melt(
        id_vars=["Season"],
        value_vars=["First Team", "Second Team", "Third Team"],
        var_name="team",
        value_name="players",
    )
    # Expand the comma‑separated player names in each cell
    rows: List[Dict[str, str]] = []
    for _, row in long_df.iterrows():
        season = row["Season"]
        team_name = row["team"]
        # Extract numeric rank (e.g. 'First Team' -> 1)
        rank = {
            "First Team": 1,
            "Second Team": 2,
            "Third Team": 3,
        }.get(team_name, None)
        player_list = [p.strip() for p in str(row["players"]).split(",") if p]
        for player in player_list:
            rows.append(
                {
                    "season_end_year": int(season.split("-")[-1]),
                    "player": player,
                    "team_rank": rank,
                }
            )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    raw_dir = args.raw_dir
    os.makedirs(raw_dir, exist_ok=True)
    seasons = list(range(args.start, args.end + 1))
    for year in seasons:
        basic_path = os.path.join(raw_dir, f"{year}_totals.csv")
        adv_path = os.path.join(raw_dir, f"{year}_advanced.csv")
        if not os.path.exists(basic_path):
            print(f"Fetching totals for {year}…")
            df_totals = fetch_season_totals(year)
            df_totals.to_csv(basic_path, index=False)
        else:
            print(f"Totals for {year} already exist – skipping.")
        if not os.path.exists(adv_path):
            print(f"Fetching advanced stats for {year}…")
            df_adv = fetch_season_advanced(year)
            df_adv.to_csv(adv_path, index=False)
        else:
            print(f"Advanced stats for {year} already exist – skipping.")
    # Fetch All‑NBA selections
    all_nba_path = os.path.join(raw_dir, "all_nba.csv")
    if not os.path.exists(all_nba_path):
        print("Scraping All‑NBA selections…")
        try:
            df_all_nba = scrape_all_nba()
            df_all_nba.to_csv(all_nba_path, index=False)
        except Exception as e:
            print(f"Failed to scrape All‑NBA selections: {e}")
    else:
        print("All‑NBA selections already exist – skipping.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Basketball‑Reference data")
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start season (end year, e.g. 2000 for the 1999‑00 season)",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End season (end year, inclusive)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory to store raw season and All‑NBA data",
    )
    args = parser.parse_args()
    main(args)
