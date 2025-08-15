"""
Combine raw per‑season Basketball‑Reference data into a single master table.

This script merges basic totals, advanced statistics and award information for
each player‑season.  It adds engineered features, binary targets for
All‑NBA selection, and auxiliary fields such as ``Years_in_League``.  The
resulting DataFrame is saved to ``data/processed`` for downstream modeling.

Usage:

    python src/data/build_master.py --raw-dir data/raw --out-dir data/processed
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_season_files(raw_dir: Path) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Load all season CSVs from the raw directory into a nested dictionary.

    Returns a dictionary keyed by season_end_year.  Each value is another
    dictionary with keys ``totals`` and ``advanced`` containing the respective
    DataFrames.  Only years for which both files exist are included.
    """
    files = list(raw_dir.glob("*_totals.csv"))
    seasons: Dict[int, Dict[str, pd.DataFrame]] = {}
    for totals_file in files:
        year = int(totals_file.stem.split("_")[0])
        adv_file = raw_dir / f"{year}_advanced.csv"
        if not adv_file.exists():
            continue
        df_totals = pd.read_csv(totals_file)
        df_adv = pd.read_csv(adv_file)
        seasons[year] = {"totals": df_totals, "advanced": df_adv}
    return seasons


def merge_totals_advanced(df_totals: pd.DataFrame, df_adv: pd.DataFrame) -> pd.DataFrame:
    """Merge basic and advanced DataFrames for a single season.

    We join on ``name`` and ``team`` (if available) along with ``season_end_year``.
    Some fields may be duplicated; we suffix advanced columns with ``_adv``.
    """
    # Determine join keys
    join_keys = ["name", "season_end_year"]
    if "team" in df_totals.columns and "team" in df_adv.columns:
        join_keys.append("team")
    # Suffix overlapping columns from the advanced table
    overlapping = [c for c in df_adv.columns if c in df_totals.columns and c not in join_keys]
    df_adv_renamed = df_adv.rename(columns={c: f"{c}_adv" for c in overlapping})
    merged = pd.merge(df_totals, df_adv_renamed, on=join_keys, how="outer")
    return merged


def add_award_targets(df: pd.DataFrame, all_nba: pd.DataFrame) -> pd.DataFrame:
    """Add binary award targets to the master table.

    ``was_all_nba`` is 1 if the player made an All‑NBA team in year *Y*,
    ``target_all_nba_next`` is 1 if the player makes an All‑NBA team in year *Y+1*.
    """
    df = df.copy()
    # Normalize player names for joining: strip periods and accents
    df["player_lower"] = df["name"].str.lower().str.replace(".", "", regex=False)
    all_nba = all_nba.copy()
    all_nba["player_lower"] = all_nba["player"].str.lower().str.replace(".", "", regex=False)
    # Add was_all_nba
    df["was_all_nba"] = df.apply(
        lambda row: 1
        if ((all_nba["season_end_year"] == row["season_end_year"]) & (all_nba["player_lower"] == row["player_lower"]))
        .any()
        else 0,
        axis=1,
    )
    # Shift forward to create next season target
    df = df.sort_values(["name", "season_end_year"])
    df["target_all_nba_next"] = df.groupby("name")["was_all_nba"].shift(-1)
    df["target_all_nba_next"] = df["target_all_nba_next"].fillna(0).astype(int)
    df = df.drop(columns=["player_lower"])
    return df


def compute_years_in_league(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Years_in_League based on the first season a player appears.

    For each player, ``Years_in_League`` is defined as ``season_end_year - first_season_end_year``.
    Rookie seasons therefore have value 0.
    """
    df = df.copy()
    first_year = df.groupby("name")["season_end_year"].transform("min")
    df["Years_in_League"] = df["season_end_year"] - first_year
    return df


def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features required by the specification.

    The following interactions are computed when the requisite columns are present:

    * ``Points_x_Assists`` = ``points`` × ``assists`` (requires `points`/`PTS` and `assists`/`AST`)
    * ``Steals_x_Blocks`` = ``steals`` × ``blocks`` (requires `steals`/`STL` and `blocks`/`BLK`)
    * ``Usage_x_Efficiency`` = ``USAGE_PERCENTAGE`` × ``TRUE_SHOOTING_PERCENTAGE`` (or TS%)
    * ``Minutes_x_BoxImpact`` = ``minutes_played`` × ``box_plus_minus`` (MP × BPM)

    Missing columns are silently ignored.
    """
    df = df.copy()
    # Helper to fetch a column with possible aliases
    def get_column(possible: List[str]) -> str | None:
        for col in possible:
            if col in df.columns:
                return col
        return None

    # Points x Assists
    pts_col = get_column(["points", "PTS"])
    ast_col = get_column(["assists", "AST"])
    if pts_col and ast_col:
        df["Points_x_Assists"] = df[pts_col] * df[ast_col]
    # Steals x Blocks
    stl_col = get_column(["steals", "STL"])
    blk_col = get_column(["blocks", "BLK"])
    if stl_col and blk_col:
        df["Steals_x_Blocks"] = df[stl_col] * df[blk_col]
    # Usage x Efficiency
    usg_col = get_column(["USAGE_PERCENTAGE", "USG%", "USG_PERCENTAGE"])
    ts_col = get_column([
        "TRUE_SHOOTING_PERCENTAGE",
        "TS%",
        "TS_PERCENTAGE",
    ])
    if usg_col and ts_col:
        df["Usage_x_Efficiency"] = df[usg_col] * df[ts_col]
    # Minutes x BoxImpact
    mp_col = get_column(["minutes_played", "MP"])
    bpm_col = get_column(["box_plus_minus", "BPM"])
    if mp_col and bpm_col:
        df["Minutes_x_BoxImpact"] = df[mp_col] * df[bpm_col]
    return df


def impute_missing_rates(df: pd.DataFrame, minute_col: str = "MP", threshold: int = 250) -> pd.DataFrame:
    """Impute rate statistics for low‑minute players.

    Players with minutes below ``threshold`` may have NA or extreme rate stats.  For
    each numeric column we compute the median among players with minutes above the
    threshold and fill missing values for low‑minute players with that value.
    """
    df = df.copy()
    if minute_col not in df.columns:
        return df
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    high_min_mask = df[minute_col] >= threshold
    medians = df.loc[high_min_mask, numeric_cols].median()
    for col in numeric_cols:
        df.loc[~high_min_mask, col] = df.loc[~high_min_mask, col].fillna(medians[col])
    return df


def main(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load raw season data
    seasons = load_season_files(raw_dir)
    if not seasons:
        raise RuntimeError(f"No season files found in {raw_dir}")
    frames: List[pd.DataFrame] = []
    for year, tables in seasons.items():
        merged = merge_totals_advanced(tables["totals"], tables["advanced"])
        frames.append(merged)
    master = pd.concat(frames, ignore_index=True, sort=False)
    # Load All‑NBA selections if available
    all_nba_path = raw_dir / "all_nba.csv"
    if all_nba_path.exists():
        all_nba = pd.read_csv(all_nba_path)
        master = add_award_targets(master, all_nba)
    else:
        master["was_all_nba"] = 0
        master["target_all_nba_next"] = 0
    # Add experience and interactions
    master = compute_years_in_league(master)
    master = engineer_interactions(master)
    master = impute_missing_rates(master)
    # Sort chronologically and save
    master = master.sort_values(["season_end_year", "name"]).reset_index(drop=True)
    # Persist both CSV and Parquet for convenience
    master_csv = out_dir / "master.csv"
    master_parquet = out_dir / "master.parquet"
    master.to_csv(master_csv, index=False)
    master.to_parquet(master_parquet, index=False)
    # Write a simple data dictionary
    dict_path = out_dir / "data_dictionary.json"
    data_dict = {
        "columns": [
            {"name": c, "dtype": str(master[c].dtype)} for c in master.columns
        ]
    }
    with open(dict_path, "w") as f:
        json.dump(data_dict, f, indent=2)
    print(f"Master dataset saved to {master_csv} and {master_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master player‑season dataset")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw totals, advanced and All‑NBA CSVs",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Directory to write the processed master dataset",
    )
    args = parser.parse_args()
    main(args)