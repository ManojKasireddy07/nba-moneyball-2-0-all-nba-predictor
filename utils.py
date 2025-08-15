"""
Utility functions for the Streamlit app.

These helpers centralize data loading and common calculations so that
``streamlit_app.py`` remains focused on UI layout and interactions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@lru_cache(maxsize=1)
def load_master(path: str = "data/processed/master.csv") -> pd.DataFrame:
    """Load the processed master dataset with caching."""
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_metrics(path: str = "reports/metrics_summary.json") -> dict:
    """Load aggregated metrics from JSON with caching."""
    import json

    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


@lru_cache(maxsize=1)
def load_global_importance(path: str = "reports/global_feature_importance.csv") -> pd.DataFrame:
    """Load global feature importance CSV with caching."""
    p = Path(path)
    return pd.read_csv(p)


@lru_cache(maxsize=1)
def load_predictions(path: str = "reports/preds_xgboost.csv") -> pd.DataFrame:
    """Load prediction probabilities from CSV with caching."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def get_player_row(df: pd.DataFrame, player: str, season: int) -> Optional[pd.Series]:
    """Return the row for a given player and season_end_year."""
    matches = df[(df["name"] == player) & (df["season_end_year"] == season)]
    if matches.empty:
        return None
    return matches.iloc[0]


def compute_all_nba_average(df: pd.DataFrame, season: int, columns: List[str]) -> pd.Series:
    """Compute the average of selected columns among All‑NBA players for a season."""
    mask = (df["season_end_year"] == season) & (df["was_all_nba"] == 1)
    return df.loc[mask, columns].mean()