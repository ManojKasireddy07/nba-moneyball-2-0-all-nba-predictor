"""
Feature engineering utilities.

This module contains helper functions for scaling numeric features and
constructing train/test matrices.  Although scikit‑learn is not a hard
dependency of this project, these functions emulate similar functionality
using pandas and NumPy so that they can be executed in minimal
environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    """A simple column‑wise standardization utility.

    Attributes
    ----------
    means: Dict[str, float]
        The computed mean of each feature during ``fit``.
    stds: Dict[str, float]
        The computed standard deviation of each feature during ``fit``.
    """

    means: Dict[str, float]
    stds: Dict[str, float]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.means:
            std = self.stds[col]
            if std == 0:
                df[col] = df[col] - self.means[col]
            else:
                df[col] = (df[col] - self.means[col]) / std
        return df

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Iterable[str]) -> "StandardScaler":
        means = {}
        stds = {}
        for col in columns:
            means[col] = df[col].mean()
            stds[col] = df[col].std(ddof=0)
        return cls(means=means, stds=stds)


def scale_numeric_features(
    df: pd.DataFrame, numeric_cols: Iterable[str]
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize selected numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to be transformed.
    numeric_cols : Iterable[str]
        Columns to standardize.

    Returns
    -------
    Tuple[pandas.DataFrame, StandardScaler]
        The transformed DataFrame and the fitted scaler.
    """
    scaler = StandardScaler.fit(df, numeric_cols)
    df_scaled = scaler.transform(df)
    return df_scaled, scaler