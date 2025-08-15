"""
Compute model explanations using XGBoost’s built‑in contribution scores.

While the ``shap`` library provides a unified framework for model
interpretability, the environment used to develop this project may not
include it.  XGBoost exposes a ``pred_contribs`` flag on the booster’s
``predict`` method which returns SHAP‑like contribution values for each
feature plus a bias term.  We use these contributions to construct a
global feature importance bar chart based on the mean absolute value of
contributions across the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb

from ..features.feature_engineering import scale_numeric_features
from .train import prepare_features


def compute_global_importance(
    model: xgb.Booster, X: np.ndarray, feature_names: List[str]
) -> pd.DataFrame:
    """Compute mean absolute contribution of each feature.

    Parameters
    ----------
    model : xgboost.Booster
        Trained XGBoost booster.
    X : numpy.ndarray
        Feature matrix used to compute contributions.
    feature_names : list of str
        Names of the features corresponding to the columns of ``X``.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``feature`` and ``mean_abs_contrib`` sorted by
        descending importance.
    """
    contribs = model.predict(xgb.DMatrix(X), pred_contribs=True)
    # Last column is the bias term
    contribs = contribs[:, :-1]
    mean_abs = np.abs(contribs).mean(axis=0)
    df_importance = pd.DataFrame(
        {"feature": feature_names, "mean_abs_contrib": mean_abs}
    ).sort_values("mean_abs_contrib", ascending=False)
    return df_importance


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    master_path = data_dir / "master.csv"
    if not master_path.exists():
        raise RuntimeError(f"Master dataset not found at {master_path}")
    master = pd.read_csv(master_path)
    # Use all seasons up to the last for training the final model
    # Filter to numeric features and targets
    features_df, feature_cols = prepare_features(master)
    X = features_df.values
    y = master["target_all_nba_next"].values
    # Train a robust gradient boosted model on the full dataset
    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        n_jobs=4,
    )
    model.fit(X, y)
    # Compute global importance
    booster = model.get_booster()
    importance_df = compute_global_importance(booster, X, feature_cols)
    # Persist numeric importance
    importance_path = reports_dir / "global_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved global feature importance to {importance_path}")
    # Plot top 15 features using Plotly
    top_n = importance_df.head(15)
    fig = px.bar(
        top_n,
        x="mean_abs_contrib",
        y="feature",
        orientation="h",
        title="Global Feature Importance (Mean |Contribution|)",
        labels={"mean_abs_contrib": "Mean Absolute Contribution", "feature": "Feature"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    fig_path = fig_dir / "shap_global_importance.html"
    fig.write_html(str(fig_path))
    print(f"Saved SHAP global importance plot to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global feature importance")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing the master.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to write importance files",
    )
    args = parser.parse_args()
    main(args)
